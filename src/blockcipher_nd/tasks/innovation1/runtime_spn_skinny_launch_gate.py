from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_rtg2a_skinny64_general_gf2_medium_65536_seed1_launch_gate_20260724"
SEED0_RUN_ID = "i1_rtg2a_skinny64_general_gf2_medium_65536_seed0_20260724"
SEED1_RUN_ID = "i1_rtg2a_skinny64_general_gf2_medium_65536_seed1_20260724"
SEED0_TRAINING_COMMIT = "81d0a67ff93385083ea71c1b030831809971ded3"
SEED0_DECISION = "innovation1_rtg2a_skinny_medium_seed0_supported"
SEED0_PLAN = (
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_medium_rtg2a_65536_seed0.csv"
)
SEED1_PLAN = (
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_medium_rtg2a_65536_seed1.csv"
)
SEED1_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_rtg2a_skinny64_general_gf2_medium_65536_seed1_gpu0_20260724.json"
)
REQUIRED_FALLBACK_FILES = frozenset(
    {
        "RAW_RETRIEVAL_NOTICE.txt",
        "gate.json",
        "results.jsonl",
        "validation.local.json",
        "visual_qa_passed.marker",
    }
)
PROTECTED_TRAINING_PATHS = (
    "scripts/train",
    "src/blockcipher_nd/cli/train.py",
    "src/blockcipher_nd/engine",
    "src/blockcipher_nd/data",
    "src/blockcipher_nd/models",
    "src/blockcipher_nd/training",
    "src/blockcipher_nd/registry",
    "src/blockcipher_nd/ciphers/spn/skinny.py",
    SEED0_PLAN,
    SEED1_PLAN,
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_skinny_seed1_launch_gate(
    *,
    seed0_root: Path,
    repository: Path,
    source_commit: str,
    upstream_ref: str = "origin/main",
) -> dict[str, Any]:
    artifact_names = {path.name for path in seed0_root.iterdir()} if seed0_root.is_dir() else set()
    seed0_gate = _read_json(seed0_root / "gate.json")
    seed0_validation = _read_json(seed0_root / "validation.local.json")
    source_commit_valid = bool(_COMMIT_RE.fullmatch(source_commit))
    source_commit_exists = source_commit_valid and _git_ok(
        repository, "cat-file", "-e", f"{source_commit}^{{commit}}"
    )
    upstream_exists = _git_ok(repository, "rev-parse", "--verify", upstream_ref)
    source_commit_published = (
        source_commit_exists
        and upstream_exists
        and _git_ok(
            repository,
            "merge-base",
            "--is-ancestor",
            source_commit,
            upstream_ref,
        )
    )
    training_commit_exists = _git_ok(
        repository, "cat-file", "-e", f"{SEED0_TRAINING_COMMIT}^{{commit}}"
    )
    protected_changes = (
        _git_lines(
            repository,
            "diff",
            "--name-only",
            f"{SEED0_TRAINING_COMMIT}..{source_commit}",
            "--",
            *PROTECTED_TRAINING_PATHS,
        )
        if source_commit_exists and training_commit_exists
        else []
    )
    seed1_plan_identical = _git_blob_equal(
        repository,
        SEED0_TRAINING_COMMIT,
        source_commit,
        SEED1_PLAN,
    )
    seed1_remote_protocol_identical = _json_blob_equal_except(
        repository,
        SEED0_TRAINING_COMMIT,
        source_commit,
        SEED1_REMOTE_CONFIG,
        ignored_keys={"launch_policy"},
    )
    return adjudicate_runtime_spn_skinny_seed1_launch(
        source_commit=source_commit,
        upstream_ref=upstream_ref,
        artifact_names=artifact_names,
        seed0_gate=seed0_gate,
        seed0_validation=seed0_validation,
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_commit_published=source_commit_published,
        training_commit_exists=training_commit_exists,
        protected_changes=protected_changes,
        seed1_plan_identical=seed1_plan_identical,
        seed1_remote_protocol_identical=seed1_remote_protocol_identical,
    )


def adjudicate_runtime_spn_skinny_seed1_launch(
    *,
    source_commit: str,
    upstream_ref: str,
    artifact_names: set[str],
    seed0_gate: dict[str, Any],
    seed0_validation: dict[str, Any],
    source_commit_valid: bool,
    source_commit_exists: bool,
    source_commit_published: bool,
    training_commit_exists: bool,
    protected_changes: list[str],
    seed1_plan_identical: bool,
    seed1_remote_protocol_identical: bool,
) -> dict[str, Any]:
    aucs = seed0_gate.get("aucs")
    margins = seed0_gate.get("margins")
    protocol_checks = seed0_gate.get("protocol_checks")
    research_checks = seed0_gate.get("research_checks")
    evidence_checks = {
        "fallback_artifacts_complete": REQUIRED_FALLBACK_FILES <= artifact_names,
        "seed0_gate_identity_exact": (
            seed0_gate.get("run_id") == SEED0_RUN_ID
            and seed0_gate.get("seed") == 0
            and seed0_gate.get("status") == "pass"
            and seed0_gate.get("decision") == SEED0_DECISION
        ),
        "seed0_protocol_checks_pass": (
            isinstance(protocol_checks, dict)
            and bool(protocol_checks)
            and all(value is True for value in protocol_checks.values())
        ),
        "seed0_research_checks_pass": (
            isinstance(research_checks, dict)
            and bool(research_checks)
            and all(value is True for value in research_checks.values())
        ),
        "seed0_metrics_finite": _finite_map(aucs, ("true", "corrupted", "independent"))
        and _finite_map(margins, ("true_minus_corrupted", "true_minus_independent")),
        "seed0_local_validation_pass": (
            seed0_validation.get("status") == "pass"
            and seed0_validation.get("expected_rows") == 3
            and seed0_validation.get("result_rows") == 3
            and seed0_validation.get("errors") == []
        ),
    }
    equivalence_checks = {
        "seed0_training_commit_exists": training_commit_exists,
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
        "protected_training_paths_unchanged": not protected_changes,
        "seed1_plan_byte_identical": seed1_plan_identical,
        "seed1_remote_protocol_identical": seed1_remote_protocol_identical,
    }
    publication_checks = {
        "source_commit_published_to_upstream": source_commit_published,
    }
    should_ssh = all(evidence_checks.values()) and all(equivalence_checks.values())
    ssh_allowed = source_commit_published
    launch_authorized = should_ssh and ssh_allowed
    if launch_authorized:
        status = "pass"
        decision = "innovation1_rtg2a_seed1_remote_launch_authorized"
        next_action = (
            "start the committed local tmux watcher with this exact pushed source commit"
        )
    elif should_ssh:
        status = "hold"
        decision = "innovation1_rtg2a_seed1_source_not_published"
        next_action = (
            "publish the exact source commit through the normal configured Git remote; do not "
            "use a dirty overlay, alternate transfer route, or remote launch from unpublished code"
        )
    else:
        status = "fail"
        decision = "innovation1_rtg2a_seed1_launch_evidence_invalid"
        next_action = "repair only the failed local launch checks before any SSH contact"
    return {
        "run_id": RUN_ID,
        "task": "innovation1_rtg2a_skinny_seed1_controlled_launch_gate",
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "upstream_ref": upstream_ref,
        "seed0_training_commit": SEED0_TRAINING_COMMIT,
        "seed0_run_id": SEED0_RUN_ID,
        "seed1_run_id": SEED1_RUN_ID,
        "evidence_checks": evidence_checks,
        "equivalence_checks": equivalence_checks,
        "publication_checks": publication_checks,
        "protected_training_paths": list(PROTECTED_TRAINING_PATHS),
        "protected_changes": protected_changes,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact unless should_ssh and ssh_allowed are both true",
            "remote launch from an unpublished commit",
            "scp or dirty-overlay source publication",
            "change the frozen seed1 data, model, training, or plan protocol",
        ],
        "claim_scope": (
            "local RTG2-A seed1 launch authorization only; no remote contact, training, "
            "research result, formal-scale claim, attack, SOTA, or breakthrough"
        ),
    }


def _finite_map(value: object, keys: tuple[str, ...]) -> bool:
    return isinstance(value, dict) and all(
        _is_finite_number(value.get(key)) for key in keys
    )


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _git_ok(repository: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _git_lines(repository: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["<git-diff-failed>"]
    return [line for line in result.stdout.splitlines() if line]


def _git_blob(repository: Path, commit: str, path: str) -> bytes | None:
    if not _COMMIT_RE.fullmatch(commit):
        return None
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_blob_equal(repository: Path, left: str, right: str, path: str) -> bool:
    left_blob = _git_blob(repository, left, path)
    right_blob = _git_blob(repository, right, path)
    return left_blob is not None and left_blob == right_blob


def _json_blob_equal_except(
    repository: Path,
    left: str,
    right: str,
    path: str,
    *,
    ignored_keys: set[str],
) -> bool:
    try:
        left_value = json.loads((_git_blob(repository, left, path) or b"").decode("utf-8"))
        right_value = json.loads((_git_blob(repository, right, path) or b"").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(left_value, dict) or not isinstance(right_value, dict):
        return False
    for key in ignored_keys:
        left_value.pop(key, None)
        right_value.pop(key, None)
    return left_value == right_value
