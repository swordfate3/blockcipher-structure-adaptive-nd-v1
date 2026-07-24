from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_launch_gate_20260725"
SEED0_RUN_ID = "i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725"
SEED1_RUN_ID = "i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725"
SEED0_DECISION = "innovation1_rtg3a_skinny_formal_seed0_supported"
SEED0_TRAINING_COMMIT = "131e6387e164fb8679b1bcb9bf46887cf049fca0"
SEED0_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_formal_rtg3a_1000000_seed0.csv"
)
SEED1_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_formal_rtg3a_1000000_seed1.csv"
)
SEED1_REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_gpu0_20260725.json"
)
REQUIRED_SEED0_FILES = frozenset(
    {
        "checkpoint-verification.local.json",
        "curves.svg",
        "gate.local.json",
        "history.local.csv",
        "results.jsonl",
        "retrieved_from_verified_result_branch.marker",
        "validation.local.json",
        "visual_qa_passed.marker",
    }
)
EXPECTED_MODELS = (
    "skinny64_runtime_e4_equivariant_true",
    "skinny64_runtime_e4_equivariant_corrupted",
    "skinny64_runtime_e4_equivariant_independent",
)
EXPECTED_CHECKPOINTS = frozenset(
    {
        "row0001_skinny64_runtime_e4_equivariant_true_seed0.pt",
        "row0002_skinny64_runtime_e4_equivariant_corrupted_seed0.pt",
        "row0003_skinny64_runtime_e4_equivariant_independent_seed0.pt",
    }
)
REQUIRED_SOURCE_ASSETS = (
    SEED1_PLAN,
    SEED1_REMOTE_CONFIG,
    Path(
        "docs/experiments/"
        "innovation1-skinny64-runtime-e4-rtg3a-1000000-seed1-plan.md"
    ),
    Path(
        "configs/remote/generated/"
        "run_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "launch_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725.sh"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rtg3a_seed1_after_seed0_20260725.sh"
    ),
    Path("scripts/check-runtime-spn-skinny-rtg3a-seed1-launch"),
    Path(
        "src/blockcipher_nd/cli/"
        "check_runtime_spn_skinny_rtg3a_seed1_launch.py"
    ),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "runtime_spn_skinny_rtg3a_seed1_launch.py"
    ),
)
PROTECTED_TRAINING_PATHS = (
    "scripts/train",
    "scripts/gate-runtime-spn-skinny-medium",
    "src/blockcipher_nd/cli/train.py",
    "src/blockcipher_nd/cli/gate_runtime_spn_skinny_medium.py",
    "src/blockcipher_nd/engine",
    "src/blockcipher_nd/data",
    "src/blockcipher_nd/models",
    "src/blockcipher_nd/training",
    "src/blockcipher_nd/registry",
    "src/blockcipher_nd/ciphers/spn/skinny.py",
    "src/blockcipher_nd/tasks/innovation1/runtime_spn_skinny_attribution.py",
    "src/blockcipher_nd/tasks/innovation1/runtime_spn_skinny_medium.py",
    SEED0_PLAN.as_posix(),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_skinny_rtg3a_seed1_launch_gate(
    *,
    seed0_root: Path,
    repository: Path,
    source_commit: str,
    readiness_status: str,
    upstream_ref: str = "origin/main",
) -> dict[str, Any]:
    artifact_names = (
        {path.name for path in seed0_root.iterdir()} if seed0_root.is_dir() else set()
    )
    seed0_gate = _read_json(seed0_root / "gate.local.json")
    seed0_validation = _read_json(seed0_root / "validation.local.json")
    checkpoint_verification = _read_json(
        seed0_root / "checkpoint-verification.local.json"
    )
    results = _read_jsonl(seed0_root / "results.jsonl")
    history = _read_csv(seed0_root / "history.local.csv")
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
        else ["<training-commit-unavailable>"]
    )
    source_assets_committed = source_commit_exists and all(
        _git_ok(repository, "cat-file", "-e", f"{source_commit}:{path.as_posix()}")
        for path in REQUIRED_SOURCE_ASSETS
    )
    source_assets_match = source_assets_committed and all(
        _git_blob(repository, source_commit, path) == _read_bytes(repository / path)
        for path in REQUIRED_SOURCE_ASSETS
    )
    return adjudicate_runtime_spn_skinny_rtg3a_seed1_launch(
        source_commit=source_commit,
        upstream_ref=upstream_ref,
        artifact_names=artifact_names,
        seed0_gate=seed0_gate,
        seed0_validation=seed0_validation,
        checkpoint_verification=checkpoint_verification,
        results=results,
        history=history,
        readiness_status=readiness_status,
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_commit_published=source_commit_published,
        training_commit_exists=training_commit_exists,
        protected_changes=protected_changes,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        plans_match_seed_only=_plans_match_seed_only(
            repository / SEED0_PLAN,
            repository / SEED1_PLAN,
        ),
    )


def adjudicate_runtime_spn_skinny_rtg3a_seed1_launch(
    *,
    source_commit: str,
    upstream_ref: str,
    artifact_names: set[str],
    seed0_gate: dict[str, Any],
    seed0_validation: dict[str, Any],
    checkpoint_verification: dict[str, Any],
    results: list[dict[str, Any]],
    history: list[dict[str, str]],
    readiness_status: str,
    source_commit_valid: bool,
    source_commit_exists: bool,
    source_commit_published: bool,
    training_commit_exists: bool,
    protected_changes: list[str],
    source_assets_committed: bool,
    source_assets_match: bool,
    plans_match_seed_only: bool,
) -> dict[str, Any]:
    protocol_checks = seed0_gate.get("protocol_checks")
    research_checks = seed0_gate.get("research_checks")
    checkpoint_entries = checkpoint_verification.get("entries")
    evidence_checks = {
        "verified_seed0_artifacts_complete": REQUIRED_SEED0_FILES <= artifact_names,
        "seed0_gate_identity_exact": (
            seed0_gate.get("run_id") == SEED0_RUN_ID
            and seed0_gate.get("phase") == "rtg3a"
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
        "seed0_metrics_finite": _finite_map(
            seed0_gate.get("aucs"), ("true", "corrupted", "independent")
        )
        and _finite_map(
            seed0_gate.get("margins"),
            ("true_minus_corrupted", "true_minus_independent"),
        ),
        "seed0_local_validation_exact_three_rows": (
            seed0_validation.get("status") == "pass"
            and seed0_validation.get("expected_rows") == 3
            and seed0_validation.get("result_rows") == 3
            and seed0_validation.get("errors") == []
        ),
        "seed0_results_exact_three_roles": _results_exact(results),
        "seed0_history_exact_fifteen_rows": _history_exact(history),
        "seed0_checkpoint_replay_exact_three": (
            checkpoint_verification.get("status") == "pass"
            and checkpoint_verification.get("file_set_exact") is True
            and checkpoint_verification.get("errors") == []
            and set(checkpoint_verification.get("actual_files", []))
            == EXPECTED_CHECKPOINTS
            and set(checkpoint_verification.get("expected_files", []))
            == EXPECTED_CHECKPOINTS
            and isinstance(checkpoint_entries, list)
            and len(checkpoint_entries) == 3
            and {
                entry.get("filename")
                for entry in checkpoint_entries
                if isinstance(entry, dict)
            }
            == EXPECTED_CHECKPOINTS
            and all(_checkpoint_entry_passes(entry) for entry in checkpoint_entries)
        ),
    }
    readiness_checks = {
        "remote_disk_cache_readiness_pass": readiness_status == "pass",
        "seed0_training_commit_exists": training_commit_exists,
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
        "protected_training_paths_unchanged": not protected_changes,
        "seed1_plan_differs_only_by_seed_and_identity": plans_match_seed_only,
        "required_source_assets_committed": source_assets_committed,
        "committed_assets_match_worktree": source_assets_match,
    }
    publication_checks = {
        "source_commit_published_to_upstream": source_commit_published,
    }
    should_ssh = all(evidence_checks.values()) and all(readiness_checks.values())
    ssh_allowed = source_commit_published
    launch_authorized = should_ssh and ssh_allowed
    if launch_authorized:
        status = "pass"
        decision = "innovation1_rtg3a_seed1_remote_launch_authorized"
        next_action = (
            "start the committed RTG3-A seed1 local tmux monitor with this exact "
            "pushed commit"
        )
    elif should_ssh:
        status = "hold"
        decision = "innovation1_rtg3a_seed1_source_not_published"
        next_action = (
            "publish and verify the exact source commit through the configured Git "
            "remote; do not use an alternate transfer route"
        )
    else:
        status = "fail"
        decision = "innovation1_rtg3a_seed1_launch_evidence_invalid"
        next_action = (
            "repair only the failed seed0 evidence, protocol-equivalence, readiness, "
            "or publication checks; do not contact the remote"
        )
    return {
        "run_id": RUN_ID,
        "task": "innovation1_rtg3a_skinny_seed1_controlled_launch_gate",
        "remote_run_id": SEED1_RUN_ID,
        "seed0_run_id": SEED0_RUN_ID,
        "seed0_training_commit": SEED0_TRAINING_COMMIT,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "upstream_ref": upstream_ref,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "publication_checks": publication_checks,
        "protected_training_paths": list(PROTECTED_TRAINING_PATHS),
        "protected_changes": protected_changes,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact unless should_ssh and ssh_allowed are both true",
            "launch seed1 before verified-branch retrieval and local seed0 re-adjudication",
            "launch seed1 without exact three-row, fifteen-history-row checkpoint replay",
            "launch seed1 before visual-qa-redraw records visual_qa_passed.marker",
            "change any frozen data, model, optimizer, epoch, or control protocol",
            "launch from an unpublished commit or source overlay",
        ],
        "claim_scope": (
            "local RTG3-A seed1 launch authorization only; no second-seed result, "
            "two-seed formal conclusion, paper reproduction, attack, SOTA, "
            "breakthrough, or universal-SPN claim"
        ),
    }


def _plans_match_seed_only(seed0_path: Path, seed1_path: Path) -> bool:
    seed0_rows = _read_csv(seed0_path)
    seed1_rows = _read_csv(seed1_path)
    if len(seed0_rows) != 3 or len(seed1_rows) != 3:
        return False
    ignored = {"network", "family", "seed", "evidence", "literature"}
    for seed0, seed1 in zip(seed0_rows, seed1_rows, strict=True):
        if seed0.get("seed") != "0" or seed1.get("seed") != "1":
            return False
        if seed0.get("samples_per_class") != "1000000":
            return False
        fields = set(seed0) | set(seed1)
        if any(seed0.get(field) != seed1.get(field) for field in fields - ignored):
            return False
    return True


def _results_exact(results: list[dict[str, Any]]) -> bool:
    return (
        len(results) == 3
        and [row.get("model") for row in results] == list(EXPECTED_MODELS)
        and {row.get("seed") for row in results} == {0}
        and {row.get("samples_per_class") for row in results} == {1_000_000}
    )


def _history_exact(history: list[dict[str, str]]) -> bool:
    if len(history) != 15:
        return False
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in history:
        by_model.setdefault(row.get("model", ""), []).append(row)
    if set(by_model) != set(EXPECTED_MODELS):
        return False
    for model in EXPECTED_MODELS:
        rows = by_model[model]
        try:
            epochs = [int(float(row["epoch"])) for row in rows]
        except (KeyError, TypeError, ValueError):
            return False
        if epochs != [1, 2, 3, 4, 5] or {row.get("seed") for row in rows} != {"0"}:
            return False
    return True


def _checkpoint_entry_passes(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    checks = value.get("checks")
    return (
        value.get("filename") in EXPECTED_CHECKPOINTS
        and value.get("model") in EXPECTED_MODELS
        and isinstance(value.get("sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]))
        and isinstance(checks, dict)
        and bool(checks)
        and all(check is True for check in checks.values())
    )


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []
    return values if all(isinstance(value, dict) for value in values) else []


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _git_ok(repository: Path, *args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


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


def _git_blob(repository: Path, commit: str, path: Path) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


__all__ = [
    "RUN_ID",
    "adjudicate_runtime_spn_skinny_rtg3a_seed1_launch",
    "build_runtime_spn_skinny_rtg3a_seed1_launch_gate",
]
