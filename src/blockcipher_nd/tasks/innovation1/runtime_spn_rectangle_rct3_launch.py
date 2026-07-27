from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_attribution import (
    MODELS,
    adjudicate_runtime_spn_rectangle_scale,
)


RUN_ID = "i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_launch_gate_20260727"
REMOTE_RUN_ID = "i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727"
RCT2_RUN_ID = "i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725"
RCT2_DECISION = "innovation1_rct2_rectangle_medium_seed_passed"
RCT2_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv"
)
RCT3_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_scale_rct3_262144_seed0.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_rct3_rectangle80_runtime_e4_scale_262144_seed0_gpu1_20260727.json"
)
REQUIRED_SOURCE_ASSETS = (
    RCT2_PLAN,
    RCT3_PLAN,
    Path("configs/runtime/spn/rectangle64.json"),
    Path(
        "docs/experiments/innovation1-rectangle80-runtime-e4-rct3-262144-seed0-plan.md"
    ),
    REMOTE_CONFIG,
    Path(
        "configs/remote/generated/"
        "run_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "launch_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.sh"
    ),
    Path("configs/remote/generated/monitor_i1_rct3_after_rct2_20260727.sh"),
    Path("scripts/check-runtime-spn-rectangle-rct3-launch"),
    Path("scripts/gate-runtime-spn-rectangle-medium"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_rectangle_rct3_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_rectangle_medium.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_attribution.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_rct3_launch.py"),
)
PROTECTED_SOURCE_PATHS = (
    RCT2_PLAN,
    RCT3_PLAN,
    Path("configs/runtime/spn/rectangle64.json"),
    REMOTE_CONFIG,
    Path("scripts/train"),
    Path("scripts/check-runtime-spn-rectangle-rct3-launch"),
    Path("scripts/gate-runtime-spn-rectangle-medium"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_rectangle_rct3_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_rectangle_medium.py"),
    Path("src/blockcipher_nd/data"),
    Path("src/blockcipher_nd/engine"),
    Path("src/blockcipher_nd/models/structure/spn/runtime_parameterized.py"),
    Path("src/blockcipher_nd/models/structure/spn/runtime_structure.py"),
    Path("src/blockcipher_nd/planning/matrix.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_attribution.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_rct3_launch.py"),
    Path("src/blockcipher_nd/training"),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_rectangle_rct3_launch_gate(
    *,
    rct2_root: Path,
    repository: Path,
    source_commit: str,
    readiness_status: str,
    upstream_ref: str = "origin/main",
) -> dict[str, Any]:
    stored_gate = _read_json(rct2_root / "gate.local.json")
    validation = _read_json(rct2_root / "validation.local.json")
    rows = _read_jsonl(rct2_root / "results.jsonl")
    recomputed_gate = adjudicate_runtime_spn_rectangle_scale(
        run_id=RCT2_RUN_ID,
        rows=rows,
        expected_seed=0,
        phase="rct2",
    )
    rct2_authority = {
        "gate_identity_exact": (
            stored_gate.get("run_id") == RCT2_RUN_ID
            and stored_gate.get("status") == "pass"
            and stored_gate.get("decision") == RCT2_DECISION
        ),
        "gate_recomputed_exact": bool(rows) and stored_gate == recomputed_gate,
        "protocol_checks_pass": bool(stored_gate.get("protocol_checks"))
        and all(stored_gate.get("protocol_checks", {}).values()),
        "research_checks_pass": bool(stored_gate.get("research_checks"))
        and all(stored_gate.get("research_checks", {}).values()),
        "plan_validation_pass": (
            validation.get("status") == "pass"
            and validation.get("expected_rows") == 3
            and validation.get("result_rows") == 3
            and validation.get("errors") == []
        ),
        "verified_result_branch_retrieval": (
            rct2_root / "retrieved_from_verified_result_branch.marker"
        ).is_file(),
        "visual_qa_passed": (rct2_root / "visual_qa_passed.marker").is_file(),
        "results_sha256": _sha256_file(rct2_root / "results.jsonl"),
        "gate_sha256": _sha256_file(rct2_root / "gate.local.json"),
    }

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
    source_assets_committed = source_commit_exists and all(
        _git_ok(repository, "cat-file", "-e", f"{source_commit}:{path.as_posix()}")
        for path in REQUIRED_SOURCE_ASSETS
    )
    source_assets_match = source_assets_committed and all(
        _git_blob(repository, source_commit, path) == _read_bytes(repository / path)
        for path in REQUIRED_SOURCE_ASSETS
    )
    protected_paths_unchanged = source_commit_exists and _git_ok(
        repository,
        "diff",
        "--quiet",
        f"{source_commit}..HEAD",
        "--",
        *(path.as_posix() for path in PROTECTED_SOURCE_PATHS),
    )
    protected_worktree_clean = (
        source_commit_exists
        and not _git_output(
            repository,
            "status",
            "--porcelain",
            "--",
            *(path.as_posix() for path in PROTECTED_SOURCE_PATHS),
        ).strip()
    )

    return adjudicate_runtime_spn_rectangle_rct3_launch(
        source_commit=source_commit,
        upstream_ref=upstream_ref,
        rct2_authority=rct2_authority,
        readiness_status=readiness_status,
        plans_match_scale_only=_plans_match_scale_only(
            repository / RCT2_PLAN,
            repository / RCT3_PLAN,
        ),
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_commit_published=source_commit_published,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        protected_paths_unchanged=protected_paths_unchanged,
        protected_worktree_clean=protected_worktree_clean,
    )


def adjudicate_runtime_spn_rectangle_rct3_launch(
    *,
    source_commit: str,
    upstream_ref: str,
    rct2_authority: dict[str, Any],
    readiness_status: str,
    plans_match_scale_only: bool,
    source_commit_valid: bool,
    source_commit_exists: bool,
    source_commit_published: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    protected_paths_unchanged: bool,
    protected_worktree_clean: bool,
) -> dict[str, Any]:
    authority_checks = {
        key: value is True
        for key, value in rct2_authority.items()
        if key not in {"results_sha256", "gate_sha256"}
    }
    evidence_checks = {
        "rct2_authority_complete": bool(authority_checks)
        and all(authority_checks.values()),
        "rct3_matches_rct2_except_scale_and_identity": plans_match_scale_only,
    }
    readiness_checks = {
        "remote_config_readiness_pass": readiness_status == "pass",
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
        "required_source_assets_committed": source_assets_committed,
        "committed_assets_match_worktree": source_assets_match,
        "protected_training_paths_unchanged": protected_paths_unchanged,
        "protected_worktree_clean": protected_worktree_clean,
    }
    publication_checks = {
        "source_commit_published_to_upstream": source_commit_published,
    }
    local_evidence_valid = all(evidence_checks.values()) and all(
        readiness_checks.values()
    )
    should_ssh = local_evidence_valid
    ssh_allowed = source_commit_published
    launch_authorized = should_ssh and ssh_allowed

    if not local_evidence_valid:
        status = "fail"
        decision = "innovation1_rct3_rectangle_launch_evidence_invalid"
        next_action = (
            "wait for or repair only the failed local RCT2 authority, RCT3 plan, "
            "readiness, or protected-source check; do not contact the remote host"
        )
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_rct3_rectangle_source_not_published"
        next_action = (
            "publish and verify the exact source commit; do not use scp or a "
            "dirty source overlay"
        )
    else:
        status = "pass"
        decision = "innovation1_rct3_rectangle_remote_launch_authorized"
        next_action = (
            "bootstrap the run-owned clean clone from this exact commit, launch "
            "RCT3 on GPU1, confirm a durable start artifact, and hand off to the "
            "local result watcher"
        )

    return {
        "run_id": RUN_ID,
        "task": "innovation1_rct3_rectangle_controlled_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "upstream_ref": upstream_ref,
        "remote_config_readiness": readiness_status,
        "rct2_authority": rct2_authority,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact before the RCT2 gate, retrieval, validation, and visual QA pass",
            "launch from the historical dirty remote clone",
            "launch from an unpublished or worktree-drifted source",
            "scp or dirty-overlay source publication",
            "change the frozen RCT2 model, data, controls, epochs, or thresholds",
            "add seed1 before the single-seed RCT3 result is adjudicated",
            "call 262144/class formal or paper-scale evidence",
        ],
        "claim_scope": (
            "local RCT3 seed0 launch authorization only; no remote result, formal "
            "scale, attack, SOTA, breakthrough, or universal-SPN claim"
        ),
    }


def _plans_match_scale_only(rct2_path: Path, rct3_path: Path) -> bool:
    try:
        with rct2_path.open(newline="", encoding="utf-8") as handle:
            rct2_rows = list(csv.DictReader(handle))
        with rct3_path.open(newline="", encoding="utf-8") as handle:
            rct3_rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(rct2_rows) != 3 or len(rct3_rows) != 3:
        return False
    rct2_by_model = {row.get("model_key"): row for row in rct2_rows}
    rct3_by_model = {row.get("model_key"): row for row in rct3_rows}
    if set(rct2_by_model) != set(MODELS.values()) or set(rct3_by_model) != set(
        MODELS.values()
    ):
        return False
    ignored = {"network", "family", "samples_per_class", "evidence", "literature"}
    for model in MODELS.values():
        medium = rct2_by_model[model]
        scale = rct3_by_model[model]
        if medium.get("samples_per_class") != "65536":
            return False
        if scale.get("samples_per_class") != "262144":
            return False
        fields = set(medium) | set(scale)
        if any(medium.get(field) != scale.get(field) for field in fields - ignored):
            return False
    return True


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
    return [value for value in values if isinstance(value, dict)]


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


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


def _git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else "__git_command_failed__"


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
    "adjudicate_runtime_spn_rectangle_rct3_launch",
    "build_runtime_spn_rectangle_rct3_launch_gate",
]
