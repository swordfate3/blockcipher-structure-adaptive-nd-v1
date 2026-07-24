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
    adjudicate_runtime_spn_rectangle_attribution,
)


RUN_ID = "i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_launch_gate_20260725"
REMOTE_RUN_ID = "i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725"
RCT1_RUN_ID = (
    "i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725"
)
RCT1_DECISION = "innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported"
RCT1_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_noncontiguous_attribution_"
    "rct1_2048_seed0_seed1.csv"
)
RCT2_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_rct2_rectangle80_runtime_e4_medium_65536_seed0_gpu0_20260725.json"
)
REQUIRED_SOURCE_ASSETS = (
    RCT1_PLAN,
    RCT2_PLAN,
    Path("configs/runtime/spn/rectangle64.json"),
    Path(
        "docs/experiments/innovation1-rectangle80-runtime-e4-rct2-65536-seed0-plan.md"
    ),
    REMOTE_CONFIG,
    Path(
        "configs/remote/generated/"
        "run_i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "launch_i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rct2_rectangle80_runtime_e4_medium_65536_seed0_20260725.sh"
    ),
    Path("configs/remote/generated/monitor_i1_rct2_after_rtg3a_20260725.sh"),
    Path("scripts/check-runtime-spn-rectangle-rct2-launch"),
    Path("scripts/gate-runtime-spn-rectangle-medium"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_rectangle_rct2_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_rectangle_medium.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_rct2_launch.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_attribution.py"),
)
PROTECTED_SOURCE_PATHS = (
    RCT1_PLAN,
    RCT2_PLAN,
    Path("configs/runtime/spn/rectangle64.json"),
    REMOTE_CONFIG,
    Path("scripts/train"),
    Path("scripts/check-runtime-spn-rectangle-rct2-launch"),
    Path("scripts/gate-runtime-spn-rectangle-medium"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_rectangle_rct2_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_rectangle_medium.py"),
    Path("src/blockcipher_nd/data"),
    Path("src/blockcipher_nd/engine"),
    Path("src/blockcipher_nd/models/structure/spn/runtime_parameterized.py"),
    Path("src/blockcipher_nd/models/structure/spn/runtime_structure.py"),
    Path("src/blockcipher_nd/planning/matrix.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_attribution.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_rectangle_rct2_launch.py"),
    Path("src/blockcipher_nd/training"),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_rectangle_rct2_launch_gate(
    *,
    rct1_root: Path,
    repository: Path,
    source_commit: str,
    readiness_status: str,
    rtg3_session_count: int,
    upstream_ref: str = "origin/main",
) -> dict[str, Any]:
    stored_gate = _read_json(rct1_root / "gate.json")
    validation = _read_json(rct1_root / "validation.json")
    rows = _read_jsonl(rct1_root / "results.jsonl")
    recomputed_gate = adjudicate_runtime_spn_rectangle_attribution(
        run_id=RCT1_RUN_ID,
        rows=rows,
    )
    rct1_authority = {
        "gate_identity_exact": (
            stored_gate.get("run_id") == RCT1_RUN_ID
            and stored_gate.get("status") == "pass"
            and stored_gate.get("decision") == RCT1_DECISION
        ),
        "gate_recomputed_exact": bool(rows) and stored_gate == recomputed_gate,
        "validation_exact": (
            validation.get("run_id") == RCT1_RUN_ID
            and validation.get("status") == "pass"
            and validation.get("checks") == stored_gate.get("protocol_checks")
        ),
        "visual_qa_passed": (rct1_root / "visual_qa_passed.marker").is_file(),
        "results_sha256": _sha256_file(rct1_root / "results.jsonl"),
        "gate_sha256": _sha256_file(rct1_root / "gate.json"),
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

    return adjudicate_runtime_spn_rectangle_rct2_launch(
        source_commit=source_commit,
        upstream_ref=upstream_ref,
        rct1_authority=rct1_authority,
        readiness_status=readiness_status,
        rtg3_session_count=rtg3_session_count,
        plans_match_scale_only=_plans_match_scale_only(
            repository / RCT1_PLAN,
            repository / RCT2_PLAN,
        ),
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_commit_published=source_commit_published,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        protected_paths_unchanged=protected_paths_unchanged,
        protected_worktree_clean=protected_worktree_clean,
    )


def adjudicate_runtime_spn_rectangle_rct2_launch(
    *,
    source_commit: str,
    upstream_ref: str,
    rct1_authority: dict[str, Any],
    readiness_status: str,
    rtg3_session_count: int,
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
        for key, value in rct1_authority.items()
        if key not in {"results_sha256", "gate_sha256"}
    }
    evidence_checks = {
        "rct1_authority_complete": bool(authority_checks)
        and all(authority_checks.values()),
        "rct2_matches_rct1_except_scale_and_identity": plans_match_scale_only,
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
    lane_checks = {
        "rtg3_session_count_nonnegative": rtg3_session_count >= 0,
        "rtg3_remote_lane_released": rtg3_session_count == 0,
    }
    publication_checks = {
        "source_commit_published_to_upstream": source_commit_published,
    }
    local_evidence_valid = all(evidence_checks.values()) and all(
        readiness_checks.values()
    )
    lane_state_valid = lane_checks["rtg3_session_count_nonnegative"]
    should_ssh = (
        local_evidence_valid
        and lane_state_valid
        and lane_checks["rtg3_remote_lane_released"]
    )
    ssh_allowed = source_commit_published
    launch_authorized = should_ssh and ssh_allowed

    if not local_evidence_valid or not lane_state_valid:
        status = "fail"
        decision = "innovation1_rct2_rectangle_launch_evidence_invalid"
        next_action = "repair only the failed local authority or readiness check"
    elif not lane_checks["rtg3_remote_lane_released"]:
        status = "hold"
        decision = "innovation1_rct2_rectangle_waiting_for_rtg3_remote_lane"
        next_action = (
            "wait until every local i1_rtg3a tmux session exits; do not contact "
            "the remote host for RCT2"
        )
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_rct2_rectangle_source_not_published"
        next_action = (
            "publish and verify the exact source commit; do not use scp or a "
            "dirty source overlay"
        )
    else:
        status = "pass"
        decision = "innovation1_rct2_rectangle_remote_launch_authorized"
        next_action = (
            "bootstrap the run-owned clean clone from this exact commit, launch "
            "RCT2 on GPU0, confirm a durable start artifact, and hand off to the "
            "local result watcher"
        )

    return {
        "run_id": RUN_ID,
        "task": "innovation1_rct2_rectangle_controlled_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "upstream_ref": upstream_ref,
        "remote_config_readiness": readiness_status,
        "rct1_authority": rct1_authority,
        "rtg3_session_count": rtg3_session_count,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "lane_checks": lane_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact while any i1_rtg3a tmux session remains",
            "launch from the historical dirty remote clone",
            "launch from an unpublished or worktree-drifted source",
            "scp or dirty-overlay source publication",
            "change the frozen RCT1 model, data, controls, epochs, or thresholds",
            "advance to seed1 or 262144/class before the retrieved seed0 gate",
        ],
        "claim_scope": (
            "local RCT2 seed0 launch authorization only; no remote result, formal "
            "scale, attack, SOTA, breakthrough, or universal-SPN claim"
        ),
    }


def _plans_match_scale_only(rct1_path: Path, rct2_path: Path) -> bool:
    try:
        with rct1_path.open(newline="", encoding="utf-8") as handle:
            rct1_rows = list(csv.DictReader(handle))
        with rct2_path.open(newline="", encoding="utf-8") as handle:
            rct2_rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(rct1_rows) != 6 or len(rct2_rows) != 3:
        return False
    rct1_seed0 = {
        row.get("model_key"): row for row in rct1_rows if row.get("seed") == "0"
    }
    rct2_by_model = {row.get("model_key"): row for row in rct2_rows}
    if set(rct1_seed0) != set(MODELS.values()) or set(rct2_by_model) != set(
        MODELS.values()
    ):
        return False
    ignored = {"network", "family", "samples_per_class", "evidence", "literature"}
    for model in MODELS.values():
        local = rct1_seed0[model]
        medium = rct2_by_model[model]
        if local.get("samples_per_class") != "2048":
            return False
        if medium.get("samples_per_class") != "65536":
            return False
        fields = set(local) | set(medium)
        if any(local.get(field) != medium.get(field) for field in fields - ignored):
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
    "adjudicate_runtime_spn_rectangle_rct2_launch",
    "build_runtime_spn_rectangle_rct2_launch_gate",
]
