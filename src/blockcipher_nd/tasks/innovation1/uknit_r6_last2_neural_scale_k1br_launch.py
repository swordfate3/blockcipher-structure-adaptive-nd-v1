from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


RUN_ID = "i1_uknit_r6_last2_neural_scale_k1br_launch_gate_20260730"
REMOTE_RUN_ID = "i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730"
K1BP_RUN_ID = "i1_uknit_r6_last_round_key_hypothesis_k1bp_seed2_seed3_seed4_20260730"
K1BP_DECISION = "innovation1_uknit_r6_k1bp_single_cell_sparse_anchor_not_supported"
K1BP_GATE_SHA256 = "846e53ed0e68077d6c103faff2c4849241043e35dce64ccaca9ec29c19df0735"
PLAN = Path(
    "configs/experiment/innovation1/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3_gpu1_20260730.json"
)
GENERATED = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    PLAN,
    REMOTE_CONFIG,
    Path("docs/experiments/innovation1-uknit-r6-last2-neural-scale-k1br-plan.md"),
    GENERATED / "run_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.cmd",
    GENERATED / "launch_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.cmd",
    GENERATED / "monitor_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.sh",
    Path("scripts/check-uknit-r6-last2-neural-scale-k1br-launch"),
    Path("scripts/gate-uknit-r6-last2-neural-scale-k1br"),
    Path("scripts/package-uknit-r6-last2-neural-scale-k1br"),
    Path("scripts/plot-uknit-r6-last2-neural-scale-k1br"),
    Path("src/blockcipher_nd/cli/check_uknit_r6_last2_neural_scale_k1br_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_r6_last2_neural_scale_k1br.py"),
    Path("src/blockcipher_nd/cli/package_uknit_r6_last2_neural_scale_k1br.py"),
    Path("src/blockcipher_nd/cli/plot_uknit_r6_last2_neural_scale_k1br.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r6_last2_neural_scale_k1br.py"),
    Path(
        "src/blockcipher_nd/tasks/innovation1/uknit_r6_last2_neural_scale_k1br_launch.py"
    ),
    Path("configs/runtime/spn/uknit64.json"),
    Path("src/blockcipher_nd/models/structure/spn/position_histogram_residual.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
)
PROTECTED_SOURCE_PATHS = (
    *REQUIRED_SOURCE_ASSETS,
    Path("scripts/train"),
    Path("scripts/validate-results"),
    Path("src/blockcipher_nd/data/cache"),
    Path("src/blockcipher_nd/engine"),
    Path("src/blockcipher_nd/training"),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_k1br_launch_gate(
    *,
    k1bp_root: Path,
    repository: Path,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
) -> dict[str, Any]:
    source_gate = _read_json(k1bp_root / "gate.json")
    source_authority = {
        "gate_identity_exact": (
            source_gate.get("run_id") == K1BP_RUN_ID
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == K1BP_DECISION
            and not source_gate.get("failed_protocol_checks")
        ),
        "gate_digest_exact": _sha256_file(k1bp_root / "gate.json") == K1BP_GATE_SHA256,
    }
    config = _read_json(repository / REMOTE_CONFIG)
    exception_frozen = config.get("user_requested_data_scarcity_exception") is True
    source_valid = bool(_COMMIT_RE.fullmatch(source_commit))
    remote_valid = bool(_COMMIT_RE.fullmatch(remote_main_sha))
    source_exists = source_valid and _git_ok(
        repository, "cat-file", "-e", f"{source_commit}^{{commit}}"
    )
    assets_committed = source_exists and all(
        _git_ok(repository, "cat-file", "-e", f"{source_commit}:{path.as_posix()}")
        for path in REQUIRED_SOURCE_ASSETS
    )
    assets_match = assets_committed and all(
        _git_blob(repository, source_commit, path) == _read_bytes(repository / path)
        for path in REQUIRED_SOURCE_ASSETS
    )
    protected_clean = not _git_output(
        repository,
        "status",
        "--porcelain",
        "--",
        *(path.as_posix() for path in PROTECTED_SOURCE_PATHS),
    ).strip()
    evidence_checks = {
        "k1bp_source_authority_exact": all(source_authority.values()),
        "user_requested_data_scarcity_exception_frozen": exception_frozen,
        "remote_config_readiness_pass": readiness_status == "pass",
        "source_commit_valid": source_valid,
        "source_commit_exists": source_exists,
        "required_source_assets_committed": assets_committed,
        "committed_assets_match_worktree": assets_match,
        "protected_worktree_clean": protected_clean,
    }
    publication_checks = {
        "remote_main_sha_valid": remote_valid,
        "source_commit_equals_exact_github_main": source_valid
        and remote_valid
        and source_commit == remote_main_sha,
    }
    should_ssh = all(evidence_checks.values())
    ssh_allowed = all(publication_checks.values())
    authorized = should_ssh and ssh_allowed
    if not should_ssh:
        status, decision = "fail", "innovation1_uknit_r6_k1br_launch_evidence_invalid"
        next_action = (
            "repair only the failed source, exception, readiness, or commit binding"
        )
    elif not ssh_allowed:
        status, decision = "hold", "innovation1_uknit_r6_k1br_source_not_published"
        next_action = "push and verify the exact source commit on GitHub main"
    else:
        status, decision = "pass", "innovation1_uknit_r6_k1br_remote_launch_authorized"
        next_action = "launch the clean run-owned clone on physical GPU1 and hand off to tmux monitoring"
    return {
        "run_id": RUN_ID,
        "task": "innovation1_uknit_r6_k1br_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "remote_main_sha": remote_main_sha,
        "source_authority": source_authority,
        "evidence_checks": evidence_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": authorized,
        "next_action": next_action,
        "claim_scope": "local launch authorization for a single-seed remote 262144/class r6 diagnostic only",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


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
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout


def _git_blob(repository: Path, commit: str, path: Path) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout


__all__ = ["PLAN", "REMOTE_CONFIG", "REMOTE_RUN_ID", "RUN_ID", "build_k1br_launch_gate"]
