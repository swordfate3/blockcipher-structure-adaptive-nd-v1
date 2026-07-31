from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_uknit_r6_pair_amplification_k1bv_launch_gate_20260731"
REMOTE_RUN_ID = "i1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_20260731"
PLAN = Path("configs/experiment/innovation1/innovation1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4.csv")
REMOTE_CONFIG = Path("configs/remote/innovation1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_gpu0_20260731.json")
GENERATED = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    PLAN, REMOTE_CONFIG,
    Path("docs/experiments/innovation1-uknit-r6-pair-amplification-k1bv-plan.md"),
    GENERATED / f"run_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"launch_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"monitor_{REMOTE_RUN_ID}.sh",
    Path("scripts/check-uknit-r6-pair-amplification-k1bv"),
    Path("scripts/check-uknit-r6-pair-amplification-k1bv-launch"),
    Path("scripts/gate-uknit-r6-pair-amplification-k1bv"),
    Path("scripts/package-uknit-r6-pair-amplification-k1bv"),
    Path("scripts/plot-uknit-r6-pair-amplification-k1bv"),
    Path("src/blockcipher_nd/cli/check_uknit_r6_pair_amplification_k1bv.py"),
    Path("src/blockcipher_nd/cli/check_uknit_r6_pair_amplification_k1bv_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_r6_pair_amplification_k1bv.py"),
    Path("src/blockcipher_nd/cli/package_uknit_r6_pair_amplification_k1bv.py"),
    Path("src/blockcipher_nd/cli/plot_uknit_r6_pair_amplification_k1bv.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r6_pair_amplification_k1bv.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r6_pair_amplification_k1bv_launch.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_family_ctspn_k1t.py"),
    Path("src/blockcipher_nd/models/structure/spn/position_histogram_residual.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
)
PROTECTED_SOURCE_PATHS = (*REQUIRED_SOURCE_ASSETS, Path("scripts/train"), Path("scripts/validate-results"), Path("src/blockcipher_nd/data/cache"), Path("src/blockcipher_nd/engine"), Path("src/blockcipher_nd/training"))
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_launch_gate(
    *, repository: Path, source_commit: str, remote_main_sha: str,
    model_readiness: dict[str, Any], remote_readiness_status: str,
) -> dict[str, Any]:
    source_valid = bool(_COMMIT_RE.fullmatch(source_commit))
    remote_valid = bool(_COMMIT_RE.fullmatch(remote_main_sha))
    source_exists = source_valid and _git_ok(repository, "cat-file", "-e", f"{source_commit}^{{commit}}")
    assets_committed = source_exists and all(_git_ok(repository, "cat-file", "-e", f"{source_commit}:{path.as_posix()}") for path in REQUIRED_SOURCE_ASSETS)
    assets_match = assets_committed and all(_git_blob(repository, source_commit, path) == (repository / path).read_bytes() for path in REQUIRED_SOURCE_ASSETS)
    protected_clean = source_exists and not _git_output(repository, "status", "--porcelain", "--", *(path.as_posix() for path in PROTECTED_SOURCE_PATHS)).strip()
    readiness = {
        "zero_training_model_readiness_pass": model_readiness.get("status") == "pass" and model_readiness.get("training_performed") is False and bool(model_readiness.get("checks")) and all(model_readiness.get("checks", {}).values()),
        "remote_config_readiness_pass": remote_readiness_status == "pass",
        "source_commit_valid": source_valid,
        "source_commit_exists": source_exists,
        "required_source_assets_committed": assets_committed,
        "committed_assets_match_worktree": assets_match,
        "protected_worktree_clean": protected_clean,
    }
    publication = {
        "remote_main_sha_valid": remote_valid,
        "source_commit_equals_exact_github_main": source_valid and remote_valid and source_commit == remote_main_sha,
    }
    should_ssh = all(readiness.values())
    ssh_allowed = all(publication.values())
    authorized = should_ssh and ssh_allowed
    if not should_ssh:
        status, decision = "fail", "innovation1_uknit_k1bv_launch_readiness_invalid"
        next_action = "repair only failed readiness, source, or asset checks"
    elif not ssh_allowed:
        status, decision = "hold", "innovation1_uknit_k1bv_source_not_published"
        next_action = "publish and verify the exact K1-BV source commit on GitHub main"
    else:
        status, decision = "pass", "innovation1_uknit_k1bv_remote_launch_authorized"
        next_action = "perform one bounded GPU0/run-root check, launch the clean pinned clone, confirm durable start, and hand off to tmux"
    return {
        "run_id": RUN_ID, "remote_run_id": REMOTE_RUN_ID,
        "status": status, "decision": decision,
        "source_commit": source_commit, "remote_main_sha": remote_main_sha,
        "readiness_checks": readiness, "publication_checks": publication,
        "should_ssh": should_ssh, "ssh_allowed": ssh_allowed,
        "launch_authorized": authorized, "next_action": next_action,
        "claim_scope": "local K1-BV remote-launch authorization only; no training result",
    }


def _git_ok(repository: Path, *args: str) -> bool:
    return subprocess.run(["git", *args], cwd=repository, capture_output=True, check=False).returncode == 0


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository, capture_output=True, text=True, check=False).stdout


def _git_blob(repository: Path, commit: str, path: Path) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path.as_posix()}"], cwd=repository, capture_output=True, check=True).stdout


__all__ = ["REMOTE_CONFIG", "REMOTE_RUN_ID", "RUN_ID", "build_launch_gate"]
