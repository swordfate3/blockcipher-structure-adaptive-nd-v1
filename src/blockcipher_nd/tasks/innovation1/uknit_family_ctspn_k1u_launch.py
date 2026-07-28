from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = (
    "i1_uknit_family_ctspn_position_residual_"
    "k1u_medium_65536_seed3_seed4_launch_gate_20260728"
)
REMOTE_RUN_ID = (
    "i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728"
)
K1T_RUN_ID = (
    "i1_uknit_family_ctspn_deterministic_position_residual_"
    "k1t_2048_seed3_seed4_20260728"
)
K1T_DECISION = (
    "innovation1_uknit_family_ctspn_k1t_deterministic_position_residual_supported"
)
K1T_DIGESTS = {
    "gate.json": "f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf",
    "validation.json": "57601722e4a1f14fd2cf1081886213fbc43c7958ada3406e92070b430a3f884e",
    "results.jsonl": "adafb1217298ade5ad7bda4aff5a53742e951e5c737babb30a449362e948563a",
    "controls.jsonl": "1172d5ac2711bd3b8523b9a378e579b59561c66d842255000b1b46c8a45d780e",
}
K1T_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_deterministic_position_residual_"
    "k1t_2048_seed3_seed4.csv"
)
K1U_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_position_residual_"
    "k1u_medium_65536_seed3_seed4.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_uknit_k1u_position_residual_medium_65536_"
    "seed3_seed4_gpu1_20260728.json"
)
GENERATED_ROOT = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    K1T_PLAN,
    K1U_PLAN,
    Path("configs/runtime/spn/uknit64.json"),
    Path(
        "docs/experiments/"
        "innovation1-uknit-family-ctspn-position-residual-k1u-medium-plan.md"
    ),
    REMOTE_CONFIG,
    GENERATED_ROOT
    / "run_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.cmd",
    GENERATED_ROOT
    / "launch_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.cmd",
    GENERATED_ROOT
    / "monitor_i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728.sh",
    Path("scripts/check-uknit-family-ctspn-k1u-launch"),
    Path("scripts/gate-uknit-family-ctspn-k1u"),
    Path("scripts/package-uknit-family-ctspn-k1u"),
    Path("scripts/plot-uknit-family-ctspn-k1u"),
    Path("src/blockcipher_nd/cli/check_uknit_family_ctspn_k1u_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_family_ctspn_k1u.py"),
    Path("src/blockcipher_nd/cli/package_uknit_family_ctspn_k1u.py"),
    Path("src/blockcipher_nd/cli/plot_uknit_family_ctspn_k1u.py"),
    Path("src/blockcipher_nd/models/structure/spn/position_histogram_residual.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_family_ctspn_k1u.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_family_ctspn_k1u_launch.py"),
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


def build_k1u_launch_gate(
    *,
    k1t_root: Path,
    repository: Path,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
) -> dict[str, Any]:
    stored_gate = _read_json(k1t_root / "gate.json")
    validation = _read_json(k1t_root / "validation.json")
    k1t_authority = {
        "gate_identity_exact": (
            stored_gate.get("run_id") == K1T_RUN_ID
            and stored_gate.get("status") == "pass"
            and stored_gate.get("decision") == K1T_DECISION
            and stored_gate.get("remote_scale") == "authorized_65536_per_class"
        ),
        "validation_exact_pass": (
            validation.get("run_id") == K1T_RUN_ID
            and validation.get("status") == "pass"
            and validation.get("training_rows") == 6
            and validation.get("evaluation_rows") == 24
            and not validation.get("errors")
        ),
        "all_protocol_checks_pass": bool(stored_gate.get("protocol_checks"))
        and all(stored_gate.get("protocol_checks", {}).values()),
        "all_research_checks_pass": bool(stored_gate.get("research_checks"))
        and all(stored_gate.get("research_checks", {}).values()),
        "visual_qa_passed": (k1t_root / "visual_qa_passed.marker").is_file(),
        "digests": {name: _sha256_file(k1t_root / name) for name in K1T_DIGESTS},
    }
    k1t_authority["digests_exact"] = k1t_authority["digests"] == K1T_DIGESTS

    source_commit_valid = bool(_COMMIT_RE.fullmatch(source_commit))
    remote_main_valid = bool(_COMMIT_RE.fullmatch(remote_main_sha))
    source_commit_exists = source_commit_valid and _git_ok(
        repository, "cat-file", "-e", f"{source_commit}^{{commit}}"
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

    return adjudicate_k1u_launch(
        source_commit=source_commit,
        remote_main_sha=remote_main_sha,
        readiness_status=readiness_status,
        k1t_authority=k1t_authority,
        plans_match_scale_only=_plans_match_scale_only(
            repository / K1T_PLAN, repository / K1U_PLAN
        ),
        source_commit_valid=source_commit_valid,
        remote_main_valid=remote_main_valid,
        source_commit_exists=source_commit_exists,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        protected_paths_unchanged=protected_paths_unchanged,
        protected_worktree_clean=protected_worktree_clean,
    )


def adjudicate_k1u_launch(
    *,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
    k1t_authority: dict[str, Any],
    plans_match_scale_only: bool,
    source_commit_valid: bool,
    remote_main_valid: bool,
    source_commit_exists: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    protected_paths_unchanged: bool,
    protected_worktree_clean: bool,
) -> dict[str, Any]:
    authority_checks = {
        key: value is True for key, value in k1t_authority.items() if key != "digests"
    }
    evidence_checks = {
        "k1t_authority_complete": bool(authority_checks)
        and all(authority_checks.values()),
        "k1u_matches_k1t_except_scale_validation_and_identity": (
            plans_match_scale_only
        ),
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
        "remote_main_sha_valid": remote_main_valid,
        "source_commit_equals_exact_github_main": (
            source_commit_valid
            and remote_main_valid
            and source_commit == remote_main_sha
        ),
    }
    local_evidence_valid = all(evidence_checks.values()) and all(
        readiness_checks.values()
    )
    should_ssh = local_evidence_valid
    ssh_allowed = all(publication_checks.values())
    launch_authorized = should_ssh and ssh_allowed

    if not local_evidence_valid:
        status = "fail"
        decision = "innovation1_uknit_family_ctspn_k1u_launch_evidence_invalid"
        next_action = "repair only the failed K1-T authority, protocol, or source gate"
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1u_source_not_published"
        next_action = (
            "publish and verify the exact K1-U source commit on GitHub main; do not "
            "use scp or a dirty source overlay"
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1u_remote_launch_authorized"
        next_action = (
            "perform one bounded GPU1 and run-root check, bootstrap the exact clean "
            "clone, launch K1-U, confirm durable start/cache progress, and hand off "
            "to the local tmux result monitor"
        )

    return {
        "run_id": RUN_ID,
        "task": "innovation1_uknit_family_ctspn_k1u_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "remote_main_sha": remote_main_sha,
        "remote_config_readiness": readiness_status,
        "k1t_authority": k1t_authority,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "launch from an unpublished or worktree-drifted source",
            "scp or dirty-overlay source publication",
            "local execution of the 65536/class matrix",
            "change K1-T model, data, controls, pairs, epochs, keys, or thresholds",
            "advance to 262144/class before retrieved K1-U adjudication",
        ],
        "claim_scope": (
            "local K1-U remote-launch authorization only; no remote result, formal "
            "scale, attack, SOTA, transfer, or universal-SPN claim"
        ),
    }


def _plans_match_scale_only(k1t_path: Path, k1u_path: Path) -> bool:
    try:
        with k1t_path.open(newline="", encoding="utf-8") as handle:
            k1t_rows = list(csv.DictReader(handle))
        with k1u_path.open(newline="", encoding="utf-8") as handle:
            k1u_rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(k1t_rows) != 6 or len(k1u_rows) != 6:
        return False
    ignored = {
        "network",
        "family",
        "samples_per_class",
        "train_samples_total",
        "validation_samples_total",
        "evidence",
        "literature",
    }
    for local, medium in zip(k1t_rows, k1u_rows, strict=True):
        if local.get("samples_per_class") != "2048":
            return False
        if medium.get("samples_per_class") != "65536":
            return False
        if local.get("validation_samples_total") not in {"", None}:
            return False
        if medium.get("validation_samples_total") != "65536":
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


__all__ = [
    "K1T_DIGESTS",
    "K1T_PLAN",
    "K1U_PLAN",
    "REMOTE_CONFIG",
    "REMOTE_RUN_ID",
    "RUN_ID",
    "_plans_match_scale_only",
    "adjudicate_k1u_launch",
    "build_k1u_launch_gate",
]
