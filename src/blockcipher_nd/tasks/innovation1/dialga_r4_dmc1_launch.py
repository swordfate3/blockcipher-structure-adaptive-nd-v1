from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_dialga128_runtime_e4_dmc1_r4_65536_launch_gate_20260731"
REMOTE_RUN_ID = "i1_dialga128_runtime_e4_dmc1_r4_65536_seed0_seed1_20260731"
D1_ROOT = Path(
    "outputs/local_diagnostic/"
    "i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725"
)
D2_ROOT = Path(
    "outputs/local_audits/"
    "i1_dialga128_runtime_e4_d2_same_checkpoint_20260725"
)
AUTHORITY_DIGESTS = {
    D1_ROOT / "gate.json": "e113227bbd541a3d5c11502793d5ebb5d75108c4f53e157326b5ac509cc10e67",
    D1_ROOT / "validation.json": "fbb7aba04197f0f309e9f08b4b6eb2cf051fe4721d692971b1ee785480591454",
    D1_ROOT / "results.jsonl": "67d2591dae166feaf1fdd391a0e7c40dc1ad3029993661bbcc03b54c4309cc78",
    D2_ROOT / "gate.json": "e7819ee9c1abb54b649cfcb1d3b78bb4364889563be67c96a459b69ab96a6501",
    D2_ROOT / "validation.json": "175841b1e5d6995c79e740807f36a837059c6d50cd07ce2c91f212ada76390bd",
    D2_ROOT / "results.jsonl": "9290fe7a94ca87e5800c80334a1b4bb5754e6d322904f725f9278f7418b41be5",
}
PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_dialga128_runtime_e4_dmc1_r4_65536_seed0_seed1.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_dialga_dmc1_r4_medium_65536_seed0_seed1_gpu1_20260731.json"
)
GENERATED = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    PLAN,
    REMOTE_CONFIG,
    Path("configs/runtime/spn/dialga128.json"),
    Path("docs/experiments/innovation1-dialga128-runtime-e4-dmc1-r4-medium-plan.md"),
    GENERATED / f"run_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"launch_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"monitor_{REMOTE_RUN_ID}.sh",
    Path("scripts/check-dialga-r4-dmc1-launch"),
    Path("scripts/gate-dialga-r4-dmc1"),
    Path("scripts/package-dialga-r4-dmc1"),
    Path("scripts/plot-dialga-r4-dmc1"),
    Path("src/blockcipher_nd/cli/check_dialga_r4_dmc1_launch.py"),
    Path("src/blockcipher_nd/cli/gate_dialga_r4_dmc1.py"),
    Path("src/blockcipher_nd/cli/package_dialga_r4_dmc1.py"),
    Path("src/blockcipher_nd/cli/plot_dialga_r4_dmc1.py"),
    Path("src/blockcipher_nd/tasks/innovation1/dialga_r4_dmc1.py"),
    Path("src/blockcipher_nd/tasks/innovation1/dialga_r4_dmc1_launch.py"),
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


def build_launch_gate(
    *,
    repository: Path,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
) -> dict[str, Any]:
    authority = _authority_checks(repository)
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
    protected_clean = source_exists and not _git_output(
        repository,
        "status",
        "--porcelain",
        "--",
        *(path.as_posix() for path in PROTECTED_SOURCE_PATHS),
    ).strip()
    return adjudicate_launch(
        source_commit=source_commit,
        remote_main_sha=remote_main_sha,
        readiness_status=readiness_status,
        authority=authority,
        source_commit_valid=source_valid,
        remote_main_valid=remote_valid,
        source_commit_exists=source_exists,
        source_assets_committed=assets_committed,
        source_assets_match=assets_match,
        protected_worktree_clean=protected_clean,
    )


def adjudicate_launch(
    *,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
    authority: dict[str, bool],
    source_commit_valid: bool,
    remote_main_valid: bool,
    source_commit_exists: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    protected_worktree_clean: bool,
) -> dict[str, Any]:
    evidence = {
        "d1_d2_authority_complete": bool(authority) and all(authority.values()),
    }
    readiness = {
        "remote_config_readiness_pass": readiness_status == "pass",
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
        "required_source_assets_committed": source_assets_committed,
        "committed_assets_match_worktree": source_assets_match,
        "protected_worktree_clean": protected_worktree_clean,
    }
    publication = {
        "remote_main_sha_valid": remote_main_valid,
        "source_commit_equals_exact_github_main": source_commit_valid
        and remote_main_valid
        and source_commit == remote_main_sha,
    }
    should_ssh = all(evidence.values()) and all(readiness.values())
    ssh_allowed = all(publication.values())
    authorized = should_ssh and ssh_allowed
    if not should_ssh:
        status = "fail"
        decision = "innovation1_dialga_dmc1_launch_evidence_invalid"
        next_action = "repair only failed D1/D2 authority, readiness, or source checks"
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_dialga_dmc1_source_not_published"
        next_action = "publish and verify the exact DMC1 source commit on GitHub main"
    else:
        status = "pass"
        decision = "innovation1_dialga_dmc1_remote_launch_authorized"
        next_action = (
            "perform one bounded GPU1/run-root check, launch the exact clean clone, "
            "confirm durable start, and hand off to tmux monitoring"
        )
    return {
        "run_id": RUN_ID,
        "task": "innovation1_dialga_dmc1_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "remote_main_sha": remote_main_sha,
        "remote_config_readiness": readiness_status,
        "authority_checks": authority,
        "evidence_checks": evidence,
        "readiness_checks": readiness,
        "publication_checks": publication,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": authorized,
        "next_action": next_action,
        "claim_scope": "local DMC1 remote-launch authorization only; no remote result claim",
        "blocked_actions": [
            "launching unpublished or worktree-drifted source",
            "using scp or dirty overlay for source publication",
            "running 65536/class locally",
            "changing the frozen DMC1 protocol",
        ],
    }


def _authority_checks(repository: Path) -> dict[str, bool]:
    checks = {
        f"sha256_{path.parent.name}_{path.name}": _sha256(repository / path) == digest
        for path, digest in AUTHORITY_DIGESTS.items()
    }
    try:
        d1_gate = _read_json(repository / D1_ROOT / "gate.json")
        d2_gate = _read_json(repository / D2_ROOT / "gate.json")
        d1_validation = _read_json(repository / D1_ROOT / "validation.json")
        d2_validation = _read_json(repository / D2_ROOT / "validation.json")
    except (OSError, json.JSONDecodeError):
        checks["authority_json_readable"] = False
        return checks
    checks.update(
        {
            "authority_json_readable": True,
            "d1_gate_exact_pass": d1_gate.get("status") == "pass"
            and d1_gate.get("decision")
            == "innovation1_dialga_runtime_e4_d1_two_seed_supported",
            "d2_gate_exact_pass": d2_gate.get("status") == "pass"
            and d2_gate.get("decision")
            == "innovation1_dialga_runtime_e4_d2_functional_topology_use_supported",
            "d1_validation_pass": d1_validation.get("status") == "pass",
            "d2_validation_pass": d2_validation.get("status") == "pass",
        }
    )
    return checks


def _git_ok(repository: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, check=False
    ).returncode == 0


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True, check=False
    ).stdout


def _git_blob(repository: Path, commit: str, path: Path) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


__all__ = [
    "PLAN",
    "REMOTE_CONFIG",
    "REMOTE_RUN_ID",
    "adjudicate_launch",
    "build_launch_gate",
]
