from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_rtg3b_present80_one_to_one_formal_1000000_seed0_launch_gate_20260726"
REMOTE_RUN_ID = "i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726"
C2_RUN_ID = "i1_runtime_spn_method_boundary_c2_20260726"
C2_DECISION = "innovation1_runtime_spn_method_boundary_frozen"
BASE_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_present_runtime_e4_transfer_t1_2048_seed0.csv"
)
FORMAL_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_present80_runtime_e4_formal_rtg3b_1000000_seed0.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_rtg3b_present80_one_to_one_formal_1000000_seed0_gpu0_20260726.json"
)
REQUIRED_SOURCE_ASSETS = (
    FORMAL_PLAN,
    REMOTE_CONFIG,
    Path(
        "docs/experiments/innovation1-present80-runtime-e4-rtg3b-1000000-seed0-plan.md"
    ),
    Path(
        "configs/remote/generated/"
        "run_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "launch_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.sh"
    ),
    Path("scripts/check-runtime-spn-present-rtg3b-launch"),
    Path("scripts/gate-runtime-spn-present-transfer"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_present_rtg3b_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_present_transfer.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_present_rtg3b_launch.py"),
    Path("src/blockcipher_nd/tasks/innovation1/runtime_spn_present_transfer.py"),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_present_rtg3b_launch_gate(
    *,
    c2_root: Path,
    t1_seed0_root: Path,
    t1_seed1_root: Path,
    readiness_report: Path,
    repository: Path,
    source_commit: str,
    remote: str = "origin",
    branch: str = "main",
) -> dict[str, Any]:
    c2_gate = _read_json(c2_root / "gate.json")
    c2_validation = _read_json(c2_root / "validation.json")
    t1_gates = (
        _read_json(t1_seed0_root / "gate.json"),
        _read_json(t1_seed1_root / "gate.json"),
    )
    readiness = _read_json(readiness_report)
    source_commit_valid = bool(_COMMIT_RE.fullmatch(source_commit))
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
    remote_sha = _git_remote_sha(repository, remote=remote, branch=branch)
    return adjudicate_runtime_spn_present_rtg3b_launch(
        source_commit=source_commit,
        remote=remote,
        branch=branch,
        remote_sha=remote_sha,
        c2_gate=c2_gate,
        c2_validation=c2_validation,
        t1_gates=t1_gates,
        readiness=readiness,
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        plans_match_scale_only=plans_match_scale_only(
            repository / BASE_PLAN,
            repository / FORMAL_PLAN,
        ),
    )


def adjudicate_runtime_spn_present_rtg3b_launch(
    *,
    source_commit: str,
    remote: str,
    branch: str,
    remote_sha: str | None,
    c2_gate: dict[str, Any],
    c2_validation: dict[str, Any],
    t1_gates: tuple[dict[str, Any], dict[str, Any]],
    readiness: dict[str, Any],
    source_commit_valid: bool,
    source_commit_exists: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    plans_match_scale_only: bool,
) -> dict[str, Any]:
    c2_requirements = c2_gate.get("requirement_status")
    evidence_checks = {
        "c2_identity_and_validation_pass": (
            c2_gate.get("run_id") == C2_RUN_ID
            and c2_gate.get("status") == "pass"
            and c2_gate.get("decision") == C2_DECISION
            and c2_validation.get("status") == "pass"
        ),
        "c2_one_to_one_requirement_supported": (
            isinstance(c2_requirements, dict)
            and c2_requirements.get("R5") == "supported"
            and c2_gate.get("universal_runtime_spn_supported") is False
        ),
        "present_t1_seed0_supported": _t1_gate_supported(t1_gates[0], seed=0),
        "present_t1_seed1_supported": _t1_gate_supported(t1_gates[1], seed=1),
    }
    readiness_checks = {
        "formal_plan_matches_t1_except_scale_and_identity": plans_match_scale_only,
        "generic_remote_readiness_pass": readiness.get("status") == "pass",
        "generic_remote_readiness_has_no_errors": readiness.get("errors") == [],
        "generic_remote_readiness_scale_exact": (
            readiness.get("max_samples_per_class") == 1_000_000
            and readiness.get("expected_rows") == 3
            and readiness.get("plan_rows") == 3
        ),
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
        "required_source_assets_committed": source_assets_committed,
        "committed_assets_match_worktree": source_assets_match,
    }
    publication_checks = {
        "live_remote_sha_available": remote_sha is not None,
        "live_remote_sha_matches_source_commit": remote_sha == source_commit,
    }
    should_ssh = all(evidence_checks.values()) and all(readiness_checks.values())
    ssh_allowed = all(publication_checks.values())
    launch_authorized = should_ssh and ssh_allowed
    if launch_authorized:
        status = "pass"
        decision = "innovation1_rtg3b_present_seed0_remote_launch_authorized"
        next_action = (
            "start the committed RTG3-B local tmux watcher with this exact "
            "live-verified source commit"
        )
    elif should_ssh:
        status = "hold"
        decision = "innovation1_rtg3b_present_seed0_source_not_live_verified"
        next_action = (
            "wait for GitHub connectivity and rerun this gate until the live remote "
            "main SHA exactly matches source_commit; do not use an alternate transfer"
        )
    else:
        status = "fail"
        decision = "innovation1_rtg3b_present_seed0_launch_evidence_invalid"
        next_action = "repair only failed local evidence/readiness checks before SSH"
    return {
        "run_id": RUN_ID,
        "task": "innovation1_rtg3b_present_seed0_controlled_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "remote": remote,
        "branch": branch,
        "live_remote_sha": remote_sha,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact unless should_ssh and ssh_allowed are both true",
            "remote launch from a commit not live-verified on origin/main",
            "scp, dirty overlay, alternate remote, force push, or SSH transport switch",
            "change T1 fields other than scale and run identity",
            "launch seed1 before a complete retrieved seed0 research pass",
        ],
        "claim_scope": (
            "local RTG3-B seed0 launch authorization only; no remote result, "
            "Zhang/Wang reproduction, attack, SOTA, breakthrough, zero-step transfer, "
            "or universal-SPN claim"
        ),
    }


def plans_match_scale_only(base_path: Path, formal_path: Path) -> bool:
    try:
        with base_path.open(newline="", encoding="utf-8") as handle:
            base_rows = list(csv.DictReader(handle))
        with formal_path.open(newline="", encoding="utf-8") as handle:
            formal_rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(base_rows) != 3 or len(formal_rows) != 3:
        return False
    ignored = {"network", "family", "samples_per_class", "evidence", "literature"}
    for base, formal in zip(base_rows, formal_rows, strict=True):
        if base.get("samples_per_class") != "2048":
            return False
        if formal.get("samples_per_class") != "1000000":
            return False
        fields = set(base) | set(formal)
        if any(base.get(field) != formal.get(field) for field in fields - ignored):
            return False
    return True


def _t1_gate_supported(gate: dict[str, Any], *, seed: int) -> bool:
    protocol = gate.get("protocol_checks")
    research = gate.get("research_checks")
    return bool(
        gate.get("seed") == seed
        and gate.get("status") == "pass"
        and gate.get("decision")
        == f"innovation1_runtime_spn_present_transfer_seed{seed}_supported"
        and isinstance(protocol, dict)
        and protocol
        and all(value is True for value in protocol.values())
        and isinstance(research, dict)
        and research
        and all(value is True for value in research.values())
    )


def _git_remote_sha(repository: Path, *, remote: str, branch: str) -> str | None:
    result = subprocess.run(
        ["git", "ls-remote", "--refs", remote, f"refs/heads/{branch}"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    fields = result.stdout.strip().split()
    return fields[0] if len(fields) == 2 and _COMMIT_RE.fullmatch(fields[0]) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
    "adjudicate_runtime_spn_present_rtg3b_launch",
    "build_runtime_spn_present_rtg3b_launch_gate",
    "plans_match_scale_only",
]
