from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_launch_gate_20260725"
REMOTE_RUN_ID = "i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725"
JOINT_RUN_ID = (
    "i1_rtg2b_skinny64_general_gf2_scale_262144_joint_seed0_seed1_20260724"
)
JOINT_DECISION = "innovation1_rtg2b_skinny_scale_two_seed_supported"
BASE_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed0.csv"
)
FORMAL_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_skinny64_runtime_e4_formal_rtg3a_1000000_seed0.csv"
)
ROUTE_DECISION = Path(
    "docs/experiments/innovation1-runtime-spn-post-rtg2b-route-decision-plan.md"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_gpu0_20260725.json"
)
REQUIRED_SOURCE_ASSETS = (
    FORMAL_PLAN,
    ROUTE_DECISION,
    Path(
        "docs/experiments/"
        "innovation1-skinny64-runtime-e4-rtg3a-1000000-seed0-plan.md"
    ),
    REMOTE_CONFIG,
    Path(
        "configs/remote/generated/"
        "run_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "launch_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725.cmd"
    ),
    Path(
        "configs/remote/generated/"
        "monitor_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725.sh"
    ),
    Path("scripts/check-runtime-spn-skinny-rtg3a-launch"),
    Path("scripts/gate-runtime-spn-skinny-medium"),
    Path("src/blockcipher_nd/cli/check_runtime_spn_skinny_rtg3a_launch.py"),
    Path("src/blockcipher_nd/cli/gate_runtime_spn_skinny_medium.py"),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "runtime_spn_skinny_rtg3a_launch.py"
    ),
    Path(
        "src/blockcipher_nd/tasks/innovation1/runtime_spn_skinny_medium.py"
    ),
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_runtime_spn_skinny_rtg3a_launch_gate(
    *,
    joint_root: Path,
    repository: Path,
    source_commit: str,
    readiness_status: str,
    upstream_ref: str = "origin/main",
) -> dict[str, Any]:
    joint_gate = _read_json(joint_root / "gate.json")
    joint_validation = _read_json(joint_root / "validation.json")
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
    return adjudicate_runtime_spn_skinny_rtg3a_launch(
        source_commit=source_commit,
        upstream_ref=upstream_ref,
        joint_gate=joint_gate,
        joint_validation=joint_validation,
        readiness_status=readiness_status,
        route_selects_formal=_route_selects_formal(repository / ROUTE_DECISION),
        source_commit_valid=source_commit_valid,
        source_commit_exists=source_commit_exists,
        source_commit_published=source_commit_published,
        source_assets_committed=source_assets_committed,
        source_assets_match=source_assets_match,
        plans_match_scale_only=_plans_match_scale_only(
            repository / BASE_PLAN,
            repository / FORMAL_PLAN,
        ),
    )


def adjudicate_runtime_spn_skinny_rtg3a_launch(
    *,
    source_commit: str,
    upstream_ref: str,
    joint_gate: dict[str, Any],
    joint_validation: dict[str, Any],
    readiness_status: str,
    route_selects_formal: bool,
    source_commit_valid: bool,
    source_commit_exists: bool,
    source_commit_published: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    plans_match_scale_only: bool,
) -> dict[str, Any]:
    joint_protocol = joint_gate.get("protocol_checks")
    joint_research = joint_gate.get("research_checks")
    evidence_checks = {
        "rtg2b_joint_gate_identity_exact": (
            joint_gate.get("run_id") == JOINT_RUN_ID
            and joint_gate.get("status") == "pass"
            and joint_gate.get("decision") == JOINT_DECISION
        ),
        "rtg2b_joint_protocol_checks_pass": (
            isinstance(joint_protocol, dict)
            and bool(joint_protocol)
            and all(value is True for value in joint_protocol.values())
        ),
        "rtg2b_joint_research_checks_pass": (
            isinstance(joint_research, dict)
            and bool(joint_research)
            and all(value is True for value in joint_research.values())
        ),
        "rtg2b_joint_validation_pass": joint_validation.get("status") == "pass",
        "post_x2_route_decision_selects_formal": route_selects_formal,
    }
    readiness_checks = {
        "rtg3a_plan_matches_rtg2b_except_scale_and_identity": plans_match_scale_only,
        "remote_disk_cache_readiness_pass": readiness_status == "pass",
        "source_commit_valid": source_commit_valid,
        "source_commit_exists": source_commit_exists,
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
        decision = "innovation1_rtg3a_seed0_remote_launch_authorized"
        next_action = (
            "start the committed RTG3-A local tmux watcher with this exact pushed commit"
        )
    elif should_ssh:
        status = "hold"
        decision = "innovation1_rtg3a_seed0_source_not_published"
        next_action = (
            "publish and verify the exact source commit through the configured Git remote; "
            "do not use a dirty overlay or alternate transfer route"
        )
    else:
        status = "fail"
        decision = "innovation1_rtg3a_seed0_launch_evidence_invalid"
        next_action = "repair only the failed local launch checks before any SSH contact"
    return {
        "run_id": RUN_ID,
        "task": "innovation1_rtg3a_skinny_seed0_controlled_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "status": status,
        "decision": decision,
        "source_commit": source_commit,
        "upstream_ref": upstream_ref,
        "evidence_checks": evidence_checks,
        "readiness_checks": readiness_checks,
        "publication_checks": publication_checks,
        "should_ssh": should_ssh,
        "ssh_allowed": ssh_allowed,
        "launch_authorized": launch_authorized,
        "next_action": next_action,
        "blocked_actions": [
            "SSH contact unless should_ssh and ssh_allowed are both true",
            "remote launch from an unpublished or unverified commit",
            "scp or dirty-overlay source publication",
            "change any frozen RTG2-B field other than sample scale and run identity",
            "launch RTG3-A seed1 before the seed0 research gate passes",
        ],
        "claim_scope": (
            "local RTG3-A seed0 launch authorization only; no remote training result, "
            "paper reproduction, attack, SOTA, breakthrough, or universal-SPN claim"
        ),
    }


def _plans_match_scale_only(base_path: Path, formal_path: Path) -> bool:
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
        if base.get("samples_per_class") != "262144":
            return False
        if formal.get("samples_per_class") != "1000000":
            return False
        fields = set(base) | set(formal)
        if any(base.get(field) != formal.get(field) for field in fields - ignored):
            return False
    return True


def _route_selects_formal(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(
        re.search(r"^status\s*=\s*pass$", text, flags=re.MULTILINE)
        and re.search(
            r"^decision\s*=\s*select_route_a_formal_skinny_scale$",
            text,
            flags=re.MULTILINE,
        )
    )


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
    "adjudicate_runtime_spn_skinny_rtg3a_launch",
    "build_runtime_spn_skinny_rtg3a_launch_gate",
]
