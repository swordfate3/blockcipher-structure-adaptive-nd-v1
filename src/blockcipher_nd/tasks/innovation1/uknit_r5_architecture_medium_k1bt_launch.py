from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_ID = "i1_uknit_r5_neural_architecture_medium_k1bt_65536_seed3_seed4_launch_gate_20260731"
REMOTE_RUN_ID = "i1_uknit_r5_neural_architecture_medium_k1bt_16pair_65536_seed3_seed4_20260731"
K1BS_RUN_ID = "i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_2048_seed3_seed4_20260731"
K1BS_DECISION = "innovation1_uknit_k1bs_structure_expert_retained"
K1BS_DIGESTS = {
    "gate.json": "66366f93683a8839d1081047d0a75b7cbb980dd896a8e43de9d7277d5d1faf0b",
    "validation.json": "8d2b36d3e262a6b317d53c6779d8ed337acdf26886d071584af5118039c1fe43",
    "results.jsonl": "5604ffbc4e4495780605931283c2b472296a6e1f5066dd0dabc740534713347d",
    "architecture_comparison.csv": "bc41801e29dcf6c694c6faff01f4f28519c67fd3caf9890e863564eab3fc23c2",
    "visual_qa_passed.marker": "7b42b3c8d54c1581206f289ec11518576717feaffeabe412c894f27fda7bd263",
}
K1BS_PLAN = Path("configs/experiment/innovation1/innovation1_uknit_r5_neural_architecture_ablation_k1bs_16pair_2048_seed3_seed4.csv")
K1BT_PLAN = Path("configs/experiment/innovation1/innovation1_uknit_r5_neural_architecture_medium_k1bt_16pair_65536_seed3_seed4.csv")
REMOTE_CONFIG = Path("configs/remote/innovation1_uknit_k1bt_architecture_medium_65536_seed3_seed4_gpu1_20260731.json")
GENERATED = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    K1BS_PLAN, K1BT_PLAN, Path("configs/runtime/spn/uknit64.json"),
    Path("docs/experiments/innovation1-uknit-r5-neural-architecture-medium-k1bt-plan.md"),
    REMOTE_CONFIG,
    GENERATED / f"run_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"launch_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"monitor_{REMOTE_RUN_ID}.sh",
    Path("scripts/check-uknit-r5-architecture-medium-k1bt-launch"),
    Path("scripts/gate-uknit-r5-architecture-medium-k1bt"),
    Path("scripts/package-uknit-r5-architecture-medium-k1bt"),
    Path("scripts/plot-uknit-r5-architecture-medium-k1bt"),
    Path("src/blockcipher_nd/cli/check_uknit_r5_architecture_medium_k1bt_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_r5_architecture_medium_k1bt.py"),
    Path("src/blockcipher_nd/cli/package_uknit_r5_architecture_medium_k1bt.py"),
    Path("src/blockcipher_nd/cli/plot_uknit_r5_architecture_medium_k1bt.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r5_architecture_medium_k1bt.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r5_architecture_medium_k1bt_launch.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r5_architecture_ablation_k1bs.py"),
    Path("src/blockcipher_nd/models/structure/spn/position_histogram_residual.py"),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
)
PROTECTED_SOURCE_PATHS = (*REQUIRED_SOURCE_ASSETS, Path("scripts/train"), Path("scripts/validate-results"), Path("src/blockcipher_nd/data/cache"), Path("src/blockcipher_nd/engine"), Path("src/blockcipher_nd/training"))
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def build_launch_gate(*, k1bs_root: Path, repository: Path, source_commit: str, remote_main_sha: str, readiness_status: str) -> dict[str, Any]:
    gate = _read_json(k1bs_root / "gate.json")
    validation = _read_json(k1bs_root / "validation.json")
    authority = {
        "gate_identity_exact": gate.get("run_id") == K1BS_RUN_ID and gate.get("status") == "pass" and gate.get("decision") == K1BS_DECISION and gate.get("remote_scale") == "candidate",
        "validation_exact_pass": validation.get("run_id") == K1BS_RUN_ID and validation.get("status") == "pass" and validation.get("result_rows") == 8 and validation.get("expected_rows") == 8 and not validation.get("errors"),
        "all_protocol_checks_pass": bool(gate.get("protocol_checks")) and all(gate.get("protocol_checks", {}).values()),
        "all_research_checks_pass": bool(gate.get("research_checks")) and all(gate.get("research_checks", {}).values()),
        "digests_exact": {name: _sha256(k1bs_root / name) for name in K1BS_DIGESTS} == K1BS_DIGESTS,
    }
    source_valid = bool(_COMMIT_RE.fullmatch(source_commit))
    remote_valid = bool(_COMMIT_RE.fullmatch(remote_main_sha))
    source_exists = source_valid and _git_ok(repository, "cat-file", "-e", f"{source_commit}^{{commit}}")
    assets_committed = source_exists and all(_git_ok(repository, "cat-file", "-e", f"{source_commit}:{path.as_posix()}") for path in REQUIRED_SOURCE_ASSETS)
    assets_match = assets_committed and all(_git_blob(repository, source_commit, path) == _read_bytes(repository / path) for path in REQUIRED_SOURCE_ASSETS)
    protected_clean = source_exists and not _git_output(repository, "status", "--porcelain", "--", *(path.as_posix() for path in PROTECTED_SOURCE_PATHS)).strip()
    return adjudicate_launch(
        source_commit=source_commit, remote_main_sha=remote_main_sha,
        readiness_status=readiness_status, k1bs_authority=authority,
        plans_match_selected_models_and_scale_only=_plans_match(repository / K1BS_PLAN, repository / K1BT_PLAN),
        source_commit_valid=source_valid, remote_main_valid=remote_valid,
        source_commit_exists=source_exists, source_assets_committed=assets_committed,
        source_assets_match=assets_match, protected_worktree_clean=protected_clean,
    )


def adjudicate_launch(*, source_commit: str, remote_main_sha: str, readiness_status: str, k1bs_authority: dict[str, bool], plans_match_selected_models_and_scale_only: bool, source_commit_valid: bool, remote_main_valid: bool, source_commit_exists: bool, source_assets_committed: bool, source_assets_match: bool, protected_worktree_clean: bool) -> dict[str, Any]:
    evidence = {
        "k1bs_authority_complete": bool(k1bs_authority) and all(k1bs_authority.values()),
        "k1bt_matches_selected_k1bs_models_except_scale_and_identity": plans_match_selected_models_and_scale_only,
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
        "source_commit_equals_exact_github_main": source_commit_valid and remote_main_valid and source_commit == remote_main_sha,
    }
    should_ssh = all(evidence.values()) and all(readiness.values())
    ssh_allowed = all(publication.values())
    authorized = should_ssh and ssh_allowed
    if not should_ssh:
        status, decision = "fail", "innovation1_uknit_k1bt_launch_evidence_invalid"
        next_action = "repair only failed K1-BS authority, plan, readiness, or source checks"
    elif not ssh_allowed:
        status, decision = "hold", "innovation1_uknit_k1bt_source_not_published"
        next_action = "publish and verify the exact K1-BT source commit on GitHub main"
    else:
        status, decision = "pass", "innovation1_uknit_k1bt_remote_launch_authorized"
        next_action = "perform one bounded GPU1/run-root check, launch the exact clean clone, confirm durable start, and hand off to tmux monitoring"
    return {
        "run_id": RUN_ID, "task": "innovation1_uknit_k1bt_remote_launch_gate", "remote_run_id": REMOTE_RUN_ID,
        "status": status, "decision": decision, "source_commit": source_commit, "remote_main_sha": remote_main_sha,
        "remote_config_readiness": readiness_status, "k1bs_authority": k1bs_authority,
        "evidence_checks": evidence, "readiness_checks": readiness, "publication_checks": publication,
        "should_ssh": should_ssh, "ssh_allowed": ssh_allowed, "launch_authorized": authorized,
        "next_action": next_action,
        "claim_scope": "local K1-BT remote-launch authorization only; no remote result or formal-scale claim",
        "blocked_actions": ["launching unpublished or worktree-drifted source", "using scp or dirty overlay for source publication", "running 65536/class locally", "changing the frozen K1-BT protocol"],
    }


def _plans_match(k1bs_path: Path, k1bt_path: Path) -> bool:
    try:
        with k1bs_path.open(newline="", encoding="utf-8") as handle:
            local = list(csv.DictReader(handle))
        with k1bt_path.open(newline="", encoding="utf-8") as handle:
            medium = list(csv.DictReader(handle))
    except OSError:
        return False
    selected = [row for row in local if row.get("model_key") in {"runtime_spn_ct_k1t_position_histogram_true", "autond_dbitnet2023"}]
    if len(selected) != 4 or len(medium) != 4:
        return False
    ignored = {"network", "family", "samples_per_class", "train_samples_total", "validation_samples_total", "evidence", "literature"}
    for old, new in zip(selected, medium, strict=True):
        if old.get("samples_per_class") != "2048" or old.get("validation_samples_total") != "2048":
            return False
        if new.get("samples_per_class") != "65536" or new.get("validation_samples_total") != "32768":
            return False
        if any(old.get(field) != new.get(field) for field in (set(old) | set(new)) - ignored):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
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
    return subprocess.run(["git", *args], cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False).stdout


def _git_blob(repository: Path, commit: str, path: Path) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path.as_posix()}"], cwd=repository, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout


__all__ = ["K1BS_PLAN", "K1BT_PLAN", "REMOTE_CONFIG", "REMOTE_RUN_ID", "RUN_ID", "_plans_match", "adjudicate_launch", "build_launch_gate"]
