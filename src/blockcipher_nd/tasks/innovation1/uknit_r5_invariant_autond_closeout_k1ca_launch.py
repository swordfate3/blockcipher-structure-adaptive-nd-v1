from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca import (
    candidate_protocol_frozen,
    read_tasks,
)


RUN_ID = "i1_uknit_r5_k1ca_closeout_launch_gate_20260803"
REMOTE_RUN_ID = "i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803"
K1U_GATE = Path(
    "outputs/remote_results_incomplete/"
    "i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728/"
    "gate.json"
)
K1U_GATE_SHA256 = "79a5f3652b8a6125af8c987cb8b1df075fc8e992e73cdb5dc61dedbfbdb6c3ed"
PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_r5_k1ca_invariant_autond_closeout_262144_seed3_seed4.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_uknit_k1ca_invariant_autond_closeout_262144_seed3_seed4_gpu0_20260803.json"
)
EXPERIMENT_PLAN = Path(
    "docs/experiments/innovation1-uknit-r5-k1ca-invariant-autond-paper-closeout-plan.md"
)
PAPER_CONTRACT = Path("paper/chinese-core-innovation1/claim_evidence_matrix.md")
GENERATED = Path("configs/remote/generated")
REQUIRED_SOURCE_ASSETS = (
    PLAN,
    REMOTE_CONFIG,
    EXPERIMENT_PLAN,
    PAPER_CONTRACT,
    Path("configs/runtime/spn/uknit64.json"),
    GENERATED / f"run_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"launch_{REMOTE_RUN_ID}.cmd",
    GENERATED / f"monitor_{REMOTE_RUN_ID}.sh",
    Path("scripts/check-uknit-r5-k1ca-closeout-launch"),
    Path("scripts/gate-uknit-r5-k1ca-closeout"),
    Path("scripts/package-uknit-r5-k1ca-closeout"),
    Path("src/blockcipher_nd/cli/check_uknit_r5_k1ca_closeout_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_r5_k1ca_closeout.py"),
    Path("src/blockcipher_nd/cli/package_uknit_r5_k1ca_closeout.py"),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "uknit_r5_invariant_autond_closeout_k1ca.py"
    ),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "uknit_r5_invariant_autond_closeout_k1ca_launch.py"
    ),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_family_ctspn_k1t.py"),
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


def build_launch_gate(
    *,
    repository: Path,
    k1u_root: Path,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
) -> dict[str, Any]:
    authority = _k1u_authority(k1u_root)
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
    protected_clean = (
        source_exists
        and not _git_output(
            repository,
            "status",
            "--porcelain",
            "--",
            *(path.as_posix() for path in PROTECTED_SOURCE_PATHS),
        ).strip()
    )
    plan_frozen = False
    try:
        plan_frozen = candidate_protocol_frozen(read_tasks(repository / PLAN))
    except (OSError, ValueError):
        pass
    return adjudicate_launch(
        source_commit=source_commit,
        remote_main_sha=remote_main_sha,
        readiness_status=readiness_status,
        authority=authority,
        paper_contract_frozen=_paper_contract_frozen(repository / PAPER_CONTRACT),
        four_row_plan_frozen=plan_frozen,
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
    paper_contract_frozen: bool,
    four_row_plan_frozen: bool,
    source_commit_valid: bool,
    remote_main_valid: bool,
    source_commit_exists: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    protected_worktree_clean: bool,
) -> dict[str, Any]:
    evidence = {
        "k1u_selected_invariant_authority_complete": bool(authority)
        and all(authority.values()),
        "paper_resource_contract_frozen": paper_contract_frozen,
        "four_row_closeout_plan_frozen": four_row_plan_frozen,
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
        decision = "innovation1_uknit_k1ca_launch_evidence_invalid"
        next_action = (
            "repair only the failed K1-U authority, resource contract, plan, "
            "readiness, or source check"
        )
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_uknit_k1ca_source_not_published"
        next_action = "publish and verify the exact K1-CA source commit on GitHub main"
    else:
        status = "pass"
        decision = "innovation1_uknit_k1ca_remote_launch_authorized"
        next_action = (
            "perform one bounded GPU0/run-root check, launch the exact run-owned "
            "clean clone, confirm durable start/cache progress, and hand off to tmux"
        )
    return {
        "run_id": RUN_ID,
        "task": "innovation1_uknit_k1ca_remote_launch_gate",
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
        "claim_scope": (
            "local authorization for one bounded uKNIT K1-CA paper-closeout "
            "matrix; no remote result claim"
        ),
        "blocked_actions": [
            "launching unpublished or worktree-drifted source",
            "using scp or a dirty source overlay",
            "running 262144/class locally",
            "adding rows, seeds, final tests, or changing the frozen protocol",
        ],
    }


def _k1u_authority(root: Path) -> dict[str, bool]:
    gate_path = root / "gate.json"
    checks = {"k1u_gate_sha256_exact": _sha256(gate_path) == K1U_GATE_SHA256}
    try:
        gate = _read_json(gate_path)
    except (OSError, json.JSONDecodeError, ValueError):
        checks["k1u_gate_json_readable"] = False
        return checks
    seeds = gate.get("seed_results", {})
    checks.update(
        {
            "k1u_gate_json_readable": True,
            "k1u_identity_selects_invariant": gate.get("status") == "hold"
            and gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1u_medium_signal_without_position_necessity"
            and gate.get("next_action")
            == "hold scale and replace the candidate by the simpler invariant branch",
            "k1u_protocol_checks_complete": bool(gate.get("protocol_checks"))
            and all(gate.get("protocol_checks", {}).values())
            and not gate.get("failed_protocol_checks"),
            "k1u_semantic_signal_passes_both_seeds": gate.get(
                "descriptive_diagnostics", {}
            ).get("exact_signal_both_seeds")
            is True
            and gate.get("descriptive_diagnostics", {}).get(
                "wrong_sbox_attribution_both_seeds"
            )
            is True,
            "k1u_invariant_outperforms_exact_both_seeds": all(
                float(seeds.get(str(seed), {}).get("exact_minus_invariant", 1.0)) < 0.0
                for seed in (3, 4)
            ),
        }
    )
    return checks


def _paper_contract_frozen(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    required = (
        "至多一次 `262144/class`、seed 3/4、位置不变候选与 AutoND 的四行矩阵",
        "预期产生四次跨模型复用",
        "该矩阵完成后，无论结果是否通过预设优势门，论文实验阶段均收口",
        "百万级、第三个及更多 seed、五次最终测试、更多轮数或更多网络族均不属于默认补全项",
    )
    return all(item in text for item in required)


def _git_ok(repository: Path, *args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args], cwd=repository, capture_output=True, check=False
        ).returncode
        == 0
    )


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


__all__ = [
    "K1U_GATE",
    "PLAN",
    "REMOTE_CONFIG",
    "REMOTE_RUN_ID",
    "REQUIRED_SOURCE_ASSETS",
    "RUN_ID",
    "adjudicate_launch",
    "build_launch_gate",
]
