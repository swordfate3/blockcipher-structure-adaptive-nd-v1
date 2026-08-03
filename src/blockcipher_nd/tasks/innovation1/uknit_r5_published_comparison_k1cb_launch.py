from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca import (
    RUN_ID as K1CA_RUN_ID,
    result_protocol_frozen as k1ca_result_protocol_frozen,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_published_comparison_k1cb import (
    RUN_ID as REMOTE_RUN_ID,
    plan_protocol_frozen,
    read_tasks,
)


RUN_ID = "i1_uknit_r5_k1cb_published_comparison_launch_gate_20260803"
PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_r5_k1cb_published_comparison_262144_seed3_seed4.csv"
)
REMOTE_CONFIG = Path(
    "configs/remote/"
    "innovation1_uknit_k1cb_published_comparison_262144_seed3_seed4_gpu0_20260803.json"
)
EXPERIMENT_PLAN = Path(
    "docs/experiments/"
    "innovation1-uknit-r5-k1cb-published-network-paper-comparison-plan.md"
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
    Path("scripts/check-uknit-r5-k1cb-cache"),
    Path("scripts/check-uknit-r5-k1cb-paper-comparison-launch"),
    Path("scripts/gate-uknit-r5-k1cb-paper-comparison"),
    Path("scripts/package-uknit-r5-k1cb-paper-comparison"),
    Path("src/blockcipher_nd/cli/check_uknit_r5_k1cb_cache.py"),
    Path("src/blockcipher_nd/cli/check_uknit_r5_k1cb_paper_comparison_launch.py"),
    Path("src/blockcipher_nd/cli/gate_uknit_r5_k1cb_paper_comparison.py"),
    Path("src/blockcipher_nd/cli/package_uknit_r5_k1cb_paper_comparison.py"),
    Path("src/blockcipher_nd/tasks/innovation1/uknit_r5_published_comparison_k1cb.py"),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "uknit_r5_invariant_autond_closeout_k1ca.py"
    ),
    Path(
        "src/blockcipher_nd/tasks/innovation1/"
        "uknit_r5_published_comparison_k1cb_launch.py"
    ),
    Path("src/blockcipher_nd/registry/model_families/spn.py"),
    Path("src/blockcipher_nd/models/structure/spn/gift_pairset_baselines.py"),
    Path("src/blockcipher_nd/models/structure/spn/present_zhang_wang_keras.py"),
    Path("src/blockcipher_nd/models/structure/spn/published_architecture_adapters.py"),
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
    source_k1ca_root: Path,
    source_commit: str,
    remote_main_sha: str,
    readiness_status: str,
) -> dict[str, Any]:
    authority = _k1ca_authority(source_k1ca_root)
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
        plan_frozen = plan_protocol_frozen(read_tasks(repository / PLAN))
    except (OSError, ValueError):
        pass
    return adjudicate_launch(
        source_commit=source_commit,
        remote_main_sha=remote_main_sha,
        readiness_status=readiness_status,
        authority=authority,
        paper_contract_frozen=_paper_contract_frozen(repository / PAPER_CONTRACT),
        six_row_plan_frozen=plan_frozen,
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
    six_row_plan_frozen: bool,
    source_commit_valid: bool,
    remote_main_valid: bool,
    source_commit_exists: bool,
    source_assets_committed: bool,
    source_assets_match: bool,
    protected_worktree_clean: bool,
) -> dict[str, Any]:
    evidence = {
        "k1ca_protocol_valid_cache_authority_complete": bool(authority)
        and all(authority.values()),
        "paper_comparison_resource_contract_frozen": paper_contract_frozen,
        "six_row_published_network_plan_frozen": six_row_plan_frozen,
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
        decision = "innovation1_uknit_k1cb_launch_evidence_invalid"
        next_action = (
            "repair only the failed K1-CA authority, paper contract, K1-CB plan, "
            "readiness, or committed-source check"
        )
    elif not ssh_allowed:
        status = "hold"
        decision = "innovation1_uknit_k1cb_source_not_published"
        next_action = "publish and verify the exact K1-CB source commit on GitHub main"
    else:
        status = "pass"
        decision = "innovation1_uknit_k1cb_remote_launch_authorized"
        next_action = (
            "perform one bounded remote launch and start confirmation; the remote "
            "cache audit must still stop before training on any missing or mismatched "
            "K1-CA cache"
        )
    return {
        "run_id": RUN_ID,
        "task": "innovation1_uknit_k1cb_remote_launch_gate",
        "remote_run_id": REMOTE_RUN_ID,
        "source_run_id": K1CA_RUN_ID,
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
            "local authorization for one cache-reusing uKNIT K1-CB paper "
            "comparison; no remote result or performance claim"
        ),
        "blocked_actions": [
            "launching before protocol-valid K1-CA retrieval",
            "launching unpublished or worktree-drifted source",
            "using scp or a dirty source overlay",
            "regenerating a missing or mismatched K1-CA cache",
            "adding models, seeds, final tests, or changing the frozen protocol",
        ],
    }


def _k1ca_authority(root: Path) -> dict[str, bool]:
    gate_path = _first_file(
        root,
        ("gate.local.json", "local_adjudication/gate.json", "gate.json"),
    )
    results_path = _first_file(root, ("results.jsonl", "results/results.jsonl"))
    cache_manifest_path = _first_file(
        root,
        (
            "cache_manifest.json",
            f"source/results_archive/{K1CA_RUN_ID}/cache_manifest.json",
        ),
    )
    checks = {
        "k1ca_gate_present": gate_path is not None,
        "k1ca_results_present": results_path is not None,
        "k1ca_cache_manifest_present": cache_manifest_path is not None,
    }
    try:
        gate = _read_json(gate_path) if gate_path is not None else {}
        rows = _read_jsonl(results_path) if results_path is not None else []
        cache_manifest = (
            _read_json(cache_manifest_path) if cache_manifest_path is not None else {}
        )
    except (OSError, json.JSONDecodeError, ValueError):
        checks["k1ca_authority_readable"] = False
        return checks
    protocol_checks = gate.get("protocol_checks", {})
    caches = cache_manifest.get("caches", [])
    splits = [str(item.get("split", "")) for item in caches if isinstance(item, dict)]
    checks.update(
        {
            "k1ca_authority_readable": True,
            "k1ca_gate_protocol_valid": gate.get("run_id") == K1CA_RUN_ID
            and gate.get("status") in {"pass", "hold"}
            and bool(protocol_checks)
            and all(protocol_checks.values())
            and not gate.get("failed_protocol_checks"),
            "k1ca_results_protocol_frozen": k1ca_result_protocol_frozen(rows),
            "k1ca_four_cache_manifest_entries": cache_manifest.get("count") == 4
            and isinstance(caches, list)
            and len(caches) == 4,
            "k1ca_train_validation_caches_only": splits.count("train") == 2
            and splits.count("validation") == 2
            and not any(split.startswith("final_test") for split in splits),
        }
    )
    return checks


def _paper_contract_frozen(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    required = (
        "K1-CB 公开网络主表已冻结",
        "严格复用 K1-CA 四缓存",
        "预期产生 12 次复用和零缓存创建",
        "K1-CB 不设性能淘汰门",
        "不增加数据、seed、pair、epoch 或最终测试",
    )
    return all(item in text for item in required)


def _first_file(root: Path, candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


__all__ = [
    "PLAN",
    "REMOTE_CONFIG",
    "REMOTE_RUN_ID",
    "REQUIRED_SOURCE_ASSETS",
    "RUN_ID",
    "adjudicate_launch",
    "build_launch_gate",
]
