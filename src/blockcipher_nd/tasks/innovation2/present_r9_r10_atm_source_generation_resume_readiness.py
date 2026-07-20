from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


RUN_ID = "i2_present_r9_r10_atm_source_generation_resume_readiness_20260720"
ATM_COMMIT = "b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b"
E100_GATE_SHA256 = "6875aa3995c5b1e7b26ad549cb50b65f4b90bb866f50bf0f15b6ed739cc296ef"
E100_DECISION = "innovation2_present_r9_coordinate_identity_anchor_remains_best"
EXPECTED_HASHES = {
    "Ciphers/PRESENT/PRESENT.ipynb": "8bf13c770dfd66eb3fcaa5568282d16fe82d1043d2ddc2f5209a89c023b7ca92",
    "Modelling/Search.py": "5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d",
    "Tools/AvecImplementations.py": "c0a9dd19f4c92e8ceeccde01b6e13de9599e05f2531ffb0c3ce7c31a9aacccae",
    "requirements.txt": "2485b8509e191570dba95cadc67ccd7eddb6684959d97ab4bbec7105bbd04502",
    "bitarrays/README.md": "d939b1273c29f577051ab0d43230b7835cbd6588b3cf5fe3be0a3715abf11d66",
}
EXPECTED_R9_SPLITS = (
    (1, 7, 1),
    (1, 6, 2),
    (1, 5, 3),
    (2, 6, 1),
    (2, 5, 2),
    (2, 4, 3),
    (3, 5, 1),
    (3, 4, 2),
    (3, 3, 3),
)
EXPECTED_R10_SPLITS = (
    (1, 8, 1),
    (1, 7, 2),
    (1, 6, 3),
    (2, 7, 1),
    (2, 6, 2),
    (2, 5, 3),
    (3, 6, 1),
    (3, 5, 2),
    (3, 4, 3),
)
MODULE_NAMES = {
    "pybind11": "pybind11",
    "ortools": "ortools",
    "python-sat": "pysat",
    "galois": "galois",
    "numpy": "numpy",
}


@dataclass(frozen=True)
class SourceGenerationResumeConfig:
    run_id: str = RUN_ID
    expected_threads: int = 36
    expected_limit: int = 2**10
    minimum_progress_boundaries: int = 2

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E101 run_id is frozen")
        if self.expected_threads != 36 or self.expected_limit != 2**10:
            raise ValueError("E101 notebook protocol is frozen")


def audit_source_generation_contract(
    atm_root: Path,
    *,
    actual_commit: str,
) -> dict[str, Any]:
    hashes = {relative: _sha256(atm_root / relative) for relative in EXPECTED_HASHES}
    notebook = _read_notebook_contract(atm_root / "Ciphers/PRESENT/PRESENT.ipynb")
    search = _inspect_search_source(atm_root / "Modelling/Search.py")
    oracle_source = (atm_root / "Tools/AvecImplementations.py").read_text(encoding="utf-8")
    requirements = _environment_contract(atm_root)
    split_rows = _split_coverage(atm_root, notebook)
    cost_rows = _historical_costs(atm_root / "Ciphers/PRESENT/Results")
    resume_rows = _resume_contract(notebook, search, oracle_source)
    checks = {
        "commit_matches": actual_commit == ATM_COMMIT,
        "source_hashes_match": hashes == EXPECTED_HASHES,
        "r9_splits_match": notebook["rounds"][9]["splits"] == EXPECTED_R9_SPLITS,
        "r10_splits_match": notebook["rounds"][10]["splits"] == EXPECTED_R10_SPLITS,
        "limit_matches": all(
            notebook["rounds"][rounds]["limit"] == 2**10 for rounds in (9, 10)
        ),
        "threads_match": notebook["threads"] == 36,
        "r9_public_file_count_replays": sum(
            row["rounds"] == 9 and row["pickle_present"] for row in split_rows
        )
        == 8,
        "r10_public_file_count_replays": sum(
            row["rounds"] == 10 and row["pickle_present"] for row in split_rows
        )
        == 0,
        "eight_historical_stats_parse": len(cost_rows) == 8,
    }
    metrics = {
        "declared_r9_splits": len(EXPECTED_R9_SPLITS),
        "public_r9_results": sum(
            row["rounds"] == 9 and row["pickle_present"] for row in split_rows
        ),
        "declared_r10_splits": len(EXPECTED_R10_SPLITS),
        "public_r10_results": sum(
            row["rounds"] == 10 and row["pickle_present"] for row in split_rows
        ),
        "missing_declared_splits": sum(not row["pickle_present"] for row in split_rows),
        "historical_min_seconds": min(row["seconds"] for row in cost_rows),
        "historical_median_seconds": median(row["seconds"] for row in cost_rows),
        "historical_max_seconds": max(row["seconds"] for row in cost_rows),
        "historical_min_oracle_calls": min(row["oracle_calls"] for row in cost_rows),
        "historical_median_oracle_calls": median(row["oracle_calls"] for row in cost_rows),
        "historical_max_oracle_calls": max(row["oracle_calls"] for row in cost_rows),
        "resume_contract_checks": len(resume_rows),
        "resume_contract_passes": sum(row["passed"] for row in resume_rows),
        "environment_contract_checks": len(requirements),
        "environment_contract_passes": sum(row["passed"] for row in requirements),
    }
    return {
        "actual_commit": actual_commit,
        "hashes": hashes,
        "notebook": notebook,
        "search": search,
        "checks": checks,
        "metrics": metrics,
        "split_rows": split_rows,
        "cost_rows": cost_rows,
        "resume_rows": resume_rows,
        "environment_rows": requirements,
    }


def adjudicate_source_generation_readiness(
    config: SourceGenerationResumeConfig,
    *,
    audit: dict[str, Any],
    e100_gate: dict[str, Any],
    e100_gate_hash: str,
) -> dict[str, Any]:
    protocol_checks = {
        "e100_gate_hash_matches": e100_gate_hash == E100_GATE_SHA256,
        "e100_status_hold": e100_gate.get("status") == "hold",
        "e100_decision_matches": e100_gate.get("decision") == E100_DECISION,
        **audit["checks"],
    }
    resume = {row["check"]: row["passed"] for row in audit["resume_rows"]}
    environment = {row["check"]: row["passed"] for row in audit["environment_rows"]}
    generation_checks = {
        "undeclared_high_round_target_exists": audit["metrics"]["missing_declared_splits"] > 0,
        "started_marker_present": resume["started_marker"],
        "progress_jsonl_present": resume["progress_jsonl"],
        "incremental_candidate_or_layer_cache": resume["incremental_candidate_or_layer_cache"],
        "parameter_matched_resume": resume["parameter_matched_resume"],
        "atomic_completion": resume["atomic_completion"],
        "nonblocking_incremental_result_boundary": resume[
            "nonblocking_incremental_result_boundary"
        ],
        "resume_fixture_verified": resume["resume_fixture_verified"],
        "requirements_version_pinned": environment["requirements_version_pinned"],
        "bitarray_build_abi_recorded": environment["bitarray_build_abi_recorded"],
        "all_required_modules_discoverable": environment["all_required_modules_discoverable"],
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_high_round_source_generation_audit_protocol_invalid"
        action = "repair frozen source, notebook, stats, or E100 replay"
    elif all(generation_checks.values()):
        status = "pass"
        decision = "innovation2_present_high_round_source_generation_ready"
        action = "preregister resumable R9 (3,3,3) generation only; keep R10 and remote scale closed"
    else:
        status = "hold"
        decision = "innovation2_present_high_round_resumable_runner_required"
        action = (
            "implement a route-owned per-candidate or per-layer resumable runner and verify "
            "interrupted-versus-uninterrupted equality on a small fixture before any long search"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "generation_checks": generation_checks,
        "metrics": audit["metrics"],
        "claim_scope": (
            "read-only source, historical-cost, environment, and resume audit for declared "
            "PRESENT r9/r10 ATM searches; no search or training was run, missing splits are not "
            "zero-dimensional or negative evidence, and this is not a new relation, published "
            "result, independent confirmation, PRESENT-80 evidence, distinguisher, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "runner_implementation_open": status == "hold",
            "long_search_open": status == "pass",
            "remote_scale": False,
            "training": False,
            "search_performed": False,
        },
    }


def result_rows(
    config: SourceGenerationResumeConfig,
    audit: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_r10_atm_source_generation_resume_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
        "search_performed": False,
    }
    return [
        {**common, "result_kind": "split", **row} for row in audit["split_rows"]
    ] + [
        {**common, "result_kind": "resume", **row} for row in audit["resume_rows"]
    ] + [
        {**common, "result_kind": "environment", **row}
        for row in audit["environment_rows"]
    ]


def serializable_config(config: SourceGenerationResumeConfig) -> dict[str, Any]:
    return asdict(config)


def _read_notebook_contract(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    threads = next(
        _assignment_value(source, "NThreads")
        for source in code_cells
        if re.search(r"\bNThreads\s*=", source)
    )
    rounds: dict[int, dict[str, Any]] = {}
    for round_count in (9, 10):
        marker = f"Results/R{round_count}-complex-oracle-"
        source = next(source for source in code_cells if marker in source)
        rounds[round_count] = {
            "splits": tuple(tuple(item) for item in _assignment_value(source, "splits")),
            "limit": _assignment_value(source, "limit"),
            "uses_final_pickle_reuse": "os.path.isfile(fname)" in source,
            "writes_pickle_after_search": source.index("search_integral_properties")
            < source.index("pickle.dump"),
            "writes_stats_after_search": source.index("search_integral_properties")
            < source.index("f.write"),
            "has_progress_jsonl": "progress.jsonl" in source,
            "has_started_marker": "started.marker" in source,
            "has_parameter_metadata": "metadata" in source,
        }
    return {"threads": threads, "rounds": rounds}


def _assignment_value(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return _safe_eval(node.value)
    raise ValueError(f"assignment {name} not found")


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_safe_eval(item) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _safe_eval(node.left) ** _safe_eval(node.right)
    raise ValueError(f"unsupported frozen notebook expression: {ast.dump(node)}")


def _inspect_search_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    ]
    return {
        "blocking_pool_map_calls": attributes.count("map"),
        "incremental_pool_calls": sum(
            attribute in {"imap", "imap_unordered", "apply_async"} for attribute in attributes
        ),
        "file_write_calls": sum(
            isinstance(node.func, ast.Name) and node.func.id == "open" for node in calls
        ),
        "mentions_progress": "progress" in source.lower(),
        "mentions_checkpoint": "checkpoint" in source.lower(),
        "mentions_resume": "resume" in source.lower(),
    }


def _split_coverage(atm_root: Path, notebook: dict[str, Any]) -> list[dict[str, Any]]:
    results = atm_root / "Ciphers/PRESENT/Results"
    rows: list[dict[str, Any]] = []
    for rounds in (9, 10):
        for split in notebook["rounds"][rounds]["splits"]:
            stem = f"R{rounds}-complex-oracle-{split[0]}-{split[1]}-{split[2]}"
            pickle_present = (results / f"{stem}.pkl").is_file()
            stats_present = (results / f"{stem}-stats.txt").is_file()
            rows.append(
                {
                    "rounds": rounds,
                    "split": "-".join(map(str, split)),
                    "declared": True,
                    "pickle_present": pickle_present,
                    "stats_present": stats_present,
                    "evidence_state": (
                        "published_result"
                        if rounds == 9 and pickle_present and stats_present
                        else "declared_without_public_result"
                    ),
                }
            )
    return rows


def _historical_costs(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"R9-complex-oracle-(\d+)-(\d+)-(\d+)-stats\.txt")
    for path in sorted(results_root.glob("R9-complex-oracle-*-stats.txt")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        seconds_text, calls_text = path.read_text(encoding="utf-8").strip().split()[:2]
        rows.append(
            {
                "rounds": 9,
                "split": "-".join(match.groups()),
                "seconds": float(seconds_text),
                "hours": float(seconds_text) / 3600.0,
                "oracle_calls": int(calls_text),
            }
        )
    return rows


def _resume_contract(
    notebook: dict[str, Any],
    search: dict[str, Any],
    oracle_source: str,
) -> list[dict[str, Any]]:
    notebook_rounds = notebook["rounds"].values()
    checks = {
        "started_marker": all(item["has_started_marker"] for item in notebook_rounds),
        "progress_jsonl": all(item["has_progress_jsonl"] for item in notebook_rounds),
        "incremental_candidate_or_layer_cache": search["file_write_calls"] > 0
        and search["incremental_pool_calls"] > 0,
        "parameter_matched_resume": all(item["has_parameter_metadata"] for item in notebook_rounds),
        "atomic_completion": "os.replace" in oracle_source or "Path.replace" in oracle_source,
        "nonblocking_incremental_result_boundary": search["incremental_pool_calls"] > 0
        and search["blocking_pool_map_calls"] == 0,
        "resume_fixture_verified": False,
        "final_pickle_reuse_only": all(
            item["uses_final_pickle_reuse"] for item in notebook_rounds
        ),
        "result_written_after_full_search": all(
            item["writes_pickle_after_search"] for item in notebook_rounds
        ),
    }
    required = {
        "started_marker",
        "progress_jsonl",
        "incremental_candidate_or_layer_cache",
        "parameter_matched_resume",
        "atomic_completion",
        "nonblocking_incremental_result_boundary",
        "resume_fixture_verified",
    }
    return [
        {
            "check": name,
            "required_for_generation": name in required,
            "passed": value,
        }
        for name, value in checks.items()
    ]


def _environment_contract(atm_root: Path) -> list[dict[str, Any]]:
    requirements = [
        line.strip()
        for line in (atm_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pinned = all(re.search(r"===|==", requirement) for requirement in requirements)
    module_rows = [
        {
            "check": f"module_{distribution}_discoverable",
            "required_for_generation": True,
            "passed": importlib.util.find_spec(module) is not None,
        }
        for distribution, module in MODULE_NAMES.items()
    ]
    bitset_artifacts = tuple((atm_root / "bitarrays").glob("bitset*.so"))
    rows = [
        {
            "check": "requirements_version_pinned",
            "required_for_generation": True,
            "passed": pinned,
        },
        {
            "check": "bitarray_build_command_documented",
            "required_for_generation": True,
            "passed": "g++ -O3" in (atm_root / "bitarrays/README.md").read_text(encoding="utf-8"),
        },
        {
            "check": "bitarray_compiled_artifact_present",
            "required_for_generation": True,
            "passed": bool(bitset_artifacts),
        },
        {
            "check": "bitarray_build_abi_recorded",
            "required_for_generation": True,
            "passed": False,
        },
        *module_rows,
    ]
    rows.append(
        {
            "check": "all_required_modules_discoverable",
            "required_for_generation": True,
            "passed": all(row["passed"] for row in module_rows),
        }
    )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return _sha256(path)
