from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r9_r10_atm_source_generation_resume_readiness import (
    render_source_generation_readiness,
)
from blockcipher_nd.tasks.innovation2.present_r9_r10_atm_source_generation_resume_readiness import (
    E100_DECISION,
    E100_GATE_SHA256,
    EXPECTED_R10_SPLITS,
    EXPECTED_R9_SPLITS,
    SourceGenerationResumeConfig,
    _inspect_search_source,
    _read_notebook_contract,
    adjudicate_source_generation_readiness,
)


def test_notebook_contract_extracts_rounds_limit_and_threads(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["NThreads = 36\n"]},
            {
                "cell_type": "code",
                "source": [
                    f"splits = {list(EXPECTED_R9_SPLITS)!r}\n",
                    "limit = 2**10\n",
                    "fname = f\"Results/R9-complex-oracle-{split[0]}.pkl\"\n",
                    "if os.path.isfile(fname):\n    pass\nelse:\n",
                    "    res = search_integral_properties(Avec, 64, 64, True, NThreads)\n",
                    "    with open(fname, 'wb') as f:\n        pickle.dump(res, f)\n",
                    "    with open(fname_stats, 'w') as f:\n        f.write('stats')\n",
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    f"splits = {list(EXPECTED_R10_SPLITS)!r}\n",
                    "limit = 2**10\n",
                    "fname = f\"Results/R10-complex-oracle-{split[0]}.pkl\"\n",
                    "if os.path.isfile(fname):\n    pass\nelse:\n",
                    "    res = search_integral_properties(Avec, 64, 64, True, NThreads)\n",
                    "    with open(fname, 'wb') as f:\n        pickle.dump(res, f)\n",
                    "    with open(fname_stats, 'w') as f:\n        f.write('stats')\n",
                ],
            },
        ]
    }
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    contract = _read_notebook_contract(path)
    assert contract["threads"] == 36
    assert contract["rounds"][9]["splits"] == EXPECTED_R9_SPLITS
    assert contract["rounds"][10]["limit"] == 1024
    assert contract["rounds"][9]["writes_pickle_after_search"] is True
    assert contract["rounds"][9]["has_progress_jsonl"] is False


def test_search_source_detects_blocking_map_without_resume(tmp_path: Path) -> None:
    path = tmp_path / "Search.py"
    path.write_text(
        "def search(pool, fn, values):\n    return pool.map(fn, values, 1)\n",
        encoding="utf-8",
    )
    contract = _inspect_search_source(path)
    assert contract["blocking_pool_map_calls"] == 1
    assert contract["incremental_pool_calls"] == 0
    assert contract["mentions_progress"] is False


def _audit() -> dict[str, object]:
    return {
        "checks": {"source": True},
        "metrics": {
            "missing_declared_splits": 10,
            "resume_contract_checks": 9,
            "resume_contract_passes": 2,
            "environment_contract_checks": 10,
            "environment_contract_passes": 3,
            "historical_min_seconds": 2714.0,
            "historical_median_seconds": 10000.0,
            "historical_max_seconds": 23786.0,
            "historical_min_oracle_calls": 2_666_126,
            "historical_median_oracle_calls": 100_000_000,
            "historical_max_oracle_calls": 220_841_578_547,
        },
        "resume_rows": [
            {"check": name, "passed": False}
            for name in (
                "started_marker",
                "progress_jsonl",
                "incremental_candidate_or_layer_cache",
                "parameter_matched_resume",
                "atomic_completion",
                "nonblocking_incremental_result_boundary",
                "resume_fixture_verified",
            )
        ],
        "environment_rows": [
            {"check": "requirements_version_pinned", "passed": False},
            {"check": "bitarray_build_abi_recorded", "passed": False},
            {"check": "all_required_modules_discoverable", "passed": False},
        ],
    }


def test_e101_holds_long_search_until_resumable_runner_exists() -> None:
    gate = adjudicate_source_generation_readiness(
        SourceGenerationResumeConfig(),
        audit=_audit(),
        e100_gate={"status": "hold", "decision": E100_DECISION},
        e100_gate_hash=E100_GATE_SHA256,
    )
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_high_round_resumable_runner_required"
    assert gate["next_action"]["runner_implementation_open"] is True
    assert gate["next_action"]["long_search_open"] is False
    assert gate["next_action"]["remote_scale"] is False


def test_e101_plot_explains_coverage_cost_and_resume_gap(tmp_path: Path) -> None:
    split_rows = [
        {"rounds": rounds, "pickle_present": rounds == 9 and index < 8}
        for rounds in (9, 10)
        for index in range(9)
    ]
    cost_rows = [
        {"split": f"1-{index + 1}-1", "hours": 0.75 + index, "oracle_calls": 10 ** (index + 3)}
        for index in range(8)
    ]
    resume_rows = [
        {"check": name, "required_for_generation": True, "passed": False}
        for name in (
            "started_marker",
            "progress_jsonl",
            "incremental_candidate_or_layer_cache",
            "parameter_matched_resume",
            "atomic_completion",
            "nonblocking_incremental_result_boundary",
            "resume_fixture_verified",
        )
    ]
    summary = {
        "split_coverage": split_rows,
        "historical_costs": cost_rows,
        "resume_contract": resume_rows,
        "gate": {
            "decision": "innovation2_present_high_round_resumable_runner_required",
            "metrics": {
                "historical_median_seconds": 10_000,
                "historical_max_oracle_calls": 220_841_578_547,
            },
        },
    }
    output = tmp_path / "curves.svg"
    render_source_generation_readiness(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "九/十轮ATM新来源生成就绪审计" in svg
    assert "声明18个" in svg
    assert "长任务恢复契约" in svg
    assert "先实现并验证可恢复runner" in svg
    assert "没有执行ATM搜索" in svg
