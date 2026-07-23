from __future__ import annotations

import csv
import json

from blockcipher_nd.cli import run_runtime_parameterized_spn_readiness as cli
from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_readiness import (
    RuntimeSpnReadinessConfig,
    readiness_values_are_finite,
    run_runtime_spn_readiness,
)


def test_runtime_spn_readiness_passes_all_implementation_checks() -> None:
    result = run_runtime_spn_readiness(RuntimeSpnReadinessConfig(run_id="unit"))

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["checks_passed"] == result["gate"]["checks_total"]
    assert result["gate"]["empirical_topology_superiority_tested"] is False
    assert len(result["rows"]) == 4
    assert {row["block_bits"] for row in result["rows"]} == {64, 128}
    assert {row["linear_layer_kind"] for row in result["rows"]} == {
        "permutation",
        "general_gf2",
    }
    assert len({row["parameter_count"] for row in result["rows"]}) == 1
    assert readiness_values_are_finite(result)
    assert len(result["cell_rows"]) == 16 + 16 + 16 + 32


def test_runtime_spn_readiness_cli_writes_complete_artifacts(tmp_path) -> None:
    exit_code = cli.main(
        [
            "--run-id",
            "cli-unit",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert {
        "results.jsonl",
        "progress.jsonl",
        "cells.csv",
        "contract.json",
        "summary.json",
        "gate.json",
        "curves.svg",
    } <= {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "pass"
    progress = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in progress] == ["run_start", "run_done"]
    with (tmp_path / "cells.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 80
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "运行时结构参数化 SPN 区分器就绪审判" in svg
    assert "不含训练或 AUC" in svg
    assert "15/15 项通过" in svg
