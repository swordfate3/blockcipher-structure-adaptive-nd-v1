from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from blockcipher_nd.cli.synthesize_cross_spn_e4 import main
from blockcipher_nd.planning.cross_spn_e4_synthesis import (
    build_cross_spn_e4_synthesis,
)


def _gate(
    *,
    source_seed: int,
    target_seed: int,
    scratch_delta: float,
    scratch_ci: tuple[float, float],
    source_delta: float,
    target_delta: float,
) -> dict:
    stage = "e4_r4" if source_seed == 0 else "e4_r5"
    true_auc = 0.58
    return {
        "status": "pass",
        "decision": (
            f"{stage}_target_adaptation_efficiency_confirmed"
            if scratch_delta >= 0.004 and scratch_ci[0] > 0.0
            else f"{stage}_target_adaptation_signal_unstable"
        ),
        "errors": [],
        "expected_seed": target_seed,
        "samples_per_class": 65536,
        "epochs": 1,
        "experiment_stage": stage,
        "alignment": {"status": "pass", "errors": []},
        "score_rows": 65536,
        "score_pairing": "same validation cache, identical sample_ids and labels",
        "research_decision_applied": True,
        "aucs": {
            "true_to_true": true_auc,
            "typed_scratch": true_auc - scratch_delta,
            "shuffled_to_true": true_auc - source_delta,
            "true_to_shuffled": true_auc - target_delta,
        },
        "margins": {
            "scratch_margin": scratch_delta,
            "source_topology_margin": source_delta,
            "target_topology_margin": target_delta,
        },
        "bootstrap": {
            "confidence": 0.95,
            "method": "paired label-stratified fixed-size nonparametric bootstrap",
            "replicates": 10000,
            "seed": 20260715,
            "comparisons": {
                "typed_scratch": {
                    "point_difference": scratch_delta,
                    "ci_lower": scratch_ci[0],
                    "ci_upper": scratch_ci[1],
                },
                "shuffled_to_true": {
                    "point_difference": source_delta,
                    "ci_lower": source_delta - 0.002,
                    "ci_upper": source_delta + 0.002,
                },
                "true_to_shuffled": {
                    "point_difference": target_delta,
                    "ci_lower": target_delta - 0.004,
                    "ci_upper": target_delta + 0.004,
                },
            },
        },
        "thresholds": {
            "scratch_margin": 0.004,
            "source_topology_margin": 0.005,
            "target_topology_margin": 0.003,
            "core_scratch_ci_lower_strictly_greater_than": 0.0,
        },
        "source_pretraining_cost": {"seed": source_seed},
    }


def _reports() -> list[dict]:
    return [
        _gate(
            source_seed=0,
            target_seed=2,
            scratch_delta=0.011,
            scratch_ci=(0.008, 0.014),
            source_delta=0.012,
            target_delta=0.077,
        ),
        _gate(
            source_seed=0,
            target_seed=3,
            scratch_delta=0.006,
            scratch_ci=(0.003, 0.009),
            source_delta=0.010,
            target_delta=0.081,
        ),
        _gate(
            source_seed=1,
            target_seed=4,
            scratch_delta=0.0002,
            scratch_ci=(-0.002, 0.003),
            source_delta=0.015,
            target_delta=0.069,
        ),
        _gate(
            source_seed=1,
            target_seed=5,
            scratch_delta=0.0038,
            scratch_ci=(0.001, 0.006),
            source_delta=0.013,
            target_delta=0.071,
        ),
    ]


def test_e4_synthesis_separates_robust_topology_from_conditional_scratch() -> None:
    report = build_cross_spn_e4_synthesis(_reports())

    assert report["status"] == "pass"
    assert report["decision"] == (
        "e4_typed_topology_attribution_robust_scratch_efficiency_conditional"
    )
    assert report["comparisons"]["scratch_margin"]["pass_count"] == 2
    assert report["comparisons"]["scratch_margin"]["ci_positive_count"] == 3
    assert report["comparisons"]["source_topology_margin"]["pass_count"] == 4
    assert report["comparisons"]["target_topology_margin"]["pass_count"] == 4
    assert report["source_strata"]["0"]["target_seeds"] == [2, 3]
    assert report["source_strata"]["1"]["target_seeds"] == [4, 5]
    assert report["inference_boundary"]["pooled_confidence_interval"] is False


def test_e4_synthesis_rejects_wrong_source_target_mapping() -> None:
    reports = _reports()
    reports[-1]["source_pretraining_cost"]["seed"] = 0

    report = build_cross_spn_e4_synthesis(reports)

    assert report["status"] == "fail"
    assert any("target seed 5 requires source seed 1" in error for error in report["errors"])


def test_e4_synthesis_rejects_unaligned_gate() -> None:
    reports = _reports()
    reports[0]["alignment"] = {"status": "fail", "errors": ["mismatch"]}

    report = build_cross_spn_e4_synthesis(reports)

    assert report["status"] == "fail"
    assert any("alignment must pass" in error for error in report["errors"])


def test_e4_synthesis_cli_writes_jsonl_csv_gate_and_svg(tmp_path: Path) -> None:
    gate_paths = []
    for report in _reports():
        path = tmp_path / f"seed{report['expected_seed']}.json"
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        gate_paths.append(path)
    output = tmp_path / "synthesis"

    status = main(
        [
            "--gates",
            *(str(path) for path in gate_paths),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("scratch_efficiency_conditional")
    result_rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["target_seed"] for row in result_rows] == [2, 3, 4, 5]
    with (output / "cells.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    ET.parse(output / "curves.svg")


def test_e4_synthesis_invalid_rerun_removes_stale_success_artifacts(
    tmp_path: Path,
) -> None:
    gate_paths = []
    for report in _reports():
        path = tmp_path / f"seed{report['expected_seed']}.json"
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        gate_paths.append(path)
    output = tmp_path / "synthesis"
    args = [
        "--gates",
        *(str(path) for path in gate_paths),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    invalid = json.loads(gate_paths[0].read_text(encoding="utf-8"))
    invalid["alignment"] = {"status": "fail", "errors": ["mismatch"]}
    gate_paths[0].write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    assert main(args) == 1

    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "fail"
    for name in ("results.jsonl", "cells.csv", "summary.json", "curves.svg"):
        assert not (output / name).exists()
