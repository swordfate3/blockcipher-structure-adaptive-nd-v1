from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_integral_position_prior import main
from blockcipher_nd.tasks.innovation2.integral_position_prior_audit import (
    SELECTORS,
    PositionPriorAuditConfig,
    PositionPriorThresholds,
    adjudicate_position_prior_audit,
    evaluate_position_prior_audit,
    select_position_prior_controls,
    training_output_position_priors,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    IntegralStructure,
    make_keys,
    make_structure_splits,
)


def test_training_position_priors_are_grouped_by_output_nibble() -> None:
    structures = tuple(
        IntegralStructure(
            structure_id=f"s-{index}",
            active_nibble=index,
            output_nibble=index,
            output_mask=1,
            fixed_plaintext=0,
        )
        for index in range(16)
    )
    parities = np.asarray(
        [[index & 1, index & 1] for index in range(16)],
        dtype=np.uint8,
    )

    priors = training_output_position_priors(structures, parities)

    assert priors == {index: float(index & 1) for index in range(16)}


def test_position_matched_controls_preserve_candidate_histogram() -> None:
    rows, _, _ = _source_fixture()
    priors = {index: index / 16 for index in range(16)}
    selections = select_position_prior_controls(
        rows,
        position_priors=priors,
        top_k=16,
        matched_random_seed=2026071602,
    )
    structures = {
        row["signature"]: IntegralStructure(
            structure_id=row["structure_id"],
            active_nibble=int(row["active_nibble"]),
            output_nibble=int(row["output_nibble"]),
            output_mask=int(row["output_mask"], 2),
            fixed_plaintext=int(row["signature"].rsplit("-p", 1)[1], 16),
        )
        for row in rows
    }

    def histogram(selector: str) -> Counter[int]:
        return Counter(
            structures[signature].output_nibble
            for signature in selections[selector]
        )

    assert set(selections) == set(SELECTORS)
    assert histogram("position_matched_linear") == histogram("structure_mlp")
    assert histogram("position_matched_random") == histogram("structure_mlp")


def test_position_prior_audit_readiness_reconstructs_frozen_source() -> None:
    rows, gate, summary = _source_fixture()
    result = evaluate_position_prior_audit(
        PositionPriorAuditConfig(
            run_id="position-prior-readiness-test",
            top_k=4,
            fresh_keys=8,
            key_seed=2026071601,
            matched_random_seed=2026071602,
            experiment_seed=0,
            gate_mode="position-prior-smoke",
            structure_chunk_size=4,
        ),
        ranking_rows=rows,
        ranking_gate=gate,
        source_summary=summary,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_integral_position_prior_audit_ready"
    )
    assert all(result["gate"]["readiness_checks"].values())
    assert len(result["rows"]) == 4
    assert len(result["rate_rows"]) == 16
    assert len(result["position_rows"]) == 16


def test_position_prior_gate_preserves_conditional_branch() -> None:
    selector_rows = [
        _selector_summary("structure_mlp", 0.95),
        _selector_summary("train_output_position_prior", 0.96),
        _selector_summary("position_matched_linear", 0.91),
        _selector_summary("position_matched_random", 0.90),
    ]
    gate = adjudicate_position_prior_audit(
        PositionPriorAuditConfig(
            run_id="position-prior-gate-test",
            top_k=16,
            fresh_keys=4096,
            key_seed=2026071601,
            matched_random_seed=2026071602,
        ),
        selector_rows=selector_rows,
        readiness_checks={"source_valid": True},
        thresholds=PositionPriorThresholds(),
        source_run_id="source",
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_integral_position_prior_dominant_with_conditional_residual"
    )
    assert gate["attribution_checks"] == {
        "candidate_position_prior_advantage_at_least_0_03": False,
        "candidate_matched_linear_advantage_at_least_0_02": True,
        "candidate_matched_random_advantage_at_least_0_03": True,
    }


def test_position_prior_cli_writes_complete_chinese_artifacts(
    tmp_path: Path,
) -> None:
    rows, gate, summary = _source_fixture()
    ranking_root = tmp_path / "ranking"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    ranking_root.mkdir()
    source_root.mkdir()
    with (ranking_root / "ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ranking_root / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    (source_root / "dataset_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    status = main(
        [
            "--run-id",
            "position-prior-cli-test",
            "--ranking-root",
            str(ranking_root),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--top-k",
            "2",
            "--fresh-keys",
            "8",
            "--key-seed",
            "2026071601",
            "--matched-random-seed",
            "2026071602",
            "--experiment-seed",
            "0",
            "--structure-chunk-size",
            "4",
            "--gate-mode",
            "position-prior-smoke",
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "fresh_key_rates.csv",
        "train_position_priors.csv",
        "selector_overlaps.csv",
        "gate.json",
        "metadata.json",
        "curves.svg",
        "progress.jsonl",
    ):
        assert (output_root / name).is_file()
    svg = (output_root / "curves.svg").read_text(encoding="utf-8")
    assert "创新2 E6" in svg
    assert "输出位置先验" in svg


def _source_fixture() -> tuple[
    list[dict[str, str]],
    dict[str, object],
    dict[str, object],
]:
    counts = {"train": 512, "validation": 16, "calibration": 16, "test": 128}
    offsets = {"train": 101, "validation": 301, "calibration": 701, "test": 501}
    structures = make_structure_splits(
        split_counts=counts,
        seed=0,
        structure_split_mode="geometry-disjoint",
        random_seed_offsets=offsets,
    )
    rows: list[dict[str, str]] = []
    for index, structure in enumerate(structures["test"]):
        rows.append(
            {
                "structure_id": structure.structure_id,
                "signature": structure.signature,
                "active_nibble": str(structure.active_nibble),
                "output_nibble": str(structure.output_nibble),
                "output_mask": f"{structure.output_mask:04b}",
                "anchor_rank": str(128 - index),
                "candidate_rank": str(index + 1),
            }
        )
    key_specs = {
        "train": (4, 201),
        "validation": (4, 401),
        "calibration": (4, 801),
        "test": (4, 601),
        "stability": (4, 1001),
    }
    summary_splits: dict[str, object] = {}
    for name in ("train", "validation", "calibration", "test"):
        key_count, key_seed = key_specs[name]
        summary_splits[name] = {
            "structures": counts[name],
            "geometry_ids": sorted(item.geometry_id for item in structures[name]),
            "keys_per_structure": key_count,
            "keys": [
                f"{key:020X}" for key in make_keys(count=key_count, seed=key_seed)
            ],
        }
    stability_count, stability_seed = key_specs["stability"]
    summary_splits["stability"] = {
        "structures": counts["test"],
        "keys_per_structure": stability_count,
        "keys": [
            f"{key:020X}"
            for key in make_keys(count=stability_count, seed=stability_seed)
        ],
    }
    summary = {
        "structure_split_mode": "geometry-disjoint",
        "geometry_splits_disjoint": True,
        "one_structure_per_geometry": True,
        "splits": summary_splits,
    }
    gate = {
        "run_id": "i2_present_r5_integral_parity_geometry_holdout_ranking_seed0",
        "status": "pass",
        "decision": "innovation2_integral_geometry_holdout_passed",
        "structure_split_mode": "geometry-disjoint",
    }
    return rows, gate, summary


def _selector_summary(selector: str, mean: float) -> dict[str, object]:
    return {
        "selector": selector,
        "mean_balance_rate": mean,
    }
