from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.evaluate_innovation2_integral_fresh_keys import main
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    SELECTORS,
    FreshKeyThresholds,
    FreshKeyValidationConfig,
    adjudicate_fresh_key_enrichment,
    evaluate_fresh_key_enrichment,
    present_integral_parities,
    present_integral_parity_matrix,
    present_round_key_matrix,
    select_structures,
    zero_failure_upper_bound,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    IntegralStructure,
    integral_mask_parity,
    make_keys,
)


def test_vectorized_present_integral_parity_matches_scalar() -> None:
    structures = (
        IntegralStructure(
            structure_id="a",
            active_nibble=2,
            output_nibble=7,
            output_mask=0b1011,
            fixed_plaintext=0x123456789AB000EF,
        ),
        IntegralStructure(
            structure_id="b",
            active_nibble=13,
            output_nibble=0,
            output_mask=0b0101,
            fixed_plaintext=0x120056789ABCDEF0,
        ),
    )
    keys = make_keys(count=13, seed=2026071601)
    round_keys = present_round_key_matrix(keys, rounds=5)

    for structure in structures:
        actual = present_integral_parities(
            structure,
            round_keys,
            key_chunk_size=5,
        )
        expected = np.asarray(
            [
                integral_mask_parity(Present80(rounds=5, key=key), structure)
                for key in keys
            ],
            dtype=np.uint8,
        )
        assert np.array_equal(actual, expected)

    matrix = present_integral_parity_matrix(
        structures,
        round_keys,
        structure_chunk_size=1,
    )
    assert matrix.shape == (2, 13)
    for index, structure in enumerate(structures):
        assert np.array_equal(
            matrix[index],
            present_integral_parities(structure, round_keys, key_chunk_size=5),
        )


def test_frozen_selectors_are_deterministic_and_label_independent() -> None:
    rows = _ranking_rows()
    selected = select_structures(
        rows,
        top_k=16,
        random_selector_seed=20260716,
    )
    relabeled = [{**row, "observed_balance_rate_256key": "0.123"} for row in rows]
    selected_after_relabel = select_structures(
        relabeled,
        top_k=16,
        random_selector_seed=20260716,
    )

    assert set(selected) == set(SELECTORS)
    assert all(len(signatures) == 16 for signatures in selected.values())
    assert selected == selected_after_relabel
    assert selected["structure_mlp"] != selected["fixed_random"]


def test_fresh_key_enrichment_readiness_validates_source_and_parity() -> None:
    result = evaluate_fresh_key_enrichment(
        FreshKeyValidationConfig(
            run_id="fresh-key-readiness-test",
            top_k=4,
            fresh_keys=8,
            key_seed=2026071601,
            random_selector_seed=20260716,
            gate_mode="fresh-key-smoke",
            key_chunk_size=3,
        ),
        ranking_rows=_ranking_rows(),
        ranking_gate=_ranking_gate(),
        source_summary=_source_summary(),
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_integral_fresh_key_implementation_ready"
    )
    assert all(result["gate"]["readiness_checks"].values())
    assert len(result["rows"]) == 4
    assert len(result["rate_rows"]) == 16
    assert len(result["overlap_rows"]) == 6


def test_fresh_key_enrichment_gate_passes_frozen_margins() -> None:
    rows = [
        _selector_summary("structure_mlp", 0.95, 0.80, 7),
        _selector_summary("linear_same_input", 0.89, 0.70, 3),
        _selector_summary("p_layer_reachability", 0.85, 0.65, 2),
        _selector_summary("fixed_random", 0.72, 0.50, 0),
    ]
    gate = adjudicate_fresh_key_enrichment(
        FreshKeyValidationConfig(
            run_id="fresh-key-gate-test",
            top_k=16,
            fresh_keys=4096,
            key_seed=2026071601,
            random_selector_seed=20260716,
        ),
        selector_rows=rows,
        readiness_checks={"source_valid": True},
        thresholds=FreshKeyThresholds(),
        source_run_id="source",
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_integral_fresh_key_enrichment_passed"
    assert all(gate["enrichment_checks"].values())
    assert gate["zero_failure_check"] is True
    assert abs(zero_failure_upper_bound(4096) - 0.0007311125564753995) < 1e-12


def test_fresh_key_cli_writes_complete_chinese_artifacts(tmp_path: Path) -> None:
    ranking_root = tmp_path / "ranking"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    ranking_root.mkdir()
    source_root.mkdir()
    rows = _ranking_rows()
    with (ranking_root / "ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ranking_root / "gate.json").write_text(
        json.dumps(_ranking_gate()),
        encoding="utf-8",
    )
    (source_root / "dataset_summary.json").write_text(
        json.dumps(_source_summary()),
        encoding="utf-8",
    )

    status = main(
        [
            "--run-id",
            "fresh-key-cli-test",
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
            "--random-selector-seed",
            "20260716",
            "--key-chunk-size",
            "3",
            "--gate-mode",
            "fresh-key-smoke",
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "fresh_key_rates.csv",
        "selector_overlaps.csv",
        "gate.json",
        "metadata.json",
        "curves.svg",
        "progress.jsonl",
    ):
        assert (output_root / name).is_file()
    svg = (output_root / "curves.svg").read_text(encoding="utf-8")
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    assert "创新2 E5" in svg
    assert "统计验证，不是所有密钥证明" in svg
    assert gate["decision"] == (
        "innovation2_integral_fresh_key_implementation_ready"
    )


def _ranking_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(128):
        active_nibble, output_nibble = divmod(index, 8)
        output_mask = 1 << (index % 4)
        fixed_plaintext = (0x123456789ABCDEF0 + index) & ((1 << 64) - 1)
        fixed_plaintext &= ~(0xF << (4 * active_nibble))
        structure = IntegralStructure(
            structure_id=f"test-{index:06d}",
            active_nibble=active_nibble,
            output_nibble=output_nibble,
            output_mask=output_mask,
            fixed_plaintext=fixed_plaintext,
        )
        rows.append(
            {
                "structure_id": structure.structure_id,
                "signature": structure.signature,
                "active_nibble": str(active_nibble),
                "output_nibble": str(output_nibble),
                "output_mask": f"{output_mask:04b}",
                "observed_balance_rate_256key": str(index / 128),
                "anchor_rank": str(128 - index),
                "candidate_rank": str(index + 1),
            }
        )
    return rows


def _ranking_gate() -> dict[str, object]:
    return {
        "run_id": "i2_present_r5_integral_parity_geometry_holdout_ranking_seed0",
        "status": "pass",
        "decision": "innovation2_integral_geometry_holdout_passed",
        "structure_split_mode": "geometry-disjoint",
    }


def _source_summary() -> dict[str, object]:
    return {
        "structure_split_mode": "geometry-disjoint",
        "geometry_splits_disjoint": True,
        "one_structure_per_geometry": True,
        "splits": {
            "train": {"keys": ["00000000000000000001"]},
            "validation": {"keys": ["00000000000000000002"]},
            "calibration": {"keys": ["00000000000000000003"]},
            "test": {"keys": ["00000000000000000004"]},
            "stability": {"keys": ["00000000000000000005"]},
        },
    }


def _selector_summary(
    selector: str,
    mean: float,
    minimum: float,
    zero_count: int,
) -> dict[str, object]:
    return {
        "selector": selector,
        "mean_balance_rate": mean,
        "minimum_balance_rate": minimum,
        "maximum_balance_rate": 1.0,
        "zero_observed_failure_structures": zero_count,
    }
