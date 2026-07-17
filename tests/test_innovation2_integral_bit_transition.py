from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_integral_bit_transition import main
from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    ACTIVE_BIT_WIDTHS,
    BitIntegralStructure,
    BitTransitionAuditConfig,
    adjudicate_bit_transition_audit,
    bit_integral_parity_matrix,
    make_bit_transition_structures,
    scalar_bit_integral_parity,
)
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)


def test_bit_structure_enumerates_complete_set_and_preserves_context() -> None:
    structure = BitIntegralStructure(
        structure_id="fixture",
        active_bits=(0, 3, 8, 17, 63),
        output_nibble=7,
        output_mask=0b1011,
        fixed_plaintext=0,
    )

    plaintexts = structure.plaintexts()

    assert plaintexts.shape == (32,)
    assert len(np.unique(plaintexts)) == 32
    assert all(
        int(value) & ~structure.active_word_mask == structure.fixed_plaintext
        for value in plaintexts
    )
    for active_bit in structure.active_bits:
        assert {(int(value) >> active_bit) & 1 for value in plaintexts} == {0, 1}
    features = structure.marginal_feature_vector()
    assert features.shape == (95,)
    assert int(features.sum()) == len(structure.active_bits) + 2


def test_vectorized_bit_parity_matches_scalar_present() -> None:
    structures = make_bit_transition_structures(
        active_bit_width=7,
        count=64,
        seed=31,
    )
    keys = (0x00010203040506070809, 0x00000000000000000000)
    round_keys = present_round_key_matrix(keys, rounds=6)

    matrix = bit_integral_parity_matrix(
        structures[:2],
        round_keys,
        structure_chunk_size=1,
    )

    for structure_index, structure in enumerate(structures[:2]):
        for key_index, key in enumerate(keys):
            assert int(matrix[structure_index, key_index]) == (
                scalar_bit_integral_parity(
                    structure,
                    rounds=6,
                    key=key,
                )
            )


def test_bit_transition_gate_selects_strongest_passing_width() -> None:
    config = BitTransitionAuditConfig(
        run_id="gate",
        structures_per_width=64,
        keys_per_structure=8,
    )
    rows = []
    for width, residual, correlation in (
        (5, 0.019, 0.30),
        (6, 0.031, 0.35),
        (7, 0.032, 0.42),
    ):
        rows.append(
            {
                "active_bit_width": width,
                "q1_rate": 0.40,
                "cross_half_structure_std": 0.05,
                "cross_half_combined_marginal_residual_std": residual,
                "cross_half_combined_marginal_residual_correlation": correlation,
                "mixed_structure_fraction": 1.0,
            }
        )

    gate = adjudicate_bit_transition_audit(config, rows, {"fixture": True})

    assert gate["status"] == "pass"
    assert gate["passing_active_bit_widths"] == [6, 7]
    assert gate["selected_active_bit_width"] == 6
    assert gate["next_action"]["remote_scale"] is False


def test_bit_transition_cli_writes_complete_audit_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "bit-transition"

    status = main(
        [
            "--run-id",
            "i2_bit_transition_test",
            "--output-root",
            str(output),
            "--structures-per-width",
            "64",
            "--keys-per-structure",
            "8",
            "--structure-chunk-size",
            "2",
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "structure_rates.csv",
        "marginal_predictions.csv",
        "progress.jsonl",
        "gate.json",
        "metadata.json",
        "curves.svg",
    ):
        assert (output / name).is_file()
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert [row["active_bit_width"] for row in rows] == list(ACTIVE_BIT_WIDTHS)
    assert gate["readiness_checks"]["vectorized_parity_matches_scalar"] is True
    assert "细粒度活动 bit 审计" in svg
    assert "无神经训练" in svg
