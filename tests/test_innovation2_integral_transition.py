from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_integral_transition import main
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_transition_audit import (
    MultiNibbleIntegralStructure,
    TransitionAuditConfig,
    adjudicate_transition_audit,
    make_transition_structures,
    multi_nibble_parity_matrix,
    scalar_integral_parity,
)


def test_two_nibble_structure_enumerates_complete_set_and_preserves_context() -> None:
    structure = MultiNibbleIntegralStructure(
        structure_id="fixture",
        active_nibbles=(1, 4),
        output_nibble=7,
        output_mask=0b1111,
        fixed_plaintext=0x123456789AB00F0F,
    )

    plaintexts = structure.plaintexts()

    assert plaintexts.shape == (256,)
    assert len(np.unique(plaintexts)) == 256
    assert all(int(value) & ~structure.active_word_mask == structure.fixed_plaintext for value in plaintexts)
    assert {(int(value) >> 4) & 0xF for value in plaintexts} == set(range(16))
    assert {(int(value) >> 16) & 0xF for value in plaintexts} == set(range(16))


def test_vectorized_two_nibble_parity_matches_scalar_present() -> None:
    structures = make_transition_structures(
        active_nibble_count=2,
        count=16,
        seed=17,
    )
    keys = (0x00010203040506070809, 0x00000000000000000000)
    round_keys = present_round_key_matrix(keys, rounds=6)

    matrix = multi_nibble_parity_matrix(
        structures[:2],
        round_keys,
        structure_chunk_size=1,
    )

    for structure_index, structure in enumerate(structures[:2]):
        for key_index, key in enumerate(keys):
            assert int(matrix[structure_index, key_index]) == scalar_integral_parity(
                structure,
                rounds=6,
                key=key,
            )


def test_transition_gate_advances_only_with_residual_structure() -> None:
    config = TransitionAuditConfig(
        run_id="gate",
        structures_per_width=16,
        keys_per_structure=4,
    )
    rows = [
        {
            "active_nibble_count": 1,
            "q1_rate": 0.49,
            "balance_rate_std": 0.03,
            "output_position_residual_std": 0.02,
            "excess_balance_rate_std": 0.01,
            "excess_output_position_residual_std": 0.005,
            "mixed_structure_fraction": 1.0,
        },
        {
            "active_nibble_count": 2,
            "q1_rate": 0.30,
            "balance_rate_std": 0.12,
            "output_position_residual_std": 0.08,
            "excess_balance_rate_std": 0.09,
            "excess_output_position_residual_std": 0.06,
            "mixed_structure_fraction": 0.75,
        },
    ]

    gate = adjudicate_transition_audit(config, rows, {"fixture": True})

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_r6_two_nibble_output_prediction_benchmark_ready"
    )
    assert gate["next_action"]["remote_scale"] is False


def test_transition_cli_writes_complete_local_audit_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "transition"

    status = main(
        [
            "--run-id",
            "i2_transition_test",
            "--output-root",
            str(output),
            "--structures-per-width",
            "16",
            "--keys-per-structure",
            "4",
            "--structure-chunk-size",
            "2",
        ]
    )

    assert status == 0
    for name in (
        "results.jsonl",
        "structure_rates.csv",
        "position_priors.csv",
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
    assert [row["active_nibble_count"] for row in rows] == [1, 2]
    assert gate["readiness_checks"]["vectorized_parity_matches_scalar"] is True
    assert "结构级平衡率分布" in svg
    assert "本图不含神经训练结果" in svg
