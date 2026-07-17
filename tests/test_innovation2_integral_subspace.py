from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_innovation2_stable_subspace import main
from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    bit_integral_output_xor_matrix,
    make_bit_transition_structures,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    present_round_key_matrix,
)
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    StableSubspaceAuditConfig,
    adjudicate_stable_subspace_audit,
    gf2_kernel_basis,
    gf2_rank,
    kernel_basis_valid,
)


def test_gf2_kernel_basis_matches_known_three_bit_system() -> None:
    words = np.asarray([0b011, 0b110], dtype=np.uint64)

    basis = gf2_kernel_basis(words, width=3)

    assert gf2_rank(words, width=3) == 2
    assert basis == (0b111,)
    assert kernel_basis_valid(words, basis)


def test_vectorized_output_xor_words_match_scalar_present() -> None:
    structures = make_bit_transition_structures(
        active_bit_width=5,
        count=64,
        seed=41,
    )
    keys = (0x00010203040506070809, 0x00000000000000000000)
    round_keys = present_round_key_matrix(keys, rounds=6)

    matrix = bit_integral_output_xor_matrix(
        structures[:2],
        round_keys,
        structure_chunk_size=1,
    )

    for structure_index, structure in enumerate(structures[:2]):
        for key_index, key in enumerate(keys):
            assert int(matrix[structure_index, key_index]) == (
                scalar_bit_integral_output_xor(
                    structure,
                    rounds=6,
                    key=key,
                )
            )


def test_subspace_gate_requires_r5_calibration_before_r6_advance() -> None:
    config = StableSubspaceAuditConfig(
        run_id="gate",
        structures_per_width=64,
        keys_per_structure=8,
    )
    rows = []
    rows.append(
        {
            "rounds": 4,
            "active_bit_width": 4,
            "nontrivial_joint_kernel_fraction": 1.0,
            "minimum_joint_kernel_dimension": 64,
            "maximum_joint_kernel_dimension": 64,
        }
    )
    for rounds in (5, 6):
        for width in (5, 6, 7):
            rows.append(
                {
                    "rounds": rounds,
                    "active_bit_width": width,
                    "nontrivial_joint_kernel_fraction": 0.5,
                    "nontrivial_joint_kernel_structures": 32,
                    "distinct_nontrivial_joint_kernel_signatures": 8,
                    "mean_discovery_basis_validation_survival_fraction": 0.75,
                }
            )

    gate = adjudicate_stable_subspace_audit(config, rows, {"fixture": True})

    assert gate["status"] == "pass"
    assert gate["r4_known_fixture_calibration_pass"] is True
    assert gate["r6_passing_widths"] == [5, 6, 7]
    assert gate["selected_active_bit_width"] == 5


def test_stable_subspace_cli_writes_complete_audit_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "subspace"

    status = main(
        [
            "--run-id",
            "i2_subspace_test",
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
        "subspaces.csv",
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
    assert len(rows) == 7
    assert gate["readiness_checks"]["vectorized_xor_words_match_scalar"] is True
    assert "输出平衡 mask 子空间稳定性审计" in svg
    assert "无神经训练结果" in svg
