from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.audit_innovation2_rectangle80_unit_balance_profile_readiness import (
    _validate_e87_anchor,
)
from blockcipher_nd.cli.plot_innovation2_rectangle80_unit_balance_profile_readiness import (
    main as plot_main,
)
from blockcipher_nd.ciphers.spn.rectangle import (
    RECTANGLE_SBOX,
    Rectangle80,
    rectangle_player,
    rectangle_sub_columns,
)
from blockcipher_nd.tasks.innovation2.rectangle80_unit_balance_profile_readiness import (
    OFFICIAL_ZERO_VECTOR,
    Rectangle80UnitProfileConfig,
    Rectangle80UnitProfileExpansionConfig,
    adjudicate_rectangle80_profile_checks,
    build_rectangle80_checkerboard,
    encrypt_rectangle80_words,
    rectangle80_round_keys,
    rectangle80_variable_supports,
    reconstruct_rectangle_sbox_from_anf,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    _cube_assignments,
    make_structures,
)


def test_rectangle80_matches_final_spec_zero_vector() -> None:
    assert Rectangle80(rounds=25, key=0).encrypt(0) == OFFICIAL_ZERO_VECTOR
    assert OFFICIAL_ZERO_VECTOR == 0x0874E8B1E3542D96


def test_rectangle_subcolumns_matches_appendix_b_bitslice_formula() -> None:
    rng = np.random.default_rng(87)
    for state in rng.integers(0, 1 << 64, size=32, dtype=np.uint64):
        value = int(state)
        a0, a1, a2, a3 = [
            (value >> (16 * row)) & 0xFFFF for row in range(4)
        ]
        t1 = (~a1) & 0xFFFF
        t2 = a0 & t1
        t3 = a2 ^ a3
        b0 = t2 ^ t3
        t5 = a3 | t1
        t6 = a0 ^ t5
        b1 = a2 ^ t6
        t8 = a1 ^ a2
        t9 = t3 & t6
        b3 = t8 ^ t9
        t11 = b0 | t8
        b2 = t6 ^ t11
        expected = sum(
            (row & 0xFFFF) << (16 * index)
            for index, row in enumerate((b0, b1, b2, b3))
        )
        assert rectangle_sub_columns(value) == expected


def test_rectangle_sbox_anf_and_player_match_spec() -> None:
    assert tuple(reconstruct_rectangle_sbox_from_anf(value) for value in range(16)) == RECTANGLE_SBOX
    assert sorted(rectangle_player()) == list(range(64))
    assert rectangle_player()[16] == 17
    assert rectangle_player()[32] == 44
    assert rectangle_player()[48] == 61


def test_vectorized_rectangle_four_rounds_matches_scalar_cipher() -> None:
    keys = (0, 1, (1 << 80) - 1, 0x0123456789ABCDEFFEDC)
    words = np.asarray(
        (0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210),
        dtype=np.uint64,
    )

    observed = encrypt_rectangle80_words(words, rectangle80_round_keys(keys, 4))
    expected = np.asarray(
        [
            [Rectangle80(rounds=4, key=key).encrypt(int(word)) for word in words]
            for key in keys
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(observed, expected)


def test_one_round_support_absence_implies_full_cube_xor_zero() -> None:
    active_bits = tuple(range(8))
    full_cube = (1 << len(active_bits)) - 1
    supports = rectangle80_variable_supports(active_bits, rounds=1)
    assignments = _cube_assignments(active_bits)
    ciphertexts = encrypt_rectangle80_words(
        assignments, rectangle80_round_keys((0,), rounds=1)
    )
    parity = int(np.bitwise_xor.reduce(ciphertexts[0]))

    certified = [
        bit for bit, support in enumerate(supports) if full_cube not in support
    ]
    assert certified
    assert all(not parity & (1 << bit) for bit in certified)


def test_rectangle_checkerboard_preserves_structure_and_output_balance() -> None:
    structures = tuple(
        ActiveStructure(
            index=index,
            structure_id=f"cube_{index:03d}",
            role="random_coordinate_cube",
            active_bits=tuple(range(index, index + 8)),
        )
        for index in range(8)
    )
    labels = np.full((8, 64), -1, dtype=np.int8)
    labels[0, :2] = (1, 0)
    labels[4, :2] = (0, 1)
    labels[1, :2] = (1, 0)
    labels[2, :2] = (0, 1)

    matched = build_rectangle80_checkerboard(labels, structures, attempts=4)

    assert matched["split_metrics"]["train"]["positive"] == 2
    assert matched["split_metrics"]["validation"]["negative"] == 2
    assert matched["balance"] == {
        "duplicate_edges": 0,
        "maximum_structure_class_delta": 0,
        "maximum_mask_class_delta": 0,
    }


def test_e87_gate_distinguishes_raw_width_from_matching_capacity() -> None:
    passed = {"check": True}

    assert adjudicate_rectangle80_profile_checks(
        passed, passed, passed, passed
    )[:2] == ("pass", "innovation2_rectangle80_unit_profile_ready")
    assert adjudicate_rectangle80_profile_checks(
        passed, {"check": False}, passed, passed
    )[:2] == (
        "hold",
        "innovation2_rectangle80_unit_profile_raw_labels_not_ready",
    )
    assert adjudicate_rectangle80_profile_checks(
        passed, passed, {"check": False}, passed
    )[:2] == (
        "hold",
        "innovation2_rectangle80_unit_profile_matching_not_ready",
    )
    assert adjudicate_rectangle80_profile_checks(
        {"check": False}, passed, passed, passed
    )[:2] == ("fail", "innovation2_rectangle80_unit_profile_protocol_invalid")


def test_rectangle80_validation_and_e87_protocol_are_frozen() -> None:
    with pytest.raises(ValueError, match="supports"):
        Rectangle80(rounds=26)
    with pytest.raises(ValueError, match="80 bits"):
        Rectangle80(key=1 << 80)
    with pytest.raises(ValueError, match="64 bits"):
        Rectangle80().encrypt(1 << 64)
    with pytest.raises(ValueError, match="frozen"):
        Rectangle80UnitProfileConfig(run_id="e87-test", witness_keys=8)


def test_e88_expands_only_the_structure_count() -> None:
    anchor = Rectangle80UnitProfileConfig(run_id="e87-test")
    expansion = Rectangle80UnitProfileExpansionConfig(run_id="e88-test")

    assert expansion.structure_count == 192
    for field in (
        "rounds",
        "witness_keys",
        "offsets_per_structure",
        "match_attempts",
        "structure_seed",
        "key_seed",
        "offset_seed",
    ):
        assert getattr(expansion, field) == getattr(anchor, field)
    assert make_structures(expansion)[:96] == make_structures(anchor)


def test_e88_gate_requires_protocol_and_all_width_checks() -> None:
    passed = {"check": True}

    assert adjudicate_rectangle80_profile_checks(
        passed, passed, passed, passed, experiment="e88"
    )[:2] == (
        "pass",
        "innovation2_rectangle80_unit_profile_expansion_ready",
    )
    assert adjudicate_rectangle80_profile_checks(
        passed, passed, {"check": False}, passed, experiment="e88"
    )[:2] == (
        "hold",
        "innovation2_rectangle80_unit_profile_expansion_not_ready",
    )
    assert adjudicate_rectangle80_profile_checks(
        {"check": False}, passed, passed, passed, experiment="e88"
    )[:2] == (
        "fail",
        "innovation2_rectangle80_unit_profile_expansion_protocol_invalid",
    )


def test_e88_anchor_replay_checks_all_frozen_arrays(tmp_path: Path) -> None:
    anchor = Rectangle80UnitProfileConfig(run_id="e87-test")
    expansion = Rectangle80UnitProfileExpansionConfig(run_id="e88-test")
    anchor_structures = make_structures(anchor)
    expanded_structures = make_structures(expansion)
    labels = np.zeros((192, 64), dtype=np.int8)
    prefix = np.zeros((192, 64, 39), dtype=np.float64)
    anchor_root = tmp_path / "anchor"
    anchor_root.mkdir()
    (anchor_root / "gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "innovation2_rectangle80_unit_profile_ready",
            }
        ),
        encoding="utf-8",
    )
    structure_rows = [
        {
            "index": structure.index,
            "structure_id": structure.structure_id,
            "role": structure.role,
            "active_bits": list(structure.active_bits),
            "active_mask_hex": f"0x{structure.active_mask:016X}",
            "split": "validation" if not structure.index % 4 else "train",
        }
        for structure in anchor_structures
    ]
    (anchor_root / "structures.json").write_text(
        json.dumps({"structures": structure_rows}), encoding="utf-8"
    )
    with (anchor_root / "atlas.jsonl").open("w", encoding="utf-8") as handle:
        for structure_index in range(96):
            for output_bit in range(64):
                handle.write(
                    json.dumps(
                        {
                            "structure_index": structure_index,
                            "output_bit": output_bit,
                            "label": 0,
                        }
                    )
                    + "\n"
                )
    np.save(anchor_root / "prefix_features.npy", prefix[:96])

    checks = _validate_e87_anchor(
        anchor_root,
        expanded_structures,
        {"labels": labels, "prefix_features": prefix},
    )

    assert all(checks.values())


def test_plot_writes_clear_chinese_e87_svg(tmp_path: Path) -> None:
    split = {
        "train": {
            "positive": 180,
            "negative": 180,
            "structures": 50,
            "output_bits": 32,
        },
        "validation": {
            "positive": 60,
            "negative": 60,
            "structures": 18,
            "output_bits": 24,
        },
    }
    summary = {
        "gate": {
            "decision": "innovation2_rectangle80_unit_profile_ready",
            "metrics": {
                "raw_positive": 2500,
                "raw_negative": 2200,
                "raw_unknown": 1444,
                "matched_split_metrics": split,
                "matched_marginal_baselines": {
                    "global": 0.5,
                    "output_bit": 0.53,
                    "active_bit": 0.55,
                    "strongest_auc": 0.55,
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E87" in svg
    assert "80-bit key" in svg
    assert "不是7轮论文复现" in svg


def test_plot_writes_clear_chinese_e88_svg(tmp_path: Path) -> None:
    split = {
        "train": {
            "positive": 300,
            "negative": 300,
            "structures": 100,
            "output_bits": 48,
        },
        "validation": {
            "positive": 80,
            "negative": 80,
            "structures": 30,
            "output_bits": 32,
        },
    }
    summary = {
        "metadata": {"experiment": "e88", "config": {"structure_count": 192}},
        "gate": {
            "decision": "innovation2_rectangle80_unit_profile_expansion_ready",
            "metrics": {
                "raw_positive": 8000,
                "raw_negative": 3000,
                "raw_unknown": 1288,
                "matched_split_metrics": split,
                "matched_marginal_baselines": {
                    "global": 0.5,
                    "output_bit": 0.5,
                    "active_bit": 0.5,
                    "strongest_auc": 0.5,
                },
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E88" in svg
    assert "192个8维输入cube" in svg
    assert "仅使用第3轮前缀的三行神经网络就绪实验" in svg
