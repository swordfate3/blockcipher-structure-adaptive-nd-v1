from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.ciphers.spn.gift import Gift64, _SBOX
from blockcipher_nd.cli.audit_innovation2_gift64_unit_balance_profile_readiness import (
    _matched_profile_arrays,
    _validate_e74_anchor,
)
from blockcipher_nd.cli.plot_innovation2_gift64_unit_balance_profile_readiness import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.gift64_unit_balance_profile_readiness import (
    Gift64UnitProfileConfig,
    Gift64UnitProfileExpansionConfig,
    adjudicate_gift_profile_checks,
    build_gift_checkerboard,
    encrypt_gift_words,
    gift_round_injections,
    gift_variable_supports,
    reconstruct_gift_sbox_from_anf,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    _cube_assignments,
    make_structures,
)


def test_gift_sbox_anf_reconstructs_all_inputs() -> None:
    assert tuple(reconstruct_gift_sbox_from_anf(value) for value in range(16)) == _SBOX


def test_vectorized_gift_four_rounds_matches_scalar_cipher() -> None:
    keys = (0, 1, (1 << 128) - 1, 0x0123456789ABCDEFFEDCBA9876543210)
    words = np.asarray(
        (0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210), dtype=np.uint64
    )

    observed = encrypt_gift_words(words, gift_round_injections(keys, rounds=4))
    expected = np.asarray(
        [[Gift64(rounds=4, key=key).encrypt(int(word)) for word in words] for key in keys],
        dtype=np.uint64,
    )

    np.testing.assert_array_equal(observed, expected)


def test_one_round_support_absence_implies_full_cube_xor_zero() -> None:
    active_bits = tuple(range(8))
    full_cube = (1 << len(active_bits)) - 1
    supports = gift_variable_supports(active_bits, rounds=1)
    assignments = _cube_assignments(active_bits)
    ciphertexts = encrypt_gift_words(
        assignments, gift_round_injections((0,), rounds=1)
    )
    parity = int(np.bitwise_xor.reduce(ciphertexts[0]))

    certified_bits = [bit for bit, support in enumerate(supports) if full_cube not in support]
    assert certified_bits
    assert all(not parity & (1 << bit) for bit in certified_bits)


def test_checkerboard_balances_each_selected_structure_and_output() -> None:
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

    matched = build_gift_checkerboard(labels, structures, attempts=4)

    assert matched["split_metrics"]["train"]["positive"] == 2
    assert matched["split_metrics"]["train"]["negative"] == 2
    assert matched["split_metrics"]["validation"]["positive"] == 2
    assert matched["split_metrics"]["validation"]["negative"] == 2
    assert matched["balance"] == {
        "duplicate_edges": 0,
        "maximum_structure_class_delta": 0,
        "maximum_mask_class_delta": 0,
    }
    targets, observed = _matched_profile_arrays(labels.shape, matched["rows"])
    assert int(observed.sum()) == 8
    assert np.all(targets[~observed] == -1)


def test_gate_passes_only_when_protocol_width_and_shortcuts_pass() -> None:
    passed = {"check": True}

    assert adjudicate_gift_profile_checks(passed, passed, passed)[:2] == (
        "pass",
        "innovation2_gift64_unit_balance_profile_ready",
    )
    assert adjudicate_gift_profile_checks(passed, {"check": False}, passed)[:2] == (
        "hold",
        "innovation2_gift64_unit_balance_profile_not_ready",
    )
    assert adjudicate_gift_profile_checks({"check": False}, passed, passed)[:2] == (
        "fail",
        "innovation2_gift64_unit_balance_profile_protocol_invalid",
    )
    assert adjudicate_gift_profile_checks(
        passed, passed, passed, experiment="e75"
    )[:2] == (
        "pass",
        "innovation2_gift64_unit_balance_profile_expansion_ready",
    )


def test_plot_writes_clear_chinese_e74_svg(tmp_path: Path) -> None:
    split_metrics = {
        "train": {
            "positive": 180,
            "negative": 180,
            "structures": 50,
            "output_bits": 30,
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
            "decision": "innovation2_gift64_unit_balance_profile_ready",
            "metrics": {
                "raw_positive": 2100,
                "raw_negative": 2300,
                "raw_unknown": 1744,
                "matched_split_metrics": split_metrics,
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
    assert "创新2 E74" in svg
    assert "GIFT-64四轮" in svg
    assert "严格训练标签" in svg


def test_e74_config_is_frozen() -> None:
    config = Gift64UnitProfileConfig(run_id="e74-test")
    assert config.rounds == 4
    try:
        Gift64UnitProfileConfig(run_id="e74-test", witness_keys=8)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("E74 audit dimensions must remain frozen")


def test_e75_expands_only_the_structure_count() -> None:
    anchor = Gift64UnitProfileConfig(run_id="e74-test")
    expansion = Gift64UnitProfileExpansionConfig(run_id="e75-test")

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


def test_e75_anchor_replay_checks_all_frozen_arrays(tmp_path: Path) -> None:
    config = Gift64UnitProfileConfig(run_id="e74-test")
    structures = make_structures(config)
    labels = np.zeros((192, 64), dtype=np.int8)
    prefix = np.zeros((192, 64, 39), dtype=np.float64)
    anchor_root = tmp_path / "anchor"
    anchor_root.mkdir()
    (anchor_root / "gate.json").write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "innovation2_gift64_unit_balance_profile_not_ready",
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
        for structure in structures
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

    checks = _validate_e74_anchor(
        anchor_root,
        make_structures(Gift64UnitProfileExpansionConfig(run_id="e75-test")),
        {"labels": labels, "prefix_features": prefix},
    )

    assert all(checks.values())
