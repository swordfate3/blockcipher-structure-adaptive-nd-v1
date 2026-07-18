from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.ciphers.spn.skinny import SKINNY64_SBOX, Skinny64
from blockcipher_nd.cli.plot_innovation2_skinny64_unit_balance_profile_readiness import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.skinny64_unit_balance_profile_readiness import (
    Skinny64UnitProfileConfig,
    adjudicate_skinny_profile_checks,
    encrypt_skinny_words,
    make_skinny_keys,
    reconstruct_skinny_sbox_from_anf,
    skinny_round_tweakeys,
    skinny_variable_supports,
    validate_skinny_support_fixture,
    validate_skinny_vectorized_fixture,
)
from blockcipher_nd.ciphers.spn.skinny import generate_round_constants


def test_skinny_sbox_anf_reconstructs_every_value() -> None:
    assert all(
        reconstruct_skinny_sbox_from_anf(value) == SKINNY64_SBOX[value]
        for value in range(16)
    )


def test_skinny_vectorized_four_rounds_match_scalar_cipher() -> None:
    config = Skinny64UnitProfileConfig(run_id="e81-test")
    assert validate_skinny_vectorized_fixture(config)["all_pass"] is True

    keys = make_skinny_keys(2, 123)
    words = np.asarray([0, 1, 0xFEDCBA9876543210], dtype=np.uint64)
    vector = encrypt_skinny_words(
        words,
        skinny_round_tweakeys(keys, 4),
        generate_round_constants(4),
    )
    scalar = np.asarray(
        [[Skinny64(rounds=4, key=key).encrypt(int(word)) for word in words] for key in keys],
        dtype=np.uint64,
    )
    assert np.array_equal(vector, scalar)


def test_skinny_support_fixture_covers_exact_one_round_anf() -> None:
    fixture = validate_skinny_support_fixture()
    supports = skinny_variable_supports((0, 5, 58, 63), 1)

    assert fixture["all_pass"] is True
    assert fixture["missing_terms"] == 0
    assert len(supports) == 64
    assert all(support for support in supports)


def test_skinny_profile_gate_pass_hold_and_fail() -> None:
    all_true = {"valid": True}
    status, decision, _ = adjudicate_skinny_profile_checks(
        all_true, all_true, all_true
    )
    assert (status, decision) == (
        "pass",
        "innovation2_skinny64_unit_balance_profile_ready",
    )

    raw_narrow = {
        "raw_each_class_at_least_256": False,
        "resolved_prevalence_in_0p10_0p90": False,
        "mixed_structures_at_least_32": False,
        "distinct_signatures_at_least_4": True,
    }
    status, decision, action = adjudicate_skinny_profile_checks(
        all_true, raw_narrow, all_true
    )
    assert (status, decision) == (
        "hold",
        "innovation2_skinny64_unit_balance_profile_not_ready",
    )
    assert action == "stop r4 expansion and audit an r5 label-distribution transition"

    status, decision, _ = adjudicate_skinny_profile_checks(
        {"valid": False}, all_true, all_true
    )
    assert (status, decision) == (
        "fail",
        "innovation2_skinny64_unit_balance_profile_protocol_invalid",
    )


def test_skinny_e81_protocol_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        Skinny64UnitProfileConfig(run_id="e81-test", witness_keys=8)


def test_plot_writes_clear_chinese_e81_svg(tmp_path: Path) -> None:
    summary = {
        "gate": {
            "decision": "innovation2_skinny64_unit_balance_profile_ready",
            "metrics": {
                "raw_positive": 3000,
                "raw_negative": 2000,
                "raw_unknown": 1144,
                "matched_split_metrics": {
                    "train": {
                        "positive": 180,
                        "negative": 180,
                        "structures": 50,
                        "output_bits": 48,
                    },
                    "validation": {
                        "positive": 60,
                        "negative": 60,
                        "structures": 18,
                        "output_bits": 28,
                    },
                },
                "matched_marginal_baselines": {
                    "global": 0.5,
                    "output_bit": 0.5,
                    "active_bit": 0.5,
                    "strongest_auc": 0.5,
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E81" in svg
    assert "真实64位主密钥" in svg
    assert "不是神经性能" in svg
