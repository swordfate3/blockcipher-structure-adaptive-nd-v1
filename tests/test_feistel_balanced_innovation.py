from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.ciphers.base import rol
from blockcipher_nd.ciphers.feistel.simeck import Simeck64_128
from blockcipher_nd.ciphers.feistel.simon import Simon64_128
from blockcipher_nd.engine.modeling import cipher_profile
from blockcipher_nd.evaluation.plots import _compact_label
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.models.structure.feistel import (
    BalancedFeistelLuSeNetDistinguisher,
    BalancedFeistelRoundRelationDistinguisher,
    balanced_feistel_relation_channels,
    simeck_round_function_bits,
    simon_round_function_bits,
)
from blockcipher_nd.planning.feistel_balanced_gate import (
    feistel_balanced_calibration_decision,
    feistel_balanced_relation_decision,
    feistel_lu_layout_decision,
    feistel_relation_scale_probe_decision,
    feistel_relation_scale_confirmation_decision,
    feistel_target_round_scale_probe_decision,
    gate_feistel_balanced_results,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.difference_profiles import (
    difference_for_profile,
    literature_difference_profiles,
)
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
READINESS_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_balanced_round_relation_readiness_seed0.csv"
)
DIAGNOSTIC_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_balanced_round_relation_2048_seed0.csv"
)
CALIBRATION_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_balanced_round_relation_calibration_2048_seed0.csv"
)
LU_LAYOUT_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_lu_senet_layout_calibration_2048_seed0.csv"
)
SCALE_PROBE_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_round_relation_scale_probe_8192_seed0.csv"
)
SCALE_CONFIRMATION_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_round_relation_scale_probe_8192_seed1.csv"
)
TARGET_ROUND_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_feistel_round_relation_target_round_8192_seed0.csv"
)


def _word_tensor(value: int) -> torch.Tensor:
    return torch.tensor([int_to_bits(value, 32)], dtype=torch.float32)


def _simon_f(value: int) -> int:
    return (rol(value, 8, 32) & rol(value, 1, 32)) ^ rol(value, 2, 32)


def _simeck_f(value: int) -> int:
    return (rol(value, 5, 32) & value) ^ rol(value, 1, 32)


def test_simon_and_simeck_implementations_match_official_vectors() -> None:
    key = 0x1B1A1918131211100B0A090803020100
    plaintext = 0x656B696C20646E75
    assert Simon64_128(rounds=44, key=key).encrypt(plaintext) == 0x44C8FC20B9DFA07A
    assert Simeck64_128(rounds=44, key=key).encrypt(plaintext) == 0x45CE69025F7AB7ED


def test_balanced_feistel_cipher_profiles_are_registered() -> None:
    assert cipher_profile("simon64").name == "SIMON64/128"
    assert cipher_profile("simeck64").name == "Simeck64/128"
    assert cipher_profile("simon64").structure == "Feistel-like"
    assert cipher_profile("simon64").traits == cipher_profile("simeck64").traits


def test_balanced_feistel_plot_and_index_labels_are_unambiguous() -> None:
    assert _compact_label({"model": "simon_lu_round_relation_true"}) == (
        "SIMON 真实轮关系"
    )
    assert (
        _compact_label(
            {
                "model": "multiscale_dense_resnet",
                "cipher": "Simeck64/128",
            }
        )
        == "Simeck 通用多尺度基线"
    )
    assert "2048/类归因诊断" in display_name_for_run(
        "i1_feistel_balanced_round_relation_2048_seed0"
    )


def test_tensor_round_functions_match_integer_formulas() -> None:
    values = (0, 1, 0x12345678, 0x89ABCDEF, 0xFFFFFFFF)
    for value in values:
        assert simon_round_function_bits(_word_tensor(value))[0].tolist() == [
            bool(bit) for bit in int_to_bits(_simon_f(value), 32)
        ]
        assert simeck_round_function_bits(_word_tensor(value))[0].tolist() == [
            bool(bit) for bit in int_to_bits(_simeck_f(value), 32)
        ]


def test_lu_eight_relation_channels_match_integer_formula() -> None:
    first_left = 0x12345678
    first_right = 0x89ABCDEF
    second_left = 0x0F1E2D3C
    second_right = 0x4B5A6978
    first_ciphertext = (first_left << 32) | first_right
    second_ciphertext = (second_left << 32) | second_right
    pair = torch.tensor(
        [
            [
                [
                    [*int_to_bits(first_ciphertext, 64)],
                    [*int_to_bits(second_ciphertext, 64)],
                ]
            ]
        ],
        dtype=torch.float32,
    ).reshape(1, 1, 2, 64)

    first_previous = first_left ^ _simon_f(first_right)
    second_previous = second_left ^ _simon_f(second_right)
    expected_words = (
        first_left ^ second_left,
        first_right ^ second_right,
        first_left,
        first_right,
        second_left,
        second_right,
        first_previous ^ second_previous,
        (first_right ^ _simon_f(first_previous))
        ^ (second_right ^ _simon_f(second_previous)),
    )
    channels = balanced_feistel_relation_channels(
        pair, round_function="simon", mapping_mode="true"
    )
    expected = torch.tensor(
        [[[[float(bit) for bit in int_to_bits(word, 32)] for word in expected_words]]]
    )
    assert torch.equal(channels, expected)


def test_branch_swapped_control_changes_channels_without_changing_capacity() -> None:
    features = torch.randint(0, 2, (2, 8 * 128), dtype=torch.float32)
    true = BalancedFeistelRoundRelationDistinguisher(
        input_bits=features.shape[1],
        round_function="simon",
        mapping_mode="true",
        base_channels=4,
        blocks=1,
        classifier_bits=8,
    )
    shuffled = BalancedFeistelRoundRelationDistinguisher(
        input_bits=features.shape[1],
        round_function="simon",
        mapping_mode="shuffled",
        base_channels=4,
        blocks=1,
        classifier_bits=8,
    )
    assert not torch.equal(
        true.relation_channels(features), shuffled.relation_channels(features)
    )
    assert sum(parameter.numel() for parameter in true.parameters()) == sum(
        parameter.numel() for parameter in shuffled.parameters()
    )


def test_lu_senet_layout_preserves_eight_pair_positions_and_control_capacity() -> None:
    features = torch.randint(0, 2, (2, 8 * 128), dtype=torch.float32)
    true = BalancedFeistelLuSeNetDistinguisher(
        input_bits=features.shape[1],
        round_function="simon",
        mapping_mode="true",
        base_channels=4,
        blocks=1,
        classifier_bits=8,
        se_ratio=2,
    )
    shuffled = BalancedFeistelLuSeNetDistinguisher(
        input_bits=features.shape[1],
        round_function="simon",
        mapping_mode="shuffled",
        base_channels=4,
        blocks=1,
        classifier_bits=8,
        se_ratio=2,
    )
    assert true.relation_sequence(features).shape == (2, 8, 256)
    assert true(features).shape == (2, 1)
    assert not torch.equal(
        true.relation_sequence(features), shuffled.relation_sequence(features)
    )
    assert sum(parameter.numel() for parameter in true.parameters()) == sum(
        parameter.numel() for parameter in shuffled.parameters()
    )


def test_balanced_relation_registry_models_forward_and_backpropagate() -> None:
    features = torch.randint(0, 2, (2, 8 * 128), dtype=torch.float32)
    for model_key in (
        "simon_lu_round_relation_true",
        "simon_lu_round_relation_shuffled",
        "simeck_lu_round_relation_true",
        "simeck_lu_round_relation_shuffled",
        "simon_lu_senet_layout_true",
        "simon_lu_senet_layout_shuffled",
        "simeck_lu_senet_layout_true",
        "simeck_lu_senet_layout_shuffled",
    ):
        model = build_model(
            model_key,
            input_bits=features.shape[1],
            hidden_bits=4,
            pair_bits=128,
            structure="Feistel-like",
            model_options={"blocks": 1, "classifier_bits": 8, "se_ratio": 2},
        )
        logits = model(features)
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
        logits.mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_lu_difference_profiles_are_fixed_eight_pair_profiles() -> None:
    profiles = literature_difference_profiles()
    for name, cipher in (
        ("simon64_lu2024_ordinary", "simon64"),
        ("simeck64_lu2024_ordinary", "simeck64"),
    ):
        assert profiles[name].cipher == cipher
        assert profiles[name].pairs_per_sample == 8
        assert difference_for_profile(name) == 0x40


def test_balanced_relation_plans_are_frozen_six_row_same_protocol_matrices() -> None:
    for plan, samples_per_class, validation_total, final_total, repeats in (
        (READINESS_PLAN, 64, 128, 128, 1),
        (DIAGNOSTIC_PLAN, 2048, 4096, 8192, 3),
    ):
        tasks = tasks_from_plan(
            plan,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=1,
            difference_profile=None,
            difference_member=0,
        )
        assert len(tasks) == 6
        assert {(task["cipher_key"], task["rounds"]) for task in tasks} == {
            ("simon64", 12),
            ("simeck64", 15),
        }
        for task in tasks:
            assert task["seed"] == 0
            assert task["samples_per_class"] == samples_per_class
            assert task["validation_samples_total"] == validation_total
            assert task["final_test_samples_total"] == final_total
            assert task["final_test_repeats"] == repeats
            assert task["pairs_per_sample"] == 8
            assert task["negative_mode"] == "encrypted_random_plaintexts"
            assert task["key_rotation_interval"] == 1
            assert task["sample_structure"] == "independent_pairs"

    calibration = tasks_from_plan(
        CALIBRATION_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(calibration) == 6
    assert {(task["cipher_key"], task["rounds"]) for task in calibration} == {
        ("simon64", 11),
        ("simeck64", 14),
    }
    layout = tasks_from_plan(
        LU_LAYOUT_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(layout) == 6
    assert {task["model_key"] for task in layout} == {
        "simon_lu_senet_layout_true",
        "simon_lu_senet_layout_shuffled",
        "simon_lu_round_relation_true",
        "simeck_lu_senet_layout_true",
        "simeck_lu_senet_layout_shuffled",
        "simeck_lu_round_relation_true",
    }
    scale_probe = tasks_from_plan(
        SCALE_PROBE_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(scale_probe) == 4
    assert {task["samples_per_class"] for task in scale_probe} == {8192}
    assert {task["final_test_samples_total"] for task in scale_probe} == {32768}
    confirmation = tasks_from_plan(
        SCALE_CONFIRMATION_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(confirmation) == 4
    assert {task["seed"] for task in confirmation} == {1}
    target_round = tasks_from_plan(
        TARGET_ROUND_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(target_round) == 4
    assert {(task["cipher_key"], task["rounds"]) for task in target_round} == {
        ("simon64", 12),
        ("simeck64", 15),
    }


def test_balanced_relation_decision_covers_pass_conditional_and_no_attribution() -> (
    None
):
    both = feistel_balanced_relation_decision(
        {
            "simon64": {0: {"candidate": 0.61, "shuffled": 0.58, "generic": 0.60}},
            "simeck64": {0: {"candidate": 0.64, "shuffled": 0.60, "generic": 0.642}},
        }
    )
    assert both["decision"] == "feistel_balanced_relation_two_cipher_seed0_pass"

    conditional = feistel_balanced_relation_decision(
        {
            "simon64": {0: {"candidate": 0.61, "shuffled": 0.58, "generic": 0.60}},
            "simeck64": {0: {"candidate": 0.54, "shuffled": 0.50, "generic": 0.53}},
        }
    )
    assert conditional["decision"] == "feistel_balanced_relation_cipher_conditional"

    unattributed = feistel_balanced_relation_decision(
        {
            "simon64": {0: {"candidate": 0.61, "shuffled": 0.605, "generic": 0.60}},
            "simeck64": {0: {"candidate": 0.63, "shuffled": 0.625, "generic": 0.62}},
        }
    )
    assert (
        unattributed["decision"]
        == "feistel_balanced_signal_without_relation_attribution"
    )

    calibrated = feistel_balanced_calibration_decision(
        {
            "simon64": {0: {"candidate": 0.75, "shuffled": 0.65, "generic": 0.74}},
            "simeck64": {0: {"candidate": 0.72, "shuffled": 0.68, "generic": 0.71}},
        }
    )
    assert calibrated["decision"] == "feistel_balanced_easier_round_calibrated"

    layout = feistel_lu_layout_decision(
        {
            "simon64": {0: {"candidate": 0.65, "shuffled": 0.55, "generic": 0.62}},
            "simeck64": {0: {"candidate": 0.68, "shuffled": 0.56, "generic": 0.64}},
        }
    )
    assert layout["decision"] == "feistel_lu_layout_two_cipher_calibrated"

    scale_probe = feistel_relation_scale_probe_decision(
        {
            "simon64": {0: {"candidate": 0.59, "shuffled": 0.52}},
            "simeck64": {0: {"candidate": 0.61, "shuffled": 0.53}},
        }
    )
    assert scale_probe["decision"] == "feistel_relation_scale_slope_two_cipher_pass"

    confirmation = feistel_relation_scale_confirmation_decision(
        {
            "simon64": {1: {"candidate": 0.64, "shuffled": 0.50}},
            "simeck64": {1: {"candidate": 0.82, "shuffled": 0.50}},
        }
    )
    assert confirmation["decision"] == "feistel_relation_8192_seed1_confirmation_pass"

    target_round = feistel_target_round_scale_probe_decision(
        {
            "simon64": {0: {"candidate": 0.57, "shuffled": 0.50}},
            "simeck64": {0: {"candidate": 0.62, "shuffled": 0.51}},
        }
    )
    assert target_round["decision"] == "feistel_target_round_8192_two_cipher_pass"


def _synthetic_readiness_rows() -> list[dict[str, object]]:
    tasks = tasks_from_plan(
        READINESS_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    names = {"simon64": "SIMON64/128", "simeck64": "Simeck64/128"}
    rows = []
    for task in tasks:
        row = {
            "cipher": names[task["cipher_key"]],
            "cipher_key": task["cipher_key"],
            "structure": "Feistel-like",
            "model": task["model_key"],
            "rounds": task["rounds"],
            "seed": task["seed"],
            "samples_per_class": task["samples_per_class"],
            "train_samples_total": task["train_samples_total"],
            "validation_samples_total": task["validation_samples_total"],
            "final_test_samples_total": task["final_test_samples_total"],
            "final_test_repeats": task["final_test_repeats"],
            "feature_encoding": task["feature_encoding"],
            "negative_mode": task["negative_mode"],
            "train_key": task["train_key"],
            "validation_key": task["validation_key"],
            "final_test_key": task["final_test_key"],
            "pairs_per_sample": task["pairs_per_sample"],
            "sample_structure": task["sample_structure"],
            "integral_active_nibble": task["integral_active_nibble"],
            "integral_active_nibbles": [],
            "validation_integral_active_nibbles": [],
            "key_rotation_interval": task["key_rotation_interval"],
            "difference_profile": task["difference_profile"],
            "difference_member": task["difference_member"],
            "selected_bit_indices": [],
            "history": [{}, {}],
            "training": {
                "loss": task["loss"],
                "learning_rate": task["learning_rate"],
                "optimizer": task["optimizer"],
                "weight_decay": task["weight_decay"],
                "checkpoint_metric": task["checkpoint_metric"],
                "restore_best_checkpoint": task["restore_best_checkpoint"],
                "early_stopping_patience": task["early_stopping_patience"],
                "early_stopping_min_delta": task["early_stopping_min_delta"],
                "selected_bit_indices": [],
                "model_options": task["model_options"],
                "pretraining": {"epochs_ran": 0, "round_sequence": []},
                "epochs": 2,
                "epochs_ran": 2,
                "samples_total": 128,
                "key_rotation_interval": 1,
                "pairs_per_sample": 8,
                "sample_structure": "independent_pairs",
                "feature_encoding": "ciphertext_pair_bits",
                "key_rotation_row_indexing": "global_dataset_row",
            },
            "validation": {
                "samples_total": 128,
                "key_rotation_row_indexing": "global_dataset_row",
            },
            "final_evaluation": {
                "auc_mean": 0.6,
                "repeats": 1,
                "samples_total_per_repeat": 128,
                "metrics_by_repeat": [{"auc": 0.6}],
            },
            "parameter_count": 100
            if task["model_key"] != "multiscale_dense_resnet"
            else 200,
            "trainable_parameter_count": 100
            if task["model_key"] != "multiscale_dense_resnet"
            else 200,
        }
        rows.append(row)
    return rows


def test_readiness_gate_passes_complete_rows_and_rejects_capacity_mismatch(
    tmp_path: Path,
) -> None:
    rows = _synthetic_readiness_rows()
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = gate_feistel_balanced_results(
        plan_path=READINESS_PLAN,
        results_path=results,
        expected_samples_per_class=64,
        expected_seeds=(0,),
        expected_epochs=2,
        expected_final_repeats=1,
        readiness=True,
    )
    assert report["status"] == "pass"
    assert report["decision"] == "feistel_balanced_relation_readiness_passed"
    assert report["research_decision_applied"] is False

    rows[1]["parameter_count"] = 101
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    invalid = gate_feistel_balanced_results(
        plan_path=READINESS_PLAN,
        results_path=results,
        expected_samples_per_class=64,
        expected_seeds=(0,),
        expected_epochs=2,
        expected_final_repeats=1,
        readiness=True,
    )
    assert invalid["status"] == "fail"
    assert invalid["decision"] == "invalid_feistel_balanced_protocol"
    assert any("parameter_count mismatch" in error for error in invalid["errors"])
