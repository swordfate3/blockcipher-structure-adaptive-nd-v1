from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.ciphers.feistel.sm4 import Sm4Reduced
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.models.structure.feistel import (
    Sm4WordRecurrenceDistinguisher,
    sm4_state_mapping_indices,
)
from blockcipher_nd.models.baseline import Sm4Yu2023PositionResNetDistinguisher
from blockcipher_nd.planning.feistel_sm4_gate import (
    gate_feistel_sm4_position_calibration,
    gate_feistel_sm4_protocol_audit,
    gate_feistel_sm4_results,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_OPTIONS = {
    "blocks": 3,
    "classifier_bits": 64,
    "dropout": 0.5,
    "rotation_offsets": [2, 10, 18, 24],
}
POSITION_OPTIONS = {"blocks": 5, "classifier_bits": 64, "dropout": 0.5}


def test_sm4_full_round_standard_vector() -> None:
    value = 0x0123456789ABCDEFFEDCBA9876543210
    cipher = Sm4Reduced(rounds=32, key=value)
    assert cipher.encrypt(value) == 0x681EDF34D206965E86B3E94F536E4246


def test_sm4_true_mapping_restores_chronological_state_words() -> None:
    serialized = 0x11111111222222223333333344444444
    other = 0xAAAAAAAA55555555CCCCCCCC33333333
    features = torch.tensor(
        [int_to_bits(serialized, 128) + int_to_bits(other, 128)],
        dtype=torch.float32,
    )
    model = Sm4WordRecurrenceDistinguisher(
        input_bits=256, base_channels=4, blocks=1, classifier_bits=8
    )
    words = model.canonical_words(features)
    expected = [0x44444444, 0x33333333, 0x22222222, 0x11111111]
    for index, word in enumerate(expected):
        assert words[0, 0, 0, index].tolist() == [
            float(bit) for bit in int_to_bits(word, 32)
        ]


def test_sm4_shuffled_mapping_is_fixed_and_capacity_preserving() -> None:
    true = sm4_state_mapping_indices("true")
    shuffled_a = sm4_state_mapping_indices("shuffled")
    shuffled_b = sm4_state_mapping_indices("shuffled")
    assert sorted(true.tolist()) == list(range(128))
    assert sorted(shuffled_a.tolist()) == list(range(128))
    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(true, shuffled_a)

    candidate = build_model(
        "sm4_word_recurrence_true",
        input_bits=256,
        hidden_bits=8,
        pair_bits=256,
        structure="Feistel-like",
        model_options=MODEL_OPTIONS,
    )
    shuffled = build_model(
        "sm4_word_recurrence_shuffled",
        input_bits=256,
        hidden_bits=8,
        pair_bits=256,
        structure="Feistel-like",
        model_options=MODEL_OPTIONS,
    )
    assert sum(parameter.numel() for parameter in candidate.parameters()) == sum(
        parameter.numel() for parameter in shuffled.parameters()
    )


def test_sm4_rotation_channels_follow_msb_first_rol_positions() -> None:
    first_words = [0, 0x80000000, 0, 0]
    second_words = [0, 0, 0, 0]
    serialized_words = list(reversed(first_words)) + list(reversed(second_words))
    features = torch.tensor(
        [[bit for word in serialized_words for bit in int_to_bits(word, 32)]],
        dtype=torch.float32,
    )
    model = Sm4WordRecurrenceDistinguisher(
        input_bits=256,
        base_channels=4,
        blocks=1,
        classifier_bits=8,
        rotation_offsets=(2,),
    )
    channels = model.semantic_channels(features)
    triples = channels[0, 0, 15:18]
    rotated = channels[0, 0, 18:21]
    assert triples[0, 0].item() == 1.0
    assert rotated[0, 30].item() == 1.0
    assert rotated.sum().item() == triples.sum().item()


def test_sm4_registry_models_forward_and_backpropagate() -> None:
    features = torch.randint(0, 2, (2, 256), dtype=torch.float32)
    for model_key in (
        "sm4_word_recurrence_true",
        "sm4_word_recurrence_shuffled",
        "multiscale_dense_resnet",
        "sm4_yu2023_position_resnet",
    ):
        model = build_model(
            model_key,
            input_bits=256,
            hidden_bits=8,
            pair_bits=256,
            structure="Feistel-like",
            model_options=(
                MODEL_OPTIONS if model_key.startswith("sm4_word_recurrence") else {}
            ),
        )
        logits = model(features)
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
        logits.mean().backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_sm4_yu2023_position_resnet_preserves_all_bit_positions() -> None:
    model = Sm4Yu2023PositionResNetDistinguisher(
        input_bits=256,
        channels=8,
        blocks=2,
        classifier_bits=16,
        dropout=0.5,
    )
    features = torch.randint(0, 2, (2, 256), dtype=torch.float32)
    hidden = model.position_features(features)
    assert hidden.shape == (2, 8, 128)
    assert model.flattened_width == 8 * 128
    assert model(features).shape == (2, 1)


def test_sm4_local_plan_is_frozen_two_seed_matrix() -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_sm4_r5_word_recurrence_attribution_2048_seed0_seed1.csv"
    tasks = tasks_from_plan(
        plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(tasks) == 6
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {
        "sm4_word_recurrence_true",
        "sm4_word_recurrence_shuffled",
        "multiscale_dense_resnet",
    }
    for task in tasks:
        assert task["cipher_key"] == "sm4"
        assert task["rounds"] == 5
        assert task["samples_per_class"] == 2048
        assert task["validation_samples_total"] == 2048
        assert task["final_test_samples_total"] == 4096
        assert task["final_test_repeats"] == 3
        assert task["pairs_per_sample"] == 1
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["key_rotation_interval"] == 1
        assert task["input_difference"] == 1


def _result_rows(scores: dict[int, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, seed_scores in scores.items():
        for model, auc in seed_scores.items():
            options = MODEL_OPTIONS if model.startswith("sm4_") else {}
            parameter_count = 1000 if model.startswith("sm4_") else 900
            rows.append(
                {
                    "cipher": "SM4",
                    "cipher_key": "sm4",
                    "structure": "Feistel-like",
                    "model": model,
                    "rounds": 5,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "train_samples_total": None,
                    "validation_samples_total": 2048,
                    "final_test_samples_total": 4096,
                    "final_test_repeats": 3,
                    "dataset_label_mode": "balanced_per_class",
                    "pairs_per_sample": 1,
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "train_key": None,
                    "validation_key": None,
                    "final_test_key": None,
                    "key_rotation_interval": 1,
                    "sample_structure": "independent_pairs",
                    "integral_active_nibble": 0,
                    "integral_active_nibbles": [],
                    "validation_integral_active_nibbles": [],
                    "difference_profile": "sm4_yu2023_conv_resnet",
                    "difference_member": 0,
                    "history": [{} for _ in range(10)],
                    "training": {
                        "loss": "mse",
                        "learning_rate": 0.0001,
                        "optimizer": "adam",
                        "weight_decay": 0.0,
                        "checkpoint_metric": "val_auc",
                        "restore_best_checkpoint": True,
                        "early_stopping_patience": 0,
                        "early_stopping_min_delta": 0.0,
                        "key_schedule": "rotating",
                        "selected_bit_indices": [],
                        "model_options": options,
                        "pretraining": {"epochs_ran": 0, "round_sequence": []},
                    },
                    "validation": {"key_schedule": "rotating"},
                    "final_evaluation": {
                        "auc_mean": auc,
                        "repeats": 3,
                        "metrics_by_repeat": [{}, {}, {}],
                    },
                    "parameter_count": parameter_count,
                    "trainable_parameter_count": parameter_count,
                }
            )
    return rows


def test_sm4_gate_requires_attribution_signal_and_baseline_competitiveness(
    tmp_path: Path,
) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_sm4_r5_word_recurrence_attribution_2048_seed0_seed1.csv"
    scores = {
        0: {
            "sm4_word_recurrence_true": 0.81,
            "sm4_word_recurrence_shuffled": 0.80,
            "multiscale_dense_resnet": 0.811,
        },
        1: {
            "sm4_word_recurrence_true": 0.80,
            "sm4_word_recurrence_shuffled": 0.79,
            "multiscale_dense_resnet": 0.801,
        },
    }
    rows = _result_rows(scores)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    passed = gate_feistel_sm4_results(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert passed["status"] == "pass", passed["errors"]
    assert passed["decision"] == "feistel_sm4_r5_word_recurrence_attributed"

    rows[1]["parameter_count"] = 1001
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    invalid = gate_feistel_sm4_results(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert invalid["status"] == "fail"
    assert invalid["decision"] == "invalid_feistel_sm4_protocol"
    assert any("capacity mismatch" in error for error in invalid["errors"])


def test_sm4_protocol_audit_identifies_fixed_key_dependency(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_sm4_key_schedule_signal_audit_2048_seed0.csv"
    scores = {
        (3, 0): 0.99,
        (3, 1): 0.75,
        (5, 0): 0.80,
        (5, 1): 0.51,
    }
    fixed_key = 0x0123456789ABCDEFFEDCBA9876543210
    rows = []
    for (rounds, rotation), auc in scores.items():
        key = fixed_key if rotation == 0 else None
        schedule = "fixed" if rotation == 0 else "rotating"
        rows.append(
            {
                "cipher": "SM4",
                "cipher_key": "sm4",
                "structure": "Feistel-like",
                "model": "multiscale_dense_resnet",
                "rounds": rounds,
                "seed": 0,
                "samples_per_class": 2048,
                "train_samples_total": None,
                "validation_samples_total": 2048,
                "final_test_samples_total": 4096,
                "final_test_repeats": 3,
                "dataset_label_mode": "balanced_per_class",
                "pairs_per_sample": 1,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "train_key": key,
                "validation_key": key,
                "final_test_key": key,
                "key_rotation_interval": rotation,
                "sample_structure": "independent_pairs",
                "integral_active_nibble": 0,
                "integral_active_nibbles": [],
                "validation_integral_active_nibbles": [],
                "difference_profile": "sm4_yu2023_conv_resnet",
                "difference_member": 0,
                "history": [{} for _ in range(10)],
                "training": {
                    "loss": "mse",
                    "learning_rate": 0.0001,
                    "optimizer": "adam",
                    "weight_decay": 0.0,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0,
                    "key_schedule": schedule,
                    "selected_bit_indices": [],
                    "model_options": {},
                    "pretraining": {"epochs_ran": 0, "round_sequence": []},
                },
                "validation": {"key_schedule": schedule},
                "final_evaluation": {
                    "auc_mean": auc,
                    "repeats": 3,
                    "metrics_by_repeat": [{}, {}, {}],
                },
                "parameter_count": 59809,
                "trainable_parameter_count": 59809,
            }
        )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = gate_feistel_sm4_protocol_audit(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0,),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "feistel_sm4_fixed_key_dependency_identified"
    assert report["signal"] == {
        "r3_fixed": True,
        "r3_rotating": True,
        "r5_fixed": True,
        "r5_rotating": False,
    }


def test_sm4_position_calibration_retains_cross_key_anchor(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_sm4_position_resnet_calibration_2048_seed0.csv"
    fixed_key = 0x0123456789ABCDEFFEDCBA9876543210
    scores = {
        ("sm4_yu2023_position_resnet", 0): 0.80,
        ("multiscale_dense_resnet", 0): 0.51,
        ("sm4_yu2023_position_resnet", 1): 0.75,
        ("multiscale_dense_resnet", 1): 0.50,
    }
    rows = []
    for (model, rotation), auc in scores.items():
        key = fixed_key if rotation == 0 else None
        schedule = "fixed" if rotation == 0 else "rotating"
        options = POSITION_OPTIONS if model.startswith("sm4_") else {}
        count = 300000 if model.startswith("sm4_") else 59809
        rows.append(
            {
                "cipher": "SM4",
                "cipher_key": "sm4",
                "structure": "Feistel-like",
                "model": model,
                "rounds": 5,
                "seed": 0,
                "samples_per_class": 2048,
                "train_samples_total": None,
                "validation_samples_total": 2048,
                "final_test_samples_total": 4096,
                "final_test_repeats": 3,
                "dataset_label_mode": "balanced_per_class",
                "pairs_per_sample": 1,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "train_key": key,
                "validation_key": key,
                "final_test_key": key,
                "key_rotation_interval": rotation,
                "sample_structure": "independent_pairs",
                "integral_active_nibble": 0,
                "integral_active_nibbles": [],
                "validation_integral_active_nibbles": [],
                "difference_profile": "sm4_yu2023_conv_resnet",
                "difference_member": 0,
                "history": [{} for _ in range(10)],
                "training": {
                    "loss": "mse",
                    "learning_rate": 0.0001,
                    "optimizer": "adam",
                    "weight_decay": 0.0,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0,
                    "key_schedule": schedule,
                    "selected_bit_indices": [],
                    "model_options": options,
                    "pretraining": {"epochs_ran": 0, "round_sequence": []},
                },
                "validation": {"key_schedule": schedule},
                "final_evaluation": {
                    "auc_mean": auc,
                    "repeats": 3,
                    "metrics_by_repeat": [{}, {}, {}],
                },
                "parameter_count": count,
                "trainable_parameter_count": count,
            }
        )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = gate_feistel_sm4_position_calibration(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0,),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "feistel_sm4_position_anchor_retained_cross_key"
    assert report["signal"] == {"fixed": True, "rotating": True}
