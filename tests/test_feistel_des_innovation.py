from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.ciphers.feistel.des import DES_IP, _permute
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.models.structure.feistel import (
    DesFeistelBranchInceptionPairSetDistinguisher,
    DesZhangWangOfficialLayoutDistinguisher,
    des_canonical_bit_indices,
)
from blockcipher_nd.planning.feistel_des_gate import feistel_des_decision
from blockcipher_nd.planning.feistel_des_gate import (
    gate_feistel_des_official_attribution,
    gate_feistel_des_official_calibration,
)
from blockcipher_nd.planning.feistel_des_gate import gate_feistel_des_results
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.difference_profiles import (
    difference_for_profile,
    literature_difference_profiles,
)
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_des_difference_profile_compensates_for_external_ip() -> None:
    external = difference_for_profile("des_zhang_wang2022_mcnd")
    internal = _permute(external, DES_IP, 64)
    assert external == 0x0000801000004000
    assert internal == 0x4008000004000000
    assert literature_difference_profiles()[
        "des_zhang_wang2022_mcnd"
    ].pairs_per_sample == 16


def test_des_true_mapping_restores_internal_left_right_state() -> None:
    cipher = build_cipher("des", rounds=6)
    ciphertext = cipher.encrypt(0x0123456789ABCDEF)
    other = cipher.encrypt(0x1111111111111111)
    features = torch.tensor(
        [int_to_bits(ciphertext, 64) + int_to_bits(other, 64)],
        dtype=torch.float32,
    )
    model = DesFeistelBranchInceptionPairSetDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        blocks=1,
        classifier_bits=8,
    )
    canonical_first = model.canonical_pairs(features)[0, 0, 0].reshape(-1)
    preoutput = _permute(ciphertext, DES_IP, 64)
    expected = ((preoutput & 0xFFFFFFFF) << 32) | (preoutput >> 32)
    assert canonical_first.tolist() == [float(bit) for bit in int_to_bits(expected, 64)]


def test_des_shuffled_mapping_is_a_fixed_capacity_preserving_control() -> None:
    true = des_canonical_bit_indices("true")
    shuffled_a = des_canonical_bit_indices("shuffled")
    shuffled_b = des_canonical_bit_indices("shuffled")
    assert sorted(true.tolist()) == list(range(64))
    assert sorted(shuffled_a.tolist()) == list(range(64))
    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.equal(true, shuffled_a)


def test_feistel_model_registry_rows_forward_and_backpropagate() -> None:
    features = torch.randint(0, 2, (2, 4 * 128), dtype=torch.float32)
    for model_key in (
        "des_feistel_branch_inception_true",
        "des_feistel_branch_inception_shuffled",
        "des_zhang_wang_inception_pairset",
        "des_lstm_pairset",
    ):
        model = build_model(
            model_key,
            input_bits=features.shape[1],
            hidden_bits=4,
            pair_bits=128,
            structure="Feistel-like",
            model_options={
                "blocks": 1,
                "classifier_bits": 8,
                "lstm_hidden_bits": 8,
            },
        )
        logits = model(features)
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
        logits.mean().backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_official_des_layout_matches_public_code_and_preserves_control_capacity() -> None:
    features = torch.randint(0, 2, (2, 16 * 128), dtype=torch.float32)
    official = DesZhangWangOfficialLayoutDistinguisher(
        input_bits=features.shape[1]
    )
    assert official.branch_channels(features).shape == (2, 16, 4, 32)
    assert [branch.kernel_size for branch in official.initial_branches] == [1, 4, 6]
    assert len(official.residual_blocks) == 5
    assert [
        block.layers[0].kernel_size for block in official.residual_blocks
    ] == [3, 5, 7, 9, 11]
    assert official(features).shape == (2, 1)

    true = build_model(
        "des_feistel_official_backbone_true",
        input_bits=features.shape[1],
        hidden_bits=8,
        pair_bits=128,
        structure="Feistel-like",
        model_options={},
    )
    shuffled = build_model(
        "des_feistel_official_backbone_shuffled",
        input_bits=features.shape[1],
        hidden_bits=8,
        pair_bits=128,
        structure="Feistel-like",
        model_options={},
    )
    raw_shuffled = build_model(
        "des_zhang_wang_official_layout_shuffled",
        input_bits=features.shape[1],
        hidden_bits=32,
        pair_bits=128,
        structure="Feistel-like",
        model_options={},
    )
    assert sum(parameter.numel() for parameter in true.parameters()) == sum(
        parameter.numel() for parameter in shuffled.parameters()
    )
    assert not torch.equal(true.mapping_indices, shuffled.mapping_indices)
    assert sum(parameter.numel() for parameter in official.parameters()) == sum(
        parameter.numel() for parameter in raw_shuffled.parameters()
    )
    assert not torch.equal(official.mapping_indices, raw_shuffled.mapping_indices)
    for model in (true, shuffled):
        logits = model(features)
        assert logits.shape == (2, 1)
        logits.mean().backward()


def test_feistel_des_local_plan_is_two_seed_same_protocol_matrix() -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r6_branch_inception_2048_seed0_seed1.csv"
    tasks = tasks_from_plan(
        plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(tasks) == 8
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {
        "des_feistel_branch_inception_true",
        "des_feistel_branch_inception_shuffled",
        "des_zhang_wang_inception_pairset",
        "des_lstm_pairset",
    }
    for task in tasks:
        assert task["cipher_key"] == "des"
        assert task["rounds"] == 6
        assert task["samples_per_class"] == 2048
        assert task["pairs_per_sample"] == 16
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["input_difference"] == 0x0000801000004000


def test_feistel_gate_requires_topology_attribution_and_competitiveness() -> None:
    passed = feistel_des_decision(
        {
            0: {
                "typed_true": 0.620,
                "typed_shuffled": 0.610,
                "paper_inception": 0.619,
                "lstm": 0.600,
            },
            1: {
                "typed_true": 0.615,
                "typed_shuffled": 0.607,
                "paper_inception": 0.616,
                "lstm": 0.603,
            },
        }
    )
    assert passed["decision"] == "feistel_branch_candidate_ready_for_medium_diagnostic"

    failed = feistel_des_decision(
        {
            0: {
                "typed_true": 0.620,
                "typed_shuffled": 0.619,
                "paper_inception": 0.618,
                "lstm": 0.600,
            },
            1: {
                "typed_true": 0.615,
                "typed_shuffled": 0.616,
                "paper_inception": 0.614,
                "lstm": 0.603,
            },
        }
    )
    assert failed["decision"] == "feistel_signal_without_branch_topology_attribution"


def test_readiness_gate_never_applies_a_research_decision(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r6_branch_inception_readiness_seed0.csv"
    rows = []
    for model_key in (
        "des_feistel_branch_inception_true",
        "des_feistel_branch_inception_shuffled",
        "des_zhang_wang_inception_pairset",
        "des_lstm_pairset",
    ):
        rows.append(
            {
                "cipher_key": "des",
                "model": model_key,
                "rounds": 6,
                "seed": 0,
                "samples_per_class": 64,
                "pairs_per_sample": 16,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "zhang_wang_case2_official_mcnd",
                "key_rotation_interval": 0,
                "integral_active_nibble": 0,
                "integral_active_nibbles": [],
                "validation_integral_active_nibbles": [],
                "selected_bit_indices": [],
                "training": {
                    "key_schedule": "per_pair_random",
                    "model_options": (
                        {"classifier_bits": 128, "lstm_hidden_bits": 128}
                        if model_key == "des_lstm_pairset"
                        else {
                            "blocks": 3,
                            "classifier_bits": 128,
                            "dropout": 0.0,
                            "initial_kernel_sizes": [1, 4, 6],
                        }
                    )
                },
                "validation": {"key_schedule": "per_pair_random"},
                "difference_profile": "des_zhang_wang2022_mcnd",
                "difference_member": 0,
                "final_test_repeats": 1,
                "history": [{}, {}],
                "final_evaluation": {
                    "auc_mean": 0.99,
                    "repeats": 1,
                    "metrics_by_repeat": [{"auc": 0.99}],
                },
                "parameter_count": 1,
                "trainable_parameter_count": 1,
            }
        )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = gate_feistel_des_results(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=64,
        expected_seeds=(0,),
        expected_epochs=2,
        expected_final_repeats=1,
    )
    assert report["status"] == "pass"
    assert report["decision"] == "feistel_des_readiness_passed"
    assert report["research_decision_applied"] is False

    rows[1]["parameter_count"] = 2
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    invalid = gate_feistel_des_results(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=64,
        expected_seeds=(0,),
        expected_epochs=2,
        expected_final_repeats=1,
    )
    assert invalid["status"] == "fail"
    assert invalid["decision"] == "invalid_feistel_des_protocol"
    assert any("parameter_count mismatch" in error for error in invalid["errors"])


def test_official_calibration_gate_requires_both_des5_seeds(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r5_official_layout_2048_seed0_seed1.csv"
    rows = []
    for seed, auc in ((0, 0.65), (1, 0.61)):
        rows.append(
            {
                "cipher": "DES",
                "cipher_key": "des",
                "structure": "Feistel-like",
                "model": "des_zhang_wang_official_layout",
                "rounds": 5,
                "seed": seed,
                "samples_per_class": 2048,
                "pairs_per_sample": 16,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "zhang_wang_case2_official_mcnd",
                "key_rotation_interval": 0,
                "integral_active_nibble": 0,
                "integral_active_nibbles": [],
                "validation_integral_active_nibbles": [],
                "selected_bit_indices": [],
                "difference_profile": "des_zhang_wang2022_mcnd",
                "difference_member": 0,
                "final_test_repeats": 3,
                "history": [{} for _ in range(10)],
                "training": {
                    "key_schedule": "per_pair_random",
                    "loss": "mse",
                    "learning_rate": 0.0001,
                    "optimizer": "adam",
                    "weight_decay": 0.0008,
                    "checkpoint_metric": "val_loss",
                    "restore_best_checkpoint": True,
                    "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0,
                    "selected_bit_indices": [],
                    "model_options": {
                        "blocks": 5,
                        "initial_kernel_sizes": [1, 4, 6],
                    },
                    "pretraining": {"epochs_ran": 0, "round_sequence": []},
                },
                "validation": {"key_schedule": "per_pair_random"},
                "final_evaluation": {
                    "auc_mean": auc,
                    "repeats": 3,
                    "metrics_by_repeat": [{}, {}, {}],
                },
                "parameter_count": 649793,
                "trainable_parameter_count": 649793,
            }
        )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    passed = gate_feistel_des_official_calibration(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert passed["status"] == "pass", passed["errors"]
    assert passed["decision"] == "feistel_des5_official_calibration_passed"

    rows[1]["final_evaluation"]["auc_mean"] = 0.59
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    failed = gate_feistel_des_official_calibration(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert failed["status"] == "pass"
    assert failed["decision"] == "feistel_des5_official_calibration_inconclusive"


def _official_attribution_rows(
    *, rounds: int, role_scores: dict[int, dict[str, float]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, scores in role_scores.items():
        for model, auc in scores.items():
            parameter_count = (
                649793
                if model.startswith("des_zhang_wang_official_layout")
                else 651201
            )
            rows.append(
                {
                    "cipher": "DES",
                    "cipher_key": "des",
                    "structure": "Feistel-like",
                    "model": model,
                    "rounds": rounds,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "key_rotation_interval": 0,
                    "integral_active_nibble": 0,
                    "integral_active_nibbles": [],
                    "validation_integral_active_nibbles": [],
                    "selected_bit_indices": [],
                    "difference_profile": "des_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "final_test_repeats": 3,
                    "history": [{} for _ in range(10)],
                    "training": {
                        "key_schedule": "per_pair_random",
                        "loss": "mse",
                        "learning_rate": 0.0001,
                        "optimizer": "adam",
                        "weight_decay": 0.0008,
                        "checkpoint_metric": "val_loss",
                        "restore_best_checkpoint": True,
                        "early_stopping_patience": 0,
                        "early_stopping_min_delta": 0.0,
                        "selected_bit_indices": [],
                        "model_options": {
                            "blocks": 5,
                            "initial_kernel_sizes": [1, 4, 6],
                        },
                        "pretraining": {"epochs_ran": 0, "round_sequence": []},
                    },
                    "validation": {"key_schedule": "per_pair_random"},
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


def test_official_attribution_gate_requires_true_mapping_on_both_seeds(
    tmp_path: Path,
) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r6_official_backbone_attribution_2048_seed0_seed1.csv"
    role_scores = {
        0: {
            "des_feistel_official_backbone_true": 0.62,
            "des_feistel_official_backbone_shuffled": 0.61,
            "des_zhang_wang_official_layout": 0.621,
        },
        1: {
            "des_feistel_official_backbone_true": 0.61,
            "des_feistel_official_backbone_shuffled": 0.60,
            "des_zhang_wang_official_layout": 0.611,
        },
    }
    rows = _official_attribution_rows(rounds=6, role_scores=role_scores)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    passed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert passed["status"] == "pass", passed["errors"]
    assert passed["decision"] == "feistel_des6_official_branch_attribution_passed"

    rows[3]["final_evaluation"]["auc_mean"] = 0.60
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    failed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
    )
    assert failed["status"] == "pass"
    assert failed["decision"] == "feistel_des6_signal_without_topology_attribution"


def test_des5_official_attribution_plan_and_strong_signal_gate(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r5_official_backbone_attribution_2048_seed0_seed1.csv"
    tasks = tasks_from_plan(
        plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(tasks) == 6
    assert {task["rounds"] for task in tasks} == {5}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == set(
        {
            "des_feistel_official_backbone_true",
            "des_feistel_official_backbone_shuffled",
            "des_zhang_wang_official_layout",
        }
    )
    assert all(task["samples_per_class"] == 2048 for task in tasks)
    assert all(task["pairs_per_sample"] == 16 for task in tasks)
    assert all(
        task["negative_mode"] == "encrypted_random_plaintexts" for task in tasks
    )

    role_scores = {
        0: {
            "des_feistel_official_backbone_true": 0.970,
            "des_feistel_official_backbone_shuffled": 0.950,
            "des_zhang_wang_official_layout": 0.971,
        },
        1: {
            "des_feistel_official_backbone_true": 0.960,
            "des_feistel_official_backbone_shuffled": 0.940,
            "des_zhang_wang_official_layout": 0.961,
        },
    }
    rows = _official_attribution_rows(rounds=5, role_scores=role_scores)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    passed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
        expected_rounds=5,
    )
    assert passed["status"] == "pass", passed["errors"]
    assert passed["decision"] == "feistel_des5_official_branch_attribution_passed"
    assert passed["minimum_signal_auc"] == 0.90

    rows[3]["final_evaluation"]["auc_mean"] = 0.93
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    failed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
        expected_rounds=5,
    )
    assert failed["status"] == "pass"
    assert failed["decision"] == "feistel_des5_signal_without_topology_attribution"


def test_des5_official_raw_mapping_plan_and_gate(tmp_path: Path) -> None:
    plan = ROOT / "configs/experiment/innovation1/innovation1_feistel_des_r5_official_raw_mapping_attribution_2048_seed0_seed1.csv"
    tasks = tasks_from_plan(
        plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    assert len(tasks) == 4
    assert {task["rounds"] for task in tasks} == {5}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {
        "des_zhang_wang_official_layout",
        "des_zhang_wang_official_layout_shuffled",
    }
    assert all(task["samples_per_class"] == 2048 for task in tasks)
    assert all(task["pairs_per_sample"] == 16 for task in tasks)
    assert all(
        task["negative_mode"] == "encrypted_random_plaintexts" for task in tasks
    )

    role_scores = {
        0: {
            "des_zhang_wang_official_layout": 0.970,
            "des_zhang_wang_official_layout_shuffled": 0.950,
        },
        1: {
            "des_zhang_wang_official_layout": 0.960,
            "des_zhang_wang_official_layout_shuffled": 0.940,
        },
    }
    rows = _official_attribution_rows(rounds=5, role_scores=role_scores)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    passed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
        expected_rounds=5,
        raw_mapping_only=True,
    )
    assert passed["status"] == "pass", passed["errors"]
    assert passed["decision"] == "feistel_des5_official_raw_mapping_attributed"
    assert passed["gates"] == {
        "topology_attributed": True,
        "signal_present": True,
    }

    rows[3]["final_evaluation"]["auc_mean"] = 0.958
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    failed = gate_feistel_des_official_attribution(
        plan_path=plan,
        results_path=results,
        expected_samples_per_class=2048,
        expected_seeds=(0, 1),
        expected_epochs=10,
        expected_final_repeats=3,
        expected_rounds=5,
        raw_mapping_only=True,
    )
    assert failed["status"] == "pass"
    assert failed["decision"] == "feistel_des5_official_raw_mapping_not_attributed"
