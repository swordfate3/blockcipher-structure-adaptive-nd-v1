from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.ciphers.feistel.des import DES_IP, _permute
from blockcipher_nd.features.encoders.bitwise import int_to_bits
from blockcipher_nd.models.structure.feistel import (
    DesFeistelBranchInceptionPairSetDistinguisher,
    des_canonical_bit_indices,
)
from blockcipher_nd.planning.feistel_des_gate import feistel_des_decision
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
