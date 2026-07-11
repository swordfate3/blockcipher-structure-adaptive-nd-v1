from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_state_matrix_conv2d_gate import (
    gate_invp_state_matrix_conv2d,
)


ANCHOR = "present_nibble_invp_only_spn_only"
CANDIDATE = "present_nibble_invp_state_matrix_conv2d_spn_only"
SHUFFLED = "present_nibble_shuffled_p_state_matrix_conv2d_spn_only"
DELTA = "present_nibble_delta_state_matrix_conv2d_spn_only"


def _row(
    model: str,
    auc: float,
    *,
    seed: int = 0,
    parameter_count: int = 1000,
) -> dict[str, Any]:
    options = (
        {"spn_mixer_depth": 2, "activation": "relu", "norm": "layernorm"}
        if model == ANCHOR
        else {
            "conv_depth": 3,
            "kernel_size": 3,
            "activation": "relu",
            "norm": "batchnorm2d",
            "dropout": 0.0,
        }
    )
    return {
        "selected_model": model,
        "rounds": 7,
        "seed": seed,
        "samples_per_class": 8192,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11" * 10, 16),
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "parameter_count": 900 if model == ANCHOR else parameter_count,
        "trainable_parameter_count": 900 if model == ANCHOR else parameter_count,
        "metrics": {
            "auc": auc,
            "accuracy": 0.6,
            "calibrated_accuracy": 0.61,
            "loss": 0.68,
        },
        "training": {
            "input_bits": 2048,
            "pair_bits": 128,
            "train_rows": 16384,
            "validation_rows": 8192,
            "epochs": 10,
            "checkpoint_metric": "val_auc",
            "selected_checkpoint": "best",
            "restore_best_checkpoint": True,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "lr_scheduler": "official_cyclic",
            "max_learning_rate": 0.002,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.0001,
            "dataset_cache_root": f"outputs/local_cache/seed{seed}",
            "model_options": options,
        },
        "validation": {
            "samples_per_class": 4096,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "zhang_wang_case2_official_mcnd",
        },
    }


def _write(
    path: Path,
    aucs: dict[str, float],
    *,
    seed: int = 0,
) -> None:
    rows = [_row(model, auc, seed=seed) for model, auc in aucs.items()]
    _write_rows(path, rows)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_gate_promotes_clear_seed0_result(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.60, CANDIDATE: 0.61, SHUFFLED: 0.605, DELTA: 0.604})

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["status"] == "pass"
    assert report["decision"] == "promote_seed1"
    assert report["seeds"]["0"]["architecture_margin"] == 0.010000000000000009
    assert report["seeds"]["0"]["topology_margin"] == 0.0050000000000000044
    assert report["seeds"]["0"]["representation_margin"] == 0.006000000000000005
    assert report["parameter_counts"]["candidate_to_anchor_ratio"] == 1000 / 900
    assert report["claim_scope"] == (
        "8192/class strict PRESENT r7 architecture-attribution diagnostic; "
        "not formal, paper-scale, or breakthrough evidence"
    )


def test_gate_stops_when_candidate_loses_to_anchor(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.61, CANDIDATE: 0.60, SHUFFLED: 0.59, DELTA: 0.58})

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["decision"] == "stop_conv2d_route"
    assert report["next_action"] == "keep_token_mixer_anchor_and_do_not_scale_conv2d"


def test_gate_stops_generic_locality_when_shuffled_p_beats_candidate(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.59, CANDIDATE: 0.61, SHUFFLED: 0.6105, DELTA: 0.60})

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["decision"] == "stop_generic_locality"


def test_gate_stops_invp_attribution_when_delta_beats_candidate(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.59, CANDIDATE: 0.61, SHUFFLED: 0.60, DELTA: 0.611})

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["decision"] == "stop_invp_attribution"


def test_gate_keeps_submargin_seed0_local(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.60, CANDIDATE: 0.61, SHUFFLED: 0.608, DELTA: 0.606})

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["decision"] == "weak_or_fragile_no_scale"


def test_gate_promotes_two_seeds_passing_joint_gates(tmp_path: Path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    _write(seed0, {ANCHOR: 0.60, CANDIDATE: 0.61, SHUFFLED: 0.606, DELTA: 0.605}, seed=0)
    _write(seed1, {ANCHOR: 0.60, CANDIDATE: 0.607, SHUFFLED: 0.604, DELTA: 0.603}, seed=1)

    report = gate_invp_state_matrix_conv2d([seed0, seed1], expected_seeds=(0, 1))

    assert report["decision"] == "promote_medium_65536"
    assert report["next_action"] == "run_65536_per_class_two_seed_medium_confirmation"


def test_gate_marks_two_seed_control_submargin_unstable(tmp_path: Path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    _write(seed0, {ANCHOR: 0.60, CANDIDATE: 0.61, SHUFFLED: 0.606, DELTA: 0.605}, seed=0)
    _write(seed1, {ANCHOR: 0.60, CANDIDATE: 0.607, SHUFFLED: 0.6055, DELTA: 0.604}, seed=1)

    report = gate_invp_state_matrix_conv2d([seed0, seed1], expected_seeds=(0, 1))

    assert report["decision"] == "unstable_no_remote_scale"


def test_gate_rejects_parameter_and_protocol_mismatch(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _row(ANCHOR, 0.60),
        _row(CANDIDATE, 0.61),
        _row(SHUFFLED, 0.605, parameter_count=1001),
        _row(DELTA, 0.604),
    ]
    rows[-1]["negative_mode"] = "random_ciphertext"
    _write_rows(results, rows)

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("conv2d_parameter_count_mismatch" in error for error in report["errors"])
    assert any("negative_mode" in error for error in report["errors"])
    assert report["next_action"] == "repair_protocol_and_rerun_same_matrix"
    assert report["claim_scope"] == "invalid strict-protocol architecture evidence"


def test_gate_rejects_trainable_count_cache_and_nonfinite_metric(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _row(ANCHOR, 0.60),
        _row(CANDIDATE, 0.61),
        _row(SHUFFLED, 0.605),
        _row(DELTA, 0.604),
    ]
    rows[1]["trainable_parameter_count"] = 999
    rows[2]["training"]["dataset_cache_root"] = ""
    rows[3]["metrics"]["loss"] = float("inf")
    _write_rows(results, rows)

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["status"] == "fail"
    assert any("conv2d_trainable_parameter_count_mismatch" in error for error in report["errors"])
    assert any("dataset_cache_root" in error for error in report["errors"])
    assert any("metrics.loss" in error for error in report["errors"])


def test_gate_rejects_duplicate_and_missing_model_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _row(ANCHOR, 0.60),
        _row(CANDIDATE, 0.61),
        _row(CANDIDATE, 0.61),
        _row(SHUFFLED, 0.605),
    ]
    _write_rows(results, rows)

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["status"] == "fail"
    assert any("candidate" in error and "rows=2" in error for error in report["errors"])
    assert any("delta_only" in error and "rows=0" in error for error in report["errors"])


def test_gate_rejects_protocol_values_with_wrong_json_types(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _row(ANCHOR, 0.60),
        _row(CANDIDATE, 0.61),
        _row(SHUFFLED, 0.605),
        _row(DELTA, 0.604),
    ]
    rows[0]["seed"] = 0.0
    rows[1]["training"]["restore_best_checkpoint"] = 1
    _write_rows(results, rows)

    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))

    assert report["status"] == "fail"
    assert any("seed" in error for error in report["errors"])
    assert any("restore_best_checkpoint" in error for error in report["errors"])
