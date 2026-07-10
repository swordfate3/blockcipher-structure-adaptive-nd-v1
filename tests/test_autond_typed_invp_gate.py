from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from blockcipher_nd.cli.gate_autond_typed_invp import main
from blockcipher_nd.planning.autond_typed_invp_gate import gate_autond_typed_invp


AUTOND = "autond_dbitnet2023"
INVP = "present_nibble_invp_only_spn_only"
SHUFFLED = "present_nibble_shuffled_paligned_spn_only"
DELTA = "present_nibble_delta_only_spn_only"


def _stage(rounds: int, step_before: int) -> dict[str, object]:
    return {
        "rounds": rounds,
        "checkpoint_metric": "val_loss",
        "dataset_label_mode": "random_labels_total",
        "optimizer_state_reused": rounds != 5,
        "optimizer_state_step_before": step_before,
        "optimizer_state_step_after": step_before + 192,
        "train_rows": 16_384,
        "train_positive_rows": 8_200,
        "train_negative_rows": 8_184,
        "validation_rows": 4_096,
        "validation_positive_rows": 2_050,
        "validation_negative_rows": 2_046,
    }


def _result_row(model: str, aucs: list[float]) -> dict[str, object]:
    stages = [_stage(rounds, index * 192) for index, rounds in enumerate((5, 6, 7, 8))]
    repeats = [
        {
            "repeat": index + 1,
            "seed": 50_000 + index,
            "samples_total": 4_096,
            "positive_rows": 2_048,
            "negative_rows": 2_048,
            "accuracy": 0.5 + index * 0.001,
            "auc": auc,
        }
        for index, auc in enumerate(aucs)
    ]
    accuracies = [float(item["accuracy"]) for item in repeats]
    return {
        "selected_model": model,
        "rounds": 9,
        "seed": 0,
        "train_samples_total": 16_384,
        "validation_samples_total": 4_096,
        "final_test_samples_total": 4_096,
        "final_test_repeats": 3,
        "dataset_label_mode": "random_labels_total",
        "negative_mode": "random_ciphertext",
        "key_rotation_interval": 1,
        "sample_structure": "independent_pairs",
        "pairs_per_sample": 1,
        "feature_encoding": "ciphertext_pair_bits",
        "training": {
            "checkpoint_metric": "val_loss",
            "dataset_label_mode": "random_labels_total",
            "optimizer_state_transition": "carry_across_stages",
            "optimizer_state_reused": True,
            "optimizer_state_step_before": 768,
            "optimizer_state_step_after": 960,
            "train_rows": 16_384,
            "train_positive_rows": 8_200,
            "train_negative_rows": 8_184,
            "validation_rows": 4_096,
            "validation_positive_rows": 2_050,
            "validation_negative_rows": 2_046,
            "pretraining": {
                "round_sequence": [5, 6, 7, 8],
                "optimizer_state_transition": "carry_across_stages",
                "curriculum_stages": stages,
            },
        },
        "final_evaluation": {
            "repeats": 3,
            "samples_total_per_repeat": 4_096,
            "seeds": [50_000, 50_001, 50_002],
            "metrics_by_repeat": repeats,
            "accuracy_mean": mean(accuracies),
            "accuracy_std": pstdev(accuracies),
            "auc_mean": mean(aucs),
            "auc_std": pstdev(aucs),
        },
    }


def _write_rows(path: Path, aucs_by_model: dict[str, list[float]]) -> None:
    rows = [_result_row(model, aucs) for model, aucs in aucs_by_model.items()]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_typed_invp_gate_supports_candidate_above_all_controls(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_rows(
        results,
        {
            AUTOND: [0.52, 0.53, 0.51],
            INVP: [0.57, 0.58, 0.56],
            SHUFFLED: [0.53, 0.54, 0.52],
            DELTA: [0.54, 0.53, 0.53],
        },
    )

    report = gate_autond_typed_invp(results)

    assert report["status"] == "pass"
    assert report["decision"] == "strong_local_support"
    assert report["candidate_margin_vs_best_control_auc"] >= 0.01
    assert report["candidate_above_all_controls_by_repeat"] is True
    assert report["next_action"] == "run_identical_seed1_local_gate"


def test_typed_invp_gate_marks_ordered_submargin_result_weak(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_rows(
        results,
        {
            AUTOND: [0.54, 0.54, 0.54],
            INVP: [0.55, 0.55, 0.55],
            SHUFFLED: [0.545, 0.545, 0.545],
            DELTA: [0.54, 0.54, 0.54],
        },
    )

    report = gate_autond_typed_invp(results)

    assert report["status"] == "pass"
    assert report["decision"] == "weak_or_fragile"
    assert report["candidate_margin_vs_best_control_auc"] == 0.0050000000000000044
    assert report["next_action"] == "run_seed1_bounded_variance_adjudication"


def test_typed_invp_gate_stops_when_delta_control_matches_candidate(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_rows(
        results,
        {
            AUTOND: [0.53, 0.53, 0.53],
            INVP: [0.55, 0.55, 0.55],
            SHUFFLED: [0.54, 0.54, 0.54],
            DELTA: [0.56, 0.56, 0.56],
        },
    )

    report = gate_autond_typed_invp(results)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_public_typed_adapter"
    assert report["candidate_margins_auc"]["delta_only"] < 0.0
    assert report["next_action"] == "do_not_scale_or_redesign_public_typed_adapter"


def test_typed_invp_gate_downgrades_outlier_driven_mean(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_rows(
        results,
        {
            AUTOND: [0.54, 0.54, 0.54],
            INVP: [0.60, 0.55, 0.55],
            SHUFFLED: [0.54, 0.56, 0.54],
            DELTA: [0.54, 0.54, 0.54],
        },
    )

    report = gate_autond_typed_invp(results)

    assert report["candidate_margin_vs_best_control_auc"] >= 0.01
    assert report["candidate_above_all_controls_by_repeat"] is False
    assert report["decision"] == "weak_or_fragile"


def test_typed_invp_gate_rejects_broken_optimizer_continuity(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_rows(
        results,
        {
            AUTOND: [0.52, 0.53, 0.51],
            INVP: [0.57, 0.58, 0.56],
            SHUFFLED: [0.53, 0.54, 0.52],
            DELTA: [0.54, 0.53, 0.53],
        },
    )
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    stages = rows[1]["training"]["pretraining"]["curriculum_stages"]
    stages[2]["optimizer_state_step_before"] = 385
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = gate_autond_typed_invp(results)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("optimizer_step_continuity" in error for error in report["errors"])
    assert report["next_action"] == "repair_protocol_and_rerun_same_matrix"


def test_typed_invp_gate_cli_writes_report(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "gate.json"
    _write_rows(
        results,
        {
            AUTOND: [0.52, 0.53, 0.51],
            INVP: [0.57, 0.58, 0.56],
            SHUFFLED: [0.53, 0.54, 0.52],
            DELTA: [0.54, 0.53, 0.53],
        },
    )

    status = main(["--results", str(results), "--output", str(output)])

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["decision"] == "strong_local_support"
