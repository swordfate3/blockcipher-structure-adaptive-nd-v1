from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.fit_state_token_residual_expert import main as fit_state_token_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, load_score_artifact, write_score_artifact
from blockcipher_nd.models.structure.spn.present_state_token_residual import (
    PresentStateTokenResidualDistinguisher,
)


def test_fit_state_token_residual_expert_writes_score_artifacts(tmp_path: Path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_features, train_labels = _toy_trail_position_features(rows=12)
    validation_features, validation_labels = _toy_trail_position_features(rows=8)
    _write_feature_dir(train_dir, split="train", features=train_features, labels=train_labels)
    _write_feature_dir(validation_dir, split="validation", features=validation_features, labels=validation_labels)

    train_scores = tmp_path / "train_scores"
    validation_scores = tmp_path / "validation_scores"
    report_path = tmp_path / "report.json"
    status = fit_state_token_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-train-dir",
            str(train_scores),
            "--output-validation-dir",
            str(validation_scores),
            "--output-report",
            str(report_path),
            "--steps",
            "8",
            "--learning-rate",
            "0.01",
            "--token-dim",
            "4",
            "--hidden-bits",
            "8",
            "--batch-size",
            "4",
            "--seed",
            "7",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation_artifact = load_score_artifact(validation_scores)
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "state_token_residual_feature_artifact_local_diagnostic"
    assert report["model_key"] == "present_state_token_residual"
    assert report["feature_view"] == "trail_position_stats"
    assert report["selected_span_feature_bits"] == 731
    assert report["train_rows"] == 12
    assert report["validation_rows"] == 8
    assert validation_artifact.metadata["model_key"] == "present_state_token_residual"
    assert validation_artifact.metadata["feature_model"] == "state_token_residual"
    assert validation_artifact.metadata["selected_span_feature_bits"] == 731
    assert validation_artifact.probabilities.shape == (8,)
    assert np.isfinite(validation_artifact.probabilities).all()


def test_fit_state_token_residual_expert_requires_trail_position_stats(tmp_path: Path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    features, labels = _toy_trail_position_features(rows=4)
    _write_feature_dir(train_dir, split="train", features=features, labels=labels, feature_view="raw")
    _write_feature_dir(validation_dir, split="validation", features=features, labels=labels, feature_view="raw")

    report_path = tmp_path / "report.json"
    try:
        fit_state_token_main(
            [
                "--train-feature-dir",
                str(train_dir),
                "--validation-feature-dir",
                str(validation_dir),
                "--output-validation-dir",
                str(tmp_path / "scores"),
                "--output-report",
                str(report_path),
                "--steps",
                "1",
            ]
        )
    except ValueError as exc:
        assert "feature_view must be trail_position_stats" in str(exc)
    else:
        raise AssertionError("expected feature_view validation failure")


def test_fit_state_token_residual_expert_can_shuffle_token_coordinates(tmp_path: Path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_features, train_labels = _toy_trail_position_features(rows=12)
    validation_features, validation_labels = _toy_trail_position_features(rows=8)
    _write_feature_dir(train_dir, split="train", features=train_features, labels=train_labels)
    _write_feature_dir(validation_dir, split="validation", features=validation_features, labels=validation_labels)

    validation_scores = tmp_path / "coordinate_shuffle_scores"
    report_path = tmp_path / "coordinate_shuffle_report.json"
    status = fit_state_token_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_scores),
            "--output-report",
            str(report_path),
            "--steps",
            "4",
            "--learning-rate",
            "0.01",
            "--token-dim",
            "4",
            "--hidden-bits",
            "8",
            "--batch-size",
            "4",
            "--seed",
            "7",
            "--shuffle-token-coordinates",
            "--token-coordinate-shuffle-seed",
            "11",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation_artifact = load_score_artifact(validation_scores)
    assert status == 0
    assert report["decision"] == "state_token_residual_token_coordinate_shuffle_control"
    assert report["token_coordinate_control"] == {
        "shuffle_token_coordinates": True,
        "token_coordinate_shuffle_seed": 11,
    }
    assert validation_artifact.metadata["token_coordinates_shuffled"] is True
    assert validation_artifact.metadata["token_coordinate_shuffle_seed"] == 11
    assert np.isfinite(validation_artifact.probabilities).all()


def test_fit_state_token_residual_correction_expert_adds_to_frozen_base(tmp_path: Path):
    from blockcipher_nd.cli.fit_state_token_residual_correction_expert import (
        main as fit_correction_main,
    )

    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_features, train_labels = _toy_trail_position_features(rows=12)
    validation_features, validation_labels = _toy_trail_position_features(rows=8)
    _write_feature_dir(train_dir, split="train", features=train_features, labels=train_labels)
    _write_feature_dir(validation_dir, split="validation", features=validation_features, labels=validation_labels)

    left_train = tmp_path / "left_train"
    right_train = tmp_path / "right_train"
    left_validation = tmp_path / "left_validation"
    right_validation = tmp_path / "right_validation"
    train_base_left = np.linspace(0.35, 0.65, len(train_labels), dtype=np.float32)
    train_base_right = np.linspace(0.40, 0.60, len(train_labels), dtype=np.float32)
    validation_base_left = np.linspace(0.35, 0.65, len(validation_labels), dtype=np.float32)
    validation_base_right = np.linspace(0.40, 0.60, len(validation_labels), dtype=np.float32)
    for path, labels, probabilities, model_key in [
        (left_train, train_labels, train_base_left, "trail"),
        (right_train, train_labels, train_base_right, "raw117"),
        (left_validation, validation_labels, validation_base_left, "trail"),
        (right_validation, validation_labels, validation_base_right, "raw117"),
    ]:
        _write_score_dir(path, labels=labels, probabilities=probabilities, model_key=model_key)

    train_scores = tmp_path / "correction_train_scores"
    validation_scores = tmp_path / "correction_validation_scores"
    report_path = tmp_path / "correction_report.json"
    status = fit_correction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--train-base-artifacts",
            str(left_train),
            str(right_train),
            "--validation-base-artifacts",
            str(left_validation),
            str(right_validation),
            "--output-train-dir",
            str(train_scores),
            "--output-validation-dir",
            str(validation_scores),
            "--output-report",
            str(report_path),
            "--steps",
            "10",
            "--learning-rate",
            "0.01",
            "--token-dim",
            "4",
            "--hidden-bits",
            "8",
            "--batch-size",
            "4",
            "--seed",
            "7",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation_artifact = load_score_artifact(validation_scores)
    base_logits = np.stack([
        _logit(validation_base_left),
        _logit(validation_base_right),
    ], axis=1).mean(axis=1)
    assert status == 0
    assert report["status"] == "pass"
    assert report["model_key"] == "present_state_token_residual_correction"
    assert report["base_model_order"] == ["trail", "raw117"]
    assert report["correction_initialization"] == {"zero_initialize_correction_head": True}
    assert report["validation_base_logit_mean_metrics"]["auc"] < 1.0
    assert "delta_validation_corrected_vs_base_logit_mean_auc" in report
    assert validation_artifact.metadata["feature_model"] == "state_token_residual_logit_correction"
    assert validation_artifact.metadata["base_fusion"] == "logit_mean"
    assert validation_artifact.metadata["base_model_order"] == ["trail", "raw117"]
    assert validation_artifact.metadata["correction_head_zero_initialized"] is True
    assert validation_artifact.metadata["score_split"] == "validation"
    assert not np.allclose(validation_artifact.logits, base_logits)
    assert np.isfinite(validation_artifact.probabilities).all()


def test_fit_state_token_residual_correction_expert_requires_strict_negatives(tmp_path: Path):
    from blockcipher_nd.cli.fit_state_token_residual_correction_expert import (
        main as fit_correction_main,
    )

    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_features, train_labels = _toy_trail_position_features(rows=12)
    validation_features, validation_labels = _toy_trail_position_features(rows=8)
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=train_labels,
        negative_mode="random_ciphertexts",
    )
    _write_feature_dir(validation_dir, split="validation", features=validation_features, labels=validation_labels)

    left_train = tmp_path / "left_train"
    right_train = tmp_path / "right_train"
    left_validation = tmp_path / "left_validation"
    right_validation = tmp_path / "right_validation"
    for path, labels, model_key in [
        (left_train, train_labels, "trail"),
        (right_train, train_labels, "raw117"),
        (left_validation, validation_labels, "trail"),
        (right_validation, validation_labels, "raw117"),
    ]:
        _write_score_dir(path, labels=labels, probabilities=np.full(len(labels), 0.5), model_key=model_key)

    try:
        fit_correction_main(
            [
                "--train-feature-dir",
                str(train_dir),
                "--validation-feature-dir",
                str(validation_dir),
                "--train-base-artifacts",
                str(left_train),
                str(right_train),
                "--validation-base-artifacts",
                str(left_validation),
                str(right_validation),
                "--output-validation-dir",
                str(tmp_path / "scores"),
                "--output-report",
                str(tmp_path / "report.json"),
                "--steps",
                "1",
            ]
        )
    except ValueError as exc:
        assert "negative_mode must be encrypted_random_plaintexts" in str(exc)
    else:
        raise AssertionError("expected non-strict negative mode to be rejected")


def test_state_token_residual_correction_head_can_start_at_zero():
    from blockcipher_nd.cli.fit_state_token_residual_correction_expert import (
        _zero_initialize_correction_head,
    )

    model = PresentStateTokenResidualDistinguisher(input_bits=3708, token_dim=4, hidden_bits=8)
    features = torch.randn(5, 3708)

    _zero_initialize_correction_head(model)

    logits = model(features).squeeze(1)
    assert torch.allclose(logits, torch.zeros_like(logits))


def _toy_trail_position_features(*, rows: int) -> tuple[np.ndarray, np.ndarray]:
    model = PresentStateTokenResidualDistinguisher(input_bits=3708, token_dim=4, hidden_bits=8)
    features = np.zeros((rows, 3708), dtype=np.float32)
    labels = np.array([0.0, 1.0] * ((rows + 1) // 2), dtype=np.float32)[:rows]
    signal_index = int(model.span_feature_indices[0])
    features[:, signal_index] = labels * 2.0 - 1.0
    features[:, int(model.span_feature_indices[1])] = np.linspace(-0.5, 0.5, rows, dtype=np.float32)
    return features, labels


def _write_feature_dir(
    path: Path,
    *,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_view: str = "trail_position_stats",
    negative_mode: str = "encrypted_random_plaintexts",
) -> None:
    path.mkdir(parents=True)
    np.save(path / "features.npy", features.astype(np.float32, copy=False))
    np.save(path / "labels.npy", labels.astype(np.float32, copy=False))
    np.save(path / "sample_ids.npy", np.array([str(index) for index in range(len(labels))], dtype=str))
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "kind": "bit_sensitivity_feature_matrix",
                "split": split,
                "feature_view": feature_view,
                "cipher": "PRESENT-80",
                "rounds": 8,
                "samples_per_class": int(len(labels) // 2),
                "pairs_per_sample": 16,
                "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
                "negative_mode": negative_mode,
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
                "difference_profile": "present_zhang_wang2022_mcnd",
                "difference_member": 0,
                "train_key": "0x00000000000000000000",
                "validation_key": "0x11111111111111111111",
                "feature_view_metadata": {"view": feature_view, "output_feature_bits": int(features.shape[1])},
            }
        ),
        encoding="utf-8",
    )


def _write_score_dir(
    path: Path,
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_key: str,
) -> None:
    write_score_artifact(
        path,
        EnsembleScoreArtifact(
            labels=labels.astype(np.float32, copy=False),
            probabilities=probabilities.astype(np.float32, copy=False),
            logits=_logit(probabilities).astype(np.float32, copy=False),
            sample_ids=np.array([str(index) for index in range(len(labels))], dtype=str),
            metadata={
                "model_key": model_key,
                "run_id": model_key,
                "cipher": "PRESENT-80",
                "rounds": 8,
                "feature_view": "trail_position_stats",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
            },
        ),
    )


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))
