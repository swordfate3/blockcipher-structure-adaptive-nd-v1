from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.fit_state_token_residual_expert import main as fit_state_token_main
from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
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
                "negative_mode": "encrypted_random_plaintexts",
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
