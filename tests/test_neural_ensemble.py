from __future__ import annotations

import json

import numpy as np
import pytest

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
    write_score_artifact,
)


def artifact_metadata(model_key: str = "model_a") -> dict[str, object]:
    return {
        "cipher": "PRESENT-80",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x00000000000000000000",
        "validation_key": "0x11111111111111111111",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "model_key": model_key,
        "model_options": {},
        "run_id": f"run_{model_key}",
        "checkpoint_path": f"/tmp/{model_key}.pt",
        "git_commit": "test",
    }


def test_score_artifact_round_trip(tmp_path):
    artifact_dir = tmp_path / "score_artifact"
    artifact = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        logits=np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata(),
    )

    write_score_artifact(artifact_dir, artifact)
    loaded = load_score_artifact(artifact_dir)

    np.testing.assert_array_equal(loaded.labels, artifact.labels)
    np.testing.assert_allclose(loaded.probabilities, artifact.probabilities)
    np.testing.assert_allclose(loaded.logits, artifact.logits)
    np.testing.assert_array_equal(loaded.sample_ids, artifact.sample_ids)
    assert loaded.metadata["negative_mode"] == "encrypted_random_plaintexts"
    assert json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))["model_key"] == "model_a"


def test_evaluate_frozen_score_ensemble_reports_fixed_rules():
    left = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
        logits=np.array([-2.2, -0.8, 0.8, 2.2], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata("left"),
    )
    right = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        logits=np.array([-1.4, -0.4, 0.4, 1.4], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata("right"),
    )

    summary = evaluate_frozen_score_ensemble([left, right])

    assert summary["status"] == "pass"
    assert [row["mode"] for row in summary["ensembles"]] == [
        "probability_mean",
        "logit_mean",
        "sum_logodds",
        "auc_positive_weighted_logit_mean",
        "rank_average",
    ]
    assert summary["best_single"]["model_key"] == "left"
    assert summary["claim_scope"].startswith("application-level")
    assert summary["diversity"]["pairwise"][0]["left"] == "left"


def test_evaluate_frozen_score_ensemble_rejects_misaligned_labels():
    left = EnsembleScoreArtifact(
        labels=np.array([0, 1], dtype=np.float32),
        probabilities=np.array([0.2, 0.8], dtype=np.float32),
        logits=np.array([-1.0, 1.0], dtype=np.float32),
        sample_ids=np.array(["0", "1"], dtype=str),
        metadata=artifact_metadata("left"),
    )
    right = EnsembleScoreArtifact(
        labels=np.array([1, 0], dtype=np.float32),
        probabilities=np.array([0.8, 0.2], dtype=np.float32),
        logits=np.array([1.0, -1.0], dtype=np.float32),
        sample_ids=np.array(["0", "1"], dtype=str),
        metadata=artifact_metadata("right"),
    )

    with pytest.raises(ValueError, match="labels differ"):
        evaluate_frozen_score_ensemble([left, right])
