from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2.gift64_output_prediction_discovery import (
    MODEL_NAMES,
    Gift64DiscoveryTrainingConfig,
    adjudicate_gift64_discovery,
    evaluate_gift64_output_split,
    freeze_gift64_candidates,
    gift64_candidate_sha256,
    select_gift64_discovery_candidates,
    train_gift64_discovery_matrix,
)


def test_formal_gx1_training_protocol_is_frozen() -> None:
    config = Gift64DiscoveryTrainingConfig.formal()

    assert config.seed == 11
    assert config.hidden_dim == 1936
    assert config.epochs == 100
    assert config.batch_size == 250
    assert config.candidate_limit == 8
    assert config.minimum_fresh_confirmed == 4
    with pytest.raises(ValueError, match="formal GX1"):
        Gift64DiscoveryTrainingConfig(
            mode="formal", epochs=99, batch_size=250, device="cuda"
        )


def test_tiny_training_uses_matched_mlp_initialization_and_shuffle_only_labels(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(11)
    source = {
        "metadata": {
            "cipher": "GIFT-64",
            "rounds": 3,
            "seed": 11,
            "key_seed": 11,
            "train_rows": 8,
            "discovery_rows": 8,
        },
        "train_features": rng.integers(0, 2, size=(8, 64)).astype(np.float32),
        "train_targets": rng.integers(0, 2, size=(8, 64)).astype(np.float32),
        "discovery_features": rng.integers(0, 2, size=(8, 64)).astype(np.float32),
        "discovery_targets": rng.integers(0, 2, size=(8, 64)).astype(np.float32),
    }
    config = Gift64DiscoveryTrainingConfig(
        hidden_dim=4, epochs=1, batch_size=4, device="cpu"
    )

    training = train_gift64_discovery_matrix(config, source, tmp_path)
    evaluation = evaluate_gift64_output_split(
        config,
        tmp_path,
        source["discovery_features"],
        source["discovery_targets"],
        split="discovery",
    )
    candidates = select_gift64_discovery_candidates(
        config, evaluation["per_bit_rows"]
    )
    frozen = freeze_gift64_candidates(candidates, tmp_path)
    fresh_evaluation = evaluate_gift64_output_split(
        config,
        tmp_path,
        source["discovery_features"],
        source["discovery_targets"],
        split="fresh_confirmation",
    )
    gate = adjudicate_gift64_discovery(
        config,
        {"source_valid": True},
        {"fresh_valid": True},
        training,
        evaluation["per_bit_rows"],
        fresh_evaluation["per_bit_rows"],
        candidates,
        candidate_sha256=frozen["candidate_sha256"],
    )

    rows = {row["model"]: row for row in training["rows"]}
    checkpoints = {row["model"]: row for row in training["checkpoints"]}
    assert set(rows) == set(MODEL_NAMES)
    assert len(training["history"]) == 2
    assert rows[MODEL_NAMES[0]]["parameters"] == rows[MODEL_NAMES[1]]["parameters"]
    assert rows[MODEL_NAMES[0]]["train_labels_shuffled"] is False
    assert rows[MODEL_NAMES[1]]["train_labels_shuffled"] is True
    assert checkpoints[MODEL_NAMES[0]]["initial_state_sha256"] == checkpoints[
        MODEL_NAMES[1]
    ]["initial_state_sha256"]
    assert all(row["sha256"] for row in checkpoints.values())
    assert len(evaluation["per_bit_rows"]) == 2 * 64
    assert len(evaluation["full_output_rows"]) == 2
    assert gate["status"] in {"pass", "hold"}
    assert all(gate["protocol_checks"]["execution"].values())


def test_candidate_freeze_is_deterministic_and_precedes_fresh(tmp_path: Path) -> None:
    config = Gift64DiscoveryTrainingConfig()
    discovery_rows = _metric_rows("discovery", strong_bits=set(range(8)))

    candidates = select_gift64_discovery_candidates(config, discovery_rows)
    frozen = freeze_gift64_candidates(candidates, tmp_path)

    assert candidates["candidate_msb_indices"] == list(range(8))
    assert all(
        row["shuffle_control_scope"] == "architecture_matched"
        for row in candidates["candidates"]
    )
    candidate_bytes = (tmp_path / "candidates.json").read_bytes()
    assert frozen["candidate_sha256"] == hashlib.sha256(candidate_bytes).hexdigest()
    assert frozen["event"] == "candidates_frozen_before_fresh_generation"
    assert frozen["candidate_sha256"] in (tmp_path / "candidates.sha256").read_text(
        encoding="ascii"
    )
    assert json.loads(candidate_bytes)["confirmation_split"] == (
        "fresh_not_generated_or_read_when_frozen"
    )


def test_fresh_gate_requires_four_matched_control_confirmations() -> None:
    config = Gift64DiscoveryTrainingConfig()
    discovery_rows = _metric_rows("discovery", strong_bits=set(range(8)))
    candidates = select_gift64_discovery_candidates(config, discovery_rows)
    training = _training(config)

    passed = adjudicate_gift64_discovery(
        config,
        {"source_valid": True},
        {"fresh_valid": True},
        training,
        discovery_rows,
        _metric_rows("fresh_confirmation", strong_bits={0, 1, 2, 3}),
        candidates,
        candidate_sha256=gift64_candidate_sha256(candidates),
    )
    held = adjudicate_gift64_discovery(
        config,
        {"source_valid": True},
        {"fresh_valid": True},
        training,
        discovery_rows,
        _metric_rows("fresh_confirmation", strong_bits={0, 1, 2}),
        candidates,
        candidate_sha256=gift64_candidate_sha256(candidates),
    )

    assert passed["status"] == "pass"
    assert passed["metrics"]["fresh_confirmed_msb_indices"] == [0, 1, 2, 3]
    assert passed["next_action"]["next_adjudication"] == (
        "gx2_selected8_architecture_screen"
    )
    assert held["status"] == "hold"
    assert held["next_action"]["next_adjudication"] == (
        "close_gift_output_position_route"
    )


def test_invalid_source_fails_closed_before_performance_interpretation() -> None:
    config = Gift64DiscoveryTrainingConfig()
    discovery_rows = _metric_rows("discovery", strong_bits=set(range(8)))
    candidates = select_gift64_discovery_candidates(config, discovery_rows)

    gate = adjudicate_gift64_discovery(
        config,
        {"source_valid": False},
        {"fresh_valid": True},
        _training(config),
        discovery_rows,
        _metric_rows("fresh_confirmation", strong_bits=set(range(8))),
        candidates,
        candidate_sha256=gift64_candidate_sha256(candidates),
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation2_gift64_r3_output_prediction_protocol_invalid"
    )
    assert gate["next_action"]["next_adjudication"] == "repair_gx1_protocol_only"


def test_missing_fairness_evidence_fails_closed() -> None:
    config = Gift64DiscoveryTrainingConfig()
    discovery_rows = _metric_rows("discovery", strong_bits=set(range(8)))
    candidates = select_gift64_discovery_candidates(config, discovery_rows)
    training = _training(config)
    training["checkpoints"][0].pop("initial_state_sha256")

    gate = adjudicate_gift64_discovery(
        config,
        {"source_valid": True},
        {"fresh_valid": True},
        training,
        discovery_rows,
        _metric_rows("fresh_confirmation", strong_bits=set(range(8))),
        candidates,
        candidate_sha256=gift64_candidate_sha256(candidates),
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["execution"][
        "matched_models_share_initialization"
    ] is False


def test_candidate_hash_must_match_frozen_payload() -> None:
    config = Gift64DiscoveryTrainingConfig()
    discovery_rows = _metric_rows("discovery", strong_bits=set(range(8)))
    candidates = select_gift64_discovery_candidates(config, discovery_rows)

    gate = adjudicate_gift64_discovery(
        config,
        {"source_valid": True},
        {"fresh_valid": True},
        _training(config),
        discovery_rows,
        _metric_rows("fresh_confirmation", strong_bits=set(range(8))),
        candidates,
        candidate_sha256="a" * 64,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["execution"][
        "candidate_hash_matches_payload"
    ] is False


def _metric_rows(split: str, *, strong_bits: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        for bit in range(64):
            strong = model == MODEL_NAMES[0] and bit in strong_bits
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "msb_index": bit,
                    "threshold_accuracy": 0.515 if strong else 0.5,
                    "majority_accuracy": 0.5,
                    "accuracy_minus_majority": 0.015 if strong else 0.0,
                    "auc": 0.52 if strong else 0.5,
                    "mse": 0.24 if strong else 0.25,
                    "invalid_numpy_rint_rate": 0.0,
                }
            )
    return rows


def _training(config: Gift64DiscoveryTrainingConfig) -> dict[str, object]:
    rows = [
        {
            "model": model,
            "cipher": "GIFT-64",
            "key_seed": 11,
            "secret_key_scope": "single_fixed_unknown_key",
            "parameters": 100,
            "test_target_identity": "true_full_gift64_ciphertext_targets",
            "discovery_mse": 0.25,
            "discovery_bit_match": 0.5,
            "discovery_macro_auc": 0.5,
            "discovery_exact_match": 0.0,
            "discovery_exact_match_count": 0,
            "discovery_invalid_rounded_cell_rate": 0.0,
            "discovery_majority_bit_match": 0.5,
        }
        for model in MODEL_NAMES
    ]
    checkpoints = [
        {
            "model": model,
            "sha256": "b" * 64,
            "initial_state_sha256": "c" * 64,
        }
        for model in MODEL_NAMES
    ]
    history = [
        {"model": model, "epoch": epoch}
        for model in MODEL_NAMES
        for epoch in range(1, config.epochs + 1)
    ]
    return {"rows": rows, "checkpoints": checkpoints, "history": history}
