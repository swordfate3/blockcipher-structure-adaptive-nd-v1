from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_data import (
    Speck32OutputPredictionDataConfig,
    prepare_speck32_output_prediction_data,
    validate_speck32_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    BILSTM_MODEL_NAME,
    FCNN_MODEL_NAME,
)
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_training import (
    A1_MODEL_NAMES,
    A2_LOGICAL_MODEL_NAMES,
    A2_NEW_MODEL_NAMES,
    FULL_MATRIX_MODEL_NAMES,
    Speck32OutputPredictionTrainingConfig,
    adjudicate_speck32_arx1_a,
    build_speck32_source_manifest,
    train_speck32_arx1_a1,
    train_speck32_arx1_a2,
)
from blockcipher_nd.tasks.innovation2.speck32_rotation_carry_model import (
    ROTATION_CARRY_MODEL_NAME,
    ROTATION_CARRY_SHUFFLE_MODEL_NAME,
    WRONG_ROTATION_CARRY_MODEL_NAME,
)


class TinyOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Sequential(nn.Linear(32, 32), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features.float())


def test_formal_arx1_a_training_contract_is_frozen() -> None:
    config = Speck32OutputPredictionTrainingConfig.arx1_a()

    assert config.seed == 21
    assert config.epochs == 100
    assert config.batch_size == 128
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.01
    assert config.candidate_channels == 400
    with pytest.raises(ValueError, match="formal ARX1-A"):
        replace(config, epochs=99)


def test_a1_a2_reuse_candidate_and_freeze_all_fairness_evidence(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    source = _source(output_root)
    config = Speck32OutputPredictionTrainingConfig(
        run_id="arx1_tiny",
        epochs=1,
        batch_size=4,
    )
    builders = _tiny_builders()

    a1 = train_speck32_arx1_a1(
        config,
        source,
        output_root,
        model_builders=builders,
    )
    candidate_checkpoint = (
        output_root / "models" / (f"{ROTATION_CARRY_MODEL_NAME}_final.pt")
    )
    candidate_bytes = candidate_checkpoint.read_bytes()
    candidate_mtime = candidate_checkpoint.stat().st_mtime_ns
    a2 = train_speck32_arx1_a2(
        config,
        source,
        output_root,
        model_builders=builders,
    )
    data_checks = validate_speck32_output_prediction_data(
        Speck32OutputPredictionDataConfig(
            run_id="arx1_tiny",
            train_rows=12,
            test_rows=8,
            chunk_rows=5,
        ),
        source,
    )
    gate = adjudicate_speck32_arx1_a(config, data_checks, a1, a2)

    assert {row["model"] for row in a1["rows"]} == set(A1_MODEL_NAMES)
    assert {row["model"] for row in a2["logical_rows"]} == set(A2_LOGICAL_MODEL_NAMES)
    assert {row["model"] for row in a2["full_matrix_rows"]} == set(
        FULL_MATRIX_MODEL_NAMES
    )
    assert len(a2["full_matrix_per_bit_rows"]) == 5 * 32
    assert a2["reused_models"] == [ROTATION_CARRY_MODEL_NAME]
    assert a2["new_models"] == list(A2_NEW_MODEL_NAMES)
    assert candidate_checkpoint.read_bytes() == candidate_bytes
    assert candidate_checkpoint.stat().st_mtime_ns == candidate_mtime
    assert all(a2["fairness_checks"].values())
    checkpoint_index = {row["model"]: row for row in a2["checkpoints"]}
    candidate_initials = {
        checkpoint_index[name]["initial_state_sha256"]
        for name in A2_LOGICAL_MODEL_NAMES
    }
    candidate_schedules = {
        checkpoint_index[name]["batch_schedule_sha256"]
        for name in A2_LOGICAL_MODEL_NAMES
    }
    assert len(candidate_initials) == 1
    assert len(candidate_schedules) == 1
    assert checkpoint_index[ROTATION_CARRY_SHUFFLE_MODEL_NAME][
        "label_permutation_sha256"
    ]
    assert (
        checkpoint_index[WRONG_ROTATION_CARRY_MODEL_NAME]["label_permutation_sha256"]
        is None
    )
    row_index = {row["model"]: row for row in a2["full_matrix_rows"]}
    assert all(
        row["test_target_identity"] == "true_full_speck32_ciphertext_targets"
        for row in row_index.values()
    )
    assert row_index[ROTATION_CARRY_SHUFFLE_MODEL_NAME]["train_labels_shuffled"] is True
    assert row_index[ROTATION_CARRY_MODEL_NAME]["train_labels_shuffled"] is False
    assert row_index[ROTATION_CARRY_MODEL_NAME]["execution"] == (
        "reused_arx1_a1_result_and_checkpoint"
    )
    assert (output_root / "source_manifest.json").is_file()
    assert (output_root / "a1_bundle.sha256").is_file()
    assert (output_root / "a2_bundle.sha256").is_file()
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_arx1_speck32_readiness_passed"
    assert all(gate["protocol_checks"]["execution"].values())
    assert gate["next_action"]["remote_scale"] is False


def test_a2_fails_closed_without_a1_or_after_checkpoint_tampering(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    source = _source(output_root)
    config = Speck32OutputPredictionTrainingConfig(
        run_id="arx1_tiny",
        epochs=1,
        batch_size=4,
    )
    builders = _tiny_builders()

    with pytest.raises(ValueError, match="requires a complete A1 bundle"):
        train_speck32_arx1_a2(
            config,
            source,
            output_root,
            model_builders=builders,
        )

    train_speck32_arx1_a1(
        config,
        source,
        output_root,
        model_builders=builders,
    )
    checkpoint = output_root / "models" / f"{ROTATION_CARRY_MODEL_NAME}_final.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="final checkpoint hash mismatch"):
        train_speck32_arx1_a2(
            config,
            source,
            output_root,
            model_builders=builders,
        )


def test_a2_rejects_data_source_changed_after_a1(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    source = _source(output_root)
    config = Speck32OutputPredictionTrainingConfig(
        run_id="arx1_tiny",
        epochs=1,
        batch_size=4,
    )
    builders = _tiny_builders()
    train_speck32_arx1_a1(
        config,
        source,
        output_root,
        model_builders=builders,
    )
    metadata_path = output_root / "data" / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["audit_note"] = "changed after A1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match A2 training source"):
        train_speck32_arx1_a2(
            config,
            source,
            output_root,
            model_builders=builders,
        )


def test_interrupted_a1_resumes_with_contiguous_history_and_same_final_state(
    tmp_path: Path,
) -> None:
    config = Speck32OutputPredictionTrainingConfig(
        run_id="arx1_resume",
        epochs=2,
        batch_size=4,
    )
    builders = _tiny_builders()
    resumed_root = tmp_path / "resumed"
    resumed_source = _source(resumed_root)
    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted_source = _source(uninterrupted_root)

    class PlannedInterruption(RuntimeError):
        pass

    def interrupt(event: str, payload: dict[str, object]) -> None:
        if (
            event == "epoch_done"
            and payload["model"] == FCNN_MODEL_NAME
            and payload["epoch"] == 1
        ):
            raise PlannedInterruption

    with pytest.raises(PlannedInterruption):
        train_speck32_arx1_a1(
            config,
            resumed_source,
            resumed_root,
            progress=interrupt,
            model_builders=builders,
        )

    resumed = train_speck32_arx1_a1(
        config,
        resumed_source,
        resumed_root,
        model_builders=builders,
    )
    uninterrupted = train_speck32_arx1_a1(
        config,
        uninterrupted_source,
        uninterrupted_root,
        model_builders=builders,
    )

    for model_name in A1_MODEL_NAMES:
        resumed_history = [
            row["epoch"] for row in resumed["history"] if row["model"] == model_name
        ]
        assert resumed_history == [1, 2]
    resumed_checkpoints = {row["model"]: row for row in resumed["checkpoints"]}
    uninterrupted_checkpoints = {
        row["model"]: row for row in uninterrupted["checkpoints"]
    }
    assert {
        name: row["final_state_sha256"] for name, row in resumed_checkpoints.items()
    } == {
        name: row["final_state_sha256"]
        for name, row in uninterrupted_checkpoints.items()
    }


def test_source_manifest_hashes_all_disk_backed_source_files(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    source = _source(output_root)
    config = Speck32OutputPredictionTrainingConfig(run_id="arx1_manifest")

    manifest = build_speck32_source_manifest(config, source, output_root)

    assert manifest["manifest_sha256"]
    assert {row["name"] for row in manifest["files"]} == {
        "cache_metadata.json",
        "plaintexts.npy",
        "features.npy",
        "full_targets.npy",
    }
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])
    assert all(row["bytes"] > 0 for row in manifest["files"])


def test_formal_gate_advances_only_valid_single_key_output_signal() -> None:
    config = Speck32OutputPredictionTrainingConfig.arx1_a(device="cpu")
    a1, a2 = _formal_gate_bundles(config)

    passed = adjudicate_speck32_arx1_a(config, {"data_valid": True}, a1, a2)
    invalid_a2 = deepcopy(a2)
    invalid_a2["a1_bundle_sha256"] = "f" * 64
    invalid = adjudicate_speck32_arx1_a(
        config,
        {"data_valid": True},
        a1,
        invalid_a2,
    )

    assert passed["status"] == "pass"
    assert passed["decision"] == ("innovation2_arx1_speck32_key21_output_gate_passed")
    assert passed["metrics"]["strongest_generic_model"] == BILSTM_MODEL_NAME
    assert passed["metrics"]["joint_bit_gate_count"] == 32
    assert passed["next_action"]["next_adjudication"] == (
        "arx1_b_independent_key_confirmation"
    )
    assert passed["next_action"]["remote_scale"] is True
    assert invalid["status"] == "fail"
    assert (
        invalid["protocol_checks"]["execution"]["a1_bundle_hash_is_verified_by_a2"]
        is False
    )


def _source(output_root: Path) -> dict[str, object]:
    return prepare_speck32_output_prediction_data(
        Speck32OutputPredictionDataConfig(
            run_id="arx1_tiny",
            train_rows=12,
            test_rows=8,
            chunk_rows=5,
        ),
        output_root,
    )


def _tiny_builders() -> dict[str, type[TinyOutputModel]]:
    return {
        FCNN_MODEL_NAME: TinyOutputModel,
        BILSTM_MODEL_NAME: TinyOutputModel,
        ROTATION_CARRY_MODEL_NAME: TinyOutputModel,
        WRONG_ROTATION_CARRY_MODEL_NAME: TinyOutputModel,
        ROTATION_CARRY_SHUFFLE_MODEL_NAME: TinyOutputModel,
    }


def _history(model_names: tuple[str, ...], epochs: int) -> list[dict[str, object]]:
    return [
        {"model": name, "epoch": epoch}
        for name in model_names
        for epoch in range(1, epochs + 1)
    ]


def _formal_gate_bundles(
    config: Speck32OutputPredictionTrainingConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    source_hash = "a" * 64
    bundle_hash = "b" * 64
    rows: list[dict[str, object]] = []
    per_bit_rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for model_name in FULL_MATRIX_MODEL_NAMES:
        is_candidate = model_name == ROTATION_CARRY_MODEL_NAME
        is_shuffle = model_name == ROTATION_CARRY_SHUFFLE_MODEL_NAME
        is_candidate_family = model_name in A2_LOGICAL_MODEL_NAMES
        bap_avg = 0.80 if is_candidate else 0.50 if is_shuffle else 0.60
        macro_auc = 0.80 if is_candidate else 0.50 if is_shuffle else 0.60
        row: dict[str, object] = {
            "model": model_name,
            "parameters": 3_732_032 if is_candidate_family else 100,
            "batch_schedule_sha256": "c" * 64,
            "train_labels_shuffled": is_shuffle,
            "label_permutation_sha256": "d" * 64 if is_shuffle else None,
            "test_target_identity": "true_full_speck32_ciphertext_targets",
            "sample_classification": False,
            "bap_avg": bap_avg,
            "macro_auc": macro_auc,
            "bce": 0.5,
            "mse": 0.2,
            "exact_match": 0.0,
            "invalid_probability_rate": 0.0,
            "train_rows": 1 << 20,
            "test_rows": 1 << 15,
            "epochs": 100,
            "batch_size": 128,
        }
        if is_candidate:
            row["execution"] = "reused_arx1_a1_result_and_checkpoint"
        rows.append(row)
        checkpoints.append(
            {
                "model": model_name,
                "sha256": "e" * 64,
                "latest_sha256": "e" * 64,
                "final_state_sha256": "f" * 64,
                "initial_state_sha256": (
                    "1" * 64 if is_candidate_family else model_name
                ),
            }
        )
        for bit in range(32):
            per_bit_rows.append(
                {
                    "model": model_name,
                    "msb_index": bit,
                    "auc": macro_auc,
                    "accuracy_minus_majority": 0.10 if is_candidate else 0.0,
                }
            )
    a1 = {
        "history": _history(A1_MODEL_NAMES, config.epochs),
        "source_manifest": {"manifest_sha256": source_hash},
        "bundle_sha256": bundle_hash,
    }
    a2 = {
        "full_matrix_rows": rows,
        "full_matrix_per_bit_rows": per_bit_rows,
        "checkpoints": checkpoints,
        "new_history": _history(A2_NEW_MODEL_NAMES, config.epochs),
        "reused_models": [ROTATION_CARRY_MODEL_NAME],
        "source_manifest": {"manifest_sha256": source_hash},
        "a1_bundle_sha256": bundle_hash,
    }
    return a1, a2
