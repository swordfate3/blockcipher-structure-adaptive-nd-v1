from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_metrics import (
    evaluate_speck32_output_scores,
)
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    BILSTM_PARAMETER_COUNT,
    BILSTM_MODEL_NAME,
    FCNN_PARAMETER_COUNT,
    FCNN_MODEL_NAME,
    Speck32JeongBiLstm,
    Speck32JeongFcnn,
    jeong_anchor_protocols,
)
from blockcipher_nd.tasks.innovation2.speck32_rotation_carry_model import (
    ROTATION_CARRY_MODEL_NAME,
    ROTATION_CARRY_SHUFFLE_MODEL_NAME,
    WRONG_ROTATION_CARRY_MODEL_NAME,
    Speck32RotationCarryPredictor,
    rotation_carry_protocols,
)


A1_MODEL_NAMES = (
    FCNN_MODEL_NAME,
    BILSTM_MODEL_NAME,
    ROTATION_CARRY_MODEL_NAME,
)
A2_NEW_MODEL_NAMES = (
    WRONG_ROTATION_CARRY_MODEL_NAME,
    ROTATION_CARRY_SHUFFLE_MODEL_NAME,
)
A2_LOGICAL_MODEL_NAMES = (
    ROTATION_CARRY_MODEL_NAME,
    *A2_NEW_MODEL_NAMES,
)
FULL_MATRIX_MODEL_NAMES = (*A1_MODEL_NAMES, *A2_NEW_MODEL_NAMES)
SOURCE_FILENAMES = (
    "cache_metadata.json",
    "plaintexts.npy",
    "features.npy",
    "full_targets.npy",
)
ProgressCallback = Callable[[str, dict[str, Any]], None]
ModelBuilder = Callable[[], nn.Module]


@dataclass(frozen=True)
class Speck32OutputPredictionTrainingConfig:
    run_id: str = "i2_output_prediction_arx1_speck32_r3_readiness_seed21"
    mode: str = "readiness"
    seed: int = 21
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    candidate_channels: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"readiness", "arx1_a"}:
            raise ValueError("mode must be readiness or arx1_a")
        if (
            min(
                self.seed,
                self.epochs,
                self.batch_size,
                self.candidate_channels,
            )
            <= 0
        ):
            raise ValueError("seed, epochs, batch size, and channels must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning rate must be positive and weight decay non-negative"
            )
        if self.mode == "arx1_a" and (
            self.seed != 21
            or self.epochs != 100
            or self.batch_size != 128
            or self.learning_rate != 1e-3
            or self.weight_decay != 0.01
            or self.candidate_channels != 400
        ):
            raise ValueError("formal ARX1-A training protocol is frozen")

    @classmethod
    def arx1_a(
        cls,
        *,
        run_id: str = "i2_output_prediction_arx1a_speck32_r3_key21",
        device: str = "cuda",
    ) -> Speck32OutputPredictionTrainingConfig:
        return cls(
            run_id=run_id,
            mode="arx1_a",
            epochs=100,
            batch_size=128,
            candidate_channels=400,
            device=device,
        )


def train_speck32_arx1_a1(
    config: Speck32OutputPredictionTrainingConfig,
    source: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
    model_builders: Mapping[str, ModelBuilder] | None = None,
) -> dict[str, Any]:
    arrays = _validated_source_arrays(config, source)
    source_manifest = build_speck32_source_manifest(config, source, output_root)
    builders = _model_builders(config, model_builders)
    results = [
        _train_one_model(
            config,
            model_name=model_name,
            model_builder=builders[model_name],
            shuffle_labels=False,
            train_features=arrays["train_features"],
            train_targets=arrays["train_targets"],
            test_features=arrays["test_features"],
            test_targets=arrays["test_targets"],
            source_metadata=source["metadata"],
            source_manifest_sha256=source_manifest["manifest_sha256"],
            output_root=output_root,
            progress=progress,
        )
        for model_name in A1_MODEL_NAMES
    ]
    bundle = {
        "bundle_version": 1,
        "stage": "arx1_a1",
        "run_id": config.run_id,
        "training_config": serializable_training_config(config),
        "source_manifest": source_manifest,
        "rows": [result["row"] for result in results],
        "per_bit_rows": [row for result in results for row in result["per_bit_rows"]],
        "history": [row for result in results for row in result["history"]],
        "checkpoints": [result["checkpoint"] for result in results],
    }
    bundle_record = _write_hashed_json(output_root / "a1_bundle.json", bundle)
    bundle["bundle_path"] = bundle_record["path"]
    bundle["bundle_sha256"] = bundle_record["sha256"]
    if progress is not None:
        progress(
            "arx1_a1_done",
            {
                "models": list(A1_MODEL_NAMES),
                "bundle_sha256": bundle_record["sha256"],
            },
        )
    return bundle


def train_speck32_arx1_a2(
    config: Speck32OutputPredictionTrainingConfig,
    source: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
    model_builders: Mapping[str, ModelBuilder] | None = None,
) -> dict[str, Any]:
    arrays = _validated_source_arrays(config, source)
    source_manifest = build_speck32_source_manifest(config, source, output_root)
    a1_bundle = _load_and_validate_a1_bundle(config, source_manifest, output_root)
    builders = _model_builders(config, model_builders)
    new_results = [
        _train_one_model(
            config,
            model_name=model_name,
            model_builder=builders[model_name],
            shuffle_labels=model_name == ROTATION_CARRY_SHUFFLE_MODEL_NAME,
            train_features=arrays["train_features"],
            train_targets=arrays["train_targets"],
            test_features=arrays["test_features"],
            test_targets=arrays["test_targets"],
            source_metadata=source["metadata"],
            source_manifest_sha256=source_manifest["manifest_sha256"],
            output_root=output_root,
            progress=progress,
        )
        for model_name in A2_NEW_MODEL_NAMES
    ]
    a1_rows = _index_by_model(a1_bundle["rows"], expected=A1_MODEL_NAMES)
    a1_per_bit = _index_per_bit(a1_bundle["per_bit_rows"], expected=A1_MODEL_NAMES)
    a1_checkpoints = _index_by_model(a1_bundle["checkpoints"], expected=A1_MODEL_NAMES)
    new_rows = {result["row"]["model"]: result["row"] for result in new_results}
    new_checkpoints = {
        result["checkpoint"]["model"]: result["checkpoint"] for result in new_results
    }
    candidate_initial = a1_checkpoints[ROTATION_CARRY_MODEL_NAME][
        "initial_state_sha256"
    ]
    control_initials = [
        new_checkpoints[name]["initial_state_sha256"] for name in A2_NEW_MODEL_NAMES
    ]
    if any(value != candidate_initial for value in control_initials):
        raise ValueError(
            "ARX1-A2 matched candidate controls do not share initialization"
        )
    candidate_parameters = int(a1_rows[ROTATION_CARRY_MODEL_NAME]["parameters"])
    if any(
        int(new_rows[name]["parameters"]) != candidate_parameters
        for name in A2_NEW_MODEL_NAMES
    ):
        raise ValueError("ARX1-A2 candidate controls do not have equal parameters")
    candidate_schedule = a1_rows[ROTATION_CARRY_MODEL_NAME]["batch_schedule_sha256"]
    if any(
        new_rows[name]["batch_schedule_sha256"] != candidate_schedule
        for name in A2_NEW_MODEL_NAMES
    ):
        raise ValueError("ARX1-A2 candidate controls do not share batch order")
    reused_row = {
        **a1_rows[ROTATION_CARRY_MODEL_NAME],
        "execution": "reused_arx1_a1_result_and_checkpoint",
        "a1_bundle_sha256": a1_bundle["bundle_sha256"],
    }
    reused_per_bit = [
        {
            **a1_per_bit[(ROTATION_CARRY_MODEL_NAME, bit)],
            "execution": "reused_arx1_a1_result_and_checkpoint",
            "a1_bundle_sha256": a1_bundle["bundle_sha256"],
        }
        for bit in range(32)
    ]
    logical_rows = [reused_row, *[new_rows[name] for name in A2_NEW_MODEL_NAMES]]
    logical_per_bit_rows = [
        *reused_per_bit,
        *[row for result in new_results for row in result["per_bit_rows"]],
    ]
    full_matrix_rows = [
        a1_rows[FCNN_MODEL_NAME],
        a1_rows[BILSTM_MODEL_NAME],
        reused_row,
        *[new_rows[name] for name in A2_NEW_MODEL_NAMES],
    ]
    full_matrix_per_bit_rows = [
        *[a1_per_bit[(name, bit)] for name in A1_MODEL_NAMES for bit in range(32)],
        *[row for result in new_results for row in result["per_bit_rows"]],
    ]
    all_checkpoints = [
        *[a1_checkpoints[name] for name in A1_MODEL_NAMES],
        *[new_checkpoints[name] for name in A2_NEW_MODEL_NAMES],
    ]
    bundle = {
        "bundle_version": 1,
        "stage": "arx1_a2",
        "run_id": config.run_id,
        "training_config": serializable_training_config(config),
        "source_manifest": source_manifest,
        "a1_bundle_path": a1_bundle["bundle_path"],
        "a1_bundle_sha256": a1_bundle["bundle_sha256"],
        "reused_models": [ROTATION_CARRY_MODEL_NAME],
        "new_models": list(A2_NEW_MODEL_NAMES),
        "logical_rows": logical_rows,
        "logical_per_bit_rows": logical_per_bit_rows,
        "full_matrix_rows": full_matrix_rows,
        "full_matrix_per_bit_rows": full_matrix_per_bit_rows,
        "new_history": [row for result in new_results for row in result["history"]],
        "checkpoints": all_checkpoints,
        "fairness_checks": {
            "a1_candidate_checkpoint_reused": True,
            "matched_candidate_initial_state_sha256": candidate_initial,
            "matched_candidate_initialization_equal": True,
            "matched_candidate_parameter_counts_equal": True,
            "matched_candidate_batch_schedule_equal": True,
            "shuffle_changes_training_target_order_only": True,
            "all_test_targets_are_true_speck32_ciphertext": True,
        },
    }
    bundle_record = _write_hashed_json(output_root / "a2_bundle.json", bundle)
    bundle["bundle_path"] = bundle_record["path"]
    bundle["bundle_sha256"] = bundle_record["sha256"]
    if progress is not None:
        progress(
            "arx1_a2_done",
            {
                "reused_models": [ROTATION_CARRY_MODEL_NAME],
                "new_models": list(A2_NEW_MODEL_NAMES),
                "bundle_sha256": bundle_record["sha256"],
            },
        )
    return bundle


def build_speck32_source_manifest(
    config: Speck32OutputPredictionTrainingConfig,
    source: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    data_root_value = source.get("data_root")
    if data_root_value is None:
        raise ValueError("SPECK32 training source must provide data_root")
    data_root = Path(data_root_value)
    if not data_root.is_dir():
        raise ValueError("SPECK32 training data_root does not exist")
    files: list[dict[str, Any]] = []
    for filename in SOURCE_FILENAMES:
        path = data_root / filename
        if not path.is_file():
            raise ValueError(f"missing SPECK32 source file: {filename}")
        files.append(
            {
                "name": filename,
                "path": _relative_path(path, output_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    metadata = source.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("status") != "complete":
        raise ValueError("SPECK32 source metadata must be complete")
    payload = {
        "manifest_version": 1,
        "task": "innovation2_output_prediction",
        "cipher": "SPECK32/64",
        "rounds": 3,
        "run_id": config.run_id,
        "seed": config.seed,
        "key_seed": int(metadata.get("key_seed", -1)),
        "train_rows": int(metadata.get("train_rows", -1)),
        "test_rows": int(metadata.get("test_rows", -1)),
        "target": "32_msb_first_true_speck32_ciphertext_bits",
        "sample_classification": False,
        "files": files,
    }
    manifest_sha256 = _json_sha256(payload)
    manifest = {**payload, "manifest_sha256": manifest_sha256}
    _write_json(output_root / "source_manifest.json", manifest)
    return manifest


def serializable_training_config(
    config: Speck32OutputPredictionTrainingConfig,
) -> dict[str, Any]:
    return asdict(config)


def adjudicate_speck32_arx1_a(
    config: Speck32OutputPredictionTrainingConfig,
    data_checks: dict[str, bool],
    a1: dict[str, Any],
    a2: dict[str, Any],
) -> dict[str, Any]:
    rows = _index_by_model(
        a2.get("full_matrix_rows", []), expected=FULL_MATRIX_MODEL_NAMES
    )
    per_bit = _index_per_bit(
        a2.get("full_matrix_per_bit_rows", []),
        expected=FULL_MATRIX_MODEL_NAMES,
    )
    checkpoints = _index_by_model(
        a2.get("checkpoints", []),
        expected=FULL_MATRIX_MODEL_NAMES,
    )
    history = [*a1.get("history", []), *a2.get("new_history", [])]
    expected_history = {
        (name, epoch)
        for name in FULL_MATRIX_MODEL_NAMES
        for epoch in range(1, config.epochs + 1)
    }
    actual_history = {
        (str(row.get("model")), int(row.get("epoch", -1))) for row in history
    }
    candidate_models = A2_LOGICAL_MODEL_NAMES
    execution_checks = {
        "five_matrix_rows_complete": set(rows) == set(FULL_MATRIX_MODEL_NAMES),
        "five_by_32_per_bit_rows_complete": len(per_bit) == 5 * 32,
        "history_is_complete_and_contiguous": actual_history == expected_history,
        "five_final_checkpoint_hashes_present": all(
            len(str(checkpoints[name].get("sha256", ""))) == 64
            and len(str(checkpoints[name].get("latest_sha256", ""))) == 64
            and len(str(checkpoints[name].get("final_state_sha256", ""))) == 64
            for name in FULL_MATRIX_MODEL_NAMES
        ),
        "a2_reuses_a1_candidate": a2.get("reused_models") == [ROTATION_CARRY_MODEL_NAME]
        and rows[ROTATION_CARRY_MODEL_NAME].get("execution")
        == "reused_arx1_a1_result_and_checkpoint",
        "candidate_controls_share_initialization": len(
            {checkpoints[name].get("initial_state_sha256") for name in candidate_models}
        )
        == 1,
        "candidate_controls_have_equal_parameters": len(
            {int(rows[name].get("parameters", -1)) for name in candidate_models}
        )
        == 1,
        "all_models_share_batch_schedule": len(
            {
                rows[name].get("batch_schedule_sha256")
                for name in FULL_MATRIX_MODEL_NAMES
            }
        )
        == 1,
        "only_shuffle_control_changes_training_target_order": all(
            bool(rows[name].get("train_labels_shuffled"))
            == (name == ROTATION_CARRY_SHUFFLE_MODEL_NAME)
            for name in FULL_MATRIX_MODEL_NAMES
        )
        and bool(
            rows[ROTATION_CARRY_SHUFFLE_MODEL_NAME].get("label_permutation_sha256")
        ),
        "all_test_targets_are_true_ciphertext_outputs": all(
            rows[name].get("test_target_identity")
            == "true_full_speck32_ciphertext_targets"
            and rows[name].get("sample_classification") is False
            for name in FULL_MATRIX_MODEL_NAMES
        ),
        "all_metrics_are_finite": all(
            math.isfinite(float(rows[name].get(field, math.nan)))
            for name in FULL_MATRIX_MODEL_NAMES
            for field in (
                "bap_avg",
                "macro_auc",
                "bce",
                "mse",
                "exact_match",
                "invalid_probability_rate",
            )
        ),
        "a1_and_a2_source_hashes_match": a1.get("source_manifest", {}).get(
            "manifest_sha256"
        )
        == a2.get("source_manifest", {}).get("manifest_sha256"),
        "a1_bundle_hash_is_verified_by_a2": a2.get("a1_bundle_sha256")
        == a1.get("bundle_sha256"),
    }
    formal_candidate_protocol = rotation_carry_protocols(400)[ROTATION_CARRY_MODEL_NAME]
    formal_checks = {
        "formal_candidate_parameter_gate": bool(
            formal_candidate_protocol["within_bilstm_five_percent"]
        ),
        "formal_budget_matches": True,
    }
    if config.mode == "arx1_a":
        protocol = rotation_carry_protocols(config.candidate_channels)[
            ROTATION_CARRY_MODEL_NAME
        ]
        formal_checks = {
            "formal_candidate_parameter_gate": bool(
                protocol["within_bilstm_five_percent"]
            )
            and int(rows[ROTATION_CARRY_MODEL_NAME]["parameters"])
            == int(protocol["parameters"]),
            "formal_budget_matches": all(
                int(rows[name]["train_rows"]) == 1 << 20
                and int(rows[name]["test_rows"]) == 1 << 15
                and int(rows[name]["epochs"]) == 100
                and int(rows[name]["batch_size"]) == 128
                for name in FULL_MATRIX_MODEL_NAMES
            ),
        }
    protocol_valid = (
        bool(data_checks)
        and all(data_checks.values())
        and all(execution_checks.values())
        and all(formal_checks.values())
    )
    candidate = rows[ROTATION_CARRY_MODEL_NAME]
    shuffle = rows[ROTATION_CARRY_SHUFFLE_MODEL_NAME]
    wrong = rows[WRONG_ROTATION_CARRY_MODEL_NAME]
    bit_passes: list[int] = []
    for bit in range(32):
        candidate_bit = per_bit[(ROTATION_CARRY_MODEL_NAME, bit)]
        shuffle_bit = per_bit[(ROTATION_CARRY_SHUFFLE_MODEL_NAME, bit)]
        if (
            float(candidate_bit["auc"]) >= 0.55
            and float(candidate_bit["accuracy_minus_majority"]) >= 0.005
            and float(candidate_bit["auc"]) - float(shuffle_bit["auc"]) >= 0.015
        ):
            bit_passes.append(bit)
    output_gate = {
        "candidate_bap_avg_at_least_0_55": float(candidate["bap_avg"]) >= 0.55,
        "candidate_macro_auc_at_least_0_55": float(candidate["macro_auc"]) >= 0.55,
        "candidate_minus_shuffle_bap_avg_at_least_0_03": (
            float(candidate["bap_avg"]) - float(shuffle["bap_avg"])
        )
        >= 0.03,
        "candidate_minus_shuffle_macro_auc_at_least_0_03": (
            float(candidate["macro_auc"]) - float(shuffle["macro_auc"])
        )
        >= 0.03,
        "at_least_16_bits_pass_joint_gate": len(bit_passes) >= 16,
    }
    generic_names = (FCNN_MODEL_NAME, BILSTM_MODEL_NAME)
    strongest_generic = max(
        generic_names,
        key=lambda name: (
            float(rows[name]["bap_avg"]),
            float(rows[name]["macro_auc"]),
            name == BILSTM_MODEL_NAME,
        ),
    )
    metrics = {
        "strongest_generic_model": strongest_generic,
        "strongest_generic_bap_avg": float(rows[strongest_generic]["bap_avg"]),
        "strongest_generic_macro_auc": float(rows[strongest_generic]["macro_auc"]),
        "candidate_bap_avg": float(candidate["bap_avg"]),
        "candidate_macro_auc": float(candidate["macro_auc"]),
        "shuffle_bap_avg": float(shuffle["bap_avg"]),
        "shuffle_macro_auc": float(shuffle["macro_auc"]),
        "wrong_rotation_bap_avg": float(wrong["bap_avg"]),
        "wrong_rotation_macro_auc": float(wrong["macro_auc"]),
        "candidate_minus_generic_bap_avg": float(candidate["bap_avg"])
        - float(rows[strongest_generic]["bap_avg"]),
        "candidate_minus_shuffle_bap_avg": float(candidate["bap_avg"])
        - float(shuffle["bap_avg"]),
        "candidate_minus_shuffle_macro_auc": float(candidate["macro_auc"])
        - float(shuffle["macro_auc"]),
        "candidate_minus_wrong_bap_avg": float(candidate["bap_avg"])
        - float(wrong["bap_avg"]),
        "joint_bit_gate_count": len(bit_passes),
        "joint_bit_gate_msb_indices": bit_passes,
    }
    if not protocol_valid:
        status = "fail"
        decision = "innovation2_arx1_speck32_training_protocol_invalid"
        next_adjudication = "repair_arx1_protocol_only"
        action = (
            "repair only data, source, checkpoint, resume, metric, or fairness evidence"
        )
    elif config.mode == "readiness":
        status = "pass"
        decision = "innovation2_arx1_speck32_readiness_passed"
        next_adjudication = "wait_for_present_and_gift_condition_closure"
        action = (
            "keep ARX1 performance execution conditionally closed until PRESENT and "
            "GIFT predecessor branches are adjudicated"
        )
    elif all(output_gate.values()):
        status = "pass"
        decision = "innovation2_arx1_speck32_key21_output_gate_passed"
        next_adjudication = "arx1_b_independent_key_confirmation"
        action = (
            "freeze the strongest generic model from A1 and rerun the four-row "
            "candidate/control matrix with key_seed 22"
        )
    elif all(
        float(rows[name]["bap_avg"]) < 0.55 and float(rows[name]["macro_auc"]) < 0.55
        for name in generic_names
    ):
        status = "hold"
        decision = "innovation2_arx1_speck32_generic_anchor_not_calibrated"
        next_adjudication = "audit_jeong_protocol_before_performance_claim"
        action = (
            "audit the Jeong paper-family implementation and protocol gap; do not "
            "interpret this budget as a SPECK output-prediction ceiling"
        )
    else:
        status = "hold"
        decision = "innovation2_arx1_speck32_rotation_carry_gate_not_passed"
        next_adjudication = "close_current_rotation_carry_candidate"
        action = (
            "retain any calibrated generic output signal but stop this rotation/carry "
            "candidate; do not add models, epochs, or selected bits post hoc"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": {
            "data": data_checks,
            "execution": execution_checks,
            "formal": formal_checks,
        },
        "output_gate": output_gate,
        "metrics": metrics,
        "claim_scope": (
            "single-fixed-key SPECK32/64 r3 full true ciphertext output prediction; "
            "not independent-key confirmation, paper-exact reproduction, selected-bit "
            "discovery, higher-round boundary, ARX-wide conclusion, or SOTA"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "remote_scale": status == "pass" and config.mode == "arx1_a",
            "target": "full_32_bit_true_speck32_ciphertext_output",
            "sample_classification": False,
        },
    }


def _train_one_model(
    config: Speck32OutputPredictionTrainingConfig,
    *,
    model_name: str,
    model_builder: ModelBuilder,
    shuffle_labels: bool,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    source_metadata: dict[str, Any],
    source_manifest_sha256: str,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_240_000 + config.seed)
    model = model_builder().to(config.device)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    _validate_formal_model(config, model_name, parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCELoss()
    config_hash = _training_config_hash(config, model_name)
    batch_schedule_sha256 = _batch_schedule_sha256(config, len(train_features))
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    latest_path = model_root / f"{model_name}_latest.pt"
    final_path = model_root / f"{model_name}_final.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location=config.device,
            weights_only=False,
        )
        _validate_resume_checkpoint(
            checkpoint,
            model_name=model_name,
            config_hash=config_hash,
            source_manifest_sha256=source_manifest_sha256,
            initial_state_sha256=initial_state_sha256,
            batch_schedule_sha256=batch_schedule_sha256,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
    labels = train_targets
    label_permutation_sha256: str | None = None
    if shuffle_labels:
        permutation = np.random.default_rng(1_250_000 + config.seed).permutation(
            len(train_targets)
        )
        if np.array_equal(permutation, np.arange(len(permutation))):
            permutation = np.roll(permutation, 1)
        label_permutation_sha256 = hashlib.sha256(
            np.ascontiguousarray(permutation).tobytes()
        ).hexdigest()
        labels = np.asarray(train_targets[permutation], dtype=np.float32)
    feature_tensor = torch.from_numpy(train_features)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs + 1):
        generator = torch.Generator().manual_seed(_epoch_batch_seed(config, epoch))
        loader = DataLoader(
            TensorDataset(feature_tensor, label_tensor),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )
        model.train()
        total_loss = 0.0
        total_cells = 0
        for features, targets in loader:
            features = features.to(config.device)
            targets = targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(features)
            loss = criterion(probabilities, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * targets.numel()
            total_cells += targets.numel()
        history_row = {
            "run_id": config.run_id,
            "stage": "arx1_a2" if model_name in A2_NEW_MODEL_NAMES else "arx1_a1",
            "model": model_name,
            "epoch": epoch,
            "train_bce": total_loss / max(1, total_cells),
            "batch_order_seed": _epoch_batch_seed(config, epoch),
            "batch_schedule_sha256": batch_schedule_sha256,
        }
        history.append(history_row)
        torch.save(
            {
                "checkpoint_version": 1,
                "model": model_name,
                "config_hash": config_hash,
                "source_manifest_sha256": source_manifest_sha256,
                "initial_state_sha256": initial_state_sha256,
                "batch_schedule_sha256": batch_schedule_sha256,
                "label_permutation_sha256": label_permutation_sha256,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            latest_path,
        )
        if progress is not None:
            progress("epoch_done", history_row)
    training_seconds = time.monotonic() - started
    probabilities = _predict_probabilities(
        model,
        test_features,
        batch_size=config.batch_size,
        device=config.device,
    )
    evaluated = evaluate_speck32_output_scores(
        model_name,
        "test",
        probabilities,
        test_targets,
    )
    final_state_sha256 = _state_dict_sha256(model.state_dict())
    if final_path.exists():
        final_checkpoint = torch.load(
            final_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            final_checkpoint.get("config_hash") != config_hash
            or final_checkpoint.get("source_manifest_sha256") != source_manifest_sha256
            or final_checkpoint.get("final_state_sha256") != final_state_sha256
        ):
            raise ValueError(f"existing final checkpoint mismatch for {model_name}")
    else:
        torch.save(
            {
                "checkpoint_version": 1,
                "model": model_name,
                "config_hash": config_hash,
                "source_manifest_sha256": source_manifest_sha256,
                "initial_state_sha256": initial_state_sha256,
                "final_state_sha256": final_state_sha256,
                "batch_schedule_sha256": batch_schedule_sha256,
                "label_permutation_sha256": label_permutation_sha256,
                "epoch": config.epochs,
                "model_state": model.state_dict(),
            },
            final_path,
        )
    checkpoint_row = {
        "model": model_name,
        "latest_path": _relative_path(latest_path, output_root),
        "latest_sha256": _sha256(latest_path),
        "path": _relative_path(final_path, output_root),
        "sha256": _sha256(final_path),
        "config_hash": config_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "batch_schedule_sha256": batch_schedule_sha256,
        "label_permutation_sha256": label_permutation_sha256,
        "epoch": config.epochs,
    }
    protocol = _model_protocol(model_name, config.candidate_channels)
    summary = evaluated["summary"]
    row = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "arx1_speck32_r3_full32_output",
        "stage": "arx1_a2" if model_name in A2_NEW_MODEL_NAMES else "arx1_a1",
        "cipher": "SPECK32/64",
        "rounds": 3,
        "model": model_name,
        "architecture": protocol["architecture"],
        "target": "full_32_bit_true_speck32_ciphertext_output",
        "sample_classification": False,
        "secret_key_scope": "single_fixed_unknown_key",
        "seed": config.seed,
        "key_seed": int(source_metadata["key_seed"]),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "parameters": parameters,
        "loss": "binary_cross_entropy",
        "optimizer": "adamw",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "paper_reports_weight_decay": False,
        "weight_decay_source": "pytorch_adamw_default_explicitly_frozen",
        "checkpoint_selection": "final_epoch",
        "final_epoch": config.epochs,
        "train_labels_shuffled": shuffle_labels,
        "label_permutation_sha256": label_permutation_sha256,
        "test_target_identity": "true_full_speck32_ciphertext_targets",
        "source_manifest_sha256": source_manifest_sha256,
        "config_hash": config_hash,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "batch_schedule_sha256": batch_schedule_sha256,
        "training_seconds_this_invocation": training_seconds,
        **summary,
    }
    per_bit_rows = [
        {
            "run_id": config.run_id,
            "stage": row["stage"],
            "source_manifest_sha256": source_manifest_sha256,
            **metric_row,
        }
        for metric_row in evaluated["per_bit_rows"]
    ]
    return {
        "row": row,
        "per_bit_rows": per_bit_rows,
        "history": history,
        "checkpoint": checkpoint_row,
    }


def _load_and_validate_a1_bundle(
    config: Speck32OutputPredictionTrainingConfig,
    source_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    bundle_path = output_root / "a1_bundle.json"
    digest_path = output_root / "a1_bundle.sha256"
    if not bundle_path.is_file() or not digest_path.is_file():
        raise ValueError("ARX1-A2 requires a complete A1 bundle and SHA256 file")
    actual_digest = _sha256(bundle_path)
    recorded_digest = digest_path.read_text(encoding="ascii").split()[0]
    if actual_digest != recorded_digest:
        raise ValueError("ARX1-A1 bundle SHA256 mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle.get("stage") != "arx1_a1"
        or bundle.get("run_id") != config.run_id
        or bundle.get("training_config") != serializable_training_config(config)
        or bundle.get("source_manifest", {}).get("manifest_sha256")
        != source_manifest["manifest_sha256"]
    ):
        raise ValueError("ARX1-A1 bundle does not match A2 training source or config")
    rows = _index_by_model(bundle.get("rows", []), expected=A1_MODEL_NAMES)
    checkpoints = _index_by_model(
        bundle.get("checkpoints", []), expected=A1_MODEL_NAMES
    )
    _index_per_bit(bundle.get("per_bit_rows", []), expected=A1_MODEL_NAMES)
    expected_history = {
        (name, epoch)
        for name in A1_MODEL_NAMES
        for epoch in range(1, config.epochs + 1)
    }
    actual_history = {
        (str(row.get("model")), int(row.get("epoch", -1)))
        for row in bundle.get("history", [])
    }
    if actual_history != expected_history:
        raise ValueError("ARX1-A1 history is incomplete or non-contiguous")
    for model_name in A1_MODEL_NAMES:
        checkpoint = checkpoints[model_name]
        for path_field, hash_field, checkpoint_kind in (
            ("latest_path", "latest_sha256", "latest"),
            ("path", "sha256", "final"),
        ):
            path = output_root / str(checkpoint[path_field])
            if not path.is_file() or _sha256(path) != checkpoint.get(hash_field):
                raise ValueError(
                    f"ARX1-A1 {checkpoint_kind} checkpoint hash mismatch for "
                    f"{model_name}"
                )
        if (
            rows[model_name].get("source_manifest_sha256")
            != source_manifest["manifest_sha256"]
        ):
            raise ValueError(f"ARX1-A1 source hash mismatch for {model_name}")
    return {
        **bundle,
        "bundle_path": _relative_path(bundle_path, output_root),
        "bundle_sha256": actual_digest,
    }


def _validated_source_arrays(
    config: Speck32OutputPredictionTrainingConfig,
    source: dict[str, Any],
) -> dict[str, np.ndarray]:
    metadata = source.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SPECK32 source metadata is required")
    expected = {
        "cipher": "SPECK32/64",
        "rounds": 3,
        "seed": config.seed,
        "task": "innovation2_output_prediction",
        "target": "32_msb_first_true_speck32_ciphertext_bits",
        "sample_classification": False,
        "status": "complete",
    }
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError("SPECK32 source metadata does not match training protocol")
    arrays = {
        name: np.asarray(source[name], dtype=np.float32)
        for name in (
            "train_features",
            "train_targets",
            "test_features",
            "test_targets",
        )
    }
    for split in ("train", "test"):
        features = arrays[f"{split}_features"]
        targets = arrays[f"{split}_targets"]
        if (
            features.ndim != 2
            or features.shape[1] != 32
            or targets.shape != features.shape
            or len(features) == 0
        ):
            raise ValueError(f"SPECK32 {split} arrays must be non-empty [rows, 32]")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"SPECK32 {split} arrays must be finite")
        if not np.all((features == 0.0) | (features == 1.0)):
            raise ValueError(f"SPECK32 {split} features must be binary plaintext bits")
        if not np.all((targets == 0.0) | (targets == 1.0)):
            raise ValueError(f"SPECK32 {split} targets must be binary ciphertext bits")
    if int(metadata.get("train_rows", -1)) != len(arrays["train_features"]):
        raise ValueError("SPECK32 train row metadata does not match arrays")
    if int(metadata.get("test_rows", -1)) != len(arrays["test_features"]):
        raise ValueError("SPECK32 test row metadata does not match arrays")
    if config.mode == "arx1_a" and (
        int(metadata.get("key_seed", -1)) != 21
        or len(arrays["train_features"]) != 1 << 20
        or len(arrays["test_features"]) != 1 << 15
    ):
        raise ValueError("formal ARX1-A source protocol is frozen")
    return arrays


def _model_builders(
    config: Speck32OutputPredictionTrainingConfig,
    overrides: Mapping[str, ModelBuilder] | None,
) -> dict[str, ModelBuilder]:
    builders: dict[str, ModelBuilder] = {
        FCNN_MODEL_NAME: Speck32JeongFcnn,
        BILSTM_MODEL_NAME: Speck32JeongBiLstm,
        ROTATION_CARRY_MODEL_NAME: lambda: Speck32RotationCarryPredictor.correct(
            channels=config.candidate_channels
        ),
        WRONG_ROTATION_CARRY_MODEL_NAME: (
            lambda: Speck32RotationCarryPredictor.wrong_rotation(
                channels=config.candidate_channels
            )
        ),
        ROTATION_CARRY_SHUFFLE_MODEL_NAME: (
            lambda: Speck32RotationCarryPredictor.correct(
                channels=config.candidate_channels
            )
        ),
    }
    if overrides is not None:
        unknown = set(overrides) - set(builders)
        if unknown:
            raise ValueError(
                f"unknown SPECK32 model builder overrides: {sorted(unknown)}"
            )
        builders.update(overrides)
    return builders


def _model_protocol(model_name: str, candidate_channels: int) -> dict[str, Any]:
    protocols = {
        **jeong_anchor_protocols(),
        **rotation_carry_protocols(candidate_channels),
    }
    try:
        return protocols[model_name]
    except KeyError as error:
        raise ValueError(f"unknown SPECK32 output model: {model_name}") from error


def _validate_formal_model(
    config: Speck32OutputPredictionTrainingConfig,
    model_name: str,
    parameters: int,
) -> None:
    if config.mode != "arx1_a":
        return
    protocol = _model_protocol(model_name, config.candidate_channels)
    expected_parameters = {
        FCNN_MODEL_NAME: FCNN_PARAMETER_COUNT,
        BILSTM_MODEL_NAME: BILSTM_PARAMETER_COUNT,
    }.get(model_name, protocol.get("parameters"))
    if expected_parameters is not None and int(expected_parameters) != parameters:
        raise ValueError(f"formal parameter count mismatch for {model_name}")
    if model_name in A2_LOGICAL_MODEL_NAMES and not protocol.get(
        "within_bilstm_five_percent"
    ):
        raise ValueError("formal rotation/carry model violates BiLSTM parameter gate")


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model_name: str,
    config_hash: str,
    source_manifest_sha256: str,
    initial_state_sha256: str,
    batch_schedule_sha256: str,
) -> None:
    if (
        checkpoint.get("model") != model_name
        or checkpoint.get("config_hash") != config_hash
        or checkpoint.get("source_manifest_sha256") != source_manifest_sha256
        or checkpoint.get("initial_state_sha256") != initial_state_sha256
        or checkpoint.get("batch_schedule_sha256") != batch_schedule_sha256
    ):
        raise ValueError(f"checkpoint contract mismatch for {model_name}")
    epoch = int(checkpoint.get("epoch", -1))
    history = checkpoint.get("history")
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(
        range(1, epoch + 1)
    ):
        raise ValueError(f"checkpoint history is non-contiguous for {model_name}")


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            probabilities.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32)


def _training_config_hash(
    config: Speck32OutputPredictionTrainingConfig,
    model_name: str,
) -> str:
    return _json_sha256(
        {
            **serializable_training_config(config),
            "model_name": model_name,
            "checkpoint_selection": "final_epoch",
            "loss": "binary_cross_entropy",
            "optimizer": "adamw",
        }
    )


def _batch_schedule_sha256(
    config: Speck32OutputPredictionTrainingConfig,
    train_rows: int,
) -> str:
    return _json_sha256(
        {
            "sampler": "torch_random_sampler",
            "train_rows": train_rows,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "epoch_seed_formula": "1260000 + seed + epoch",
            "seed": config.seed,
        }
    )


def _epoch_batch_seed(
    config: Speck32OutputPredictionTrainingConfig,
    epoch: int,
) -> int:
    return 1_260_000 + config.seed + epoch


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _index_by_model(
    rows: list[dict[str, Any]],
    *,
    expected: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("model")): row for row in rows}
    if set(indexed) != set(expected) or len(rows) != len(expected):
        raise ValueError(f"expected model rows exactly: {list(expected)}")
    return indexed


def _index_per_bit(
    rows: list[dict[str, Any]],
    *,
    expected: tuple[str, ...],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {
        (str(row.get("model")), int(row.get("msb_index", -1))): row for row in rows
    }
    expected_keys = {(name, bit) for name in expected for bit in range(32)}
    if set(indexed) != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("SPECK32 per-bit rows are incomplete or duplicated")
    return indexed


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    _write_json(path, payload)
    digest = _sha256(path)
    digest_path = path.with_suffix(".sha256")
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return {"path": path.name, "sha256": digest}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "A1_MODEL_NAMES",
    "A2_LOGICAL_MODEL_NAMES",
    "A2_NEW_MODEL_NAMES",
    "FULL_MATRIX_MODEL_NAMES",
    "Speck32OutputPredictionTrainingConfig",
    "adjudicate_speck32_arx1_a",
    "build_speck32_source_manifest",
    "serializable_training_config",
    "train_speck32_arx1_a1",
    "train_speck32_arx1_a2",
]
