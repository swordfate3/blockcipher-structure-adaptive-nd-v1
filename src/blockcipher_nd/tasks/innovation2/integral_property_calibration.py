from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    INPUT_BITS,
    INTEGRAL_SET_SIZE,
    IntegralParitySplit,
    LinearParityPredictor,
    ProgressCallback,
    build_integral_split_from_structures,
    make_keys,
    make_structure_splits,
    summarize_splits,
)
from blockcipher_nd.training import (
    TrainingConfig,
    evaluate_binary_classifier,
    predict_binary_probabilities,
    train_binary_classifier,
)


@dataclass(frozen=True)
class IntegralCalibrationConfig:
    run_id: str
    train_structures: int
    validation_structures: int
    calibration_structures: int
    test_structures: int
    train_keys: int
    validation_keys: int
    calibration_keys: int
    test_keys: int
    stability_test_keys: int
    epochs: int
    seed: int = 0
    rounds: int = 5
    batch_size: int = 256
    hidden_bits: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    gate_mode: str = "calibration"
    structure_split_mode: str = "random-disjoint"

    def __post_init__(self) -> None:
        for name in (
            "train_structures",
            "validation_structures",
            "calibration_structures",
            "test_structures",
            "train_keys",
            "validation_keys",
            "calibration_keys",
            "test_keys",
            "stability_test_keys",
            "epochs",
            "batch_size",
            "hidden_bits",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rounds != 5:
            raise ValueError("the frozen calibration protocol requires PRESENT r5")
        if self.gate_mode not in {"calibration-smoke", "calibration"}:
            raise ValueError(
                "gate_mode must be calibration-smoke or calibration"
            )
        if self.structure_split_mode not in {
            "random-disjoint",
            "geometry-disjoint",
        }:
            raise ValueError(
                "structure_split_mode must be random-disjoint or geometry-disjoint"
            )


@dataclass(frozen=True)
class MonotoneAffineLogitCalibration:
    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope <= 0.0:
            raise ValueError("calibration slope must be finite and positive")
        if not np.isfinite(self.intercept):
            raise ValueError("calibration intercept must be finite")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = probabilities_to_logits(probabilities)
        calibrated_logits = np.clip(
            self.slope * logits + self.intercept,
            -60.0,
            60.0,
        )
        return (1.0 / (1.0 + np.exp(-calibrated_logits))).astype(
            np.float32,
            copy=False,
        )


def fit_monotone_affine_logit_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> MonotoneAffineLogitCalibration:
    if len(probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("calibration probabilities and labels must align")
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.array([0, 1], dtype=unique_labels.dtype)):
        raise ValueError("calibration labels must contain both classes")

    logits = torch.from_numpy(probabilities_to_logits(probabilities)).double()
    targets = torch.from_numpy(labels.astype(np.float64, copy=False)).double()
    log_slope = torch.zeros((), dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_slope, intercept),
        lr=0.5,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        slope = torch.exp(torch.clamp(log_slope, min=-8.0, max=8.0))
        loss = loss_fn(slope * logits + intercept, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    slope = float(
        torch.exp(torch.clamp(log_slope, min=-8.0, max=8.0)).detach()
    )
    return MonotoneAffineLogitCalibration(
        slope=slope,
        intercept=float(intercept.detach()),
    )


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    return np.log(clipped) - np.log1p(-clipped)


def run_integral_calibration_experiment(
    config: IntegralCalibrationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    split_specs = {
        "train": (config.train_structures, config.train_keys, 101, 201),
        "validation": (
            config.validation_structures,
            config.validation_keys,
            301,
            401,
        ),
        "calibration": (
            config.calibration_structures,
            config.calibration_keys,
            701,
            801,
        ),
        "test": (config.test_structures, config.test_keys, 501, 601),
    }
    structures_by_split = make_structure_splits(
        split_counts={name: values[0] for name, values in split_specs.items()},
        seed=config.seed,
        structure_split_mode=config.structure_split_mode,
        random_seed_offsets={
            name: values[2] for name, values in split_specs.items()
        },
    )
    splits = {
        name: build_integral_split_from_structures(
            name=name,
            rounds=config.rounds,
            structures=structures_by_split[name],
            keys=make_keys(count=values[1], seed=config.seed + values[3]),
            structure_split_mode=config.structure_split_mode,
            progress_callback=progress_callback,
        )
        for name, values in split_specs.items()
    }
    splits["stability"] = build_integral_split_from_structures(
        name="stability",
        rounds=config.rounds,
        structures=splits["test"].structures,
        keys=make_keys(
            count=config.stability_test_keys,
            seed=config.seed + 1001,
        ),
        structure_split_mode=config.structure_split_mode,
        progress_callback=progress_callback,
    )
    dataset_summary = summarize_calibration_splits(
        splits,
        structure_split_mode=config.structure_split_mode,
    )

    models = (
        ("anchor", "linear_same_input", False),
        ("candidate", "structure_mlp", False),
        ("control", "structure_mlp_shuffled_labels", True),
    )
    rows: list[dict[str, Any]] = []
    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    observation_rows: list[dict[str, Any]] = []
    for role, model_name, shuffle_labels in models:
        model_seed = config.seed if role == "anchor" else config.seed + 1000
        row, predictions = _fit_calibrated_model_row(
            config,
            role=role,
            model_name=model_name,
            shuffle_labels=shuffle_labels,
            splits=splits,
            model_seed=model_seed,
            progress_callback=progress_callback,
        )
        rows.append(row)
        model_predictions[model_name] = predictions
        observation_rows.extend(
            _observation_prediction_rows(
                model_name=model_name,
                splits=splits,
                predictions=predictions,
            )
        )

    gate = adjudicate_calibration(config, rows, dataset_summary)
    return {
        "rows": rows,
        "gate": gate,
        "dataset_summary": dataset_summary,
        "structure_rate_rows": _calibration_structure_rate_rows(
            splits["test"],
            splits["stability"],
            model_predictions,
        ),
        "observation_prediction_rows": observation_rows,
    }


def summarize_calibration_splits(
    splits: dict[str, IntegralParitySplit],
    *,
    structure_split_mode: str = "random-disjoint",
) -> dict[str, Any]:
    required = {"train", "validation", "calibration", "test", "stability"}
    if set(splits) != required:
        raise ValueError(f"calibration splits must be {sorted(required)}")
    main = {name: split for name, split in splits.items() if name != "stability"}
    summary = summarize_splits(
        main,
        structure_split_mode=structure_split_mode,
    )
    all_key_sets = [set(split.keys) for split in splits.values()]
    all_keys_disjoint = all(
        all_key_sets[left].isdisjoint(all_key_sets[right])
        for left in range(len(all_key_sets))
        for right in range(left + 1, len(all_key_sets))
    )
    test_signatures = tuple(item.signature for item in splits["test"].structures)
    stability_signatures = tuple(
        item.signature for item in splits["stability"].structures
    )
    summary["key_splits_disjoint"] = all_keys_disjoint
    summary["stability_structures_match_test"] = (
        test_signatures == stability_signatures
    )
    summary["splits"]["stability"] = _split_summary(splits["stability"])
    return summary


def _split_summary(split: IntegralParitySplit) -> dict[str, Any]:
    labels = split.dataset.labels
    return {
        "structures": len(split.structures),
        "keys_per_structure": len(split.keys),
        "rows": int(len(labels)),
        "q0_rows": int(np.sum(labels == 0)),
        "q1_rows": int(np.sum(labels == 1)),
        "q1_rate": float(labels.mean()),
        "all_zero_structures": int(np.sum(split.observed_q1_rates == 0.0)),
        "all_one_structures": int(np.sum(split.observed_q1_rates == 1.0)),
        "key_variable_structures": int(
            np.sum(
                (split.observed_q1_rates > 0.0)
                & (split.observed_q1_rates < 1.0)
            )
        ),
        "keys": [f"{key:020X}" for key in split.keys],
    }


def _fit_calibrated_model_row(
    config: IntegralCalibrationConfig,
    *,
    role: str,
    model_name: str,
    shuffle_labels: bool,
    splits: dict[str, IntegralParitySplit],
    model_seed: int,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_split = splits["train"]
    fit_dataset = train_split.dataset
    if shuffle_labels:
        shuffled_labels = np.random.default_rng(model_seed + 77).permutation(
            train_split.dataset.labels
        )
        fit_dataset = DifferentialDataset(
            features=train_split.dataset.features,
            labels=shuffled_labels.astype(np.uint8, copy=False),
            metadata={
                **train_split.dataset.metadata,
                "fit_train_labels_shuffled": True,
            },
        )

    torch.manual_seed(model_seed)
    model: nn.Module
    if model_name == "linear_same_input":
        model = LinearParityPredictor(INPUT_BITS)
    else:
        model = MlpDistinguisher(INPUT_BITS, hidden_bits=config.hidden_bits)
    _emit(
        progress_callback,
        "model_start",
        model=model_name,
        role=role,
        fit_train_labels_shuffled=shuffle_labels,
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=model_seed,
        device=config.device,
        optimizer="adam",
        weight_decay=config.weight_decay,
        checkpoint_metric="val_auc",
        restore_best_checkpoint=True,
        loss="bce",
        train_eval_interval=1,
    )

    def model_progress(event: str, payload: dict[str, Any]) -> None:
        _emit(
            progress_callback,
            event,
            model=model_name,
            role=role,
            **payload,
        )

    training_result = train_binary_classifier(
        model,
        fit_dataset,
        splits["validation"].dataset,
        training_config,
        progress_callback=model_progress,
    )
    true_train_metrics = evaluate_binary_classifier(
        model,
        train_split.dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    test_metrics = evaluate_binary_classifier(
        model,
        splits["test"].dataset,
        batch_size=config.batch_size,
        device=config.device,
    )

    raw_probabilities = {
        name: predict_binary_probabilities(
            model,
            splits[name].dataset,
            batch_size=config.batch_size,
            device=config.device,
        )
        for name in ("validation", "calibration", "test", "stability")
    }
    calibrator = fit_monotone_affine_logit_calibration(
        raw_probabilities["calibration"],
        splits["calibration"].dataset.labels,
    )
    calibrated_probabilities = {
        name: calibrator.transform(probabilities)
        for name, probabilities in raw_probabilities.items()
    }
    predictions = {
        **{f"raw_{name}": values for name, values in raw_probabilities.items()},
        **{
            f"calibrated_{name}": values
            for name, values in calibrated_probabilities.items()
        },
    }

    raw_test_rates = aggregate_structure_probabilities(
        splits["test"], raw_probabilities["test"]
    )
    calibrated_test_rates = aggregate_structure_probabilities(
        splits["test"], calibrated_probabilities["test"]
    )
    raw_stability_rates = aggregate_structure_probabilities(
        splits["stability"], raw_probabilities["stability"]
    )
    calibrated_stability_rates = aggregate_structure_probabilities(
        splits["stability"], calibrated_probabilities["stability"]
    )
    observed_test_rates = splits["test"].observed_q1_rates.astype(
        np.float64, copy=False
    )
    observed_stability_rates = splits["stability"].observed_q1_rates.astype(
        np.float64, copy=False
    )
    row: dict[str, Any] = {
        "run_id": config.run_id,
        "cipher": "PRESENT-80",
        "rounds": config.rounds,
        "seed": config.seed,
        "role": role,
        "task": "innovation2_integral_property_calibration",
        "model": model_name,
        "selected_model": model_name,
        "samples_per_class": None,
        "pairs_per_sample": INTEGRAL_SET_SIZE,
        "integral_set_size": INTEGRAL_SET_SIZE,
        "input_bits": INPUT_BITS,
        "input_view": "active_output_mask_fixed_context",
        "structure_split_mode": config.structure_split_mode,
        "key_omitted_from_features": True,
        "fit_train_labels_shuffled": shuffle_labels,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "train_structures": len(train_split.structures),
        "train_keys_per_structure": len(train_split.keys),
        "train_rows": int(len(train_split.dataset.labels)),
        "validation": _split_dimensions(splits["validation"]),
        "calibration": _split_dimensions(splits["calibration"]),
        "test": _split_dimensions(splits["test"]),
        "stability": _split_dimensions(splits["stability"]),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "history": training_result.history,
        "train_loss": true_train_metrics["loss"],
        "train_accuracy": true_train_metrics["accuracy"],
        "train_auc": true_train_metrics["auc"],
        "val_loss": training_result.final_metrics["loss"],
        "val_accuracy": training_result.final_metrics["accuracy"],
        "val_auc": training_result.final_metrics["auc"],
        "val_calibrated_accuracy": training_result.final_metrics[
            "calibrated_accuracy"
        ],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_calibrated_accuracy": test_metrics["calibrated_accuracy"],
        "calibration_slope": calibrator.slope,
        "calibration_intercept": calibrator.intercept,
        "calibration_raw_log_loss": binary_log_loss(
            splits["calibration"].dataset.labels,
            raw_probabilities["calibration"],
        ),
        "calibration_calibrated_log_loss": binary_log_loss(
            splits["calibration"].dataset.labels,
            calibrated_probabilities["calibration"],
        ),
        "test_raw_structure_rate_mae_32key": mean_absolute_error(
            raw_test_rates, observed_test_rates
        ),
        "test_calibrated_structure_rate_mae_32key": mean_absolute_error(
            calibrated_test_rates, observed_test_rates
        ),
        "stability_raw_structure_rate_mae_256key": mean_absolute_error(
            raw_stability_rates, observed_stability_rates
        ),
        "stability_calibrated_structure_rate_mae_256key": mean_absolute_error(
            calibrated_stability_rates, observed_stability_rates
        ),
        "observed_rate_32_256_mae": mean_absolute_error(
            observed_test_rates, observed_stability_rates
        ),
        "training": training_result.metadata,
        "claim_scope": (
            "PRESENT r5 local independent-calibration and 256-fresh-key "
            "structure-rate stability diagnostic only; not an exact integral proof, "
            "not remote, and not paper-scale output-prediction evidence"
        ),
    }
    _emit(
        progress_callback,
        "model_done",
        model=model_name,
        role=role,
        test_auc=row["test_auc"],
        stability_calibrated_structure_rate_mae_256key=row[
            "stability_calibrated_structure_rate_mae_256key"
        ],
    )
    return row, predictions


def aggregate_structure_probabilities(
    split: IntegralParitySplit,
    probabilities: np.ndarray,
) -> np.ndarray:
    if len(probabilities) != len(split.dataset.labels):
        raise ValueError("probabilities must align with split observations")
    predicted_rates = np.empty(len(split.structures), dtype=np.float64)
    for structure_index in range(len(split.structures)):
        predicted_rates[structure_index] = float(
            probabilities[split.structure_indices == structure_index].mean()
        )
    return predicted_rates


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    targets = np.asarray(labels, dtype=np.float64)
    return float(
        np.mean(-(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped)))
    )


def mean_absolute_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def adjudicate_calibration(
    config: IntegralCalibrationConfig,
    rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    by_role = {str(row["role"]): row for row in rows}
    candidate = by_role["candidate"]
    linear = by_role["anchor"]
    shuffled = by_role["control"]
    split_has_both_labels = all(
        split["q0_rows"] > 0 and split["q1_rows"] > 0
        for split in dataset_summary["splits"].values()
    )
    finite_metrics = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in (
            "test_auc",
            "calibration_slope",
            "calibration_intercept",
            "stability_calibrated_structure_rate_mae_256key",
        )
    )
    positive_slopes = all(float(row["calibration_slope"]) > 0.0 for row in rows)
    readiness_checks = {
        "main_structure_splits_disjoint": bool(
            dataset_summary["structure_splits_disjoint"]
        ),
        "all_key_splits_disjoint": bool(dataset_summary["key_splits_disjoint"]),
        "geometry_splits_disjoint_when_required": bool(
            config.structure_split_mode != "geometry-disjoint"
            or dataset_summary["geometry_splits_disjoint"]
        ),
        "one_structure_per_geometry_when_required": bool(
            config.structure_split_mode != "geometry-disjoint"
            or dataset_summary["one_structure_per_geometry"]
        ),
        "stability_structures_match_test": bool(
            dataset_summary["stability_structures_match_test"]
        ),
        "all_splits_have_both_labels": split_has_both_labels,
        "all_metrics_finite": finite_metrics,
        "calibration_slopes_positive": positive_slopes,
        "three_model_rows_present": len(rows) == 3,
    }
    candidate_auc = float(candidate["test_auc"])
    linear_auc = float(linear["test_auc"])
    shuffled_auc = float(shuffled["test_auc"])
    candidate_mae = float(
        candidate["stability_calibrated_structure_rate_mae_256key"]
    )
    linear_mae = float(linear["stability_calibrated_structure_rate_mae_256key"])
    observed_rate_stability = float(candidate["observed_rate_32_256_mae"])
    diagnostic_checks = {
        "calibrated_candidate_256key_mae_at_most_0_09": candidate_mae <= 0.09,
        "calibrated_candidate_mae_margin_at_least_0_015": (
            linear_mae - candidate_mae >= 0.015
        ),
        "candidate_linear_auc_margin_at_least_0_02": (
            candidate_auc - linear_auc >= 0.02
        ),
        "observed_32_256_rate_mae_at_most_0_05": observed_rate_stability <= 0.05,
        "shuffled_auc_near_chance": abs(shuffled_auc - 0.5) <= 0.05,
    }
    readiness_passed = all(readiness_checks.values())
    if config.gate_mode == "calibration-smoke":
        status = "pass" if readiness_passed else "fail"
        if config.structure_split_mode == "geometry-disjoint":
            decision = (
                "innovation2_integral_geometry_holdout_implementation_ready"
                if readiness_passed
                else "innovation2_integral_geometry_holdout_smoke_invalid"
            )
            next_action = (
                "Run the frozen local E4 geometry-holdout calibration and 256-key "
                "stability diagnostic."
                if readiness_passed
                else "Repair geometry ownership or artifact generation before E4."
            )
        else:
            decision = (
                "innovation2_integral_calibration_implementation_ready"
                if readiness_passed
                else "innovation2_integral_calibration_smoke_invalid"
            )
            next_action = (
                "Run the frozen local E1 calibration and 256-key stability diagnostic."
                if readiness_passed
                else "Repair split ownership, calibration, or artifact generation before E1."
            )
    elif not readiness_passed or not diagnostic_checks["shuffled_auc_near_chance"]:
        status = "fail"
        decision = "innovation2_integral_calibration_invalid_control"
        next_action = (
            "Audit split ownership, calibration fitting, and shuffled-label behavior; "
            "the E1 result is invalid."
        )
    elif all(diagnostic_checks.values()):
        status = "pass"
        decision = "innovation2_integral_calibration_advance_seed1_geometry"
        next_action = (
            "Run seed1 with held-out active/output/mask geometry combinations and add "
            "a classical integral baseline; keep the same calibration protocol."
        )
    elif not diagnostic_checks["observed_32_256_rate_mae_at_most_0_05"]:
        status = "hold"
        decision = "innovation2_integral_rate_target_unstable"
        next_action = (
            "Replace point-probability regression with an interval or ranking target; "
            "do not add training structures, epochs, seeds, or remote GPU budget."
        )
    else:
        status = "hold"
        decision = "innovation2_integral_calibration_insufficient"
        next_action = (
            "Add only PRESENT P-layer reachability features to the same E1 matrix; "
            "do not increase structures, epochs, seeds, or remote GPU budget."
        )
    return {
        "status": status,
        "decision": decision,
        "gate_mode": config.gate_mode,
        "structure_split_mode": config.structure_split_mode,
        "run_id": config.run_id,
        "readiness_checks": readiness_checks,
        "diagnostic_checks": diagnostic_checks,
        "metrics": {
            "candidate_test_auc": candidate_auc,
            "linear_test_auc": linear_auc,
            "shuffled_test_auc": shuffled_auc,
            "candidate_linear_auc_margin": candidate_auc - linear_auc,
            "candidate_calibrated_256key_mae": candidate_mae,
            "linear_calibrated_256key_mae": linear_mae,
            "linear_candidate_calibrated_256key_mae_margin": linear_mae
            - candidate_mae,
            "observed_32_256_rate_mae": observed_rate_stability,
        },
        "next_action": next_action,
        "claim_scope": (
            "local PRESENT r5 independent-calibration and 256-fresh-key "
            f"structure-rate stability diagnostic with {config.structure_split_mode} "
            "splits only"
        ),
    }


def _calibration_structure_rate_rows(
    test_split: IntegralParitySplit,
    stability_split: IntegralParitySplit,
    model_predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_test = test_split.observed_q1_rates.astype(np.float64, copy=False)
    observed_stability = stability_split.observed_q1_rates.astype(
        np.float64, copy=False
    )
    aggregated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, predictions in model_predictions.items():
        aggregated[model_name] = (
            aggregate_structure_probabilities(
                test_split, predictions["raw_test"]
            ),
            aggregate_structure_probabilities(
                test_split, predictions["calibrated_test"]
            ),
        )
    for index, structure in enumerate(test_split.structures):
        row: dict[str, Any] = {
            "structure_id": structure.structure_id,
            "signature": structure.signature,
            "geometry_id": structure.geometry_id,
            "structure_split_mode": test_split.dataset.metadata.get(
                "structure_split_mode", "random-disjoint"
            ),
            "active_nibble": structure.active_nibble,
            "output_nibble": structure.output_nibble,
            "output_mask": f"{structure.output_mask:04b}",
            "fixed_plaintext": f"0x{structure.fixed_plaintext:016X}",
            "observed_q1_rate_32key": float(observed_test[index]),
            "observed_q1_rate_256key": float(observed_stability[index]),
            "observed_rate_absolute_delta": float(
                abs(observed_test[index] - observed_stability[index])
            ),
        }
        for model_name, (raw_rates, calibrated_rates) in aggregated.items():
            row[f"{model_name}_raw_predicted_q1_rate"] = float(raw_rates[index])
            row[f"{model_name}_calibrated_predicted_q1_rate"] = float(
                calibrated_rates[index]
            )
        rows.append(row)
    return rows


def _observation_prediction_rows(
    *,
    model_name: str,
    splits: dict[str, IntegralParitySplit],
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("validation", "calibration", "test", "stability"):
        split = splits[split_name]
        raw = predictions[f"raw_{split_name}"]
        calibrated = predictions[f"calibrated_{split_name}"]
        raw_logits = probabilities_to_logits(raw)
        calibrated_logits = probabilities_to_logits(calibrated)
        key_count = len(split.keys)
        for index, label in enumerate(split.dataset.labels):
            structure_index = int(split.structure_indices[index])
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "structure_id": split.structures[
                        structure_index
                    ].structure_id,
                    "structure_index": structure_index,
                    "key_index": index % key_count,
                    "label_q": int(label),
                    "raw_logit": float(raw_logits[index]),
                    "raw_probability": float(raw[index]),
                    "calibrated_logit": float(calibrated_logits[index]),
                    "calibrated_probability": float(calibrated[index]),
                }
            )
    return rows


def _split_dimensions(split: IntegralParitySplit) -> dict[str, int]:
    return {
        "structures": len(split.structures),
        "keys_per_structure": len(split.keys),
        "rows": int(len(split.dataset.labels)),
    }


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)
