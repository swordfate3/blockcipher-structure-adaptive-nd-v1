from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    _bitwise_design,
    ridge_loocv_predictions,
)
from blockcipher_nd.training.metrics import binary_auc


EXPECTED_SOURCE_DECISION = (
    "innovation2_equal_prevalence_context_label_shortcut_dominated"
)
EXPECTED_SOURCE_TASK = "innovation2_equal_prevalence_context_mask_label_readiness"
SplitMode = Literal["context", "mask", "dual"]


@dataclass(frozen=True)
class ContextGroupSplitConfig:
    run_id: str
    seed: int = 0
    ridge_alpha: float = 1.0
    folds: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.folds != 4:
            raise ValueError("the frozen E17c audit requires four folds")


def run_context_group_split_audit(
    config: ContextGroupSplitConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_label_rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_checks = validate_source(source_gate, source_metadata, source_label_rows)
    parsed_rows = parse_label_rows(source_label_rows)
    labels = np.asarray(
        [int(row["balanced_label"]) for row in parsed_rows], dtype=np.float64
    )
    context_values = [int(row["context_value"]) for row in parsed_rows]
    mask_values = [int(row["mask_value"]) for row in parsed_rows]
    context_ids = [int(row["context_id"]) for row in parsed_rows]
    mask_ids = [int(row["mask_index"]) for row in parsed_rows]
    design = _bitwise_design(context_values, mask_values)
    context_folds, mask_folds, fold_search_attempts = make_class_complete_group_folds(
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        folds=config.folds,
        seed=config.seed + 6401,
    )

    rowwise_predictions = ridge_loocv_predictions(
        design, labels, alpha=config.ridge_alpha
    )
    context_predictions, context_diagnostics = grouped_ridge_predictions(
        design,
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=context_folds,
        mask_folds=mask_folds,
        mode="context",
        alpha=config.ridge_alpha,
    )
    mask_predictions, mask_diagnostics = grouped_ridge_predictions(
        design,
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=context_folds,
        mask_folds=mask_folds,
        mode="mask",
        alpha=config.ridge_alpha,
    )
    dual_predictions, dual_diagnostics = grouped_ridge_predictions(
        design,
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=context_folds,
        mask_folds=mask_folds,
        mode="dual",
        alpha=config.ridge_alpha,
    )
    rng = np.random.default_rng(config.seed + 6301)
    shuffled_labels = labels[rng.permutation(len(labels))]
    shuffled_predictions, shuffle_diagnostics = grouped_ridge_predictions(
        design,
        shuffled_labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=context_folds,
        mask_folds=mask_folds,
        mode="dual",
        alpha=config.ridge_alpha,
    )
    metric_rows = [
        metric_row(config, "rowwise_loocv_bitwise", "逐行LOOCV位模式", rowwise_predictions, labels),
        metric_row(
            config,
            "context_disjoint_bitwise",
            "context组外位模式",
            context_predictions,
            labels,
        ),
        metric_row(
            config,
            "mask_disjoint_bitwise",
            "mask组外位模式",
            mask_predictions,
            labels,
        ),
        metric_row(
            config,
            "dual_disjoint_bitwise",
            "context+mask双轴组外",
            dual_predictions,
            labels,
        ),
        metric_row(
            config,
            "label_shuffle_dual_disjoint",
            "标签打乱双轴组外",
            shuffled_predictions,
            shuffled_labels,
        ),
    ]
    diagnostics = {
        "context_disjoint": context_diagnostics,
        "mask_disjoint": mask_diagnostics,
        "dual_disjoint": dual_diagnostics,
        "label_shuffle_dual_disjoint": shuffle_diagnostics,
    }
    readiness = {
        **source_checks,
        "context_folds_have_four_contexts_each": sorted(
            list(context_folds.values()).count(fold) for fold in range(config.folds)
        )
        == [4, 4, 4, 4],
        "mask_folds_have_eight_masks_each": sorted(
            list(mask_folds.values()).count(fold) for fold in range(config.folds)
        )
        == [8, 8, 8, 8],
        "all_group_protocols_assign_every_row_once": all(
            item["all_rows_assigned_once"] for item in diagnostics.values()
        ),
        "all_group_train_test_splits_have_both_classes": all(
            item["all_train_test_splits_have_both_classes"]
            for item in diagnostics.values()
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in metric_rows
            for key in ("accuracy", "brier", "auc", "directional_auc")
        ),
    }
    gate = adjudicate_context_group_split(
        config,
        metric_rows,
        readiness,
    )
    return {
        "rows": metric_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_context_mask_group_disjoint_shortcut_readiness",
            "source_run_id": source_gate.get("run_id"),
            "rows": len(parsed_rows),
            "contexts": len(context_folds),
            "masks": len(mask_folds),
            "folds": config.folds,
            "context_folds": {str(key): value for key, value in context_folds.items()},
            "mask_folds": {str(key): value for key, value in mask_folds.items()},
            "fold_search_attempts": fold_search_attempts,
            "diagnostics": diagnostics,
            "ridge_alpha": config.ridge_alpha,
            "seed": config.seed,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def validate_source(
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_label_rows: list[dict[str, str]],
) -> dict[str, bool]:
    required_columns = {
        "context_id",
        "context_value",
        "mask_index",
        "mask_value",
        "balanced_label",
    }
    return {
        "source_gate_is_e17b_shortcut_hold": (
            source_gate.get("status") == "hold"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e17b_equal_prevalence": (
            source_metadata.get("task") == EXPECTED_SOURCE_TASK
            and source_metadata.get("training_performed") is False
            and int(source_metadata.get("contexts", -1)) == 16
            and int(source_metadata.get("candidate_masks", -1)) == 32
            and int(source_metadata.get("label_rows", -1)) == 512
        ),
        "source_label_rows_complete": len(source_label_rows) == 512
        and all(required_columns.issubset(row) for row in source_label_rows),
    }


def parse_label_rows(rows: list[dict[str, str]]) -> list[dict[str, int]]:
    parsed = [
        {
            "context_id": int(row["context_id"]),
            "context_value": int(row["context_value"]),
            "mask_index": int(row["mask_index"]),
            "mask_value": int(row["mask_value"]),
            "balanced_label": int(row["balanced_label"]),
        }
        for row in rows
    ]
    contexts = {row["context_id"] for row in parsed}
    masks = {row["mask_index"] for row in parsed}
    grid = {(row["context_id"], row["mask_index"]) for row in parsed}
    if contexts != set(range(16)) or masks != set(range(32)):
        raise ValueError("label rows must contain context ids 0..15 and mask ids 0..31")
    if len(grid) != 512 or len(parsed) != 512:
        raise ValueError("label rows must form a unique 16 by 32 grid")
    if {row["balanced_label"] for row in parsed} != {0, 1}:
        raise ValueError("label rows must contain both classes")
    return parsed


def make_group_folds(
    values: tuple[int, ...],
    *,
    folds: int,
    seed: int,
) -> dict[int, int]:
    if not values or len(values) % folds:
        raise ValueError("group count must be nonzero and divisible by folds")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(values, dtype=np.int64)
    rng.shuffle(shuffled)
    return {
        int(value): index % folds for index, value in enumerate(shuffled.tolist())
    }


def make_class_complete_group_folds(
    labels: np.ndarray,
    *,
    context_ids: list[int],
    mask_ids: list[int],
    folds: int,
    seed: int,
    max_attempts: int = 10000,
) -> tuple[dict[int, int], dict[int, int], int]:
    contexts = tuple(sorted(set(context_ids)))
    masks = tuple(sorted(set(mask_ids)))
    if len(contexts) % folds or len(masks) % folds:
        raise ValueError("context and mask group counts must be divisible by folds")
    rng = np.random.default_rng(seed)
    context_array = np.asarray(contexts, dtype=np.int64)
    mask_array = np.asarray(masks, dtype=np.int64)
    for attempt in range(1, max_attempts + 1):
        rng.shuffle(context_array)
        rng.shuffle(mask_array)
        context_folds = {
            int(value): index % folds
            for index, value in enumerate(context_array.tolist())
        }
        mask_folds = {
            int(value): index % folds
            for index, value in enumerate(mask_array.tolist())
        }
        context_fold_rows = np.asarray(
            [context_folds[value] for value in context_ids], dtype=np.int64
        )
        mask_fold_rows = np.asarray(
            [mask_folds[value] for value in mask_ids], dtype=np.int64
        )
        class_complete = True
        for context_fold in range(folds):
            for mask_fold in range(folds):
                test = (context_fold_rows == context_fold) & (
                    mask_fold_rows == mask_fold
                )
                train = (context_fold_rows != context_fold) & (
                    mask_fold_rows != mask_fold
                )
                if (
                    set(labels[test].astype(int)) != {0, 1}
                    or set(labels[train].astype(int)) != {0, 1}
                ):
                    class_complete = False
                    break
            if not class_complete:
                break
        if class_complete:
            return context_folds, mask_folds, attempt
    raise ValueError(
        f"unable to find class-complete balanced folds after {max_attempts} attempts"
    )


def grouped_ridge_predictions(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    context_ids: list[int],
    mask_ids: list[int],
    context_folds: dict[int, int],
    mask_folds: dict[int, int],
    mode: SplitMode,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    context_fold_rows = np.asarray(
        [context_folds[value] for value in context_ids], dtype=np.int64
    )
    mask_fold_rows = np.asarray(
        [mask_folds[value] for value in mask_ids], dtype=np.int64
    )
    predictions = np.zeros(len(labels), dtype=np.float64)
    assignments = np.zeros(len(labels), dtype=np.int64)
    class_checks: list[bool] = []
    fold_pairs = (
        [(fold, -1) for fold in sorted(set(context_folds.values()))]
        if mode == "context"
        else [(-1, fold) for fold in sorted(set(mask_folds.values()))]
        if mode == "mask"
        else [
            (context_fold, mask_fold)
            for context_fold in sorted(set(context_folds.values()))
            for mask_fold in sorted(set(mask_folds.values()))
        ]
    )
    for context_fold, mask_fold in fold_pairs:
        if mode == "context":
            test = context_fold_rows == context_fold
            train = ~test
        elif mode == "mask":
            test = mask_fold_rows == mask_fold
            train = ~test
        else:
            test = (context_fold_rows == context_fold) & (
                mask_fold_rows == mask_fold
            )
            train = (context_fold_rows != context_fold) & (
                mask_fold_rows != mask_fold
            )
        if not np.any(test) or not np.any(train):
            raise ValueError(f"empty train or test split for {mode}")
        class_checks.append(
            set(labels[train].astype(int)) == {0, 1}
            and set(labels[test].astype(int)) == {0, 1}
        )
        predictions[test] = ridge_fit_predict(
            design[train],
            labels[train],
            design[test],
            alpha=alpha,
        )
        assignments[test] += 1
    return predictions, {
        "mode": mode,
        "split_count": len(fold_pairs),
        "all_rows_assigned_once": bool(np.all(assignments == 1)),
        "all_train_test_splits_have_both_classes": all(class_checks),
        "minimum_assignments": int(assignments.min()),
        "maximum_assignments": int(assignments.max()),
    }


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    penalty = np.eye(train_x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_x.T @ train_x + penalty,
        train_x.T @ train_y,
    )
    return np.clip(test_x @ coefficients, 0.0, 1.0)


def metric_row(
    config: ContextGroupSplitConfig,
    key: str,
    label: str,
    predictions: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    auc = float(binary_auc(target, predictions))
    predicted_labels = (predictions >= 0.5).astype(np.uint8)
    return {
        "run_id": config.run_id,
        "baseline": key,
        "baseline_label": label,
        "accuracy": float(np.mean(predicted_labels == target)),
        "brier": float(np.mean((predictions - target) ** 2)),
        "auc": auc,
        "directional_auc": max(auc, 1.0 - auc),
        "rows": len(target),
    }


def adjudicate_context_group_split(
    config: ContextGroupSplitConfig,
    metric_rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    metrics = {str(row["baseline"]): row for row in metric_rows}
    shortcut_checks = {
        "context_disjoint_directional_auc_below_0p75": (
            float(metrics["context_disjoint_bitwise"]["directional_auc"]) < 0.75
        ),
        "mask_disjoint_directional_auc_below_0p75": (
            float(metrics["mask_disjoint_bitwise"]["directional_auc"]) < 0.75
        ),
        "dual_disjoint_directional_auc_below_0p75": (
            float(metrics["dual_disjoint_bitwise"]["directional_auc"]) < 0.75
        ),
        "shuffle_dual_directional_auc_at_most_0p65": (
            float(metrics["label_shuffle_dual_disjoint"]["directional_auc"])
            <= 0.65
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_group_disjoint_protocol_invalid"
        next_action = {
            "action": "repair group folds, coverage, or E17b source validation",
            "training": False,
            "remote_scale": False,
        }
    elif all(shortcut_checks.values()):
        status = "pass"
        decision = "innovation2_group_disjoint_shortcuts_controlled"
        next_action = {
            "action": "validate context-dependent labels on fresh keys",
            "next_adjudication": "E18 fresh-key context kernel stability",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_group_disjoint_shortcut_generalizes"
        next_action = {
            "action": "stop current context-mask label family and redesign structures",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": {
            key: {
                "accuracy": float(row["accuracy"]),
                "brier": float(row["brier"]),
                "auc": float(row["auc"]),
                "directional_auc": float(row["directional_auc"]),
            }
            for key, row in metrics.items()
        },
        "claim_scope": (
            "deterministic group-disjoint linear shortcut readiness over the E17b "
            "512-row label grid; not a neural result or fresh-key validation"
        ),
        "next_action": next_action,
    }
