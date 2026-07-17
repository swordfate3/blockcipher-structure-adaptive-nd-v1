from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_context_group_split_readiness import (
    grouped_ridge_predictions,
)
from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    _categorical_additive_design,
    _marginal_predictions,
    ridge_loocv_predictions,
)
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_rank
from blockcipher_nd.training.metrics import binary_auc


EXPECTED_SOURCE_DECISION = "innovation2_speck_hwang_position_family_advance"
EXPECTED_SOURCE_TASK = "innovation2_speck32_hwang_fixed_position_family"
TARGET_MASK = 0x02050204


@dataclass(frozen=True)
class SpeckPositionLabelConfig:
    run_id: str
    seed: int = 0
    ridge_alpha: float = 1.0
    folds: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.seed != 0:
            raise ValueError("E28 uses the preregistered seed 0")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.folds != 4:
            raise ValueError("E28 uses four deterministic folds")


def run_position_label_audit(
    config: SpeckPositionLabelConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_checks = validate_source(source_gate, source_metadata, source_rows)
    kernels = full_position_kernels(source_rows)
    candidate_masks = bounded_flipping_masks(kernels)
    label_rows = build_label_grid(config, kernels, candidate_masks)
    labels = np.asarray(
        [int(row["balanced_label"]) for row in label_rows], dtype=np.float64
    )
    signatures = {
        tuple(
            int(row["balanced_label"])
            for row in label_rows
            if int(row["position_start"]) == position
        )
        for position in kernels
    }
    target_membership = sum(
        mask_in_kernel(TARGET_MASK, basis) for basis in kernels.values()
    )
    width_checks = {
        "at_least_eight_full_evidence_positions": len(kernels) >= 8,
        "at_least_eight_flipping_masks": len(candidate_masks) >= 8,
        "at_least_four_position_label_signatures": len(signatures) >= 4,
        "positive_rate_between_0p10_and_0p90": (
            bool(labels.size) and 0.10 <= float(labels.mean()) <= 0.90
        ),
        "target_mask_is_included_and_flips": (
            TARGET_MASK in candidate_masks and 0 < target_membership < len(kernels)
        ),
        "complete_unique_position_mask_grid": (
            len(label_rows) == len(kernels) * len(candidate_masks)
            and len(
                {
                    (int(row["position_start"]), int(row["mask_value"]))
                    for row in label_rows
                }
            )
            == len(label_rows)
        ),
        "labels_have_both_classes": bool(labels.size)
        and set(labels.astype(int)) == {0, 1},
    }
    baseline_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    fold_checks = {
        "all_group_protocols_assign_every_row_once": False,
        "all_group_train_test_splits_have_both_classes": False,
        "all_metrics_finite": False,
    }
    position_folds: dict[int, int] = {}
    mask_folds: dict[int, int] = {}
    if all(source_checks.values()) and all(width_checks.values()):
        baseline_rows, diagnostics, position_folds, mask_folds = evaluate_shortcuts(
            config, label_rows
        )
        fold_checks = {
            "all_group_protocols_assign_every_row_once": all(
                row["all_rows_assigned_once"] for row in diagnostics.values()
            ),
            "all_group_train_test_splits_have_both_classes": all(
                row["all_train_test_splits_have_both_classes"]
                for row in diagnostics.values()
            ),
            "all_metrics_finite": all(
                math.isfinite(float(row[field]))
                for row in baseline_rows
                for field in ("accuracy", "brier", "auc", "directional_auc")
            ),
        }
    readiness = {**source_checks, **width_checks, **fold_checks}
    gate = adjudicate_position_label_audit(
        config,
        baseline_rows,
        source_checks=source_checks,
        width_checks={**width_checks, **fold_checks},
        positive_rate=float(labels.mean()) if labels.size else 0.0,
        full_positions=len(kernels),
        flipping_masks=len(candidate_masks),
        distinct_signatures=len(signatures),
    )
    return {
        "rows": baseline_rows,
        "label_rows": label_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_speck_position_mask_label_width_readiness",
            "source_run_id": source_gate.get("run_id"),
            "source_task": source_metadata.get("task"),
            "full_evidence_positions": list(kernels),
            "candidate_masks": [f"0x{mask:08X}" for mask in candidate_masks],
            "label_rows": len(label_rows),
            "position_folds": {str(key): value for key, value in position_folds.items()},
            "mask_folds": {str(key): value for key, value in mask_folds.items()},
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
    source_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    required = {
        "position_start",
        "fixed_bits",
        "evidence_keys",
        "joint_nullity",
        "joint_kernel_basis",
    }
    starts = [int(row.get("position_start", -1)) for row in source_rows]
    return {
        "source_gate_passed_position_family": (
            source_gate.get("status") == "pass"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e27_position_family": (
            source_metadata.get("task") == EXPECTED_SOURCE_TASK
            and source_metadata.get("training_performed") is False
        ),
        "source_has_thirty_unique_position_rows": (
            len(source_rows) == 30 and len(set(starts)) == 30
        ),
        "full_evidence_rows_have_kernel_fields": all(
            required.issubset(row)
            for row in source_rows
            if int(row.get("evidence_keys", 0)) == 64
        ),
    }


def full_position_kernels(
    source_rows: list[dict[str, Any]],
) -> dict[int, tuple[int, ...]]:
    kernels: dict[int, tuple[int, ...]] = {}
    for row in sorted(source_rows, key=lambda item: int(item.get("position_start", -1))):
        if int(row.get("evidence_keys", 0)) != 64:
            continue
        start = int(row["position_start"])
        text = str(row.get("joint_kernel_basis", ""))
        basis = tuple(int(value, 16) for value in text.split(";") if value)
        if len(basis) != int(row.get("joint_nullity", -1)):
            raise ValueError(f"joint kernel basis/nullity mismatch for position {start}")
        if gf2_rank(np.asarray(basis, dtype=np.uint64), width=32) != len(basis):
            raise ValueError(f"joint kernel basis is dependent for position {start}")
        kernels[start] = basis
    return kernels


def bounded_flipping_masks(
    kernels: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    candidates: set[int] = set()
    for basis in kernels.values():
        candidates.update(mask for mask in basis if mask)
        for left_index, left in enumerate(basis):
            for right in basis[left_index + 1 :]:
                if left ^ right:
                    candidates.add(left ^ right)
    return tuple(
        sorted(
            mask
            for mask in candidates
            if 0
            < sum(mask_in_kernel(mask, basis) for basis in kernels.values())
            < len(kernels)
        )
    )


def mask_in_kernel(mask: int, basis: tuple[int, ...]) -> bool:
    if mask == 0:
        return True
    basis_words = np.asarray(basis, dtype=np.uint64)
    combined = np.concatenate((basis_words, np.asarray([mask], dtype=np.uint64)))
    return gf2_rank(combined, width=32) == len(basis)


def build_label_grid(
    config: SpeckPositionLabelConfig,
    kernels: dict[int, tuple[int, ...]],
    candidate_masks: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position_index, (start, basis) in enumerate(sorted(kernels.items())):
        position_mask = (1 << start) | (1 << (start + 1))
        for mask_index, mask in enumerate(candidate_masks):
            rows.append(
                {
                    "run_id": config.run_id,
                    "position_index": position_index,
                    "position_start": start,
                    "fixed_bits": f"{start},{start + 1}",
                    "position_mask_hex": f"0x{position_mask:08X}",
                    "position_mask_value": position_mask,
                    "position_weight": 2,
                    "mask_index": mask_index,
                    "mask_hex": f"0x{mask:08X}",
                    "mask_value": mask,
                    "mask_weight": mask.bit_count(),
                    "balanced_label": int(mask_in_kernel(mask, basis)),
                }
            )
    return rows


def evaluate_shortcuts(
    config: SpeckPositionLabelConfig,
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[int, int],
    dict[int, int],
]:
    labels = np.asarray([int(row["balanced_label"]) for row in rows], dtype=np.float64)
    position_ids = [int(row["position_start"]) for row in rows]
    mask_ids = [int(row["mask_index"]) for row in rows]
    position_groups = [str(value) for value in position_ids]
    position_weights = [str(row["position_weight"]) for row in rows]
    mask_groups = [str(row["mask_hex"]) for row in rows]
    mask_weights = [str(row["mask_weight"]) for row in rows]
    position_values = [int(row["position_mask_value"]) for row in rows]
    mask_values = [int(row["mask_value"]) for row in rows]
    design = bitwise_design(position_values, mask_values)

    predictions = {
        "global_rate": np.full(len(rows), labels.mean(), dtype=np.float64),
        "position_identity_marginal": _marginal_predictions(labels, position_groups),
        "position_weight_marginal": _marginal_predictions(labels, position_weights),
        "mask_identity_marginal": _marginal_predictions(labels, mask_groups),
        "mask_weight_marginal": _marginal_predictions(labels, mask_weights),
        "position_mask_additive": ridge_loocv_predictions(
            _categorical_additive_design((position_groups, mask_groups)),
            labels,
            alpha=config.ridge_alpha,
        ),
        "position_mask_bitwise_linear": ridge_loocv_predictions(
            design, labels, alpha=config.ridge_alpha
        ),
    }
    positions = tuple(sorted(set(position_ids)))
    masks = tuple(sorted(set(mask_ids)))
    position_folds = {
        value: index % config.folds for index, value in enumerate(positions)
    }
    mask_folds = {value: index % config.folds for index, value in enumerate(masks)}
    diagnostics: dict[str, Any] = {}
    for mode, key in (
        ("context", "position_disjoint_bitwise"),
        ("mask", "mask_disjoint_bitwise"),
        ("dual", "dual_disjoint_bitwise"),
    ):
        prediction, diagnostic = grouped_ridge_predictions(
            design,
            labels,
            context_ids=position_ids,
            mask_ids=mask_ids,
            context_folds=position_folds,
            mask_folds=mask_folds,
            mode=mode,
            alpha=config.ridge_alpha,
        )
        predictions[key] = prediction
        diagnostics[key] = diagnostic
    rng = np.random.default_rng(config.seed + 7301)
    shuffled = labels[rng.permutation(len(labels))]
    shuffled_prediction, shuffled_diagnostic = grouped_ridge_predictions(
        design,
        shuffled,
        context_ids=position_ids,
        mask_ids=mask_ids,
        context_folds=position_folds,
        mask_folds=mask_folds,
        mode="dual",
        alpha=config.ridge_alpha,
    )
    predictions["label_shuffle_dual_disjoint"] = shuffled_prediction
    diagnostics["label_shuffle_dual_disjoint"] = shuffled_diagnostic
    metric_rows = [
        metric_row(
            config,
            key,
            values,
            shuffled if key == "label_shuffle_dual_disjoint" else labels,
        )
        for key, values in predictions.items()
    ]
    return metric_rows, diagnostics, position_folds, mask_folds


def bitwise_design(
    position_values: list[int], mask_values: list[int]
) -> np.ndarray:
    if len(position_values) != len(mask_values):
        raise ValueError("position and mask values must have equal length")
    design = np.ones((len(position_values), 65), dtype=np.float64)
    for row_index, (position, mask) in enumerate(
        zip(position_values, mask_values, strict=True)
    ):
        for bit in range(32):
            design[row_index, 1 + bit] = (position >> bit) & 1
            design[row_index, 33 + bit] = (mask >> bit) & 1
    return design


def metric_row(
    config: SpeckPositionLabelConfig,
    key: str,
    predictions: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    auc = float(binary_auc(target, predictions))
    return {
        "run_id": config.run_id,
        "baseline": key,
        "accuracy": float(np.mean((predictions >= 0.5).astype(np.uint8) == target)),
        "brier": float(np.mean((predictions - target) ** 2)),
        "auc": auc,
        "directional_auc": max(auc, 1.0 - auc),
        "rows": len(target),
    }


def adjudicate_position_label_audit(
    config: SpeckPositionLabelConfig,
    metric_rows: list[dict[str, Any]],
    *,
    source_checks: dict[str, bool],
    width_checks: dict[str, bool],
    positive_rate: float,
    full_positions: int,
    flipping_masks: int,
    distinct_signatures: int,
) -> dict[str, Any]:
    metrics = {str(row["baseline"]): row for row in metric_rows}
    required_shortcuts = (
        "position_identity_marginal",
        "mask_identity_marginal",
        "position_mask_additive",
        "position_mask_bitwise_linear",
        "position_disjoint_bitwise",
        "mask_disjoint_bitwise",
        "dual_disjoint_bitwise",
    )
    shortcut_checks = {
        f"{key}_directional_auc_below_0p75": (
            key in metrics and float(metrics[key]["directional_auc"]) < 0.75
        )
        for key in required_shortcuts
    }
    shortcut_checks["shuffle_dual_directional_auc_at_most_0p65"] = (
        "label_shuffle_dual_disjoint" in metrics
        and float(metrics["label_shuffle_dual_disjoint"]["directional_auc"]) <= 0.65
    )
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_speck_position_label_protocol_invalid"
        next_action = {
            "action": "repair or revalidate the E27 source gate and full-evidence kernel rows",
            "training": False,
            "remote_scale": False,
        }
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_speck_position_label_grid_too_narrow"
        next_action = {
            "action": "do not train; test the preregistered ROR7-to-addition aligned family against the offset-minus-one control",
            "training": False,
            "remote_scale": False,
        }
    elif all(shortcut_checks.values()):
        status = "pass"
        decision = "innovation2_speck_position_label_grid_advance"
        next_action = {
            "action": "validate the label grid on fresh keys and expand independent structures",
            "next_adjudication": "E29 SPECK fresh-key label stability and structure expansion",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_speck_position_label_grid_shortcut_dominated"
        next_action = {
            "action": "stop the adjacent-position by mask table; do not select only passing folds",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": {
            key: {
                field: float(row[field])
                for field in ("accuracy", "brier", "auc", "directional_auc")
            }
            for key, row in metrics.items()
        },
        "full_evidence_positions": full_positions,
        "flipping_masks": flipping_masks,
        "distinct_position_signatures": distinct_signatures,
        "positive_rate": positive_rate,
        "claim_scope": (
            "local deterministic basis-plus-pairwise position-mask label-width and "
            "group-disjoint shortcut audit over verified E27 64-key kernels; not new "
            "encryption, fresh-key validation, neural training, or an all-key proof"
        ),
        "next_action": next_action,
    }
