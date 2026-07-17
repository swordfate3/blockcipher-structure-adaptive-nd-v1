from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from blockcipher_nd.training.metrics import binary_auc


EXPECTED_SOURCE_DECISION = (
    "innovation2_present_r7_active_block_kernel_diversity_ready"
)
BLOCK_IDS = ("block_0_15", "block_16_31", "block_32_47", "block_48_63")


@dataclass(frozen=True)
class OutputLabelReadinessConfig:
    run_id: str
    seed: int = 0
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")


def run_output_label_readiness_audit(
    config: OutputLabelReadinessConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_basis_rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_checks = validate_source(
        source_gate,
        source_metadata,
        source_basis_rows,
    )
    kernels = kernel_bases_from_rows(source_basis_rows)
    positive_masks = tuple(sorted({vector for basis in kernels.values() for vector in basis}))
    kernel_spans = {
        block_id: bounded_span(basis, max_dimension=16)
        for block_id, basis in kernels.items()
    }
    controls = matched_negative_masks(
        positive_masks,
        excluded_masks=set().union(*kernel_spans.values()),
        seed=config.seed,
    )
    candidate_masks = positive_masks + controls
    label_rows = build_label_rows(
        config,
        candidate_masks=candidate_masks,
        positive_masks=set(positive_masks),
        kernel_spans=kernel_spans,
    )
    baseline_rows = evaluate_baselines(config, label_rows)
    labels = np.asarray([int(row["balanced_label"]) for row in label_rows], dtype=np.uint8)
    mask_labels: dict[int, set[int]] = {}
    block_signatures: set[str] = set()
    for row in label_rows:
        mask_labels.setdefault(int(row["mask_value"]), set()).add(
            int(row["balanced_label"])
        )
    for block_id in BLOCK_IDS:
        block_signatures.add(
            "".join(
                str(int(row["balanced_label"]))
                for row in label_rows
                if row["block_id"] == block_id
            )
        )
    flipping_masks = sum(values == {0, 1} for values in mask_labels.values())
    readiness = {
        **source_checks,
        "candidate_masks_unique": len(candidate_masks) == len(set(candidate_masks)),
        "matched_controls_equal_positive_candidates": len(controls)
        == len(positive_masks),
        "all_controls_outside_every_source_kernel": all(
            mask not in span for mask in controls for span in kernel_spans.values()
        ),
        "complete_structure_mask_grid": len(label_rows)
        == len(BLOCK_IDS) * len(candidate_masks),
        "labels_have_both_classes": set(int(value) for value in labels) == {0, 1},
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in baseline_rows
            for key in ("accuracy", "brier", "auc")
        ),
    }
    gate = adjudicate_output_label_readiness(
        config,
        baseline_rows,
        readiness,
        positive_rate=float(labels.mean()),
        flipping_masks=flipping_masks,
        distinct_block_label_signatures=len(block_signatures),
    )
    return {
        "rows": baseline_rows,
        "label_rows": label_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_structure_mask_output_label_readiness",
            "source_run_id": source_gate.get("run_id"),
            "structures": len(BLOCK_IDS),
            "positive_candidate_masks": len(positive_masks),
            "matched_control_masks": len(controls),
            "candidate_masks": len(candidate_masks),
            "label_rows": len(label_rows),
            "ridge_alpha": config.ridge_alpha,
            "seed": config.seed,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def validate_source(
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_basis_rows: list[dict[str, str]],
) -> dict[str, bool]:
    required_columns = {
        "block_id",
        "basis_index",
        "vector_hex",
        "vector_weight",
    }
    source_blocks = {row.get("block_id", "") for row in source_basis_rows}
    return {
        "source_gate_passed_diversity": (
            source_gate.get("status") == "pass"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e12_kernel_diversity": (
            source_metadata.get("task")
            == "innovation2_present_r7_active_block_kernel_diversity"
            and source_metadata.get("training_performed") is False
        ),
        "source_basis_rows_complete": bool(source_basis_rows)
        and all(required_columns.issubset(row) for row in source_basis_rows),
        "source_has_four_frozen_blocks": source_blocks == set(BLOCK_IDS),
    }


def kernel_bases_from_rows(
    rows: list[dict[str, str]],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[tuple[int, int]]] = {block_id: [] for block_id in BLOCK_IDS}
    for row in rows:
        block_id = str(row["block_id"])
        if block_id not in grouped:
            raise ValueError(f"unexpected block_id: {block_id}")
        grouped[block_id].append(
            (int(row["basis_index"]), int(row["vector_hex"], 16))
        )
    bases: dict[str, tuple[int, ...]] = {}
    for block_id, values in grouped.items():
        ordered = tuple(vector for _, vector in sorted(values))
        if not ordered:
            raise ValueError(f"missing basis rows for {block_id}")
        bases[block_id] = ordered
    return bases


def bounded_span(basis: tuple[int, ...], *, max_dimension: int) -> set[int]:
    if len(basis) > max_dimension:
        raise ValueError(
            f"refusing to enumerate dimension-{len(basis)} span above {max_dimension}"
        )
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return values


def matched_negative_masks(
    positive_masks: tuple[int, ...],
    *,
    excluded_masks: set[int],
    seed: int,
) -> tuple[int, ...]:
    rng = np.random.default_rng(seed + 4319)
    controls: list[int] = []
    used = set(excluded_masks)
    for positive in positive_masks:
        weight = positive.bit_count()
        while True:
            bits = rng.choice(64, size=weight, replace=False)
            candidate = sum(1 << int(bit) for bit in bits)
            if candidate == 0 or candidate in used:
                continue
            used.add(candidate)
            controls.append(candidate)
            break
    return tuple(controls)


def build_label_rows(
    config: OutputLabelReadinessConfig,
    *,
    candidate_masks: tuple[int, ...],
    positive_masks: set[int],
    kernel_spans: dict[str, set[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_index, block_id in enumerate(BLOCK_IDS):
        for mask_index, mask in enumerate(candidate_masks):
            rows.append(
                {
                    "run_id": config.run_id,
                    "block_id": block_id,
                    "block_index": block_index,
                    "mask_index": mask_index,
                    "mask_hex": f"0x{mask:016X}",
                    "mask_value": mask,
                    "mask_weight": mask.bit_count(),
                    "mask_role": (
                        "kernel_basis_union" if mask in positive_masks else "matched_control"
                    ),
                    "balanced_label": int(mask in kernel_spans[block_id]),
                }
            )
    return rows


def evaluate_baselines(
    config: OutputLabelReadinessConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray([int(row["balanced_label"]) for row in rows], dtype=np.float64)
    block_ids = [str(row["block_id"]) for row in rows]
    mask_ids = [str(row["mask_hex"]) for row in rows]
    mask_weights = [str(row["mask_weight"]) for row in rows]
    global_predictions = np.full(len(rows), labels.mean(), dtype=np.float64)
    block_predictions = _marginal_predictions(labels, block_ids)
    mask_predictions = _marginal_predictions(labels, mask_ids)
    weight_predictions = _marginal_predictions(labels, mask_weights)
    block_weight_predictions = _crossfit_additive_predictions(
        labels,
        (block_ids, mask_weights),
        alpha=config.ridge_alpha,
    )
    block_mask_predictions = _crossfit_additive_predictions(
        labels,
        (block_ids, mask_ids),
        alpha=config.ridge_alpha,
    )
    rng = np.random.default_rng(config.seed + 4327)
    shuffled_labels = labels[rng.permutation(len(labels))]
    shuffled_predictions = _crossfit_additive_predictions(
        shuffled_labels,
        (block_ids, mask_ids),
        alpha=config.ridge_alpha,
    )
    candidates = (
        ("global_rate", "全局正例率", global_predictions, labels),
        ("active_block_marginal", "活动块边际", block_predictions, labels),
        ("mask_identity_marginal", "mask身份边际", mask_predictions, labels),
        ("mask_weight_marginal", "mask权重边际", weight_predictions, labels),
        (
            "block_weight_additive",
            "活动块+mask权重加性",
            block_weight_predictions,
            labels,
        ),
        (
            "block_mask_additive",
            "活动块+mask身份加性",
            block_mask_predictions,
            labels,
        ),
        (
            "label_shuffle_additive",
            "标签打乱加性控制",
            shuffled_predictions,
            shuffled_labels,
        ),
    )
    baseline_rows = [
        _metric_row(config, key, label, predictions, target)
        for key, label, predictions, target in candidates
    ]
    baseline_rows.append(
        {
            "run_id": config.run_id,
            "baseline": "direct_gf2_kernel_oracle",
            "baseline_label": "直接GF(2) kernel标签",
            "accuracy": 1.0,
            "brier": 0.0,
            "auc": 1.0,
            "rows": len(rows),
            "is_oracle": True,
        }
    )
    return baseline_rows


def adjudicate_output_label_readiness(
    config: OutputLabelReadinessConfig,
    baseline_rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    positive_rate: float,
    flipping_masks: int,
    distinct_block_label_signatures: int,
) -> dict[str, Any]:
    metrics = {str(row["baseline"]): row for row in baseline_rows}
    shortcut_checks = {
        "positive_rate_between_0p10_and_0p90": 0.10 <= positive_rate <= 0.90,
        "at_least_two_masks_flip_across_structures": flipping_masks >= 2,
        "at_least_two_block_label_signatures": distinct_block_label_signatures >= 2,
        "active_block_marginal_accuracy_below_0p95": (
            float(metrics["active_block_marginal"]["accuracy"]) < 0.95
        ),
        "mask_weight_marginal_accuracy_below_0p95": (
            float(metrics["mask_weight_marginal"]["accuracy"]) < 0.95
        ),
        "mask_identity_marginal_accuracy_below_0p98": (
            float(metrics["mask_identity_marginal"]["accuracy"]) < 0.98
        ),
        "block_mask_additive_accuracy_below_0p98": (
            float(metrics["block_mask_additive"]["accuracy"]) < 0.98
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_output_label_readiness_protocol_invalid"
        next_action = {
            "action": "repair E12 source validation or label construction",
            "training": False,
            "remote_scale": False,
        }
    elif all(shortcut_checks.values()):
        status = "pass"
        decision = "innovation2_output_label_interaction_ready"
        next_action = {
            "action": "expand to a larger structure family before neural training",
            "required_controls": [
                "active-block marginal",
                "mask-identity marginal",
                "block+mask additive",
                "label shuffle",
            ],
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_output_label_shortcut_dominated"
        next_action = {
            "action": "redesign structure family to break block and mask shortcuts",
            "preferred_change": (
                "vary inactive context or non-contiguous active geometry while preserving r7"
            ),
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "shortcut_checks": shortcut_checks,
        "positive_rate": positive_rate,
        "flipping_masks": flipping_masks,
        "distinct_block_label_signatures": distinct_block_label_signatures,
        "baseline_accuracies": {
            key: float(row["accuracy"])
            for key, row in metrics.items()
            if key != "direct_gf2_kernel_oracle"
        },
        "claim_scope": (
            "deterministic structure-mask label and marginal-shortcut readiness; "
            "not a neural result or cryptanalytic improvement claim"
        ),
        "next_action": next_action,
    }


def _marginal_predictions(labels: np.ndarray, groups: list[str]) -> np.ndarray:
    means = {
        group: float(labels[[index for index, value in enumerate(groups) if value == group]].mean())
        for group in sorted(set(groups))
    }
    return np.asarray([means[group] for group in groups], dtype=np.float64)


def _crossfit_additive_predictions(
    labels: np.ndarray,
    fields: tuple[list[str], ...],
    *,
    alpha: float,
) -> np.ndarray:
    matrices = [np.ones((len(labels), 1), dtype=np.float64)]
    for field in fields:
        categories = sorted(set(field))
        matrix = np.zeros((len(labels), len(categories)), dtype=np.float64)
        mapping = {category: index for index, category in enumerate(categories)}
        for row_index, category in enumerate(field):
            matrix[row_index, mapping[category]] = 1.0
        matrices.append(matrix)
    design = np.concatenate(matrices, axis=1)
    predictions = np.zeros(len(labels), dtype=np.float64)
    for held_out in range(len(labels)):
        keep = np.arange(len(labels)) != held_out
        train_x = design[keep]
        train_y = labels[keep]
        penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            train_x.T @ train_x + penalty,
            train_x.T @ train_y,
        )
        predictions[held_out] = float(design[held_out] @ coefficients)
    return np.clip(predictions, 0.0, 1.0)


def _metric_row(
    config: OutputLabelReadinessConfig,
    key: str,
    label: str,
    predictions: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted_labels = (predictions >= 0.5).astype(np.uint8)
    return {
        "run_id": config.run_id,
        "baseline": key,
        "baseline_label": label,
        "accuracy": float(np.mean(predicted_labels == target)),
        "brier": float(np.mean((predictions - target) ** 2)),
        "auc": float(binary_auc(target, predictions)),
        "rows": len(target),
        "is_oracle": False,
    }
