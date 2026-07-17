from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_output_label_readiness import (
    bounded_span,
    matched_negative_masks,
)
from blockcipher_nd.training.metrics import binary_auc


EXPECTED_SOURCE_DECISION = "innovation2_inactive_context_kernel_diversity_ready"
EXPECTED_SOURCE_TASK = "innovation2_present_r7_inactive_context_kernel_diversity"
CONTEXT_IDS = tuple(range(16))


@dataclass(frozen=True)
class ContextLabelReadinessConfig:
    run_id: str
    seed: int = 0
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")


def run_context_label_readiness_audit(
    config: ContextLabelReadinessConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_basis_rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_checks = validate_source(source_gate, source_metadata, source_basis_rows)
    kernels, contexts = kernel_bases_and_contexts_from_rows(source_basis_rows)
    positive_masks = tuple(
        sorted({vector for basis in kernels.values() for vector in basis})
    )
    kernel_spans = {
        context_id: bounded_span(basis, max_dimension=16)
        for context_id, basis in kernels.items()
    }
    controls = matched_negative_masks(
        positive_masks,
        excluded_masks=set().union(*kernel_spans.values()),
        seed=config.seed + 1000,
    )
    candidate_masks = positive_masks + controls
    label_rows = build_context_label_rows(
        config,
        contexts=contexts,
        candidate_masks=candidate_masks,
        positive_masks=set(positive_masks),
        kernel_spans=kernel_spans,
    )
    baseline_rows = evaluate_context_baselines(config, label_rows)
    labels = np.asarray(
        [int(row["balanced_label"]) for row in label_rows], dtype=np.uint8
    )
    mask_labels: dict[int, set[int]] = {}
    context_signatures: set[str] = set()
    for row in label_rows:
        mask_labels.setdefault(int(row["mask_value"]), set()).add(
            int(row["balanced_label"])
        )
    for context_id in CONTEXT_IDS:
        context_signatures.add(
            "".join(
                str(int(row["balanced_label"]))
                for row in label_rows
                if int(row["context_id"]) == context_id
            )
        )
    flipping_masks = sum(values == {0, 1} for values in mask_labels.values())
    readiness = {
        **source_checks,
        "source_basis_union_has_nine_masks": len(positive_masks) == 9,
        "candidate_masks_unique": len(candidate_masks) == len(set(candidate_masks)),
        "matched_controls_equal_positive_candidates": len(controls)
        == len(positive_masks),
        "all_controls_outside_every_source_kernel": all(
            mask not in span for mask in controls for span in kernel_spans.values()
        ),
        "complete_context_mask_grid": len(label_rows)
        == len(CONTEXT_IDS) * len(candidate_masks),
        "labels_have_both_classes": set(int(value) for value in labels) == {0, 1},
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in baseline_rows
            for key in ("accuracy", "brier", "auc")
        ),
    }
    gate = adjudicate_context_label_readiness(
        config,
        baseline_rows,
        readiness,
        positive_rate=float(labels.mean()),
        flipping_masks=flipping_masks,
        distinct_context_label_signatures=len(context_signatures),
    )
    return {
        "rows": baseline_rows,
        "label_rows": label_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_context_mask_output_label_readiness",
            "source_run_id": source_gate.get("run_id"),
            "contexts": len(CONTEXT_IDS),
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
        "context_id",
        "fixed_plaintext",
        "basis_index",
        "vector_hex",
        "vector_weight",
    }
    source_contexts = {int(row.get("context_id", -1)) for row in source_basis_rows}
    metadata_contexts = source_metadata.get("contexts")
    return {
        "source_gate_passed_context_diversity": (
            source_gate.get("status") == "pass"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e16_context_diversity": (
            source_metadata.get("task") == EXPECTED_SOURCE_TASK
            and source_metadata.get("training_performed") is False
        ),
        "source_basis_rows_complete": bool(source_basis_rows)
        and all(required_columns.issubset(row) for row in source_basis_rows),
        "source_has_sixteen_frozen_contexts": source_contexts == set(CONTEXT_IDS),
        "source_metadata_has_sixteen_contexts": isinstance(metadata_contexts, list)
        and len(metadata_contexts) == len(CONTEXT_IDS),
    }


def kernel_bases_and_contexts_from_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[int, tuple[int, ...]], dict[int, int]]:
    grouped: dict[int, list[tuple[int, int]]] = {
        context_id: [] for context_id in CONTEXT_IDS
    }
    contexts: dict[int, int] = {}
    for row in rows:
        context_id = int(row["context_id"])
        if context_id not in grouped:
            raise ValueError(f"unexpected context_id: {context_id}")
        fixed_plaintext = int(row["fixed_plaintext"], 16)
        if fixed_plaintext >> 48:
            raise ValueError("fixed plaintext context must fit in the inactive low 48 bits")
        if context_id in contexts and contexts[context_id] != fixed_plaintext:
            raise ValueError(f"inconsistent fixed plaintext for context {context_id}")
        contexts[context_id] = fixed_plaintext
        grouped[context_id].append(
            (int(row["basis_index"]), int(row["vector_hex"], 16))
        )
    bases: dict[int, tuple[int, ...]] = {}
    for context_id, values in grouped.items():
        ordered = tuple(vector for _, vector in sorted(values))
        if not ordered:
            raise ValueError(f"missing basis rows for context {context_id}")
        bases[context_id] = ordered
    if contexts.get(0) != 0 or len(set(contexts.values())) != len(CONTEXT_IDS):
        raise ValueError("contexts must be unique and context 0 must be all-zero")
    return bases, contexts


def build_context_label_rows(
    config: ContextLabelReadinessConfig,
    *,
    contexts: dict[int, int],
    candidate_masks: tuple[int, ...],
    positive_masks: set[int],
    kernel_spans: dict[int, set[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context_id in CONTEXT_IDS:
        context_value = contexts[context_id]
        for mask_index, mask in enumerate(candidate_masks):
            rows.append(
                {
                    "run_id": config.run_id,
                    "context_id": context_id,
                    "fixed_plaintext": f"0x{context_value:016X}",
                    "context_value": context_value,
                    "context_weight": context_value.bit_count(),
                    "mask_index": mask_index,
                    "mask_hex": f"0x{mask:016X}",
                    "mask_value": mask,
                    "mask_weight": mask.bit_count(),
                    "mask_role": (
                        "kernel_basis_union" if mask in positive_masks else "matched_control"
                    ),
                    "balanced_label": int(mask in kernel_spans[context_id]),
                }
            )
    return rows


def evaluate_context_baselines(
    config: ContextLabelReadinessConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray([int(row["balanced_label"]) for row in rows], dtype=np.float64)
    context_ids = [str(row["context_id"]) for row in rows]
    context_weights = [str(row["context_weight"]) for row in rows]
    mask_ids = [str(row["mask_hex"]) for row in rows]
    mask_weights = [str(row["mask_weight"]) for row in rows]
    context_values = [int(row["context_value"]) for row in rows]
    mask_values = [int(row["mask_value"]) for row in rows]

    global_predictions = np.full(len(rows), labels.mean(), dtype=np.float64)
    context_predictions = _marginal_predictions(labels, context_ids)
    context_weight_predictions = _marginal_predictions(labels, context_weights)
    mask_predictions = _marginal_predictions(labels, mask_ids)
    mask_weight_predictions = _marginal_predictions(labels, mask_weights)
    additive_design = _categorical_additive_design((context_ids, mask_ids))
    additive_predictions = ridge_loocv_predictions(
        additive_design, labels, alpha=config.ridge_alpha
    )
    bitwise_design = _bitwise_design(context_values, mask_values)
    bitwise_predictions = ridge_loocv_predictions(
        bitwise_design, labels, alpha=config.ridge_alpha
    )
    rng = np.random.default_rng(config.seed + 5327)
    shuffled_labels = labels[rng.permutation(len(labels))]
    shuffled_predictions = ridge_loocv_predictions(
        bitwise_design, shuffled_labels, alpha=config.ridge_alpha
    )
    candidates = (
        ("global_rate", "全局正例率", global_predictions, labels),
        ("context_identity_marginal", "context身份边际", context_predictions, labels),
        (
            "context_weight_marginal",
            "context汉明重量边际",
            context_weight_predictions,
            labels,
        ),
        ("mask_identity_marginal", "mask身份边际", mask_predictions, labels),
        ("mask_weight_marginal", "mask汉明重量边际", mask_weight_predictions, labels),
        (
            "context_mask_additive",
            "context+mask身份加性",
            additive_predictions,
            labels,
        ),
        (
            "context_mask_bitwise_linear",
            "48+64位模式线性",
            bitwise_predictions,
            labels,
        ),
        (
            "label_shuffle_bitwise_linear",
            "标签打乱位模式线性",
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


def adjudicate_context_label_readiness(
    config: ContextLabelReadinessConfig,
    baseline_rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    positive_rate: float,
    flipping_masks: int,
    distinct_context_label_signatures: int,
) -> dict[str, Any]:
    metrics = {str(row["baseline"]): row for row in baseline_rows}
    shortcut_checks = {
        "positive_rate_between_0p10_and_0p90": 0.10 <= positive_rate <= 0.90,
        "at_least_three_masks_flip_across_contexts": flipping_masks >= 3,
        "at_least_four_context_label_signatures": (
            distinct_context_label_signatures >= 4
        ),
        "context_identity_marginal_accuracy_below_0p95": (
            float(metrics["context_identity_marginal"]["accuracy"]) < 0.95
        ),
        "context_weight_marginal_accuracy_below_0p95": (
            float(metrics["context_weight_marginal"]["accuracy"]) < 0.95
        ),
        "mask_identity_marginal_accuracy_below_0p98": (
            float(metrics["mask_identity_marginal"]["accuracy"]) < 0.98
        ),
        "mask_weight_marginal_accuracy_below_0p95": (
            float(metrics["mask_weight_marginal"]["accuracy"]) < 0.95
        ),
        "context_mask_additive_accuracy_below_0p98": (
            float(metrics["context_mask_additive"]["accuracy"]) < 0.98
        ),
        "context_mask_bitwise_linear_accuracy_below_0p95": (
            float(metrics["context_mask_bitwise_linear"]["accuracy"]) < 0.95
        ),
        "context_identity_marginal_auc_below_0p95": (
            float(metrics["context_identity_marginal"]["auc"]) < 0.95
        ),
        "context_weight_marginal_auc_below_0p95": (
            float(metrics["context_weight_marginal"]["auc"]) < 0.95
        ),
        "mask_identity_marginal_auc_below_0p95": (
            float(metrics["mask_identity_marginal"]["auc"]) < 0.95
        ),
        "mask_weight_marginal_auc_below_0p95": (
            float(metrics["mask_weight_marginal"]["auc"]) < 0.95
        ),
        "context_mask_additive_auc_below_0p95": (
            float(metrics["context_mask_additive"]["auc"]) < 0.95
        ),
        "context_mask_bitwise_linear_auc_below_0p95": (
            float(metrics["context_mask_bitwise_linear"]["auc"]) < 0.95
        ),
    }
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_context_label_readiness_protocol_invalid"
        next_action = {
            "action": "repair E16 source validation or context-mask label construction",
            "training": False,
            "remote_scale": False,
        }
    elif all(shortcut_checks.values()):
        status = "pass"
        decision = "innovation2_context_label_interaction_ready"
        next_action = {
            "action": "validate context-dependent kernel labels on fresh keys",
            "next_adjudication": "E18 fresh-key context kernel stability",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_context_label_shortcut_dominated"
        next_action = {
            "action": "redesign candidate masks or context family before training",
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
        "distinct_context_label_signatures": distinct_context_label_signatures,
        "baseline_accuracies": {
            key: float(row["accuracy"])
            for key, row in metrics.items()
            if key != "direct_gf2_kernel_oracle"
        },
        "baseline_aucs": {
            key: float(row["auc"])
            for key, row in metrics.items()
            if key != "direct_gf2_kernel_oracle"
        },
        "claim_scope": (
            "deterministic context-mask label and shortcut readiness from E16 joint "
            "kernels; not a neural result or cryptanalytic improvement claim"
        ),
        "next_action": next_action,
    }


def ridge_loocv_predictions(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if design.ndim != 2 or design.shape[0] != len(labels):
        raise ValueError("design must have one row per label")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    solved_design = np.linalg.solve(design.T @ design + penalty, design.T)
    fitted = design @ (solved_design @ labels)
    leverage = np.sum(design * solved_design.T, axis=1)
    denominator = 1.0 - leverage
    if np.any(denominator <= 1e-9):
        raise ValueError("ridge LOOCV leverage is too close to one")
    predictions = labels - (labels - fitted) / denominator
    return np.clip(predictions, 0.0, 1.0)


def _marginal_predictions(labels: np.ndarray, groups: list[str]) -> np.ndarray:
    means = {
        group: float(
            labels[
                [index for index, value in enumerate(groups) if value == group]
            ].mean()
        )
        for group in sorted(set(groups))
    }
    return np.asarray([means[group] for group in groups], dtype=np.float64)


def _categorical_additive_design(fields: tuple[list[str], ...]) -> np.ndarray:
    rows = len(fields[0])
    matrices = [np.ones((rows, 1), dtype=np.float64)]
    for field in fields:
        if len(field) != rows:
            raise ValueError("categorical fields must have equal length")
        categories = sorted(set(field))
        mapping = {category: index for index, category in enumerate(categories)}
        matrix = np.zeros((rows, len(categories)), dtype=np.float64)
        for row_index, category in enumerate(field):
            matrix[row_index, mapping[category]] = 1.0
        matrices.append(matrix)
    return np.concatenate(matrices, axis=1)


def _bitwise_design(
    context_values: list[int],
    mask_values: list[int],
) -> np.ndarray:
    if len(context_values) != len(mask_values):
        raise ValueError("context and mask vectors must have equal length")
    design = np.ones((len(context_values), 1 + 48 + 64), dtype=np.float64)
    for row_index, (context_value, mask_value) in enumerate(
        zip(context_values, mask_values, strict=True)
    ):
        for bit in range(48):
            design[row_index, 1 + bit] = (context_value >> bit) & 1
        for bit in range(64):
            design[row_index, 1 + 48 + bit] = (mask_value >> bit) & 1
    return design


def _metric_row(
    config: ContextLabelReadinessConfig,
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
