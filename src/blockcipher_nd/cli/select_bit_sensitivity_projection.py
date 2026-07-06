from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
from blockcipher_nd.training.metrics import binary_auc


DEFAULT_TOP_K = 64
DEFAULT_GROUP_SIZE = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select train-only bit-sensitivity projection axes from aligned "
            "features and trail-position frozen score artifacts."
        )
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--control-artifact", required=True, type=Path)
    parser.add_argument("--anchor-artifact", required=True, type=Path)
    parser.add_argument("--output-mask", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--top-groups", type=int, default=None)
    parser.add_argument("--selection-split", default="train")
    return parser.parse_args(argv)


def select_bit_sensitivity_projection(
    *,
    features_path: Path,
    control_artifact_dir: Path,
    anchor_artifact_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    group_size: int = DEFAULT_GROUP_SIZE,
    top_groups: int | None = None,
    selection_split: str = "train",
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if top_groups is not None and top_groups <= 0:
        raise ValueError("top_groups must be positive")
    if selection_split != "train":
        raise ValueError("bit-sensitivity projection must be selected on the train split")

    control = load_score_artifact(control_artifact_dir)
    anchor = load_score_artifact(anchor_artifact_dir)
    _validate_alignment(control, anchor)
    features = _load_feature_matrix(features_path)
    if features.shape[0] != len(anchor.labels):
        raise ValueError(
            "feature row count must match score artifact rows: "
            f"{features.shape[0]} != {len(anchor.labels)}"
        )

    labels = anchor.labels.astype(np.float32, copy=False)
    stats = _axis_statistics(features, labels, control.probabilities, anchor.probabilities)
    if group_size == 1:
        selected, selected_axes, selected_axis_scores, selected_groups = _select_axes(stats, top_k)
        projection_unit = "axis"
    else:
        selected, selected_axes, selected_axis_scores, selected_groups = _select_groups(
            features,
            labels,
            control.probabilities,
            anchor.probabilities,
            group_size=group_size,
            top_groups=top_groups or max(1, top_k // group_size),
        )
        projection_unit = "contiguous_axis_group"
    mask = {
        "status": "pass",
        "selection_split": selection_split,
        "projection_unit": projection_unit,
        "feature_source": str(features_path),
        "control_artifact": str(control_artifact_dir),
        "anchor_artifact": str(anchor_artifact_dir),
        "selected_axes": selected_axes,
        "axis_scores": selected_axis_scores,
        "selected_groups": selected_groups,
        "selected_axis_count": len(selected_axes),
        "selected_group_count": len(selected_groups),
        "top_k": int(top_k),
        "group_size": int(group_size),
        "top_groups": int(top_groups) if top_groups is not None else None,
        "feature_shape": [int(value) for value in features.shape],
        "protocol": _protocol_metadata(anchor.metadata),
        "claim_scope": (
            "train-only bit-sensitivity mask selection for a future local projection screen; "
            "not a trained model result, not a remote-launch gate, and not formal SPN/PRESENT evidence"
        ),
    }
    return {
        "status": "pass",
        "decision": "projection_mask_ready_for_local_screen",
        "action": "use_mask_only_in_a_local_same-protocol_projection_screen_with_controls",
        "mask": mask,
        "axis_scores": selected_axis_scores,
        "summary": _summary_stats(
            stats,
            selected,
            selected_axis_count=len(selected_axes),
            selected_group_count=len(selected_groups),
        ),
        "guardrails": [
            "selection_split_must_be_train",
            "do_not_select_mask_on_validation",
            "group_selection_must_be_train_only",
            "do_not_claim_model_gain_from_axis_selection_only",
            "compare_future_projection_against_same_input_global_control",
            "require_active_nibble_and_input_difference_mismatch_controls",
        ],
        "claim_scope": mask["claim_scope"],
    }


def _validate_alignment(control: Any, anchor: Any) -> None:
    if not np.array_equal(control.labels, anchor.labels):
        raise ValueError("control and anchor labels differ")
    if not np.array_equal(control.sample_ids, anchor.sample_ids):
        raise ValueError("control and anchor sample_ids differ")
    for field in (
        "cipher",
        "rounds",
        "validation_samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "validation_key",
    ):
        if control.metadata.get(field) != anchor.metadata.get(field):
            raise ValueError(f"score artifact protocol field differs: {field}")


def _load_feature_matrix(path: Path) -> np.ndarray:
    features = np.load(path)
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions: rows x axes")
    return features.reshape(features.shape[0], -1).astype(np.float32, copy=False)


def _axis_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    control_probabilities: np.ndarray,
    anchor_probabilities: np.ndarray,
) -> np.ndarray:
    control_correct = (control_probabilities >= 0.5) == labels
    anchor_correct = (anchor_probabilities >= 0.5) == labels
    helpful = anchor_correct & ~control_correct
    harmful = control_correct & ~anchor_correct
    anchor_wrong = ~anchor_correct
    dtype = [
        ("axis", np.int64),
        ("score", np.float64),
        ("class_auc", np.float64),
        ("class_delta", np.float64),
        ("residual_delta", np.float64),
        ("anchor_error_delta", np.float64),
        ("helpful_mean", np.float64),
        ("harmful_mean", np.float64),
        ("positive_mean", np.float64),
        ("negative_mean", np.float64),
    ]
    rows = np.zeros(features.shape[1], dtype=dtype)
    positive = labels >= 0.5
    negative = ~positive
    for axis in range(features.shape[1]):
        column = features[:, axis].astype(np.float64, copy=False)
        class_auc = binary_auc(labels, column)
        class_auc = max(class_auc, 1.0 - class_auc)
        positive_mean = _masked_mean(column, positive)
        negative_mean = _masked_mean(column, negative)
        helpful_mean = _masked_mean(column, helpful)
        harmful_mean = _masked_mean(column, harmful)
        anchor_correct_mean = _masked_mean(column, anchor_correct)
        anchor_wrong_mean = _masked_mean(column, anchor_wrong)
        class_delta = abs(positive_mean - negative_mean)
        residual_delta = abs(helpful_mean - harmful_mean)
        anchor_error_delta = abs(anchor_correct_mean - anchor_wrong_mean)
        score = residual_delta * 2.0 + anchor_error_delta + class_delta + max(0.0, class_auc - 0.5)
        rows[axis] = (
            axis,
            score,
            class_auc,
            class_delta,
            residual_delta,
            anchor_error_delta,
            helpful_mean,
            harmful_mean,
            positive_mean,
            negative_mean,
        )
    return rows


def _select_axes(
    stats: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    order = np.lexsort(
        (
            stats["axis"],
            -stats["class_delta"],
            -stats["anchor_error_delta"],
            -stats["residual_delta"],
            -stats["score"],
        )
    )
    selected = stats[order[: min(top_k, len(order))]]
    selected_axes = [int(axis) for axis in selected["axis"]]
    selected_axis_scores = [_axis_row_to_dict(row) for row in selected]
    return selected, selected_axes, selected_axis_scores, []


def _select_groups(
    features: np.ndarray,
    labels: np.ndarray,
    control_probabilities: np.ndarray,
    anchor_probabilities: np.ndarray,
    *,
    group_size: int,
    top_groups: int,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    for group_id, start in enumerate(range(0, features.shape[1], group_size)):
        axes = list(range(start, min(start + group_size, features.shape[1])))
        group_features = features[:, axes].mean(axis=1, keepdims=True)
        stats = _axis_statistics(group_features, labels, control_probabilities, anchor_probabilities)
        row = _axis_row_to_dict(stats[0])
        row.update(
            {
                "group_id": group_id,
                "axis": group_id,
                "axes": axes,
                "axis_count": len(axes),
            }
        )
        groups.append(row)
    groups.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["residual_delta"]),
            -float(row["anchor_error_delta"]),
            -float(row["class_delta"]),
            int(row["group_id"]),
        )
    )
    selected_groups = groups[: min(top_groups, len(groups))]
    selected_axes = sorted({int(axis) for group in selected_groups for axis in group["axes"]})
    selected_axis_scores = [
        {
            **group,
            "axis": int(axis),
            "source_group_id": int(group["group_id"]),
        }
        for group in selected_groups
        for axis in group["axes"]
    ]
    selected = _groups_to_structured(selected_groups)
    return selected, selected_axes, selected_axis_scores, selected_groups


def _groups_to_structured(groups: list[dict[str, Any]]) -> np.ndarray:
    dtype = [
        ("score", np.float64),
        ("class_delta", np.float64),
        ("residual_delta", np.float64),
    ]
    rows = np.zeros(len(groups), dtype=dtype)
    for idx, group in enumerate(groups):
        rows[idx] = (
            float(group["score"]),
            float(group["class_delta"]),
            float(group["residual_delta"]),
        )
    return rows


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].mean())


def _axis_row_to_dict(row: np.void) -> dict[str, Any]:
    return {
        "axis": int(row["axis"]),
        "score": float(row["score"]),
        "class_auc": float(row["class_auc"]),
        "class_delta": float(row["class_delta"]),
        "residual_delta": float(row["residual_delta"]),
        "anchor_error_delta": float(row["anchor_error_delta"]),
        "helpful_mean": float(row["helpful_mean"]),
        "harmful_mean": float(row["harmful_mean"]),
        "positive_mean": float(row["positive_mean"]),
        "negative_mean": float(row["negative_mean"]),
    }


def _summary_stats(
    stats: np.ndarray,
    selected: np.ndarray,
    *,
    selected_axis_count: int,
    selected_group_count: int,
) -> dict[str, Any]:
    return {
        "axis_count": int(len(stats)),
        "selected_axis_count": int(selected_axis_count),
        "selected_group_count": int(selected_group_count),
        "max_score": float(stats["score"].max()) if len(stats) else 0.0,
        "min_selected_score": float(selected["score"].min()) if len(selected) else 0.0,
        "max_selected_score": float(selected["score"].max()) if len(selected) else 0.0,
        "mean_selected_residual_delta": float(selected["residual_delta"].mean()) if len(selected) else 0.0,
        "mean_selected_class_delta": float(selected["class_delta"].mean()) if len(selected) else 0.0,
    }


def _protocol_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "cipher",
        "rounds",
        "validation_samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "validation_key",
    ]
    return {field: metadata.get(field) for field in fields if field in metadata}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = select_bit_sensitivity_projection(
        features_path=args.features,
        control_artifact_dir=args.control_artifact,
        anchor_artifact_dir=args.anchor_artifact,
        top_k=args.top_k,
        group_size=args.group_size,
        top_groups=args.top_groups,
        selection_split=args.selection_split,
    )
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_mask.write_text(json.dumps(report["mask"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
