from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


INNOVATION_ONE_GROUP_FIELDS = (
    "cipher",
    "structure",
    "model",
    "architecture",
    "architecture_rank",
    "matching_score",
    "literature",
    "rounds",
    "difference_profile",
    "difference_member",
    "difference_source",
    "gate_mode",
    "samples_per_class",
    "pairs_per_sample",
    "key_protocol",
    "key_rotation_interval",
    "sample_structure",
    "feature_encoding",
)
INNOVATION_ONE_METRIC_FIELDS = (
    "accuracy",
    "best_accuracy",
    "calibrated_accuracy",
    "auc",
    "advantage",
    "calibrated_advantage",
    "loss",
)
HPARAM_SUMMARY_FIELDS = [
    "trial_id",
    "cipher",
    "model",
    "rounds",
    "seed",
    "trial_seeds",
    "samples_per_class",
    "pairs_per_sample",
    "difference_profile",
    "difference_member",
    "calibrated_accuracy",
    "calibrated_accuracy_std",
    "accuracy",
    "accuracy_std",
    "auc",
    "auc_std",
    "loss",
    "loss_std",
    "gate_temperature",
    "gate_hidden_bits",
    "gate_activation",
    "gate_dropout",
    "pairwise_pooling",
    "spn_token_dim",
    "spn_mixer_depth",
    "spn_token_mlp_ratio",
    "expert_activation",
    "expert_norm",
    "spn_pooling",
    "expert_dropout",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "config_json",
]
MOE_COMPONENT_FIELDS = [
    "gate_temperature",
    "gate_hidden_bits",
    "gate_activation",
    "gate_dropout",
    "pairwise_pooling",
    "spn_token_dim",
    "spn_mixer_depth",
    "spn_token_mlp_ratio",
    "expert_activation",
    "expert_norm",
    "spn_pooling",
    "expert_dropout",
]


def innovation_one_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_innovation_one_group_value(row, field) for field in INNOVATION_ONE_GROUP_FIELDS)
        groups[key].append(row)

    summary = []
    for key, group_rows in sorted(groups.items()):
        out = dict(zip(INNOVATION_ONE_GROUP_FIELDS, key))
        out["runs"] = len(group_rows)
        for metric in INNOVATION_ONE_METRIC_FIELDS:
            values = [_innovation_one_metric_value(row["metrics"], metric) for row in group_rows]
            out[f"{metric}_mean"] = round(mean(values), 10)
            out[f"{metric}_std"] = round(pstdev(values), 10) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def innovation_one_summary_fields() -> list[str]:
    fieldnames = list(INNOVATION_ONE_GROUP_FIELDS) + ["runs"]
    for metric in INNOVATION_ONE_METRIC_FIELDS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    return fieldnames


def hparam_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = [hparam_summary_row(row) for row in rows]
    summary.sort(key=lambda row: float(row.get("calibrated_accuracy") or 0.0), reverse=True)
    return summary


def hparam_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    metrics_std = row.get("metrics_std", {})
    config = row.get("config", {})
    components = row.get("moe_components", row.get("gate_config", {}))
    result: dict[str, Any] = {
        "trial_id": row.get("trial_id", ""),
        "cipher": row.get("cipher", ""),
        "model": config.get("model", ""),
        "rounds": row.get("rounds", ""),
        "seed": row.get("seed", ""),
        "trial_seeds": ",".join(str(seed) for seed in row.get("trial_seeds", [])),
        "samples_per_class": row.get("samples_per_class", ""),
        "pairs_per_sample": row.get("pairs_per_sample", ""),
        "difference_profile": row.get("difference_profile", ""),
        "difference_member": row.get("difference_member", ""),
        "calibrated_accuracy": metrics.get("calibrated_accuracy", ""),
        "calibrated_accuracy_std": metrics_std.get("calibrated_accuracy", ""),
        "accuracy": metrics.get("accuracy", ""),
        "accuracy_std": metrics_std.get("accuracy", ""),
        "auc": metrics.get("auc", ""),
        "auc_std": metrics_std.get("auc", ""),
        "loss": metrics.get("loss", ""),
        "loss_std": metrics_std.get("loss", ""),
        "learning_rate": config.get("learning_rate", config.get("lr", "")),
        "weight_decay": config.get("weight_decay", ""),
        "optimizer": config.get("optimizer", ""),
        "config_json": json.dumps(config, sort_keys=True),
    }
    for key in MOE_COMPONENT_FIELDS:
        result[key] = components.get(key, config.get(key, ""))
    return result


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _innovation_one_metric_value(metrics: dict[str, Any], metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])
    if metric in {"best_accuracy", "calibrated_accuracy"}:
        return float(metrics["accuracy"])
    if metric == "calibrated_advantage":
        return float(metrics.get("advantage", 2.0 * float(metrics["accuracy"]) - 1.0))
    raise KeyError(metric)


def _innovation_one_group_value(row: dict[str, Any], field: str) -> Any:
    if field == "architecture":
        return row.get("architecture", row["model"])
    if field == "key_protocol":
        return _innovation_one_key_protocol(row)
    return row.get(field, "")


def _innovation_one_key_protocol(row: dict[str, Any]) -> str:
    rotation = row.get("key_rotation_interval", "")
    if rotation not in {"", None}:
        try:
            if int(rotation) > 0:
                return "key_rotating_multi_key"
        except (TypeError, ValueError):
            return "key_rotating_multi_key"

    train_key = row.get("train_key", "")
    validation_key = row.get("validation_key", "")
    if train_key and validation_key and str(train_key).lower() != str(validation_key).lower():
        return "fixed_train_cross_key_validation"
    if train_key or validation_key:
        return "fixed_key"
    return "unspecified"
