from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


DEFAULT_METRICS = ("accuracy", "auc", "loss")
PLOT_COLORS = {
    "train": "#2563eb",
    "val": "#dc2626",
}
PLOT_LINESTYLES = {
    "train": "--",
    "val": "-",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "loss": "Loss",
}


def plot_jsonl_training_curves(
    results_path: Path,
    output_path: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    title: str | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_rows(results_path)
    series = training_curve_series(rows, metrics=metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_training_curves_svg(series, output_path, title=title or results_path.name)
    return {
        "rows": len(rows),
        "series": len(series),
        "output": str(output_path),
        "metrics": list(metrics),
    }


def write_history_csv(results_path: Path, output_path: Path) -> dict[str, Any]:
    rows = load_jsonl_rows(results_path)
    records = history_records(rows)
    fieldnames = [
        "run_index",
        "run_label",
        "cipher",
        "model",
        "selected_model",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return {
        "rows": len(records),
        "output": str(output_path),
    }


def training_curve_series(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for run_index, row in enumerate(rows, start=1):
        label = run_label(row, run_index)
        for metric in metrics:
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            for split, key in (("train", train_key), ("val", val_key)):
                points = [
                    (float(item["epoch"]), float(item[key]))
                    for item in row.get("history", [])
                    if key in item and item.get(key) is not None
                ]
                if points:
                    series.append(
                        {
                            "metric": metric,
                            "split": split,
                            "label": label,
                            "points": points,
                        }
                    )
    return series


def history_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_index, row in enumerate(rows, start=1):
        label = run_label(row, run_index)
        for item in row.get("history", []):
            records.append(
                {
                    "run_index": run_index,
                    "run_label": label,
                    "cipher": row.get("cipher", ""),
                    "model": row.get("model", ""),
                    "selected_model": row.get("selected_model", ""),
                    "rounds": row.get("rounds", ""),
                    "seed": row.get("seed", ""),
                    "samples_per_class": row.get("samples_per_class", ""),
                    "pairs_per_sample": row.get("pairs_per_sample", ""),
                    **item,
                }
            )
    return records


def render_training_curves_svg(
    series: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    metrics = list(dict.fromkeys(item["metric"] for item in series))
    if not metrics:
        metrics = list(DEFAULT_METRICS)
    with plt.rc_context(_plot_rc_params()):
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(12, max(3.2, 3.05 * len(metrics))),
            sharex=True,
            constrained_layout=True,
        )
        if len(metrics) == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=15, fontweight="bold")
        for axis, metric in zip(axes, metrics):
            render_metric_panel(
                axis,
                metric,
                [item for item in series if item["metric"] == metric],
            )
        axes[-1].set_xlabel("Epoch")
        handles, labels = _deduplicate_legend(axes)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(4, len(labels)),
                frameon=False,
            )
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)


def render_metric_panel(
    axis,
    metric: str,
    series: list[dict[str, Any]],
) -> None:
    all_points = [point for item in series for point in item["points"]]
    if not all_points:
        axis.set_title(_metric_title(metric), loc="left")
        axis.text(0.5, 0.5, "No history points", transform=axis.transAxes, ha="center", va="center")
        return
    min_epoch = min(point[0] for point in all_points)
    max_epoch = max(point[0] for point in all_points)
    min_value = min(point[1] for point in all_points)
    max_value = max(point[1] for point in all_points)
    axis.set_title(_metric_title(metric), loc="left", fontweight="bold")
    axis.set_ylabel(_metric_ylabel(metric))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(True, axis="y", color="#e2e8f0", linewidth=0.85)
    axis.grid(True, axis="x", color="#f1f5f9", linewidth=0.6)
    _format_y_axis(axis, metric, min_value, max_value)
    for item in series:
        color = PLOT_COLORS.get(item["split"], "#334155")
        epochs = [point[0] for point in item["points"]]
        values = [point[1] for point in item["points"]]
        line_label = f"{_compact_label(item['label'])} {item['split']}"
        axis.plot(
            epochs,
            values,
            label=line_label,
            color=color,
            linestyle=PLOT_LINESTYLES.get(item["split"], "-"),
            linewidth=2.1 if item["split"] == "val" else 1.65,
            marker="o" if len(values) <= 12 else None,
            markersize=3.6,
            alpha=0.95 if item["split"] == "val" else 0.75,
        )
        _annotate_validation_best(axis, metric, item, epochs, values, color)
    axis.set_xlim(max(0.0, min_epoch - 0.25), max_epoch + 0.35)


def _plot_rc_params() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#94a3b8",
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "axes.labelcolor": "#334155",
        "text.color": "#0f172a",
        "legend.fontsize": 9.2,
        "savefig.facecolor": "#ffffff",
        "svg.fonttype": "none",
    }


def _format_y_axis(axis, metric: str, min_value: float, max_value: float) -> None:
    if metric in {"accuracy", "auc"}:
        lower = max(0.0, min(0.45, min_value - 0.03))
        upper = min(1.0, max(0.75, max_value + 0.03))
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.axhline(0.5, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.75)
        return
    padding = max(1e-6, (max_value - min_value) * 0.08)
    axis.set_ylim(min_value - padding, max_value + padding)


def _annotate_validation_best(
    axis,
    metric: str,
    item: dict[str, Any],
    epochs: list[float],
    values: list[float],
    color: str,
) -> None:
    if item["split"] != "val" or not values:
        return
    best_index = min(range(len(values)), key=values.__getitem__) if metric == "loss" else max(
        range(len(values)),
        key=values.__getitem__,
    )
    best_epoch = epochs[best_index]
    best_value = values[best_index]
    axis.scatter([best_epoch], [best_value], s=34, color=color, edgecolor="white", linewidth=0.9, zorder=4)
    axis.annotate(
        f"best {format_value(best_value)} @ {format_epoch_tick(best_epoch)}",
        xy=(best_epoch, best_value),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": color, "alpha": 0.88},
    )


def _deduplicate_legend(axes) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _metric_title(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_ylabel(metric: str) -> str:
    if metric in {"accuracy", "auc"}:
        return f"{_metric_title(metric)} (%)"
    return _metric_title(metric)


def _compact_label(label: str) -> str:
    if ": " in label:
        label = label.split(": ", 1)[1]
    parts = label.split()
    if len(parts) <= 3:
        return label
    return " ".join(parts[-3:])


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.4g}"


def format_epoch_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return format_value(value)


def run_label(row: dict[str, Any], run_index: int) -> str:
    return (
        f"run{run_index}: {row.get('cipher', '')} "
        f"r{row.get('rounds', '')} {row.get('model', row.get('selected_model', ''))} "
        f"seed{row.get('seed', '')}"
    )


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
