from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


DEFAULT_METRICS = ("accuracy", "auc", "loss")
MODEL_COLORS = (
    "#2563EB",  # blue
    "#DC2626",  # red
    "#059669",  # green
    "#7C3AED",  # violet
    "#D97706",  # amber
    "#0891B2",  # cyan
)
PLOT_LINESTYLES = {
    "train": (0, (2.0, 2.0)),
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
            train_key = "train_eval_loss" if metric == "loss" else f"train_{metric}"
            val_key = f"val_{metric}"
            for split, key in (("train", train_key), ("val", val_key)):
                points = _finite_history_points(row.get("history", []), key)
                if points:
                    series.append(
                        {
                            "metric": metric,
                            "split": split,
                            "label": label,
                            "run_index": run_index,
                            "model": row.get("model", row.get("selected_model", "")),
                            "points": points,
                        }
                    )
    return series


def _finite_history_points(history: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in history:
        if key not in item or item.get(key) is None:
            continue
        epoch = float(item["epoch"])
        value = float(item[key])
        if math.isfinite(epoch) and math.isfinite(value):
            points.append((epoch, value))
    return points


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
        fig = plt.figure(
            figsize=(10.8, max(4.6, 2.08 * len(metrics) + 1.35)),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            len(metrics) + 1,
            1,
            height_ratios=[1.0] * len(metrics) + [0.56],
            hspace=0.28,
        )
        axes = [
            fig.add_subplot(grid[index, 0], sharex=None if index == 0 else fig.axes[0])
            for index in range(len(metrics))
        ]
        summary_axis = fig.add_subplot(grid[-1, 0])
        fig.subplots_adjust(
            left=0.085,
            right=0.84,
            top=0.88,
            bottom=0.075,
        )
        fig.suptitle(_display_title(title), fontsize=14.6, fontweight="bold", x=0.085, y=0.965, ha="left")
        fig.text(
            0.085,
            0.915,
            _plot_subtitle(series),
            fontsize=9.3,
            color="#526070",
            ha="left",
            va="top",
        )
        for axis, metric in zip(axes, metrics):
            render_metric_panel(
                axis,
                metric,
                [item for item in series if item["metric"] == metric],
            )
        axes[-1].set_xlabel("Epoch")
        _render_summary_strip(summary_axis, series)
        _render_style_legend(fig)
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
    axis.set_title(_metric_title(metric), loc="left", fontweight="bold", fontsize=11.2, pad=7)
    axis.set_ylabel(_metric_ylabel(metric))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    axis.grid(False, axis="x")
    _format_y_axis(axis, metric, min_value, max_value)
    endpoint_targets = _validation_endpoint_targets(axis, series)
    for item in series:
        epochs = [point[0] for point in item["points"]]
        values = [point[1] for point in item["points"]]
        is_validation = item["split"] == "val"
        color = _series_color(item) if is_validation else "#94A3B8"
        axis.plot(
            epochs,
            values,
            label=f"{_compact_label(item)} {'validation' if is_validation else 'train'}",
            color=color,
            linestyle=PLOT_LINESTYLES.get(item["split"], "-"),
            linewidth=2.55 if is_validation else 1.05,
            marker="o" if is_validation and len(values) <= 16 else None,
            markersize=3.7,
            markeredgewidth=0,
            alpha=0.98 if is_validation else 0.32,
            solid_capstyle="round",
            dash_capstyle="round",
        )
        _annotate_validation_best(axis, metric, item, epochs, values, _series_color(item))
        _label_validation_endpoint(
            axis,
            item,
            epochs,
            values,
            color,
            endpoint_targets.get(int(item.get("run_index", 1))),
        )
    axis.set_xlim(max(0.0, min_epoch - 0.25), max_epoch + 1.05)
    axis.margins(y=0.08)


def _plot_rc_params() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9.6,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0F172A",
        "text.color": "#0F172A",
        "legend.fontsize": 8.6,
        "savefig.facecolor": "#ffffff",
        "svg.fonttype": "none",
    }


def _format_y_axis(axis, metric: str, min_value: float, max_value: float) -> None:
    if metric in {"accuracy", "auc"}:
        lower = max(0.0, min(0.5, min_value - 0.025))
        upper = min(1.0, max(0.75, max_value + 0.025))
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.axhline(0.5, color="#94A3B8", linewidth=0.9, linestyle=(0, (1.5, 2.5)), alpha=0.78)
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
    axis.scatter([best_epoch], [best_value], s=38, color=color, edgecolor="white", linewidth=1.05, zorder=4)


def _label_validation_endpoint(
    axis,
    item: dict[str, Any],
    epochs: list[float],
    values: list[float],
    color: str,
    target_y: float | None,
) -> None:
    if item["split"] != "val" or not values or target_y is None:
        return
    label_position = (epochs[-1] + 0.12, target_y)
    axis.annotate(
        "",
        xy=(epochs[-1], values[-1]),
        xytext=label_position,
        textcoords="data",
        clip_on=False,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.8,
            "alpha": 0.65,
        },
    )
    axis.annotate(
        _compact_label(item),
        xy=label_position,
        color=color,
        fontsize=8.1,
        fontweight="bold",
        va="center",
        ha="left",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.12,rounding_size=0.04",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )


def _validation_endpoint_targets(
    axis,
    series: list[dict[str, Any]],
) -> dict[int, float]:
    endpoints = [
        item
        for item in series
        if item["split"] == "val" and item["points"]
    ]
    ordered = sorted(
        endpoints,
        key=lambda item: (
            item["points"][-1][1],
            int(item.get("run_index", 1)),
        ),
    )
    if not ordered:
        return {}

    lower, upper = axis.get_ylim()
    span = max(upper - lower, 1e-12)
    interior_lower = lower + span * 0.12
    interior_upper = upper - span * 0.12
    if len(ordered) == 1:
        item = ordered[0]
        value = item["points"][-1][1]
        return {
            int(item.get("run_index", 1)): min(
                max(value, interior_lower),
                interior_upper,
            )
        }

    step = (interior_upper - interior_lower) / (len(ordered) - 1)
    return {
        int(item.get("run_index", 1)): interior_lower + rank * step
        for rank, item in enumerate(ordered)
    }


def _render_style_legend(fig) -> None:
    split_items = [
        Line2D([0], [0], color="#475569", lw=2.1, linestyle="-", label="validation"),
        Line2D([0], [0], color="#475569", lw=1.25, linestyle=PLOT_LINESTYLES["train"], alpha=0.45, label="train"),
    ]
    fig.legend(
        handles=split_items,
        loc="upper left",
        bbox_to_anchor=(0.085, 0.898),
        frameon=False,
        ncol=2,
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.35,
    )


def _render_summary_strip(axis, series: list[dict[str, Any]]) -> None:
    axis.axis("off")
    rows = _summary_rows(series)
    if not rows:
        return
    axis.text(
        0.0,
        0.92,
        "Validation summary",
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    columns = ["Model", "Final Acc", "Best Acc", "Final AUC", "Best AUC", "Best Loss"]
    table_rows = [
        [
            row["label"],
            _format_metric_value("accuracy", row.get("accuracy_final")) if row.get("accuracy_final") is not None else "-",
            _format_best_cell("accuracy", row),
            _format_metric_value("auc", row.get("auc_final")) if row.get("auc_final") is not None else "-",
            _format_best_cell("auc", row),
            _format_best_cell("loss", row),
        ]
        for row in rows[:4]
    ]
    table = axis.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.72],
        colWidths=[0.32, 0.13, 0.15, 0.13, 0.15, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_linewidth(0.5)
        cell.PAD = 0.08
        if row_index == 0:
            cell.set_facecolor("#F8FAFC")
            cell.set_text_props(color="#334155", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_index % 2 else "#FBFDFF")
            cell.set_text_props(color="#0F172A")


def _summary_rows(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for item in series:
        if item["split"] != "val" or not item["points"]:
            continue
        key = (int(item.get("run_index", 0)), _compact_label(item))
        row = grouped.setdefault(key, {"label": key[1], "run_index": key[0]})
        values = [point[1] for point in item["points"]]
        epochs = [point[0] for point in item["points"]]
        metric = item["metric"]
        best_index = min(range(len(values)), key=values.__getitem__) if metric == "loss" else max(
            range(len(values)),
            key=values.__getitem__,
        )
        row[f"{metric}_final"] = values[-1]
        row[f"{metric}_best"] = values[best_index]
        row[f"{metric}_best_epoch"] = epochs[best_index]
    return [grouped[key] for key in sorted(grouped)]


def _format_best_cell(metric: str, row: dict[str, Any]) -> str:
    best = row.get(f"{metric}_best")
    epoch = row.get(f"{metric}_best_epoch")
    if best is None or epoch is None:
        return "-"
    return f"{_format_metric_value(metric, best)} @ e{format_epoch_tick(float(epoch))}"


def _metric_title(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_ylabel(metric: str) -> str:
    if metric in {"accuracy", "auc"}:
        return f"{_metric_title(metric)} (%)"
    return _metric_title(metric)


def _display_title(title: str) -> str:
    stem = title[:-6] if title.endswith(".jsonl") else title
    title_aliases = {
        "innovation1_spn_present_nibble_paligned_mcnd_r7_64k_screen_gpu1_20260625": (
            "Innovation 1 PRESENT r7 Medium Screen"
        ),
        "zhang_wang_present_r7_262k_official_cyclic_20260624": "Zhang-Wang PRESENT r7 262k Diagnostic",
    }
    if stem in title_aliases:
        return title_aliases[stem]
    cleaned = stem.replace("_", " ").replace("-", " ")
    cleaned = " ".join(part for part in cleaned.split() if not part.startswith("gpu") and not part.isdigit())
    return "\n".join(textwrap.wrap(cleaned, width=78, max_lines=2, placeholder="..."))


def _plot_subtitle(series: list[dict[str, Any]]) -> str:
    rows = _summary_rows(series)
    metric_names = ", ".join(_metric_title(metric) for metric in dict.fromkeys(item["metric"] for item in series))
    if not metric_names:
        metric_names = ", ".join(_metric_title(metric) for metric in DEFAULT_METRICS)
    model_word = "model" if len(rows) == 1 else "models"
    return f"{len(rows)} {model_word} compared | validation emphasized | {metric_names}"


def _series_color(item: dict[str, Any]) -> str:
    run_index = int(item.get("run_index", 1))
    return MODEL_COLORS[(run_index - 1) % len(MODEL_COLORS)]


def _compact_label(item: dict[str, Any]) -> str:
    model = str(item.get("model") or item.get("label") or "")
    aliases = {
        "present_zhang_wang_keras_mcnd": "Zhang-Wang MCND",
        "present_nibble_paligned_mcnd": "I1 nibble-P MCND",
        "present_nibble_invp_only_spn_only": "InvP token mixer",
        "present_nibble_invp_topology_residual_spn_only": "true InvP residual",
        "present_nibble_shuffled_p_topology_residual_spn_only": "shuffled P residual",
        "present_nibble_delta_topology_residual_spn_only": "Delta-only residual",
    }
    if model in aliases:
        return aliases[model]
    if ": " in model:
        model = model.split(": ", 1)[1]
    model = model.replace("_", " ")
    words = model.split()
    if len(words) > 4:
        model = " ".join(words[-4:])
    return model or f"run {item.get('run_index', '')}".strip()


def _format_metric_value(metric: str, value: float) -> str:
    if metric in {"accuracy", "auc"}:
        return f"{value * 100:.1f}%"
    return format_value(value)


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
