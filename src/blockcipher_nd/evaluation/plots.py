from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

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
MODEL_MARKERS = ("o", "s", "^", "D", "X", "P")
PLOT_LINESTYLES = {
    "train": (0, (2.0, 2.0)),
    "val": "-",
}
METRIC_LABELS = {
    "accuracy": "准确率 Accuracy",
    "auc": "AUC",
    "loss": "损失 Loss",
}


def plot_jsonl_training_curves(
    results_path: Path,
    output_path: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    title: str | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl_rows(results_path)
    series = training_curve_series(rows, metrics=metrics)
    if validation_only:
        series = [item for item in series if item["split"] == "val"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_training_curves_svg(series, output_path, title=title or results_path.name)
    return {
        "rows": len(rows),
        "series": len(series),
        "output": str(output_path),
        "metrics": list(metrics),
        "validation_only": validation_only,
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
                            "cipher": row.get("cipher", ""),
                            "rounds": row.get("rounds", ""),
                            "seed": row.get("seed", ""),
                            "runtime_structure_mode": row.get(
                                "runtime_structure_mode", ""
                            ),
                            "sbox_context_mode": row.get("training", {})
                            .get("model_options", {})
                            .get("sbox_context_mode", ""),
                            "samples_per_class": row.get("samples_per_class", ""),
                            "train_structures": row.get("train_structures", ""),
                            "train_keys_per_structure": row.get(
                                "train_keys_per_structure",
                                "",
                            ),
                            "integral_set_size": row.get("integral_set_size", ""),
                            "validation_samples_per_class": row.get(
                                "validation", {}
                            ).get(
                                "samples_per_class",
                                "",
                            ),
                            "pairs_per_sample": row.get("pairs_per_sample", ""),
                            "points": points,
                        }
                    )
    return series


def _finite_history_points(
    history: list[dict[str, Any]], key: str
) -> list[tuple[float, float]]:
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
            figsize=(12.4, max(5.8, 2.25 * len(metrics) + 2.45)),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            len(metrics) + 1,
            1,
            height_ratios=[1.0] * len(metrics) + [0.9],
            hspace=0.3,
        )
        axes = [
            fig.add_subplot(grid[index, 0], sharex=None if index == 0 else fig.axes[0])
            for index in range(len(metrics))
        ]
        summary_axis = fig.add_subplot(grid[-1, 0])
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.745,
            bottom=0.065,
        )
        fig.suptitle(
            _display_title(title),
            fontsize=15.0,
            fontweight="bold",
            x=0.075,
            y=0.975,
            ha="left",
        )
        fig.text(
            0.075,
            0.925,
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
        axes[-1].set_xlabel("训练轮次 Epoch")
        _render_summary_strip(summary_axis, series)
        _render_model_legend(fig, series)
        _render_style_legend(fig, series)
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
        axis.text(
            0.5,
            0.5,
            "No history points",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        return
    min_epoch = min(point[0] for point in all_points)
    max_epoch = max(point[0] for point in all_points)
    min_value = min(point[1] for point in all_points)
    max_value = max(point[1] for point in all_points)
    axis.set_title(
        _metric_title(metric), loc="left", fontweight="bold", fontsize=11.2, pad=7
    )
    axis.set_ylabel(_metric_ylabel(metric))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    axis.grid(False, axis="x")
    _format_y_axis(axis, metric, min_value, max_value)
    for item in series:
        epochs = [point[0] for point in item["points"]]
        values = [point[1] for point in item["points"]]
        is_validation = item["split"] == "val"
        color = _series_color(item)
        axis.plot(
            epochs,
            values,
            label=f"{_compact_label(item)} {'validation' if is_validation else 'train'}",
            color=color,
            linestyle=PLOT_LINESTYLES.get(item["split"], "-"),
            linewidth=2.45 if is_validation else 1.0,
            marker=_series_marker(item)
            if is_validation and len(values) <= 16
            else None,
            markersize=4.2,
            markeredgewidth=0,
            alpha=0.98 if is_validation else 0.2,
            zorder=3 if is_validation else 1,
            solid_capstyle="round",
            dash_capstyle="round",
        )
        _annotate_validation_best(
            axis, metric, item, epochs, values, _series_color(item)
        )
    axis.set_xlim(max(0.0, min_epoch - 0.25), max_epoch + 0.35)
    axis.margins(y=0.08)


def _plot_rc_params() -> dict[str, Any]:
    return {
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
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
        value_span = max_value - min_value
        padding = max(0.008, value_span * 0.1)
        lower = max(0.0, min(0.5, min_value) - padding)
        upper = min(1.0, max_value + padding)
        minimum_span = 0.06
        if upper - lower < minimum_span:
            center = (lower + upper) / 2.0
            lower = max(0.0, center - minimum_span / 2.0)
            upper = min(1.0, center + minimum_span / 2.0)
            if upper - lower < minimum_span:
                lower = max(0.0, upper - minimum_span)
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        axis.axhline(
            0.5, color="#94A3B8", linewidth=0.9, linestyle=(0, (1.5, 2.5)), alpha=0.78
        )
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
    best_index = (
        min(range(len(values)), key=values.__getitem__)
        if metric == "loss"
        else max(
            range(len(values)),
            key=values.__getitem__,
        )
    )
    best_epoch = epochs[best_index]
    best_value = values[best_index]
    axis.scatter(
        [best_epoch],
        [best_value],
        s=38,
        color=color,
        edgecolor="white",
        linewidth=1.05,
        zorder=4,
    )


def _render_style_legend(fig, series: list[dict[str, Any]]) -> None:
    splits = {item["split"] for item in series}
    split_items = []
    if "val" in splits:
        split_items.append(
            Line2D(
                [0],
                [0],
                color="#475569",
                lw=2.1,
                linestyle="-",
                label="验证集（实线）",
            )
        )
    if "train" in splits:
        split_items.append(
            Line2D(
                [0],
                [0],
                color="#475569",
                lw=1.25,
                linestyle=PLOT_LINESTYLES["train"],
                alpha=0.45,
                label="训练集（虚线）",
            )
        )
    if not split_items:
        return
    fig.legend(
        handles=split_items,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.835),
        frameon=False,
        ncol=2,
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.35,
    )


def _render_model_legend(fig, series: list[dict[str, Any]]) -> None:
    rows = _summary_rows(series)
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[(int(row["run_index"]) - 1) % len(MODEL_COLORS)],
            lw=2.3,
            marker=MODEL_MARKERS[(int(row["run_index"]) - 1) % len(MODEL_MARKERS)],
            markersize=5.0,
            label=str(row["label"]),
        )
        for row in rows
    ]
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.895),
        frameon=False,
        ncol=min(3, len(handles)),
        borderaxespad=0.0,
        handlelength=2.3,
        columnspacing=1.4,
        labelspacing=0.8,
        fontsize=8.8,
    )


def _render_summary_strip(axis, series: list[dict[str, Any]]) -> None:
    axis.axis("off")
    rows = _summary_rows(series)
    if not rows:
        return
    axis.text(
        0.0,
        0.92,
        "验证集结果汇总 Validation summary",
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    available_metrics = [
        metric
        for metric in ("accuracy", "auc", "loss")
        if any(row.get(f"{metric}_final") is not None for row in rows)
    ]
    columns = ["模型 / 对照角色"]
    for metric in available_metrics:
        if metric == "accuracy":
            columns.extend(("最终准确率", "最佳准确率"))
        elif metric == "auc":
            columns.extend(("最终 AUC", "最佳 AUC"))
        else:
            columns.append("最低损失")
    table_rows = []
    for row in rows:
        cells = [row["label"]]
        for metric in available_metrics:
            if metric in {"accuracy", "auc"}:
                cells.extend(
                    (
                        _format_metric_value(metric, row[f"{metric}_final"]),
                        _format_best_cell(metric, row),
                    )
                )
            else:
                cells.append(_format_best_cell(metric, row))
        table_rows.append(cells)
    role_width = 0.40 if len(columns) > 1 else 1.0
    metric_width = (1.0 - role_width) / max(1, len(columns) - 1)
    table = axis.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.76],
        colWidths=[role_width] + [metric_width] * (len(columns) - 1),
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
        best_index = (
            min(range(len(values)), key=values.__getitem__)
            if metric == "loss"
            else max(
                range(len(values)),
                key=values.__getitem__,
            )
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
    i2_bridge = re.fullmatch(
        r"i2_present_r8_high_round_integral_bridge_262144_seed(?P<seed>\d+)"
        r"(?:_gpu\d+)?(?:_\d{8})?",
        stem,
    )
    if i2_bridge:
        return (
            "创新2：PRESENT-80 8轮高轮积分神经桥接 "
            f"（262,144总训练行，seed {i2_bridge.group('seed')}）"
        )
    i2_paper_reference = re.fullmatch(
        r"i2_present_r8_high_round_integral_paper_reference_2pow21_"
        r"seed(?P<seed>\d+)(?:_gpu\d+)?(?:_\d{8})?",
        stem,
    )
    if i2_paper_reference:
        return (
            "创新2：PRESENT-80 8轮论文参考规模近似 "
            f"（2^21总训练行，seed {i2_paper_reference.group('seed')}）"
        )
    transfer_match = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r2_seed(?P<seed>\d+)",
        stem,
    )
    if transfer_match:
        return (
            "创新1：PRESENT → GIFT-64 跨 SPN 结构迁移 "
            f"（E4-R2，目标 seed {transfer_match.group('seed')}）"
        )
    r3_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_readiness_seed(?P<seed>\d+)",
        stem,
    )
    if r3_readiness:
        return (
            "创新1：PRESENT → GIFT-64 跨 SPN 结构迁移 "
            f"（E4-R3 就绪检查，目标 seed {r3_readiness.group('seed')}）"
        )
    r3_medium = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_65536_seed(?P<seed>\d+)",
        stem,
    )
    if r3_medium:
        return (
            "创新1：PRESENT → GIFT-64 跨 SPN 结构迁移 "
            f"（E4-R3 中等规模，目标 seed {r3_medium.group('seed')}）"
        )
    r4_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_readiness_seed(?P<seed>\d+)",
        stem,
    )
    if r4_readiness:
        return (
            "创新1：PRESENT → GIFT-64 目标适配效率 "
            f"（E4-R4 就绪检查，目标 seed {r4_readiness.group('seed')}）"
        )
    r4_medium = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_65536_seed(?P<seed>\d+)",
        stem,
    )
    if r4_medium:
        return (
            "创新1：PRESENT → GIFT-64 目标适配效率 "
            f"（E4-R4 中等规模，目标 seed {r4_medium.group('seed')}）"
        )
    e5_source = re.fullmatch(
        r"i1_cross_spn_e5_source_objective_8192_seed(?P<seed>\d+)",
        stem,
    )
    if e5_source:
        return (
            "创新1 E5-R0：PRESENT 源端拓扑反事实辅助目标 "
            f"（本地诊断，源 seed {e5_source.group('seed')}）"
        )
    e5_target = re.fullmatch(
        r"i1_cross_spn_e5_target_8192_source_seed(?P<source_seed>\d+)_"
        r"target_seed(?P<target_seed>\d+)",
        stem,
    )
    if e5_target:
        return (
            "创新1 E5-R0：PRESENT → GIFT-64 一轮迁移门控 "
            f"（源 seed {e5_target.group('source_seed')}，"
            f"目标 seed {e5_target.group('target_seed')}）"
        )
    if stem == "i1_cross_spn_e6_functional_margin_readiness":
        return "创新1 E6-R0：源端功能性拓扑边际目标（就绪检查）"
    if stem == "i1_cross_spn_e6_target_readiness":
        return "创新1 E6-R0：PRESENT → GIFT-64 严格迁移（就绪检查）"
    e6_source = re.fullmatch(
        r"i1_cross_spn_e6_functional_margin_8192_seed(?P<seed>\d+)",
        stem,
    )
    if e6_source:
        return (
            "创新1 E6-R0：PRESENT 源端功能性拓扑边际 "
            f"（本地诊断，源 seed {e6_source.group('seed')}）"
        )
    e6_target = re.fullmatch(
        r"i1_cross_spn_e6_target_8192_source_seed(?P<source_seed>\d+)_"
        r"target_seed(?P<target_seed>\d+)",
        stem,
    )
    if e6_target:
        return (
            "创新1 E6-R0：PRESENT → GIFT-64 功能边际迁移门控 "
            f"（源 seed {e6_target.group('source_seed')}，"
            f"目标 seed {e6_target.group('target_seed')}）"
        )
    cleaned = stem.replace("_", " ").replace("-", " ")
    cleaned = " ".join(
        part
        for part in cleaned.split()
        if not part.startswith("gpu") and not part.isdigit()
    )
    return "\n".join(textwrap.wrap(cleaned, width=78, max_lines=2, placeholder="..."))


def _plot_subtitle(series: list[dict[str, Any]]) -> str:
    rows = _summary_rows(series)
    metric_names = ", ".join(
        _metric_title(metric)
        for metric in dict.fromkeys(item["metric"] for item in series)
    )
    if not metric_names:
        metric_names = ", ".join(_metric_title(metric) for metric in DEFAULT_METRICS)
    context = series[0] if series else {}
    protocol_parts = []
    cipher_rounds = list(
        dict.fromkeys(
            (str(item.get("cipher") or ""), item.get("rounds", ""))
            for item in series
            if item.get("cipher") and item.get("rounds") != ""
        )
    )
    if cipher_rounds:
        protocol_parts.append(
            " / ".join(f"{cipher} {rounds} 轮" for cipher, rounds in cipher_rounds)
        )
    if context.get("train_structures") not in (None, ""):
        structure_text = f"训练 {int(context['train_structures']):,} 个结构"
        if context.get("train_keys_per_structure") not in (None, ""):
            structure_text += f" × {int(context['train_keys_per_structure']):,} 把密钥"
        protocol_parts.append(structure_text)
        if context.get("integral_set_size") not in (None, ""):
            protocol_parts.append(
                f"每个积分集合 {int(context['integral_set_size'])} 个明文"
            )
    else:
        if context.get("samples_per_class") not in (None, ""):
            protocol_parts.append(f"训练 {int(context['samples_per_class']):,}/类")
        if context.get("validation_samples_per_class") not in (None, ""):
            protocol_parts.append(
                f"验证 {int(context['validation_samples_per_class']):,}/类"
            )
        if context.get("pairs_per_sample") not in (None, ""):
            protocol_parts.append(f"每样本 {int(context['pairs_per_sample'])} 对")
    protocol_parts.append(f"{len(rows)} 个模型/对照")
    protocol_parts.append(f"验证集重点显示：{metric_names}")
    return "｜".join(protocol_parts)


def _series_color(item: dict[str, Any]) -> str:
    run_index = int(item.get("run_index", 1))
    return MODEL_COLORS[(run_index - 1) % len(MODEL_COLORS)]


def _series_marker(item: dict[str, Any]) -> str:
    run_index = int(item.get("run_index", 1))
    return MODEL_MARKERS[(run_index - 1) % len(MODEL_MARKERS)]


def _compact_label(item: dict[str, Any]) -> str:
    model = str(item.get("model") or item.get("label") or "")
    if item.get("cipher") == "uKNIT-BC" and model.startswith("runtime_spn_e4_"):
        seed = item.get("seed", "")
        mode = item.get("runtime_structure_mode")
        context = item.get("sbox_context_mode")
        if mode == "true" and context == "late_cell":
            return f"seed{seed}：正确归属（逐 cell）"
        if mode == "true" and context == "late_pair":
            return f"seed{seed}：全局 S盒锚点"
        if mode == "sbox_shuffled" and context == "late_cell":
            return f"seed{seed}：S盒归属打乱控制"
        if mode == "true" and context == "edge_gate":
            return f"seed{seed}：正确 S盒-拓扑门控"
        if mode == "sbox_shuffled" and context == "edge_gate":
            return f"seed{seed}：打乱 S盒-拓扑门控"
    aliases = {
        "linear_same_input": "同输入线性基线",
        "structure_mlp": "结构交互 MLP",
        "structure_mlp_shuffled_labels": "训练标签打乱 MLP 控制",
        "wu_guo_paper_family_mbconv": "论文族锚点：Wu/Guo MBConv",
        "present_integral_structured_residual": "创新2候选：结构积分残差网络",
        "same_input_flat_linear": "同输入线性基线",
        "present_integral_structured_residual_shuffled": (
            "标签打乱控制（候选同架构）"
        ),
        "present_zhang_wang_keras_mcnd": "Zhang-Wang MCND",
        "present_nibble_paligned_mcnd": "I1 nibble-P MCND",
        "present_nibble_invp_only_spn_only": "InvP token mixer",
        "present_nibble_invp_topology_residual_spn_only": "true InvP residual",
        "present_nibble_shuffled_p_topology_residual_spn_only": "shuffled P residual",
        "present_nibble_delta_topology_residual_spn_only": "Delta-only residual",
        "present_nibble_case3_invp_topology_residual_spn_only": "Case3 true InvP",
        "present_nibble_case3_shuffled_p_topology_residual_spn_only": "Case3 shuffled P",
        "present_nibble_case3_raw_topology_residual_spn_only": "Case3 raw triple",
        "gift_cross_spn_aligned_token_mixer_raw_anchor": "GIFT 原始输入基线",
        "gift_cross_spn_typed_cell_true": "GIFT 结构网络（从零训练）",
        "gift_cross_spn_typed_cell_true_from_present_true": "PRESENT 真结构 → GIFT 真结构",
        "gift_cross_spn_typed_cell_true_from_present_shuffled": "PRESENT 打乱结构 → GIFT 真结构",
        "gift_cross_spn_typed_cell_shuffled_from_present_true": "PRESENT 真结构 → GIFT 打乱结构",
        "present_cross_spn_typed_cell_e5_off": "源分类基线（辅助损失关闭）",
        "present_cross_spn_typed_cell_e5_true_shuffled": "候选：真拓扑 vs 打乱拓扑",
        "present_cross_spn_typed_cell_e5_shuffled_placebo": "安慰剂：打乱拓扑 vs 打乱拓扑",
        "gift_cross_spn_typed_cell_e5_scratch": "GIFT 从零训练",
        "gift_cross_spn_typed_cell_e5_from_present_off": "迁移基线：源辅助损失关闭",
        "gift_cross_spn_typed_cell_e5_from_present_true_shuffled": "候选迁移：源真拓扑 vs 打乱拓扑",
        "gift_cross_spn_typed_cell_e5_from_present_shuffled_placebo": "安慰剂迁移：源打乱 vs 打乱",
        "present_cross_spn_typed_cell_e6_off": "源分类基线（功能边际关闭）",
        "present_cross_spn_typed_cell_e6_functional_margin": "候选：真拓扑功能边际",
        "present_cross_spn_typed_cell_e6_shuffled_placebo": "安慰剂：打乱拓扑功能边际",
        "gift_cross_spn_typed_cell_e6_scratch": "GIFT 从零训练",
        "gift_cross_spn_typed_cell_e6_from_present_off": "迁移基线：源功能边际关闭",
        "gift_cross_spn_typed_cell_e6_from_present_functional_margin": "候选迁移：源真拓扑功能边际",
        "gift_cross_spn_typed_cell_e6_from_present_shuffled_placebo": "安慰剂迁移：源打乱功能边际",
        "simon_lu_round_relation_true": "SIMON 真实轮关系",
        "simon_lu_round_relation_target_scratch": "SIMON 目标轮等轮次从零训练",
        "simon_lu_round_relation_shuffled": "SIMON 左右错位控制",
        "simeck_lu_round_relation_true": "SIMECK 真实轮关系",
        "simeck_lu_round_relation_target_scratch": "SIMECK 目标轮等轮次从零训练",
        "simeck_lu_round_relation_shuffled": "SIMECK 左右错位控制",
        "simon_lu_senet_layout_true": "SIMON Lu-SE布局真实关系",
        "simon_lu_senet_layout_shuffled": "SIMON Lu-SE布局错位控制",
        "simeck_lu_senet_layout_true": "SIMECK Lu-SE布局真实关系",
        "simeck_lu_senet_layout_shuffled": "SIMECK Lu-SE布局错位控制",
    }
    if model in aliases:
        return aliases[model]
    if model == "multiscale_dense_resnet" and item.get("cipher") in {
        "SIMON64/128",
        "Simeck64/128",
    }:
        return f"{item['cipher'].split('64', 1)[0]} 通用多尺度基线"
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
