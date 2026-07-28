from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    ANCHOR_CELL,
    AUC_FLOOR,
    CONFIRMATION_SEEDS,
    FRESH_SPLITS,
    LABEL_SHUFFLE_MARGIN,
    RAW_MARGIN,
)


SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-Q difference-position chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1q_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1q_svg(
    gate: Mapping[str, Any],
    output: Path,
    *,
    cipher_label: str = "uKNIT",
    rounds: int = 5,
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    anchor_cell: int = ANCHOR_CELL,
) -> dict[str, Any]:
    selection = gate.get("selection", {})
    ranking = selection.get("ranking", [])
    if len(ranking) != 16:
        raise ValueError("K1-Q plot requires all sixteen discovery positions")
    selected = [int(cell) for cell in selection.get("selected_cells", [])]
    confirmation = gate.get("confirmation_summary", {})

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 11.2))
        figure.subplots_adjust(
            left=0.07,
            right=0.96,
            top=0.78,
            bottom=0.09,
            hspace=0.48,
            wspace=0.24,
        )
        figure.suptitle(
            f"创新1：移动 {cipher_label} 输入差分的位置，能否恢复第 {rounds} 轮信号",
            x=0.05,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.91,
            "固定同一个 cell 内 bit_role=1、四对密文、严格负样本和精确五阶段特征；只在 16 个 native cell 之间移动差分。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.855,
            _decision_text(gate, confirmation_seeds),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )

        _plot_discovery_exact(axes[0, 0], ranking, selected, anchor_cell)
        _plot_discovery_margin(axes[0, 1], ranking, selected, anchor_cell)
        _plot_confirmation_auc(
            axes[1, 0],
            confirmation,
            selected,
            confirmation_seeds,
            anchor_cell,
        )
        _plot_confirmation_margins(
            axes[1, 1],
            confirmation,
            selected,
            confirmation_seeds,
            anchor_cell,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 11.2,
        "language": "zh-CN",
        "panels": 4,
        "title_explains_run": True,
        "all_sixteen_positions_visible": True,
        "anchor_and_selected_positions_distinguished": True,
        "confirmation_uses_heatmaps": True,
    }


def _plot_discovery_exact(
    axis: plt.Axes,
    ranking: list[Mapping[str, Any]],
    selected: list[int],
    anchor_cell: int,
) -> None:
    ordered = sorted(ranking, key=lambda row: int(row["cell"]))
    cells = [int(row["cell"]) for row in ordered]
    for split, color, marker in (
        ("same_key_fresh", "#0F766E", "o"),
        ("cross_key_validation", "#C2410C", "s"),
    ):
        values = [float(row["fresh_splits"][split]["exact_auc"]) for row in ordered]
        axis.plot(
            cells,
            values,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5.5,
            label=SPLIT_LABELS[split],
        )
    _mark_positions(axis, selected, anchor_cell)
    values = [
        float(row["fresh_splits"][split]["exact_auc"])
        for row in ordered
        for split in FRESH_SPLITS
    ]
    if min(values) > AUC_FLOOR + 0.05:
        axis.set_ylim(min(values) - 0.02, max(values) + 0.02)
        _offscale_threshold(axis, f"发现门槛 {AUC_FLOOR:.3f}（低于放大范围）")
    else:
        axis.axhline(
            AUC_FLOOR,
            color="#2563EB",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            label=f"发现门槛 {AUC_FLOOR:.3f}",
        )
        axis.axhline(0.5, color="#9CA3AF", linestyle=(0, (2, 3)), linewidth=1)
        axis.set_ylim(min(0.47, min(values) - 0.02), max(0.58, max(values) + 0.025))
    axis.set_xticks(cells)
    axis.set_xlabel("native cell 编号（黄色是原 0x40；绿色是入选候选）")
    axis.set_ylabel("精确五阶段特征 AUC")
    axis.set_title("发现阶段：每个差分位置的 fresh AUC", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=3, loc="best", fontsize=8.5)


def _plot_discovery_margin(
    axis: plt.Axes,
    ranking: list[Mapping[str, Any]],
    selected: list[int],
    anchor_cell: int,
) -> None:
    ordered = sorted(ranking, key=lambda row: int(row["cell"]))
    cells = [int(row["cell"]) for row in ordered]
    for split, color, marker in (
        ("same_key_fresh", "#0F766E", "o"),
        ("cross_key_validation", "#C2410C", "s"),
    ):
        values = [
            float(row["fresh_splits"][split]["exact_minus_raw"]) for row in ordered
        ]
        axis.plot(
            cells,
            values,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5.5,
            label=SPLIT_LABELS[split],
        )
    _mark_positions(axis, selected, anchor_cell)
    values = [
        float(row["fresh_splits"][split]["exact_minus_raw"])
        for row in ordered
        for split in FRESH_SPLITS
    ]
    if min(values) > RAW_MARGIN + 0.05:
        axis.set_ylim(min(values) - 0.015, max(values) + 0.02)
        _offscale_threshold(axis, f"归因门槛 +{RAW_MARGIN:.3f}（低于放大范围）")
    else:
        axis.axhline(
            RAW_MARGIN,
            color="#2563EB",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            label=f"归因门槛 +{RAW_MARGIN:.3f}",
        )
        axis.axhline(0.0, color="#9CA3AF", linewidth=1)
        axis.set_ylim(min(-0.03, min(values) - 0.015), max(0.04, max(values) + 0.02))
    axis.set_xticks(cells)
    axis.set_xlabel("native cell 编号")
    axis.set_ylabel("精确特征 AUC - 原始密文 AUC")
    axis.set_title("发现阶段：信号是否来自五阶段结构", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=3, loc="best", fontsize=8.5)


def _plot_confirmation_auc(
    axis: plt.Axes,
    confirmation: Mapping[str, Any],
    selected: list[int],
    confirmation_seeds: tuple[int, ...],
    anchor_cell: int,
) -> None:
    cells = [anchor_cell, *selected] if selected else []
    if not cells:
        _empty_confirmation(axis, "发现阶段没有位置同时通过两个 fresh 门槛")
        return
    columns = [
        (str(seed), split) for seed in confirmation_seeds for split in FRESH_SPLITS
    ]
    values = np.asarray(
        [
            [
                float(confirmation[str(cell)][seed][split]["exact_auc"])
                for seed, split in columns
            ]
            for cell in cells
        ],
        dtype=float,
    )
    _heatmap(
        axis,
        values,
        row_labels=[
            f"cell {cell}" + ("（0x40）" if cell == anchor_cell else "")
            for cell in cells
        ],
        column_labels=[f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
        title="未见 seed 确认：绝对 AUC（必须全部 ≥ 0.550）",
        vmin=min(0.48, float(values.min())),
        vmax=max(0.60, float(values.max())),
        threshold=AUC_FLOOR,
    )


def _plot_confirmation_margins(
    axis: plt.Axes,
    confirmation: Mapping[str, Any],
    selected: list[int],
    confirmation_seeds: tuple[int, ...],
    anchor_cell: int,
) -> None:
    cells = [anchor_cell, *selected] if selected else []
    if not cells:
        seeds = "/".join(str(seed) for seed in confirmation_seeds)
        _empty_confirmation(
            axis,
            f"没有启动 seed{seeds} 确认，避免对空候选继续加样本",
        )
        return
    values = []
    for cell in cells:
        summaries = [
            confirmation[str(cell)][str(seed)][split]
            for seed in confirmation_seeds
            for split in FRESH_SPLITS
        ]
        values.append(
            [
                min(float(row["exact_minus_raw"]) for row in summaries),
                min(float(row["exact_minus_label_shuffle"]) for row in summaries),
            ]
        )
    matrix = np.asarray(values, dtype=float)
    _heatmap(
        axis,
        matrix,
        row_labels=[
            f"cell {cell}" + ("（0x40）" if cell == anchor_cell else "")
            for cell in cells
        ],
        column_labels=["最小\n超过原始密文", "最小\n超过标签打乱"],
        title="未见 seed 确认：四个 fresh 组合中的最差净优势",
        vmin=min(-0.04, float(matrix.min())),
        vmax=max(0.06, float(matrix.max())),
        thresholds=[RAW_MARGIN, LABEL_SHUFFLE_MARGIN],
    )


def _mark_positions(
    axis: plt.Axes,
    selected: list[int],
    anchor_cell: int,
) -> None:
    axis.axvspan(
        anchor_cell - 0.35,
        anchor_cell + 0.35,
        color="#FBBF24",
        alpha=0.18,
        linewidth=0,
    )
    for cell in selected:
        axis.axvspan(
            cell - 0.35,
            cell + 0.35,
            color="#16A34A",
            alpha=0.14,
            linewidth=0,
        )


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    threshold: float | None = None,
    thresholds: list[float] | None = None,
) -> None:
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title, loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            limit = threshold if threshold is not None else thresholds[column]
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}" if threshold is None else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if value >= limit else "normal",
                color="#111827",
            )
    axis.tick_params(length=0)


def _empty_confirmation(axis: plt.Axes, text: str) -> None:
    axis.set_axis_off()
    axis.text(
        0.5,
        0.56,
        "确认阶段未启动",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#6B7280",
    )
    axis.text(
        0.5,
        0.42,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#6B7280",
    )


def _offscale_threshold(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.99,
        0.04,
        text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#2563EB",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
    )


def _decision_text(
    gate: Mapping[str, Any],
    confirmation_seeds: tuple[int, ...],
) -> str:
    selected = gate.get("selection", {}).get("selected_cells", [])
    confirmed = gate.get("confirmed_cells", [])
    status = str(gate.get("status", ""))
    if status == "invalid":
        return "协议无效：先修复数据、缓存或裁决不变量，当前指标不能解释。"
    if confirmed:
        seeds = "/".join(str(seed) for seed in confirmation_seeds)
        return f"结论：cell {confirmed} 在 seed{seeds} 上确认通过，可进入同预算神经结构归因。"
    if selected:
        return (
            f"结论：发现阶段选中 cell {selected}，但未在全部未见 seed/密钥组合上确认。"
        )
    return "结论：16 个相同 bit 角色的位置都未通过发现门槛；停止机械位置扫描。"


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1q_svg"]
