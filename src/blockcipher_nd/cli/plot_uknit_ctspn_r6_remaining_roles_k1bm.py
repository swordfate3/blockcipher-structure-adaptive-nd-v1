from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_remaining_roles_k1bm import (
    ACTIVE_BIT_ROLES,
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
        description="Render the Chinese uKNIT r6 remaining-bit-role scan."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bm_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bm_svg(
    gate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    ranking = gate.get("selection", {}).get("ranking", [])
    if len(ranking) != 48:
        raise ValueError("K1-BM plot requires all 48 discovery candidates")
    selected = [
        int(bit) for bit in gate.get("selection", {}).get("selected_bit_indices", [])
    ]
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
            left=0.08,
            right=0.96,
            top=0.78,
            bottom=0.10,
            hspace=0.50,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1：uKNIT 第6轮剩余48个单 bit 差分扫描",
            x=0.05,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.91,
            "K1-BL 已淘汰 role1；本图固定数据、密钥、四对密文和五阶段特征，只扫描 role0/2/3 的16个位置。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.855,
            str(gate.get("decision_text_zh", "")),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )

        exact, margin = _discovery_matrices(ranking)
        _discovery_heatmap(
            axes[0, 0],
            exact,
            selected,
            ranking,
            title="发现阶段：每个角色/位置的最差 fresh AUC",
            value_label="最差 AUC",
            vmin=min(0.46, float(exact.min())),
            vmax=max(0.57, float(exact.max())),
            threshold=AUC_FLOOR,
        )
        _discovery_heatmap(
            axes[0, 1],
            margin,
            selected,
            ranking,
            title="发现阶段：精确五阶段相对原始密文的最差优势",
            value_label="最差净优势",
            vmin=min(-0.04, float(margin.min())),
            vmax=max(0.04, float(margin.max())),
            threshold=RAW_MARGIN,
            signed=True,
        )
        _confirmation_auc(axes[1, 0], selected, confirmation)
        _confirmation_margins(axes[1, 1], selected, confirmation)

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
        "all_48_candidates_visible": True,
        "selected_candidates_marked": True,
        "decision_matches_gate": True,
    }


def _discovery_matrices(
    ranking: list[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    by_coordinate = {
        (int(row["role"]), int(row["cell"])): row for row in ranking
    }
    exact = np.asarray(
        [
            [
                float(by_coordinate[(role, cell)]["minimum_fresh_exact_auc"])
                for cell in range(16)
            ]
            for role in ACTIVE_BIT_ROLES
        ],
        dtype=float,
    )
    margin = np.asarray(
        [
            [
                float(
                    by_coordinate[(role, cell)]["minimum_fresh_exact_minus_raw"]
                )
                for cell in range(16)
            ]
            for role in ACTIVE_BIT_ROLES
        ],
        dtype=float,
    )
    return exact, margin


def _discovery_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    selected: list[int],
    ranking: list[Mapping[str, Any]],
    *,
    title: str,
    value_label: str,
    vmin: float,
    vmax: float,
    threshold: float,
    signed: bool = False,
) -> None:
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(16), [str(cell) for cell in range(16)])
    axis.set_yticks(range(3), [f"role {role}" for role in ACTIVE_BIT_ROLES])
    axis.set_xlabel("native cell 编号")
    axis.set_ylabel("cell 内 bit 角色")
    axis.set_title(title, loc="left", fontweight="bold")
    selected_coordinates = {
        (int(row["role"]), int(row["cell"]))
        for row in ranking
        if int(row["bit_index"]) in selected
    }
    for row_index, role in enumerate(ACTIVE_BIT_ROLES):
        for cell in range(16):
            value = float(values[row_index, cell])
            selected_here = (role, cell) in selected_coordinates
            axis.text(
                cell,
                row_index,
                f"{value:+.3f}" if signed else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold" if selected_here or value >= threshold else "normal",
                color="#111827",
            )
            if selected_here:
                axis.add_patch(
                    plt.Rectangle(
                        (cell - 0.48, row_index - 0.46),
                        0.96,
                        0.92,
                        fill=False,
                        edgecolor="#1D4ED8",
                        linewidth=2.2,
                    )
                )
    axis.text(
        0.0,
        -0.24,
        f"蓝框是每个角色冻结的确认候选；{value_label}门槛 "
        + (f"+{threshold:.3f}" if signed else f"{threshold:.3f}"),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    axis.tick_params(length=0)


def _confirmation_auc(
    axis: plt.Axes,
    selected: list[int],
    confirmation: Mapping[str, Any],
) -> None:
    if not selected:
        _empty_panel(axis, "发现阶段没有候选，未启动 seed3/4 确认。")
        return
    columns = [
        (str(seed), split) for seed in CONFIRMATION_SEEDS for split in FRESH_SPLITS
    ]
    values = np.asarray(
        [
            [
                float(confirmation[str(bit)][seed][split]["exact_auc"])
                for seed, split in columns
            ]
            for bit in selected
        ],
        dtype=float,
    )
    _confirmation_heatmap(
        axis,
        values,
        selected,
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
        title="未见 seed 确认：绝对 AUC（每格必须 ≥ 0.550）",
        thresholds=[AUC_FLOOR] * len(columns),
        signed=False,
    )


def _confirmation_margins(
    axis: plt.Axes,
    selected: list[int],
    confirmation: Mapping[str, Any],
) -> None:
    if not selected:
        _empty_panel(axis, "64个单 bit 位置均无候选，下一步转多 bit 轨迹引导。")
        return
    values = []
    for bit in selected:
        summaries = [
            confirmation[str(bit)][str(seed)][split]
            for seed in CONFIRMATION_SEEDS
            for split in FRESH_SPLITS
        ]
        values.append(
            [
                min(float(row["exact_minus_raw"]) for row in summaries),
                min(float(row["exact_minus_label_shuffle"]) for row in summaries),
            ]
        )
    _confirmation_heatmap(
        axis,
        np.asarray(values, dtype=float),
        selected,
        ["最差\n超过原始密文", "最差\n超过标签打乱"],
        title="未见 seed 确认：最差归因优势",
        thresholds=[RAW_MARGIN, LABEL_SHUFFLE_MARGIN],
        signed=True,
    )


def _confirmation_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    selected: list[int],
    column_labels: list[str],
    *,
    title: str,
    thresholds: list[float],
    signed: bool,
) -> None:
    axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=min(-0.04 if signed else 0.48, float(values.min())),
        vmax=max(0.07 if signed else 0.60, float(values.max())),
    )
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(
        range(len(selected)),
        [f"bit {bit}\nΔ=0x{1 << bit:016x}" for bit in selected],
    )
    axis.set_title(title, loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}" if signed else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if value >= thresholds[column] else "normal",
                color="#111827",
            )
    axis.tick_params(length=0)


def _empty_panel(axis: plt.Axes, text: str) -> None:
    axis.set_axis_off()
    axis.text(
        0.5,
        0.57,
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


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1bm_svg"]
