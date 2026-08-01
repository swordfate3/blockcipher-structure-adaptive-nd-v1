from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_source_bundle_collision_k1by10 import (
    REQUIRED_TAPS,
    STAGES,
    TARGET_CELLS,
)


TAP_LABELS = {
    "linear_primitive_expert": "置换专家输出",
    "cell_fusion": "单元融合输出",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY10 per-cell collision audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--effects", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    effects = [
        json.loads(line)
        for line in args.effects.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = render_k1by10_svg(gate, effects, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by10_svg(
    gate: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    matrices = {
        (seed, tap): _effect_matrix(effects, seed=seed, tap=tap)
        for seed in (2, 3)
        for tap in REQUIRED_TAPS
    }
    values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    bound = max(0.01, float(np.nanmax(np.abs(values))) * 1.08)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 10.2))
        figure.subplots_adjust(
            left=0.07,
            right=0.91,
            top=0.75,
            bottom=0.17,
            wspace=0.16,
            hspace=0.33,
        )
        figure.suptitle(
            "创新1 K1-BY10：源单元组均值的逐单元过度平滑审计",
            x=0.045,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.045,
            0.905,
            "冻结 PRESENT-80 r7、两颗正确检查点和 K1-BY9 表示；没有训练，只分解每个阶段和目标单元的探针差值。",
            ha="left",
            fontsize=11.0,
        )
        figure.text(
            0.045,
            0.855,
            "颜色表示加入源组均值后，正确运行时相对仿射运行时的 AUC 间隔变化；蓝色为改善，红色为损失。",
            ha="left",
            fontsize=10.3,
            color="#4B5563",
        )
        image = None
        for row_index, seed in enumerate((2, 3)):
            for column_index, tap in enumerate(REQUIRED_TAPS):
                axis = axes[row_index, column_index]
                matrix = matrices[(seed, tap)]
                image = axis.imshow(
                    matrix,
                    cmap="RdBu",
                    vmin=-bound,
                    vmax=bound,
                    aspect="auto",
                    interpolation="nearest",
                )
                _mark_threshold_cells(axis, matrix, seed=seed)
                axis.set_xticks(np.arange(TARGET_CELLS))
                axis.set_xticklabels([str(cell) for cell in range(TARGET_CELLS)])
                axis.set_yticks(np.arange(len(STAGES)))
                axis.set_yticklabels(
                    ["捕获阶段0\n(程序阶段1)", "捕获阶段1\n(程序阶段0)"]
                )
                axis.set_xlabel("目标单元编号（仅用于诊断定位）")
                axis.set_title(
                    f"seed{seed} · {TAP_LABELS[tap]}",
                    fontsize=12.2,
                )
                axis.set_xticks(np.arange(-0.5, TARGET_CELLS, 1), minor=True)
                axis.set_yticks(np.arange(-0.5, len(STAGES), 1), minor=True)
                axis.grid(which="minor", color="#FFFFFF", linewidth=0.8)
                axis.tick_params(which="minor", bottom=False, left=False)
        if image is None:  # pragma: no cover - fixed panel makes this unreachable
            raise ValueError("K1-BY10 plot has no heatmap")
        color_axis = figure.add_axes((0.93, 0.265, 0.017, 0.40))
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("候选间隔 - 旧表示间隔（AUC）")
        figure.text(
            0.045,
            0.075,
            _decision_text(gate),
            ha="left",
            fontsize=10.8,
            color=_decision_color(gate),
            fontweight="bold",
        )
        figure.text(
            0.045,
            0.035,
            "标记：seed2 中 × 表示损失不高于 -0.005；seed3 中 ○ 表示变化不低于 0。门槛必须在同一单元、两个阶段、两个下游层同时成立。",
            ha="left",
            fontsize=9.5,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "run_id": gate.get("run_id"),
        "panels": 4,
        "seeds": [2, 3],
        "taps": list(REQUIRED_TAPS),
        "stages": list(STAGES),
        "target_cells": TARGET_CELLS,
        "color_bound": bound,
        "status": gate.get("status"),
        "research_gate_passed": gate.get("research_gate_passed"),
    }


def _effect_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    tap: str,
) -> np.ndarray:
    matrix = np.full((len(STAGES), TARGET_CELLS), np.nan, dtype=float)
    for row in rows:
        if int(row["seed"]) != seed or str(row["tap"]) != tap:
            continue
        matrix[int(row["tap_stage"]), int(row["target_cell"])] = float(
            row["candidate_minus_anchor_margin"]
        )
    if not np.isfinite(matrix).all():
        raise ValueError("K1-BY10 plot requires every seed/stage/cell effect")
    return matrix


def _mark_threshold_cells(axis: Any, matrix: np.ndarray, *, seed: int) -> None:
    for stage in STAGES:
        for cell in range(TARGET_CELLS):
            value = float(matrix[stage, cell])
            if seed == 2 and value <= -0.005:
                axis.text(
                    cell,
                    stage,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="#111827",
                )
            elif seed == 3 and value >= 0.0:
                axis.text(
                    cell,
                    stage,
                    "○",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#111827",
                )


def _decision_text(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "裁决：冻结来源、分区、逐单元探针或产物协议无效，本次结果不可解释。"
    cells = gate.get("supported_target_cells", [])
    if cells:
        labels = "、".join(str(cell) for cell in cells)
        return f"裁决：目标单元 {labels} 在两个阶段均解释跨 seed 过度平滑；只允许下一步非平均的逐单元偏差残差。"
    return "裁决：没有同一目标单元稳定解释两个阶段的跨 seed 损失；关闭等价类均值路线，回到保留单元的边条件残差。"


def _decision_color(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "#B91C1C"
    if gate.get("research_gate_passed"):
        return "#047857"
    return "#B45309"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by10_svg"]
