from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_post_expert_edge_residual_k1by12 import (
    TAPS,
)


TAP_LABELS = (
    "后专家结构残差",
    "单元融合",
    "阶段池化摘要",
    "分类前表示",
    "最终输出",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY12 post-expert audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by12_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by12_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = ("2", "3")
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != set(seeds):
        raise ValueError("K1-BY12 plot requires seed2 and seed3")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(19.5, 9.3))
        figure.subplots_adjust(
            left=0.055,
            right=0.975,
            top=0.75,
            bottom=0.18,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-BY12：冻结专家后的边结构残差审计",
            x=0.035,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.035,
            0.895,
            "冻结 PRESENT-80 r7 检查点与验证集；只在原语专家之后加入无参数、有界的运行时边消息残差。",
            ha="left",
            fontsize=11.2,
        )
        figure.text(
            0.035,
            0.842,
            "每个层级必须同时高于仿射和打乱边控制 +0.005；右图仅放大最终 AUC，不代表正式规模结果。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _margin_panel(
            axes[0],
            seed_results,
            seeds,
            field="correct_minus_affine_probe_auc",
            final_field="correct_minus_affine_final_auc",
            title="候选正确运行时 - 候选仿射运行时",
        )
        _margin_panel(
            axes[1],
            seed_results,
            seeds,
            field="correct_minus_shuffled_probe_auc",
            final_field="correct_minus_shuffled_final_auc",
            title="正确入边端点 - 打乱源单元绑定",
        )
        _final_auc_panel(axes[2], seed_results, seeds)
        figure.text(
            0.035,
            0.073,
            _decision_text(gate),
            ha="left",
            fontsize=10.8,
            color=_decision_color(gate),
            fontweight="bold",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "run_id": gate.get("run_id"),
        "panels": 3,
        "seeds": [2, 3],
        "taps": list(TAPS) + ["final_output"],
        "status": gate.get("status"),
        "research_gate_passed": gate.get("research_gate_passed"),
    }


def _margin_panel(
    axis: Any,
    seed_results: Mapping[str, Any],
    seeds: tuple[str, ...],
    *,
    field: str,
    final_field: str,
    title: str,
) -> None:
    positions = np.arange(len(TAP_LABELS), dtype=float)
    colors = ("#2563EB", "#D97706")
    series = []
    for seed in seeds:
        values = seed_results[seed]
        margins = [float(values["taps"][tap][field]) for tap in TAPS]
        margins.append(float(values[final_field]))
        series.append(margins)
    all_values = [value for margins in series for value in margins]
    for index, (seed, margins) in enumerate(zip(seeds, series, strict=True)):
        axis.plot(
            positions,
            margins,
            marker="o",
            markersize=5.5,
            linewidth=2.1,
            color=colors[index],
            label=f"seed{seed}",
        )
        for point_index, (x, value) in enumerate(
            zip(positions, margins, strict=True)
        ):
            other = series[1 - index][point_index]
            offset = 8 if value >= other else -16
            axis.annotate(
                f"{value:+.3f}",
                (x, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.1,
                color=colors[index],
            )
    axis.set_ylim(min(-0.015, min(all_values) - 0.012), max(0.025, max(all_values) + 0.016))
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.2, linestyle="--")
    axis.set_xticks(positions, TAP_LABELS, rotation=17)
    axis.set_ylabel("内部探针 AUC 差值")
    axis.set_title(title, fontsize=12.0)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _final_auc_panel(
    axis: Any,
    seed_results: Mapping[str, Any],
    seeds: tuple[str, ...],
) -> None:
    labels = ("旧本地\n正确", "候选\n正确", "候选\n仿射", "候选\n打乱边")
    fields = (
        "anchor_correct_final_auc",
        "candidate_correct_final_auc",
        "candidate_affine_final_auc",
        "candidate_shuffled_final_auc",
    )
    positions = np.arange(len(labels), dtype=float)
    colors = ("#2563EB", "#D97706")
    all_values = []
    for index, seed in enumerate(seeds):
        values = [float(seed_results[seed][field]) for field in fields]
        all_values.extend(values)
        offset = -0.07 if index == 0 else 0.07
        axis.plot(
            positions + offset,
            values,
            marker="o",
            linewidth=1.8,
            markersize=6.0,
            color=colors[index],
            label=f"seed{seed}",
        )
        for x, value in zip(positions + offset, values, strict=True):
            axis.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(0, 8 if index == 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8.1,
                color=colors[index],
            )
    axis.set_ylim(min(all_values) - 0.010, max(all_values) + 0.012)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("最终验证 AUC（局部放大）")
    axis.set_title("最终输出与冻结锚点", fontsize=12.0)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "裁决：来源、参数转移、残差边界、重标号或探针协议无效，本次结果不可解释。"
    if gate.get("research_gate_passed") is True:
        return "裁决：两颗 seed 的内部与最终控制均通过；保留后专家位置，下一步只做同预算可训练适配器。"
    return "裁决：任一 seed 或层级未稳定超过双控制；停止确定性冻结干预，转零初始化可训练适配器。"


def _decision_color(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "#B91C1C"
    if gate.get("research_gate_passed") is False:
        return "#B45309"
    return "#047857"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by12_svg"]
