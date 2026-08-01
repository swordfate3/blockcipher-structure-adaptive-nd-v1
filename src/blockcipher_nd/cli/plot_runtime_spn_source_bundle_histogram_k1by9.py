from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_source_bundle_histogram_k1by9 import (
    REPRESENTATIONS,
    TAPS,
)


TAP_LABELS = (
    "线性直方图",
    "置换专家输出",
    "单元融合",
    "阶段池化摘要",
    "分类前表示",
    "最终输出",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY9 source-bundle audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by9_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by9_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = ("2", "3")
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != set(seeds):
        raise ValueError("K1-BY9 plot requires seed2 and seed3")

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
        figure, axes = plt.subplots(1, 3, figsize=(19.0, 9.4))
        figure.subplots_adjust(
            left=0.055,
            right=0.975,
            top=0.70,
            bottom=0.20,
            wspace=0.25,
        )
        figure.suptitle(
            "创新1 K1-BY9：线性直方图加入相对源单元组上下文",
            x=0.035,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.035,
            0.895,
            "冻结 PRESENT-80 r7 正确检查点与验证集；只比较旧的本地直方图和固定 1:1 源组均值表示。",
            ha="left",
            fontsize=11.3,
        )
        figure.text(
            0.035,
            0.842,
            "纵轴差值为正确运行时减仿射运行时；绿色虚线是每个层级必须达到的 +0.005 门槛。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _margin_panel(
            axes[0],
            seed_results,
            seeds,
            representation="anchor_local",
            title="旧表示：精确重放锚点",
        )
        _margin_panel(
            axes[1],
            seed_results,
            seeds,
            representation="candidate_source_bundle_mean",
            title="候选表示：源单元组均值",
        )
        _final_auc_panel(axes[2], seed_results, seeds)
        figure.text(
            0.035,
            0.075,
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
        "representations": list(REPRESENTATIONS),
        "taps": list(TAPS) + ["final_output"],
        "status": gate.get("status"),
    }


def _margin_panel(
    axis: Any,
    seed_results: Mapping[str, Any],
    seeds: tuple[str, ...],
    *,
    representation: str,
    title: str,
) -> None:
    positions = np.arange(len(TAP_LABELS), dtype=float)
    colors = ("#2563EB", "#D97706")
    series = []
    for seed in seeds:
        values = seed_results[seed]["representations"][representation]
        margins = [
            float(values["taps"][tap]["correct_minus_affine_runtime_probe_auc"])
            for tap in TAPS
        ]
        margins.append(float(values["correct_minus_affine_runtime_final_auc"]))
        series.append(margins)
    all_values = [value for margins in series for value in margins]
    for index, (seed, margins) in enumerate(zip(seeds, series, strict=True)):
        axis.plot(
            positions,
            margins,
            marker="o",
            linewidth=2.2,
            color=colors[index],
            label=f"seed{seed}",
        )
        for point_index, (x, value) in enumerate(
            zip(positions, margins, strict=True)
        ):
            other = series[1 - index][point_index]
            if abs(value - other) < 0.014:
                is_upper = value > other or (value == other and index == 0)
                label_offset = 8 if is_upper else -16
            elif value < 0:
                label_offset = -13
            else:
                label_offset = 7
            axis.annotate(
                f"{value:+.3f}",
                (x, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.1,
                color=colors[index],
            )
    bound = max(0.02, max(abs(value) for value in all_values) * 1.45)
    axis.set_ylim(-bound, bound)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.2, linestyle="--")
    axis.set_xticks(positions, TAP_LABELS, rotation=18)
    axis.set_ylabel("正确运行时 AUC - 仿射运行时 AUC")
    axis.set_title(title, fontsize=12.3)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _final_auc_panel(
    axis: Any,
    seed_results: Mapping[str, Any],
    seeds: tuple[str, ...],
) -> None:
    labels = (
        "旧表示\n正确运行时",
        "旧表示\n仿射运行时",
        "候选表示\n正确运行时",
        "候选表示\n仿射运行时",
    )
    keys = (
        ("anchor_local", "correct_runtime_final_auc"),
        ("anchor_local", "affine_runtime_final_auc"),
        ("candidate_source_bundle_mean", "correct_runtime_final_auc"),
        ("candidate_source_bundle_mean", "affine_runtime_final_auc"),
    )
    positions = np.arange(len(labels), dtype=float)
    width = 0.34
    colors = ("#2563EB", "#D97706")
    all_values = []
    for index, seed in enumerate(seeds):
        values = [
            float(seed_results[seed]["representations"][representation][field])
            for representation, field in keys
        ]
        all_values.extend(values)
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values,
            width,
            color=colors[index],
            alpha=0.86,
            label=f"seed{seed}",
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.3)
    lower = min(0.45, min(all_values) - 0.035)
    upper = max(0.60, max(all_values) + 0.055)
    axis.set_ylim(lower, upper)
    axis.axhline(0.5, color="#6B7280", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("最终验证 AUC")
    axis.set_title("旧表示与候选表示的最终输出", fontsize=12.3)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if gate.get("status") == "invalid":
        return "裁决：来源、等价矩阵、参数交换或探针协议无效，本次结果不可解释。"
    if decision.endswith("source_bundle_histogram_repair_supported"):
        return "裁决：两颗 seed 的所有层级均保留正确结构优势，且最终性能未明显退化；进入同预算训练确认。"
    return "裁决：固定源组均值未同时修复两颗 seed 的所有层级；丢弃该表示，不调融合权重、不扩样。"


def _decision_color(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "#B91C1C"
    if gate.get("research_gate_passed") is False:
        return "#B45309"
    return "#047857"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by9_svg"]
