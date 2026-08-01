from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_same_checkpoint_runtime_swap_k1by8 import (
    TAPS,
    WEIGHT_SOURCES,
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
        description="Render the Chinese K1-BY8 same-checkpoint runtime swap."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by8_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by8_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = ("2", "3")
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != set(seeds):
        raise ValueError("K1-BY8 plot requires seed2 and seed3")

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
            "创新1 K1-BY8：同一组权重只替换运行时结构",
            x=0.035,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.035,
            0.895,
            "冻结 PRESENT-80 r7 检查点与验证集；四格交叉组合权重来源和正确/仿射运行时程序。",
            ha="left",
            fontsize=11.3,
        )
        figure.text(
            0.035,
            0.842,
            "纵轴差值均为正确运行时减仿射运行时；绿色虚线是预注册的 +0.005 结构优势门槛。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _margin_panel(
            axes[0],
            seed_results,
            seeds,
            weight_source="correct_weights",
            title="正确检查点：运行时因果差值",
        )
        _margin_panel(
            axes[1],
            seed_results,
            seeds,
            weight_source="affine_weights",
            title="仿射检查点：反向运行时交换",
        )
        _final_auc_panel(axes[2], seed_results, seeds)
        figure.text(
            0.035,
            0.075,
            _decision_text(gate),
            ha="left",
            fontsize=10.8,
            color=_decision_color(str(gate.get("status", ""))),
            fontweight="bold",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "run_id": gate.get("run_id"),
        "panels": 3,
        "seeds": [2, 3],
        "weight_sources": list(WEIGHT_SOURCES),
        "taps": list(TAPS) + ["final_output"],
        "status": gate.get("status"),
    }


def _margin_panel(
    axis: Any,
    seed_results: Mapping[str, Any],
    seeds: tuple[str, ...],
    *,
    weight_source: str,
    title: str,
) -> None:
    positions = np.arange(len(TAP_LABELS), dtype=float)
    colors = ("#2563EB", "#D97706")
    all_values = []
    for index, seed in enumerate(seeds):
        values = seed_results[seed]["weights"][weight_source]
        margins = [
            float(
                values["taps"][tap][
                    "correct_minus_affine_runtime_probe_auc"
                ]
            )
            for tap in TAPS
        ]
        margins.append(float(values["correct_minus_affine_runtime_final_auc"]))
        all_values.extend(margins)
        axis.plot(
            positions,
            margins,
            marker="o",
            linewidth=2.2,
            color=colors[index],
            label=f"seed{seed}",
        )
        for x, value in zip(positions, margins, strict=True):
            axis.annotate(
                f"{value:+.3f}",
                (x, value),
                xytext=(0, 7 if value >= 0 else -13),
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
        "正确权重\n正确运行时",
        "正确权重\n仿射运行时",
        "仿射权重\n正确运行时",
        "仿射权重\n仿射运行时",
    )
    keys = (
        ("correct_weights", "correct_runtime_final_auc"),
        ("correct_weights", "affine_runtime_final_auc"),
        ("affine_weights", "correct_runtime_final_auc"),
        ("affine_weights", "affine_runtime_final_auc"),
    )
    positions = np.arange(len(labels), dtype=float)
    width = 0.34
    colors = ("#2563EB", "#D97706")
    all_values = []
    for index, seed in enumerate(seeds):
        values = [
            float(seed_results[seed]["weights"][weight_source][field])
            for weight_source, field in keys
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
    axis.set_title("四格交换后的最终输出", fontsize=12.3)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if gate.get("status") == "invalid":
        return "裁决：来源、参数交换、运行时缓冲或探针协议无效，本次结果不可解释。"
    if decision.endswith("independent_training_variance_identified"):
        return "裁决：两颗 seed 在正确权重下都偏好正确运行时；K1-BY6 的反转主要来自独立训练耦合。"
    if decision.endswith("same_checkpoint_histogram_access_loss"):
        return "裁决：同一组正确权重下仍在最早线性直方图失去正确结构优势；应只修复该表示接口。"
    return "裁决：线性直方图保留结构优势，但下游或最终输出再次丢失；按首个失败接口继续定位。"


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "invalid": "#B91C1C"}.get(status, "#374151")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by8_svg"]
