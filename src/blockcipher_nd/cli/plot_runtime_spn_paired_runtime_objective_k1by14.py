from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_paired_runtime_objective_k1by14 import (
    EXPECTED_SEEDS,
    STRUCTURE_MARGIN,
)


DECISION_LABELS = {
    "innovation1_runtime_spn_k1by14_paired_preference_supported": (
        "通过：正确方向、同权重结构差值和未见打乱控制均成立"
    ),
    "innovation1_runtime_spn_k1by14_orientation_placebo_failed": (
        "暂缓：交换方向控制未被排除，目标可能只是在制造任意偏好"
    ),
    "innovation1_runtime_spn_k1by14_counterexample_overfit": (
        "暂缓：只学会区分训练中的仿射反例，未通过未见打乱控制"
    ),
    "innovation1_runtime_spn_k1by14_anchor_retention_failed": (
        "暂缓：成对目标破坏了原有 PRESENT 七轮信号"
    ),
    "innovation1_runtime_spn_k1by14_research_gate_failed": (
        "暂缓：至少一项逐 seed 结构偏好门槛未通过"
    ),
    "innovation1_runtime_spn_k1by14_protocol_invalid": "无效：实验协议或证据不完整",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Chinese K1-BY14 paired-runtime result figure."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by14_svg(gate, args.output)
    report_path = args.output.with_name("plot_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def render_k1by14_svg(
    gate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "svg.fonttype": "none",
            "axes.facecolor": "#FFFFFF",
            "text.color": "#111827",
        }
    )
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("K1-BY14 plot requires both seed result panels")

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.6), constrained_layout=True)
    figure.suptitle(
        "创新1 K1-BY14：PRESENT 七轮成对运行时结构学习",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "同一组网络参数分别换入正确、仿射错误和未见打乱结构；训练规模为 2048/class，仅作结构诊断",
        ha="center",
        fontsize=11,
    )

    labels = ("普通锚点", "正确结构", "仿射错误", "未见打乱")
    colors = ("#6B7280", "#168A72", "#D97706", "#A23B72")
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    all_values: list[float] = []
    for offset, seed in zip((-width / 2, width / 2), EXPECTED_SEEDS, strict=True):
        values = seed_results[str(seed)]
        aucs = values["auc_by_orientation_and_runtime"]["correct_oriented"]
        bars = axes[0].bar(
            x + offset,
            (
                values["ordinary_k1by3_anchor_auc"],
                aucs["correct_runtime"],
                aucs["affine_runtime"],
                aucs["heldout_shuffled"],
            ),
            width,
            label=f"seed{seed}",
            color=colors,
            alpha=0.72 if seed == EXPECTED_SEEDS[0] else 1.0,
            edgecolor="#1F2937",
            linewidth=0.5,
        )
        all_values.extend(float(bar.get_height()) for bar in bars)
        axes[0].bar_label(bars, fmt="%.4f", fontsize=8, padding=3, rotation=90)
    axes[0].set_title("正确方向 checkpoint 的同权重结构替换", fontsize=13)
    axes[0].set_ylabel("验证集 AUC")
    axes[0].set_xticks(x, labels)
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    margin_names = ("保留锚点", "胜过交换方向", "胜过仿射结构", "胜过未见打乱")
    margin_keys = (
        "anchor",
        "swapped_primary",
        "same_checkpoint_affine",
        "same_checkpoint_heldout_shuffled",
    )
    margin_x = np.arange(len(margin_names), dtype=float)
    margin_values: list[float] = []
    for offset, seed, color in zip(
        (-width / 2, width / 2),
        EXPECTED_SEEDS,
        ("#2563A6", "#168A72"),
        strict=True,
    ):
        margins = seed_results[str(seed)]["correct_oriented_margins"]
        values = [float(margins[key]) for key in margin_keys]
        margin_values.extend(values)
        bars = axes[1].bar(
            margin_x + offset,
            values,
            width,
            label=f"seed{seed}",
            color=color,
        )
        axes[1].bar_label(bars, fmt="%+.4f", fontsize=8, padding=3, rotation=90)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].axhline(
        STRUCTURE_MARGIN,
        color="#B42318",
        linestyle="--",
        linewidth=1.2,
        label="结构差值门槛 +0.005",
    )
    axes[1].set_title("正确方向相对控制的 AUC 差值", fontsize=13)
    axes[1].set_ylabel("AUC 差值（越高越好）")
    axes[1].set_xticks(margin_x, margin_names, rotation=12, ha="right")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    auc_pad = max(0.025, (max(all_values) - min(all_values)) * 0.28)
    axes[0].set_ylim(max(0.45, min(all_values) - auc_pad), min(1.0, max(all_values) + auc_pad))
    margin_pad = max(0.012, (max(margin_values) - min(margin_values)) * 0.35)
    axes[1].set_ylim(min(margin_values + [0.0]) - margin_pad, max(margin_values + [STRUCTURE_MARGIN]) + margin_pad)

    decision = str(gate.get("decision", ""))
    figure.text(
        0.5,
        0.018,
        DECISION_LABELS.get(decision, "裁决：未知"),
        ha="center",
        fontsize=11,
        color="#176B5B" if gate.get("status") == "pass" else "#8A4B08",
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", dpi=160)
    plt.close(figure)
    return {
        "status": "written",
        "output": str(output),
        "seed_count": len(EXPECTED_SEEDS),
        "decision": decision,
        "title": "创新1 K1-BY14：PRESENT 七轮成对运行时结构学习",
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by14_svg"]
