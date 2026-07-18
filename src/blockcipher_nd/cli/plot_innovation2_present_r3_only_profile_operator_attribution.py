from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E73 formal attribution.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_r3_only_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_r3_only_attribution(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立node", "错误P", "正确P")
    train_auc = [rows[mode]["train_auc"] for mode in modes]
    validation_auc = [rows[mode]["validation_auc"] for mode in modes]
    metrics = gate["metrics"]
    comparisons = (
        metrics["true_minus_independent"],
        metrics["true_minus_corrupted"],
        metrics["true_minus_e67_full_prefix"],
    )
    decisions = {
        "innovation2_present_r3_only_neural_gain_attributed": (
            "r3-only保留质量与拓扑增益；进入相同30轮seed1。"
        ),
        "innovation2_present_r3_only_quality_not_retained": (
            "r3-only未保持E67质量；保留完整39维E68。"
        ),
        "innovation2_present_r3_only_topology_not_attributed": (
            "r3-only未通过正确P归因；不进入seed1。"
        ),
        "innovation2_present_r3_only_attribution_protocol_invalid": (
            "readiness、source、contract或30轮协议无效。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14.8, 8.0))
        figure.subplots_adjust(
            left=0.08, right=0.975, top=0.70, bottom=0.28, wspace=0.32
        )
        figure.text(
            0.08,
            0.955,
            "创新2 E73：r3-only平衡谱算子30轮正式归因",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "只用第3轮13维前缀；与独立node、错误P和E67完整39维正确P做同协议比较。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.845,
            "三行均训练30轮，按validation observed-edge AUC选择checkpoint；参数比E68减少15.6%。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.34
        axes[0].bar(x - width / 2, train_auc, width, label="训练AUC", color="#94A3B8")
        axes[0].bar(x + width / 2, validation_auc, width, label="验证AUC", color="#2563EB")
        for index, value in enumerate(validation_auc):
            axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.93, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.50, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("30轮最佳checkpoint", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        comparison_labels = ("正确P-独立", "正确P-错误P", "r3-only-E67完整")
        axes[1].bar(
            np.arange(3),
            comparisons,
            color=("#0F766E", "#2563EB", "#D97706"),
            width=0.58,
        )
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(-0.02, color="#9333EA", linestyle=":", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(comparisons):
            axes[1].text(index, value + 0.006, f"{value:+.3f}", ha="center")
        lower = min(-0.08, min(comparisons) - 0.04)
        upper = max(0.15, max(comparisons) + 0.05)
        axes[1].set_ylim(lower, upper)
        axes[1].set_xticks(np.arange(3), comparison_labels)
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("拓扑归因与完整前缀锚点", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = summary["contract"]
        figure.text(
            0.08,
            0.183,
            (
                f"参数量={next(iter(contract['parameter_counts'].values()))}；"
                f"正确P train-validation gap={metrics['true_train_validation_gap']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.08,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.053,
            "证据范围：PRESENT-80四轮、8-bit活动cube的本地seed0正式方法归因；不是高轮、攻击或SOTA。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
