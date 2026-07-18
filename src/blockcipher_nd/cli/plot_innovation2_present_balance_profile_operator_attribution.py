from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E67 formal profile operator attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_profile_operator_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_profile_operator_attribution(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    labels = ("ANF前缀\nridge", "独立node", "错误P\nmixer", "正确P\nmixer")
    values = (
        float(gate["metrics"]["e65_prefix_ridge_validation_auc"]),
        float(rows["independent"]["validation_auc"]),
        float(rows["corrupted"]["validation_auc"]),
        float(rows["true"]["validation_auc"]),
    )
    deltas = (
        float(gate["metrics"]["true_minus_independent"]),
        float(gate["metrics"]["true_minus_corrupted"]),
        float(gate["metrics"]["true_minus_e65_prefix_ridge"]),
    )
    delta_labels = ("正确P -\n独立node", "正确P -\n错误P", "正确P -\nANF ridge")
    decisions = {
        "innovation2_present_profile_operator_neural_gain_attributed": (
            "候选、关系归因和确定性基线增益全部通过；允许seed1复核。"
        ),
        "innovation2_present_profile_operator_no_ridge_gain": (
            "正确拓扑贡献成立，但未超过ANF ridge；不运行seed1。"
        ),
        "innovation2_present_profile_operator_relation_not_attributed": (
            "候选未稳定领先独立node或错误P；停止该结构。"
        ),
        "innovation2_present_profile_operator_candidate_not_ready": (
            "绝对AUC或过拟合门失败；停止该结构。"
        ),
        "innovation2_present_profile_operator_attribution_protocol_invalid": (
            "source、contract、metric或正式训练协议无效。"
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E67：prefix引导的PRESENT平衡谱算子正式归因",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "30轮seed0，固定E65严格unit profile；同参数比较独立node、正确P和fair-corrupted P。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "神经增益必须同时超过同容量控制、错误拓扑和最强安全ANF-prefix ridge。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(4)
        axes[0].bar(x, values, color=["#0F766E", "#94A3B8", "#D97706", "#2563EB"])
        for index, value in enumerate(values):
            axes[0].text(index, value + 0.012, f"{value:.3f}", ha="center", fontweight="bold")
        axes[0].axhline(0.78, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.45, max(0.90, max(values) + 0.07))
        axes[0].set_ylabel("structure-disjoint验证 AUC")
        axes[0].set_title("绝对性能与三类锚点", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        dx = np.arange(3)
        axes[1].bar(dx, deltas, color=["#2563EB", "#7C3AED", "#0F766E"])
        for index, value in enumerate(deltas):
            axes[1].text(index, value + 0.006, f"{value:+.3f}", ha="center", fontweight="bold")
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.02, color="#D97706", linestyle=":", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=0.9)
        axes[1].set_xticks(dx, delta_labels)
        axes[1].set_ylim(min(-0.05, min(deltas) - 0.03), max(0.16, max(deltas) + 0.04))
        axes[1].set_ylabel("验证 AUC 差值")
        axes[1].set_title("关系归因0.03门 / ridge增益0.02门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.178,
            (
                f"正确P最佳epoch={rows['true']['best_epoch']}，train AUC="
                f"{rows['true']['train_auc']:.3f}，train-validation gap="
                f"{gate['metrics']['true_train_validation_gap']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：PRESENT-80四轮严格unit平衡谱的本地seed0归因；不是高轮结论、新攻击或SOTA。",
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
