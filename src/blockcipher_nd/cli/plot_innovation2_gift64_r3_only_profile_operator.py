from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E76 GIFT-64 r3-only profile readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_gift_r3_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_gift_r3_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立node", "错误GIFT P", "真实GIFT P")
    training = [rows[mode]["train_auc"] for mode in modes]
    validation = [rows[mode]["validation_auc"] for mode in modes]
    ridges = summary["deterministic_ridges"]
    ridge_values = (
        ridges["full39"]["validation_auc"],
        ridges["r3_only"]["validation_auc"],
    )
    margins = (
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
        gate["metrics"]["true_minus_r3_ridge"],
    )
    decisions = {
        "innovation2_gift64_r3_only_profile_readiness_passed": (
            "r3信息与真实GIFT P-layer两轮门通过；可进入冻结30轮seed0。"
        ),
        "innovation2_gift64_r3_only_prefix_not_sufficient": (
            "r3确定性信息不足；只允许审计完整39维算子。"
        ),
        "innovation2_gift64_r3_only_profile_readiness_not_passed": (
            "真实GIFT P-layer未通过两轮防退化门；停止r3-only正式训练。"
        ),
        "innovation2_gift64_r3_only_profile_protocol_invalid": (
            "E75来源、GIFT拓扑、参数公平或训练协议无效。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.2))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E76：GIFT-64真实P-layer能否提升第3轮前缀平衡谱预测",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "输入是E75每个结构的64个输出node及其13维第3轮ANF前缀；输出是64维masked平衡谱。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "三行同为4,795参数、两轮本地训练；完整39维与r3-only ridge提供确定性安全基线。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.34
        axes[0].bar(
            x - width / 2, training, width, label="训练AUC", color="#94A3B8"
        )
        axes[0].bar(
            x + width / 2, validation, width, label="验证AUC", color="#2563EB"
        )
        for index, value in enumerate(validation):
            axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.65, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("三行两轮readiness", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        ridge_x = np.arange(2)
        axes[1].bar(ridge_x, ridge_values, color=("#D97706", "#0F766E"), width=0.58)
        for index, value in enumerate(ridge_values):
            axes[1].text(
                index,
                value - 0.035,
                f"{value:.3f}",
                ha="center",
                color="#FFFFFF",
                fontweight="bold",
            )
        axes[1].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[1].set_xticks(ridge_x, ("完整r1-r3\n39维ridge", "仅r3\n13维ridge"))
        axes[1].set_ylim(0.35, 1.04)
        axes[1].set_ylabel("验证AUC")
        axes[1].set_title("确定性前缀基线", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(3)
        margin_labels = ("真实P -\n独立node", "真实P -\n错误P", "真实P -\nr3 ridge")
        axes[2].bar(
            margin_x,
            margins,
            color=("#0F766E", "#2563EB", "#7C3AED"),
            width=0.62,
        )
        axes[2].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[2].axhline(-0.03, color="#D97706", linestyle=":", linewidth=1.2)
        axes[2].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            offset = 0.006 if value >= 0 else -0.014
            axes[2].text(index, value + offset, f"{value:+.3f}", ha="center")
        axes[2].set_xticks(margin_x, margin_labels)
        axes[2].set_ylim(min(-0.10, min(margins) - 0.04), max(0.18, max(margins) + 0.05))
        axes[2].set_ylabel("验证AUC差值")
        axes[2].set_title("真实GIFT拓扑归因门", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = summary["contract"]
        figure.text(
            0.065,
            0.183,
            (
                f"参数量={next(iter(contract['parameter_counts'].values()))}；"
                f"cell重标号误差={contract['cell_relabel_max_abs_error']:.2e}；"
                "E75 matched坐标=620。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.065,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：GIFT-64四轮严格unit-profile的两轮本地readiness；不是正式神经收益、高轮、攻击或SOTA。",
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
