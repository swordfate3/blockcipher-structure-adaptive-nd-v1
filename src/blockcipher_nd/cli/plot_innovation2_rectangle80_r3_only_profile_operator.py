from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E89 RECTANGLE-80 r3-only profile readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_rectangle_r3_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_rectangle_r3_readiness(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立节点", "错误P层", "真实P层")
    training = [rows[mode]["train_auc"] for mode in modes]
    validation = [rows[mode]["validation_auc"] for mode in modes]
    ridges = summary["deterministic_ridges"]
    ridge_modes = ("local", "corrupted", "true")
    ridge_values = [ridges[mode]["validation_auc"] for mode in ridge_modes]
    margins = (
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
        gate["metrics"]["true_minus_fair_ridge"],
    )
    decisions = {
        "innovation2_rectangle80_r3_only_profile_readiness_passed": (
            "真实P层的两轮神经门和公平基线门全部通过；可进入冻结30轮seed0。"
        ),
        "innovation2_rectangle80_r3_only_topology_baseline_not_ready": (
            "公平确定性拓扑基线未确认真实P层信息；关闭当前神经路线。"
        ),
        "innovation2_rectangle80_r3_only_profile_readiness_not_passed": (
            "真实P层未同时超过神经控制和公平ridge；停止正式训练。"
        ),
        "innovation2_rectangle80_r3_only_profile_protocol_invalid": (
            "E88来源、cell-major重排、拓扑或训练协议无效；必须先修复。"
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
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.3))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E89：真实RECTANGLE P层能否提升四轮积分平衡谱预测",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "输入是E88每个结构的64个输出节点及其13维第3轮ANF前缀；输出是64维平衡/非平衡预测。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "三行均为4,795参数、两轮本地训练；ridge获得相同的本节点、同单元和P层前驱信息。",
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
            axes[0].text(
                index + width / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
            )
        axes[0].axhline(0.65, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("三行两轮神经网络", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        ridge_x = np.arange(3)
        axes[1].bar(
            ridge_x,
            ridge_values,
            color=("#94A3B8", "#D97706", "#0F766E"),
            width=0.62,
        )
        for index, value in enumerate(ridge_values):
            axes[1].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[1].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[1].set_xticks(
            ridge_x,
            ("仅本节点\n13维", "错误P层\n公平39维", "真实P层\n公平39维"),
        )
        axes[1].set_ylim(0.35, 1.04)
        axes[1].set_ylabel("验证AUC")
        axes[1].set_title("信息范围对齐的确定性基线", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(3)
        margin_labels = (
            "真实P -\n独立节点",
            "真实P -\n错误P",
            "真实P -\n公平ridge",
        )
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
            offset = 0.006
            axes[2].text(index, value + offset, f"{value:+.3f}", ha="center")
        axes[2].set_xticks(margin_x, margin_labels)
        axes[2].set_ylim(
            min(-0.10, min(margins) - 0.04),
            max(0.18, max(margins) + 0.05),
        )
        axes[2].set_ylabel("验证AUC差值")
        axes[2].set_title("真实RECTANGLE拓扑归因门", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = summary["contract"]
        figure.text(
            0.065,
            0.183,
            (
                f"参数量={next(iter(contract['parameter_counts'].values()))}；"
                f"cell重标号误差={contract['cell_relabel_max_abs_error']:.2e}；"
                "E88配平坐标=3,192。"
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
            "证据范围：RECTANGLE-80四轮严格标签的两轮本地就绪实验；不是7轮复现、正式收益、高轮攻击或SOTA。",
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
