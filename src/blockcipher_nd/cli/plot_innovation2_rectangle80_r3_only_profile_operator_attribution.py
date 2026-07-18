from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E90 RECTANGLE-80 r3-only seed0 attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_rectangle_r3_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_rectangle_r3_attribution(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立节点", "错误P层", "真实P层")
    training = [rows[mode]["train_auc"] for mode in modes]
    validation = [rows[mode]["validation_auc"] for mode in modes]
    margins = (
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
        gate["metrics"]["true_minus_e89_fair_ridge"],
    )
    fair_ridge = gate["metrics"]["e89_true_topology_ridge_auc"]
    decisions = {
        "innovation2_rectangle80_r3_only_neural_gain_attributed": (
            "真实P层的30轮质量与拓扑归因通过；可运行完全相同的seed1。"
        ),
        "innovation2_rectangle80_r3_only_quality_not_confirmed": (
            "绝对质量、过拟合或公平ridge门未过；关闭正式神经路线。"
        ),
        "innovation2_rectangle80_r3_only_topology_not_attributed": (
            "真实P层未稳定领先同参数控制；关闭正式神经路线。"
        ),
        "innovation2_rectangle80_r3_only_attribution_protocol_invalid": (
            "E88/E89来源、cell-major拓扑、参数公平或30轮协议无效。"
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
        figure, axes = plt.subplots(1, 2, figsize=(15.0, 8.1))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.28, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E90：真实RECTANGLE P层的30轮seed0神经增益能否归因",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "三行从同一seed0重新训练30轮，均为4,795参数；输入固定为13维第3轮ANF前缀。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            f"E89真实P层公平ridge AUC={fair_ridge:.6f}，作为信息范围对齐的确定性安全锚点。",
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
                value - 0.035,
                f"{value:.3f}",
                ha="center",
                color="#FFFFFF",
                fontweight="bold",
            )
        axes[0].axhline(0.80, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(
            fair_ridge,
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="E89公平ridge",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("30轮最佳checkpoint", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=3)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(3)
        margin_labels = (
            "真实P -\n独立节点",
            "真实P -\n错误P",
            "真实P -\n公平ridge",
        )
        axes[1].bar(
            margin_x,
            margins,
            color=("#0F766E", "#2563EB", "#7C3AED"),
            width=0.62,
        )
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            axes[1].text(index, value + 0.006, f"{value:+.4f}", ha="center")
        axes[1].set_xticks(margin_x, margin_labels)
        axes[1].set_ylim(
            min(-0.08, min(margins) - 0.04),
            max(0.22, max(margins) + 0.05),
        )
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("正式质量与拓扑归因门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.183,
            (
                f"真实P train-validation gap={gate['metrics']['true_train_validation_gap']:+.3f}；"
                f"错误P margin={gate['metrics']['true_minus_corrupted']:+.4f} "
                f"< +0.0300；最佳epoch={rows['true']['best_epoch']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：RECTANGLE-80四轮严格标签的30轮seed0归因；不是双seed、7轮复现、高轮攻击或SOTA。",
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
