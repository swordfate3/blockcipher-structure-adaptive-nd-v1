from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E43 PRESENT universal-balance atlas readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_present_universal_balance_atlas(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_present_universal_balance_atlas(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = summary["metrics"]
    raw_counts = [
        metrics["raw_positive"],
        metrics["raw_negative"],
        metrics["raw_unknown"],
    ]
    split_metrics = metrics["matched_split_metrics"]
    raw_auc = metrics["raw_marginal_baselines"]["strongest_auc"]
    matched_auc = metrics["matched_marginal_baselines"]["strongest_auc"]
    decisions = {
        "innovation2_present_universal_balance_atlas_ready": (
            "证书/反例标签与checkerboard反捷径门通过；可进入E44本地seed0神经归因。"
        ),
        "innovation2_present_universal_balance_atlas_too_narrow": (
            "严格标签或matched宽度不足；先扩展结构或改进sound证书。"
        ),
        "innovation2_present_universal_balance_atlas_shortcut_dominated": (
            "matched数据仍可被一元边际解释；禁止神经训练。"
        ),
        "innovation2_present_universal_balance_atlas_protocol_invalid": (
            "ANF、反例、split、mask或证书协议无效。"
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
        figure, axes = plt.subplots(1, 3, figsize=(15.6, 8.2))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.27, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E43：PRESENT 4轮全称平衡证书/反例标签是否可供神经训练",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "正类由ANF支撑缺失证明；负类由具体key与inactive offset上的masked XOR=1反例证明。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "unknown不参与训练；checkerboard同时平衡structure与output mask，阻断一元位置捷径。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        axes[0].bar(x, raw_counts, color=["#0F766E", "#2563EB", "#94A3B8"])
        for index, value in enumerate(raw_counts):
            axes[0].text(index, value + 260, f"{value:,}", ha="center", va="bottom")
        axes[0].set_xticks(x, ["可证明平衡", "反例非平衡", "未知"])
        axes[0].set_ylim(0, max(raw_counts) * 1.17)
        axes[0].set_ylabel("structure × mask 数量")
        axes[0].set_title("原始三态 atlas", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        split_names = ("train", "validation")
        positions = np.arange(2)
        width = 0.34
        positives = [split_metrics[name]["positive"] for name in split_names]
        negatives = [split_metrics[name]["negative"] for name in split_names]
        axes[1].bar(
            positions - width / 2,
            positives,
            width=width,
            color="#0F766E",
            label="平衡",
        )
        axes[1].bar(
            positions + width / 2,
            negatives,
            width=width,
            color="#2563EB",
            label="非平衡",
        )
        for index, value in enumerate(positives):
            axes[1].text(
                index - width / 2, value + 12, str(value), ha="center", va="bottom"
            )
        for index, value in enumerate(negatives):
            axes[1].text(
                index + width / 2, value + 12, str(value), ha="center", va="bottom"
            )
        axes[1].set_xticks(positions, ["训练", "验证"])
        axes[1].set_ylim(0, max(1.0, max(positives + negatives) * 1.24))
        axes[1].set_ylabel("matched标签数量")
        axes[1].set_title("checkerboard正负匹配", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        auc_values = [raw_auc, matched_auc]
        auc_x = np.arange(2)
        axes[2].bar(auc_x, auc_values, color=["#D97706", "#0F766E"])
        for index, value in enumerate(auc_values):
            axes[2].text(
                index, value + 0.025, f"{value:.3f}", ha="center", va="bottom"
            )
        axes[2].axhline(
            0.65,
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label="训练门 0.65",
        )
        axes[2].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[2].set_xticks(auc_x, ["原始atlas", "matched验证"])
        axes[2].set_ylim(0.4, 1.04)
        axes[2].set_ylabel("最强一元边际 AUC")
        axes[2].set_title("反捷径效果", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, loc="upper right")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        train_structures = split_metrics["train"]["structures"]
        validation_structures = split_metrics["validation"]["structures"]
        figure.text(
            0.065,
            0.178,
            f"matched覆盖：训练 {train_structures} 个structure，验证 {validation_structures} 个structure；两组structure互斥。",
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：真实PRESENT-80四轮标签benchmark readiness；不是高轮积分区分器、神经结果或SOTA攻击。",
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
