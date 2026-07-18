from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E73 r3-only two-seed result.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_r3_only_replication(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_r3_only_replication(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    seed0 = {row["relation_mode"]: row for row in metrics["seed0_rows"]}
    seed1 = {row["relation_mode"]: row for row in metrics["seed1_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立node", "错误P", "正确P")
    seed0_auc = [seed0[mode]["validation_auc"] for mode in modes]
    seed1_auc = [seed1[mode]["validation_auc"] for mode in modes]
    decisions = {
        "innovation2_present_r3_only_two_seed_confirmed": (
            "r3-only双seed保持质量与拓扑增益；确认为更简洁的方法。"
        ),
        "innovation2_present_r3_only_seed_not_replicated": (
            "seed1未复现；保留完整39维E68与seed0诊断。"
        ),
        "innovation2_present_r3_only_replication_protocol_invalid": (
            "seed0/source/contract或seed1协议无效。"
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
            "创新2 E73：r3-only平衡谱算子双seed复核",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "两颗seed均只用第3轮13维前缀，训练30轮；参数4795，比完整E68减少15.6%。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.845,
            "每颗seed同时比较独立node、错误P和正确P，并锚定同seed完整39维正确P。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.34
        axes[0].bar(x - width / 2, seed0_auc, width, label="seed0", color="#64748B")
        axes[0].bar(x + width / 2, seed1_auc, width, label="seed1", color="#2563EB")
        for offset, values in ((-width / 2, seed0_auc), (width / 2, seed1_auc)):
            for index, value in enumerate(values):
                axes[0].text(index + offset, value + 0.012, f"{value:.3f}", ha="center", fontsize=8.8)
        axes[0].axhline(0.93, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.50, 1.04)
        axes[0].set_ylabel("验证AUC")
        axes[0].set_title("双seed三行公平矩阵", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        true_values = (metrics["seed0_true_auc"], metrics["seed1_true_auc"], metrics["mean_true_auc"])
        full_values = (0.9530555555555555, 0.9613888888888888, 0.9572222222222222)
        labels2 = ("seed0", "seed1", "双seed平均")
        axes[1].bar(x - width / 2, full_values, width, label="完整39维E68", color="#94A3B8")
        axes[1].bar(x + width / 2, true_values, width, label="r3-only", color="#0F766E")
        for index, value in enumerate(true_values):
            axes[1].text(index + width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8.8)
        axes[1].set_xticks(x, labels2)
        axes[1].set_ylim(0.90, 0.99)
        axes[1].set_ylabel("正确P验证AUC")
        axes[1].set_title("r3-only与完整前缀", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, ncol=2)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.08,
            0.183,
            (
                f"seed1：正确P-独立={metrics['seed1_true_minus_independent']:+.3f}，"
                f"正确P-错误P={metrics['seed1_true_minus_corrupted']:+.3f}；"
                f"双seed平均相对完整前缀={metrics['mean_minus_e68_full_prefix']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
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
            "证据范围：PRESENT-80四轮、8-bit活动cube的本地双seed方法证据；不是高轮、攻击或SOTA。",
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
