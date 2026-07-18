from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E68 two-seed profile replication.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_profile_operator_replication(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_profile_operator_replication(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    seed_metrics = gate["metrics"]["seed_metrics"]
    seeds = ("seed0", "seed1")
    true_auc = [seed_metrics[seed]["true_auc"] for seed in seeds]
    independent_auc = [seed_metrics[seed]["independent_auc"] for seed in seeds]
    corrupted_auc = [seed_metrics[seed]["corrupted_auc"] for seed in seeds]
    delta_keys = (
        "true_minus_independent",
        "true_minus_corrupted",
        "true_minus_ridge",
    )
    delta_labels = ("正确P - 独立node", "正确P - 错误P", "正确P - ANF ridge")
    decisions = {
        "innovation2_present_profile_operator_two_seed_confirmed": (
            "两颗seed全部通过；保留为真实PRESENT四轮结构方法证据。"
        ),
        "innovation2_present_profile_operator_seed_not_replicated": (
            "seed1或联合门失败；仅保留seed0证据并停止该结构。"
        ),
        "innovation2_present_profile_operator_replication_protocol_invalid": (
            "seed0 source、profile、contract或seed1协议无效。"
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
            "创新2 E68：PRESENT四轮平衡谱算子能否跨随机种子复现",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "seed1只改变随机种子；数据、prefix、5679参数模型、30轮预算和全部正式门保持不变。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "每颗seed都必须独立超过同容量node、错误P和安全ANF-prefix ridge。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(2)
        width = 0.24
        axes[0].bar(x - width, independent_auc, width, label="独立node", color="#94A3B8")
        axes[0].bar(x, corrupted_auc, width, label="错误P", color="#D97706")
        axes[0].bar(x + width, true_auc, width, label="正确P", color="#2563EB")
        for offset, values in ((-width, independent_auc), (0.0, corrupted_auc), (width, true_auc)):
            for index, value in enumerate(values):
                axes[0].text(index + offset, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.7936111111111112, color="#0F766E", linestyle="--", linewidth=1.2)
        axes[0].set_xticks(x, seeds)
        axes[0].set_ylim(0.45, max(0.98, max(true_auc) + 0.06))
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("两seed绝对AUC与三种关系模式", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        dx = np.arange(3)
        offsets = (-0.16, 0.16)
        colors = ("#2563EB", "#7C3AED")
        for offset, seed, color in zip(offsets, seeds, colors, strict=True):
            values = [seed_metrics[seed][key] for key in delta_keys]
            axes[1].bar(dx + offset, values, 0.32, label=seed, color=color)
            for index, value in enumerate(values):
                axes[1].text(index + offset, value + 0.006, f"{value:+.3f}", ha="center")
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.02, color="#D97706", linestyle=":", linewidth=1.2)
        axes[1].set_xticks(dx, delta_labels)
        axes[1].tick_params(axis="x", labelrotation=5)
        axes[1].set_ylim(-0.02, max(0.22, max(seed_metrics[seed][key] for seed in seeds for key in delta_keys) + 0.04))
        axes[1].set_ylabel("验证 AUC 差值")
        axes[1].set_title("每颗seed的独立控制增益", loc="left", fontweight="bold")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        mean = gate["metrics"]["mean_metrics"]
        figure.text(
            0.075,
            0.178,
            (
                f"两seed mean true AUC={mean['true_auc']:.3f}；mean增益："
                f"独立node={mean['true_minus_independent']:+.3f}，"
                f"错误P={mean['true_minus_corrupted']:+.3f}，"
                f"ANF ridge={mean['true_minus_ridge']:+.3f}。"
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
            "证据范围：PRESENT-80四轮严格unit平衡谱的本地双seed方法结果；不是高轮结论、新攻击或SOTA。",
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
