from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E79 GIFT-64 r3-only seed1 replication."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_gift_r3_replication(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_gift_r3_replication(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    per_seed = gate["metrics"]["per_seed"]
    seed_names = ("seed0", "seed1")
    true_auc = [per_seed[seed]["true_auc"] for seed in seed_names]
    independent_auc = [per_seed[seed]["independent_auc"] for seed in seed_names]
    corrupted_auc = [per_seed[seed]["corrupted_auc"] for seed in seed_names]
    margins = {
        "独立node": [per_seed[seed]["true_minus_independent"] for seed in seed_names],
        "错误P": [per_seed[seed]["true_minus_corrupted"] for seed in seed_names],
        "公平ridge": [per_seed[seed]["true_minus_e77_ridge"] for seed in seed_names],
    }
    decisions = {
        "innovation2_gift64_r3_only_two_seed_confirmed": (
            "真实GIFT P的30轮质量与拓扑增益在两颗seed均通过；冻结该结果。"
        ),
        "innovation2_gift64_r3_only_seed_not_replicated": (
            "seed1未独立通过全部门；保留seed0与确定性证据。"
        ),
        "innovation2_gift64_r3_only_replication_protocol_invalid": (
            "E75/E78来源、参数公平或seed1协议无效。"
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
            "创新2 E79：GIFT-64真实P-layer的30轮神经增益能否由seed1独立复现",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "seed0与seed1使用相同E75标签、13维r3输入、4,795参数三行和30轮预算。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "每颗seed必须独立超过独立node、错误P和E77公平ridge；平均值不能掩盖单seed失败。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(2)
        width = 0.24
        axes[0].bar(x - width, independent_auc, width, label="独立node", color="#94A3B8")
        axes[0].bar(x, corrupted_auc, width, label="错误P", color="#2563EB")
        axes[0].bar(x + width, true_auc, width, label="真实P", color="#0F766E")
        for index, value in enumerate(true_auc):
            axes[0].text(index + width, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.80, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(
            gate["metrics"]["e77_true_topology_ridge_auc"],
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
        )
        axes[0].set_xticks(x, ("seed0", "seed1"))
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("validation AUC")
        axes[0].set_title("两颗seed的三行最佳checkpoint", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=3)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(3)
        width = 0.32
        seed0_margin = [margins[name][0] for name in margins]
        seed1_margin = [margins[name][1] for name in margins]
        axes[1].bar(
            margin_x - width / 2, seed0_margin, width, label="seed0", color="#94A3B8"
        )
        axes[1].bar(
            margin_x + width / 2, seed1_margin, width, label="seed1", color="#0F766E"
        )
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(seed1_margin):
            axes[1].text(index + width / 2, value + 0.007, f"{value:+.3f}", ha="center")
        axes[1].set_xticks(margin_x, tuple(f"真实P -\n{name}" for name in margins))
        axes[1].set_ylim(0.0, max(0.40, max(seed0_margin + seed1_margin) + 0.06))
        axes[1].set_ylabel("validation AUC差值")
        axes[1].set_title("逐seed正式归因margin", loc="left", fontweight="bold")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.183,
            (
                f"双seed mean true AUC={gate['metrics']['mean_true_auc']:.3f}；"
                f"mean真实P-错误P={gate['metrics']['mean_true_minus_corrupted']:+.3f}；"
                f"mean真实P-公平ridge={gate['metrics']['mean_true_minus_e77_ridge']:+.3f}。"
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
            "证据范围：GIFT-64四轮严格unit-profile的双seed归因；不是高轮、零样本跨密码、攻击或SOTA。",
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
