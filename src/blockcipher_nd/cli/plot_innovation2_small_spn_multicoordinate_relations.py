from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E62 relation readiness.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_small_spn_multicoordinate_relations(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_small_spn_multicoordinate_relations(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    splits = metrics["split_metrics"]
    baselines = metrics["marginal_baselines"]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.69, bottom=0.28, wspace=0.43
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E62：小型SPN严格多坐标relation标签是否可训练",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "65,536个标签盲坐标对；每条标签穷尽冻结8-bit主密钥空间的全部256把密钥。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "36个训练拓扑选择relation模板；12/12/4个未见S、未见P、双重未见拓扑只用于评估。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        round_values = metrics["per_round_selected_templates"]
        axes[0].bar(
            np.arange(4),
            round_values,
            color=["#94A3B8", "#0F766E", "#2563EB", "#94A3B8"],
            width=0.62,
        )
        axes[0].set_xticks(np.arange(4), ["2轮", "3轮", "4轮", "5轮"])
        axes[0].set_ylabel("选中relation模板")
        axes[0].set_title("3轮与4轮各保留1024条", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(round_values):
            axes[0].text(index, value + 22, str(value), ha="center", fontweight="bold")

        split_names = ["train", "unseen_sbox", "unseen_player", "dual_unseen"]
        labels = ["训练", "未见S", "未见P", "双重未见"]
        positives = [splits[name]["positive"] for name in split_names]
        negatives = [splits[name]["negative"] for name in split_names]
        x = np.arange(4)
        axes[1].bar(x, positives, color="#2563EB", width=0.62, label="positive")
        axes[1].bar(
            x,
            negatives,
            bottom=positives,
            color="#D97706",
            width=0.62,
            label="negative",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("严格标签数量")
        axes[1].set_title("所有split都有宽正负类", loc="left", fontweight="bold")
        axes[1].legend(loc="upper right", frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        heldout = ["unseen_sbox", "unseen_player", "dual_unseen"]
        auc_values = [baselines[name]["strongest_auc"] for name in heldout]
        limits = [0.80, 0.80, 0.75]
        x = np.arange(3)
        axes[2].bar(x, auc_values, color=["#0F766E", "#7C3AED", "#2563EB"], width=0.58)
        axes[2].scatter(x, limits, marker="_", s=480, linewidths=2.5, color="#DC2626", label="预注册上限")
        axes[2].set_xticks(x, ["未见S", "未见P", "双重未见"])
        axes[2].set_ylim(0.45, 0.86)
        axes[2].set_ylabel("最强拓扑无关边际 AUC")
        axes[2].set_title("边际基线未解释标签", loc="left", fontweight="bold")
        axes[2].legend(loc="upper right", frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(auc_values):
            axes[2].text(index, value + 0.012, f"{value:.3f}", ha="center", fontweight="bold")

        figure.text(
            0.07,
            0.17,
            "裁决：严格标签宽度、拓扑敏感性和反捷径门全部通过，可以训练DeepSets与RCCA。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：固定hidden64、40 epochs、seed0/1，比较DeepSets、RCCA、wrong-P与label-shuffle。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：16-bit合成SPN、全256主密钥严格relation标签；不是PRESENT/GIFT结果或攻击。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
