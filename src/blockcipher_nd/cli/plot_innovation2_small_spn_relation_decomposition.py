from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E64 relation decomposition.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_relation_decomposition(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_relation_decomposition(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    split_metrics = gate["metrics"]["split_metrics"]
    baselines = gate["metrics"]["singleton_status_baselines"]
    split_names = ["train", "unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["训练", "未见S", "未见P", "双重未见"]
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
            "创新2 E64：多坐标relation是否真的包含非平凡GF(2)消去",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "逐样本重算两个坐标的256-bit全密钥parity向量，并分解positive/negative来源。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "nontrivial positive要求两个坐标各自nonzero，但完整parity向量严格相等。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(4)
        trivial = [split_metrics[name]["trivial_positive"] for name in split_names]
        nontrivial = [
            split_metrics[name]["nontrivial_positive"] for name in split_names
        ]
        axes[0].bar(x - 0.18, trivial, width=0.36, color="#2563EB", label="both-zero正类")
        axes[0].bar(x + 0.18, nontrivial, width=0.36, color="#D97706", label="nonzero消去正类")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x, split_labels)
        axes[0].set_ylabel("positive数量（log）")
        axes[0].set_title("正类几乎全是both-balanced", loc="left", fontweight="bold")
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(nontrivial):
            axes[0].text(index + 0.18, value * 1.22, str(value), ha="center", fontweight="bold")

        one_zero = [split_metrics[name]["one_zero_negative"] for name in split_names]
        nontrivial_negative = [
            split_metrics[name]["nontrivial_negative"] for name in split_names
        ]
        axes[1].bar(x - 0.18, one_zero, width=0.36, color="#7C3AED", label="一零一非零")
        axes[1].bar(x + 0.18, nontrivial_negative, width=0.36, color="#0F766E", label="两非零不同")
        axes[1].set_yscale("log")
        axes[1].set_xticks(x, split_labels)
        axes[1].set_ylabel("negative数量（log）")
        axes[1].set_title("负类也以singleton差异为主", loc="left", fontweight="bold")
        axes[1].legend(loc="upper right", frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        heldout = ["unseen_sbox", "unseen_player", "dual_unseen"]
        auc = [baselines[name]["strongest_auc"] for name in heldout]
        limits = [0.80, 0.80, 0.75]
        hx = np.arange(3)
        axes[2].bar(hx, auc, color=["#0F766E", "#7C3AED", "#2563EB"], width=0.58)
        axes[2].scatter(hx, limits, marker="_", s=480, linewidths=2.5, color="#DC2626", label="预注册上限")
        axes[2].set_xticks(hx, ["未见S", "未见P", "双重未见"])
        axes[2].set_ylim(0.45, 1.03)
        axes[2].set_ylabel("singleton-status最强AUC")
        axes[2].set_title("both-balanced基线近乎满分", loc="left", fontweight="bold")
        axes[2].legend(loc="lower left", frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(auc):
            axes[2].text(index, value + 0.012, f"{value:.4f}", ha="center", fontweight="bold")

        figure.text(
            0.07,
            0.17,
            "裁决：非平凡消去正类极窄，singleton平衡状态近乎完全解释标签；停止多坐标网络路线。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "dual仅6条nontrivial positive，对比6152条both-zero positive；不训练新的relation模型。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：16-bit合成SPN全256主密钥精确分解；不是PRESENT/GIFT结果或神经收益。",
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
