from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DUAL_ID_BASELINE = 0.6843931010265274


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E40 same-budget no-triangle SPN-PRR ablation."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_no_triangle_ablation_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_no_triangle_ablation_svg(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    source_rows = summary["source_rows"]
    candidate_rows = [
        row for row in summary["rows"] if row["label_mode"] == "true"
    ]
    shuffle_rows = [
        row for row in summary["rows"] if row["label_mode"] == "shuffled"
    ]
    decisions = {
        "innovation2_small_spn_pair_relation_triangle_attributed": (
            "triangle稳定领先同预算局部pair更新；路径组合贡献通过归因门。"
        ),
        "innovation2_small_spn_pair_relation_triangle_not_isolated": (
            "triangle未拉开预注册差距；保留pair表示，不作路径组合特异性声明。"
        ),
        "innovation2_small_spn_pair_relation_no_triangle_not_attributed": (
            "no-triangle的label-shuffle异常；暂不解释消融。"
        ),
        "innovation2_small_spn_pair_relation_no_triangle_protocol_invalid": (
            "来源、pair局部性、参数匹配、等变性或训练协议无效。"
        ),
    }
    split_keys = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双重未见"]
    triangle_mean = [
        float(np.mean([row[f"{split}_auc"] for row in source_rows]))
        for split in split_keys
    ]
    local_mean = [
        float(np.mean([row[f"{split}_auc"] for row in candidate_rows]))
        for split in split_keys
    ]
    source_by_seed = {int(row["seed"]): row for row in source_rows}
    candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
    dual_delta = triangle_mean[2] - local_mean[2]
    shuffle_dual = shuffle_rows[0]["dual_unseen_auc"] if shuffle_rows else None

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.7,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.71, bottom=0.25, wspace=0.25
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E40：SPN-PRR的收益是否来自 i→k→j 路径组合",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "单变量消融：保留16×16 pair状态和全部参数，只将triangle消息替换为逐pair局部更新。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "数据、split、hidden64、rank8、40 epochs、readout、seed0/1与checkpoint规则完全相同。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(split_keys))
        width = 0.34
        axes[0].bar(
            x - width / 2,
            triangle_mean,
            width,
            color="#0F766E",
            label="triangle路径组合",
        )
        axes[0].bar(
            x + width / 2,
            local_mean,
            width,
            color="#D97706",
            label="no-triangle局部更新",
        )
        for family_index, values in enumerate((triangle_mean, local_mean)):
            offset = -width / 2 if family_index == 0 else width / 2
            for split_index, value in enumerate(values):
                axes[0].text(
                    split_index + offset,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                )
        axes[0].set_xticks(x, split_labels)
        axes[0].set_ylim(0.45, 1.02)
        axes[0].set_ylabel("两seed平均AUC")
        axes[0].set_title("同预算路径消融", loc="left", fontweight="bold", pad=42)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.7,
        )

        seed_x = np.arange(2)
        triangle_dual = [
            source_by_seed[seed]["dual_unseen_auc"] for seed in (0, 1)
        ]
        local_dual = [
            candidate_by_seed[seed]["dual_unseen_auc"] for seed in (0, 1)
        ]
        axes[1].bar(
            seed_x - width / 2,
            triangle_dual,
            width,
            color="#0F766E",
            label="triangle路径组合",
        )
        axes[1].bar(
            seed_x + width / 2,
            local_dual,
            width,
            color="#D97706",
            label="no-triangle局部更新",
        )
        for family_index, values in enumerate((triangle_dual, local_dual)):
            offset = -width / 2 if family_index == 0 else width / 2
            for seed, value in enumerate(values):
                axes[1].text(
                    seed + offset,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                )
        axes[1].axhline(
            DUAL_ID_BASELINE,
            color="#DC2626",
            linewidth=1.5,
            linestyle="--",
            label="dual ID边际",
        )
        axes[1].set_xticks(seed_x, ["seed0", "seed1"])
        axes[1].set_ylim(0.45, 1.02)
        axes[1].set_ylabel("双重未见拓扑AUC")
        axes[1].set_title("逐seed路径归因", loc="left", fontweight="bold", pad=42)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.7,
        )

        shuffle_text = "未运行" if shuffle_dual is None else f"{shuffle_dual:.3f}"
        figure.text(
            0.075,
            0.165,
            f"dual均值：triangle {triangle_mean[2]:.6f} / no-triangle {local_mean[2]:.6f} / 差值{dual_delta:+.6f}；label-shuffle {shuffle_text}。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.105,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.052,
            "证据范围：16-bit合成SPN路径归因；不是实际密码高轮结果、攻击或SOTA。",
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
