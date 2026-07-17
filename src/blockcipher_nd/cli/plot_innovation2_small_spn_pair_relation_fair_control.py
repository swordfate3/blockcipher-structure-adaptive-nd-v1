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
        description="Render E39 pair-relation fair-corrupted P attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_pair_relation_fair_control_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_pair_relation_fair_control_svg(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    true_rows = summary["true_rows"]
    control_rows = summary["control_rows"]
    decisions = {
        "innovation2_small_spn_pair_relation_topology_confirmed": (
            "真实P-layer收益通过公平控制；下一步只做no-triangle归因消融。"
        ),
        "innovation2_small_spn_pair_relation_topology_not_attributed": (
            "真实P-layer未稳定领先公平错误拓扑；停止拓扑收益声明。"
        ),
        "innovation2_small_spn_pair_relation_attribution_protocol_invalid": (
            "Phase A来源、fair control、seed、参数或metric协议无效。"
        ),
    }
    split_keys = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双重未见"]
    true_mean = [
        float(np.mean([row[f"{split}_auc"] for row in true_rows]))
        for split in split_keys
    ]
    control_mean = [
        float(np.mean([row[f"{split}_auc"] for row in control_rows]))
        for split in split_keys
    ]
    true_by_seed = {int(row["seed"]): row for row in true_rows}
    control_by_seed = {int(row["seed"]): row for row in control_rows}
    dual_delta = true_mean[2] - control_mean[2]

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
            "创新2 E39 Phase B：SPN-PRR是否真正使用正确P-layer",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "公平控制：对每个variant自身P-layer组合固定destination-cell rotation，不跨拓扑替换。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "true与control保持相同labels、split、hidden64、rank8、40 epochs和seed0/1。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(split_keys))
        width = 0.34
        axes[0].bar(
            x - width / 2,
            true_mean,
            width,
            color="#0F766E",
            label="真实P-layer",
        )
        axes[0].bar(
            x + width / 2,
            control_mean,
            width,
            color="#D97706",
            label="公平错误P-layer",
        )
        for index, values in enumerate((true_mean, control_mean)):
            offset = -width / 2 if index == 0 else width / 2
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
        axes[0].set_title("真实拓扑与公平控制", loc="left", fontweight="bold", pad=42)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.8,
        )

        seed_x = np.arange(2)
        true_dual = [true_by_seed[seed]["dual_unseen_auc"] for seed in (0, 1)]
        control_dual = [
            control_by_seed[seed]["dual_unseen_auc"] for seed in (0, 1)
        ]
        axes[1].bar(
            seed_x - width / 2,
            true_dual,
            width,
            color="#0F766E",
            label="真实P-layer",
        )
        axes[1].bar(
            seed_x + width / 2,
            control_dual,
            width,
            color="#D97706",
            label="公平错误P-layer",
        )
        for index, values in enumerate((true_dual, control_dual)):
            offset = -width / 2 if index == 0 else width / 2
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
        axes[1].set_title("逐seed拓扑归因", loc="left", fontweight="bold", pad=42)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.7,
        )

        figure.text(
            0.075,
            0.165,
            f"dual均值：真实{true_mean[2]:.6f} / 公平控制{control_mean[2]:.6f} / 差值{dual_delta:+.6f}。",
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
            "证据范围：16-bit合成SPN拓扑归因；不是实际密码高轮结果、攻击或SOTA。",
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
