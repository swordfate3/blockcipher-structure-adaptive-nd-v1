from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E37 expanded small-SPN topology benchmark audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_expanded_topology_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_expanded_topology_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    topology = metrics["topology"]
    fraction_thresholds = gate["thresholds"]["topology_fractions"]
    marginal_limits = gate["thresholds"]["maximum_marginal_auc"]
    split_metrics = metrics["split_metrics"]
    decisions = {
        "innovation2_small_spn_expanded_topology_benchmark_ready": (
            "扩展benchmark通过；开放最小GraphGPS/CETT同预算归因矩阵。"
        ),
        "innovation2_small_spn_expanded_topology_benchmark_not_ready": (
            "扩展benchmark未通过；停止随机P-layer机械扩展并重设数据任务。"
        ),
        "innovation2_small_spn_expanded_topology_protocol_invalid": (
            "缓存、split、train-only选择或公平控制协议无效。"
        ),
    }
    fraction_keys = [
        "train_p_sensitive_any_s",
        "train_p_sensitive_all_s",
        "heldout_p_novel_any_s",
        "dual_p_effect_cells",
        "train_interaction_cells",
        "full_interaction_cells",
    ]
    fraction_labels = [
        "训练P敏感\n至少一个S",
        "训练P敏感\n全部S",
        "未见P产生\n条件变化",
        "dual未见P\n效应",
        "训练S×P\n交互",
        "完整S×P\n交互",
    ]
    split_order = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双重未见"]

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
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
            "创新2 E37：扩展独立P-layer后，合成SPN benchmark能否支持组外学习",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "数据：4个S-box × 16个P-layer × 全部256个toy key；训练含12个独立P-layer，本轮不训练网络。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "选择：只读取36个训练拓扑，保留正类数9至27的round × structure × mask cell。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        fractions = np.asarray(
            [topology["fractions"][key] for key in fraction_keys], dtype=float
        )
        minimums = np.asarray(
            [fraction_thresholds[key] for key in fraction_keys], dtype=float
        )
        x = np.arange(len(fraction_keys))
        axes[0].bar(x, fractions, width=0.64, color="#0F766E", label="实际比例")
        axes[0].scatter(
            x,
            minimums,
            marker="_",
            s=250,
            linewidths=2.3,
            color="#DC2626",
            label="预注册最低门",
            zorder=3,
        )
        for index, value in enumerate(fractions):
            axes[0].text(
                index,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )
        axes[0].set_xticks(x, fraction_labels)
        axes[0].set_ylim(0, 1.12)
        axes[0].set_ylabel("入选base cell比例")
        axes[0].set_title(
            "P-layer敏感性与S×P交互",
            loc="left",
            fontweight="bold",
            pad=42,
        )
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.8,
        )

        auc_values = np.asarray(
            [metrics["marginal_baselines"][split]["strongest_auc"] for split in split_order]
        )
        auc_limits = np.asarray([marginal_limits[split] for split in split_order])
        auc_x = np.arange(len(split_order))
        axes[1].bar(
            auc_x,
            auc_values - 0.5,
            bottom=0.5,
            width=0.56,
            color=["#2563EB", "#D97706", "#7C3AED"],
            label="最强train-only ID边际",
        )
        axes[1].scatter(
            auc_x,
            auc_limits,
            marker="_",
            s=280,
            linewidths=2.4,
            color="#DC2626",
            label="允许上限",
            zorder=3,
        )
        for index, value in enumerate(auc_values):
            axes[1].text(
                index,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
            )
        axes[1].axhline(0.5, color="#94A3B8", linewidth=1.0)
        axes[1].set_xticks(auc_x, split_labels)
        axes[1].set_ylim(0.45, 1.02)
        axes[1].set_ylabel("AUC（越接近0.5，ID捷径越弱）")
        axes[1].set_title(
            "简单ID边际的组外解释能力",
            loc="left",
            fontweight="bold",
            pad=42,
        )
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.8,
        )

        class_text = "；".join(
            f"{label} 正{split_metrics[split]['positive']} / 负{split_metrics[split]['negative']}"
            for split, label in zip(split_order, split_labels, strict=True)
        )
        figure.text(
            0.075,
            0.165,
            (
                f"入选base cell：{metrics['selected_base_cells']}；"
                f"覆盖轮数：{metrics['per_round_selected_cells']}；{class_text}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
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
            "证据范围：16-bit合成SPN精确标签benchmark；不是神经训练、真实密码高轮结果或攻击证明。",
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
