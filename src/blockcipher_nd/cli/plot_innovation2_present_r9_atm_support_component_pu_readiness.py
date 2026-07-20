from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E98-B support-component PU evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_support_component_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_support_component_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    groups = summary["groups"]
    baselines = summary["ranking_baselines"]
    group_labels = [row["group_id"].replace("group_", "组") for row in groups]

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
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 9.2))
        figure.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.29, wspace=0.34)
        figure.text(
            0.065,
            0.958,
            "创新2 E98-B：PRESENT九轮正例/未标注排序数据就绪审判",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.897,
            "468条独立已知正关系按共享坐标组成不可拆组件，再均分为6组；每次留1组测试、其余5组训练。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.849,
            "候选是保持边缘统计的同步旋转关系，只能称未标注；图中检验无关系/坐标泄漏及简单规则是否主导。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(groups))
        heldout = [row["heldout_relations"] for row in groups]
        candidates = [row["minimum_unlabeled_candidates"] for row in groups]
        axes[0].bar(x - 0.19, heldout, 0.38, label="已知正例", color="#0F766E")
        axes[0].bar(x + 0.19, candidates, 0.38, label="最少未标注候选", color="#D97706")
        axes[0].set_xticks(x, group_labels)
        axes[0].set_ylim(0, max(heldout + candidates) * 1.18)
        axes[0].set_ylabel("每组数量")
        axes[0].set_title("6组配平且候选充足", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        overlap_names = ("关系", "组件", "完整坐标")
        overlap_values = (
            metrics["maximum_relation_overlap"],
            metrics["maximum_component_overlap"],
            metrics["maximum_support_coordinate_overlap"],
        )
        axes[1].bar(np.arange(3), overlap_values, color=("#2563EB", "#7C3AED", "#DC2626"))
        for index, value in enumerate(overlap_values):
            axes[1].scatter(index, 0.035, s=42, color="#0F766E", zorder=3)
            axes[1].text(
                index,
                0.075,
                f"{value} / 通过",
                ha="center",
                va="bottom",
                color="#0F766E",
                fontweight="bold",
            )
        axes[1].set_xticks(np.arange(3), overlap_names)
        axes[1].set_ylim(0, max(1.0, max(overlap_values) + 0.25))
        axes[1].set_ylabel("训练集与留出集重合数")
        axes[1].set_title("三层泄漏检查", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].text(
            0.02,
            0.91,
            "门槛：三项必须全部为0",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            color="#334155",
        )

        baseline_names = {
            "deterministic_hash_random": "哈希随机",
            "file_id": "组号",
            "relation_size": "关系项数",
            "exponent_weight": "指数重量",
            "exact_training_frequency": "已见关系",
            "training_coordinate_frequency": "坐标频率",
            "training_support_overlap": "支撑重合",
            "absolute_bit_position": "绝对位置",
        }
        baseline_labels = [baseline_names[row["baseline"]] for row in baselines]
        recall = [row["recall_at_5"] for row in baselines]
        mrr = [row["mean_reciprocal_rank"] for row in baselines]
        bx = np.arange(len(baselines))
        axes[2].plot(bx, recall, marker="o", linewidth=2.0, label="Recall@5", color="#0F766E")
        axes[2].plot(bx, mrr, marker="s", linewidth=2.0, label="MRR", color="#D97706")
        axes[2].axhline(0.50, color="#0F766E", linestyle="--", linewidth=1.0, alpha=0.65)
        axes[2].axhline(0.35, color="#D97706", linestyle=":", linewidth=1.2, alpha=0.75)
        axes[2].set_xticks(bx, baseline_labels, rotation=32, ha="right")
        axes[2].set_ylim(0, max(0.58, max(recall + mrr) + 0.08))
        axes[2].set_ylabel("已知正例排序指标")
        axes[2].set_title("简单捷径没有垄断任务", loc="left", fontweight="bold")
        axes[2].legend(frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        conclusion = (
            "通过：只开放E99本地神经排序门，远程仍关闭。"
            if gate["status"] == "pass"
            else "未通过：不训练E99，停止当前九轮公开语料路线。"
        )
        figure.text(
            0.065,
            0.176,
            (
                f"数据结论：{metrics['canonical_independent_relations']}条独立正关系，"
                f"{metrics['support_components']}个支撑组件，6组各"
                f"{metrics['minimum_group_positives']}条；每个正例至少"
                f"{metrics['minimum_unlabeled_candidates']}个未标注候选。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(0.065, 0.108, f"裁决：{conclusion}", ha="left", va="bottom", fontsize=9.8, color="#334155")
        figure.text(
            0.065,
            0.045,
            "证据范围：同一公开ATM语料内部的九轮结构泛化就绪门；不是神经结果、PRESENT-80密钥调度验证、区分器、攻击或SOTA。",
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
