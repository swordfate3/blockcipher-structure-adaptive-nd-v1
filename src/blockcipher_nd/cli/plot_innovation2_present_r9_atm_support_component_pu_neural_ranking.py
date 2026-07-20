from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "absolute_position",
    "summary_mlp",
    "coordinate_deepsets",
    "present_topology_set",
    "present_topology_set_label_shuffle",
)
MODEL_LABELS = {
    "absolute_position": "位置规则",
    "summary_mlp": "摘要MLP",
    "coordinate_deepsets": "坐标集合网",
    "present_topology_set": "P层拓扑网",
    "present_topology_set_label_shuffle": "拓扑网标签打乱",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E99 PRESENT r9 PU neural ranking evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_pu_neural_ranking(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_pu_neural_ranking(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    aggregates = summary["aggregate_metrics"]
    folds = summary["fold_metrics"]

    def value(model: str, seed: int, metric: str) -> float:
        return next(
            row[metric]
            for row in aggregates
            if row["model"] == model and row["seed"] == seed
        )

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
        figure.text(0.065, 0.958, "创新2 E99：PRESENT九轮已知正关系/未标注候选神经排序", ha="left", va="top", fontsize=15.2, fontweight="bold", color="#0F172A")
        figure.text(0.065, 0.897, "六折按共享坐标与旋转轨道双互斥；网络学习把每池唯一的公开已知正关系排在未标注候选之前。", ha="left", va="top", fontsize=9.8, color="#526070")
        figure.text(0.065, 0.849, "指标是Recall@5与MRR，不是二分类准确率；P层拓扑网必须同时超过位置规则、摘要和坐标集合控制。", ha="left", va="top", fontsize=9.8, color="#526070")

        x = np.arange(len(MODEL_ORDER))
        width = 0.36
        colors = ("#0F766E", "#D97706")
        for axis, metric, title, ylabel in (
            (axes[0], "recall_at_5", "已知正例进入前5名的比例", "Recall@5"),
            (axes[1], "mean_reciprocal_rank", "已知正例的平均倒数排名", "MRR"),
        ):
            for seed in (0, 1):
                values = [value(model, seed, metric) for model in MODEL_ORDER]
                axis.bar(x + (seed - 0.5) * width, values, width, label=f"seed{seed}", color=colors[seed])
            anchor = value("absolute_position", 0, metric)
            required = anchor + (0.05 if metric == "recall_at_5" else 0.03)
            axis.axhline(anchor, color="#64748B", linestyle=":", linewidth=1.2, label="位置规则锚点")
            axis.axhline(required, color="#DC2626", linestyle="--", linewidth=1.2, label="拓扑最低线")
            axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=24, ha="right")
            axis.set_ylim(0, max(required + 0.08, max(value(model, seed, metric) for model in MODEL_ORDER for seed in (0, 1)) + 0.08))
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.legend(frameon=False, fontsize=8.4)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)

        fold_x = np.arange(6)
        for seed, color, marker in ((0, "#0F766E", "o"), (1, "#D97706", "s")):
            topology = [
                next(row["recall_at_5"] for row in folds if row["model"] == "present_topology_set" and row["seed"] == seed and row["fold"] == fold)
                for fold in range(6)
            ]
            shuffled = [
                next(row["recall_at_5"] for row in folds if row["model"] == "present_topology_set_label_shuffle" and row["seed"] == seed and row["fold"] == fold)
                for fold in range(6)
            ]
            axes[2].plot(fold_x, topology, marker=marker, linewidth=2.0, color=color, label=f"真实拓扑 seed{seed}")
            axes[2].plot(fold_x, shuffled, marker=marker, linewidth=1.2, linestyle="--", alpha=0.55, color=color, label=f"标签打乱 seed{seed}")
        axes[2].axhline(0.10, color="#DC2626", linestyle=":", linewidth=1.2, label="单折最低线0.10")
        axes[2].set_xticks(fold_x, [f"折{fold}" for fold in range(6)])
        axes[2].set_ylim(0, max(0.35, axes[2].get_ylim()[1]))
        axes[2].set_ylabel("每折 Recall@5")
        axes[2].set_title("真实拓扑与标签打乱的六折稳定性", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, fontsize=8.2)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        topology_rows = [row for row in aggregates if row["model"] == "present_topology_set"]
        figure.text(0.065, 0.176, f"拓扑网：seed0 Recall@5={topology_rows[0]['recall_at_5']:.3f} / MRR={topology_rows[0]['mean_reciprocal_rank']:.3f}；seed1 Recall@5={topology_rows[1]['recall_at_5']:.3f} / MRR={topology_rows[1]['mean_reciprocal_rank']:.3f}。", ha="left", va="bottom", fontsize=9.6, color="#334155")
        decision_text = {
            "innovation2_present_r9_pu_topology_neural_signal_confirmed": "通过：九轮拓扑神经信号确认，只开放远程方案设计。",
            "innovation2_present_r9_pu_generic_neural_signal_only": "暂缓：只有通用神经信号，PRESENT拓扑归因未通过。",
            "innovation2_present_r9_pu_public_corpus_neural_route_stopped": "停止：公开九轮语料上的神经排序未稳定超过控制。",
            "innovation2_present_r9_pu_neural_ranking_protocol_invalid": "失败：来源、折、候选或指标协议无效。",
        }[gate["decision"]]
        figure.text(0.065, 0.108, f"裁决：{decision_text}", ha="left", va="bottom", fontsize=9.8, color="#334155")
        figure.text(0.065, 0.045, "证据范围：公开ATM独立轮密钥语料内部的本地PU排序；不是严格负类分类、新九轮关系、PRESENT-80密钥调度、区分器、攻击、远程规模或SOTA。", ha="left", va="bottom", fontsize=9.0, color="#526070")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
