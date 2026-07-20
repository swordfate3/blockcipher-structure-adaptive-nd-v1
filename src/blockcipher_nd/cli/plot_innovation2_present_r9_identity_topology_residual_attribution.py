from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "coordinate_anchor",
    "identity_true_p_residual",
    "identity_wrong_p_residual",
)
MODEL_LABELS = {
    "coordinate_anchor": "坐标身份锚点",
    "identity_true_p_residual": "身份+真实P残差",
    "identity_wrong_p_residual": "身份+错误P残差",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E100 identity/topology residual attribution.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_identity_topology_residual(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_identity_topology_residual(summary: dict[str, Any], output: Path) -> None:
    aggregates = summary["aggregate_metrics"]
    folds = summary["fold_metrics"]
    gate = summary["gate"]

    def value(model: str, seed: int, metric: str) -> float:
        return next(row[metric] for row in aggregates if row["model"] == model and row["seed"] == seed)

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
        figure.text(0.065, 0.958, "创新2 E100：PRESENT九轮坐标身份保持拓扑残差归因", ha="left", va="top", fontsize=15.2, fontweight="bold", color="#0F172A")
        figure.text(0.065, 0.897, "保留E99近满分的坐标身份主干，只比较真实PRESENT P-layer残差与同参数错误P残差。", ha="left", va="top", fontsize=9.8, color="#526070")
        figure.text(0.065, 0.849, "主指标改为仍有提升空间的Recall@1与MRR；候选仍是未标注关系，不是密码学负例。", ha="left", va="top", fontsize=9.8, color="#526070")

        x = np.arange(len(MODEL_ORDER))
        width = 0.36
        colors = ("#0F766E", "#D97706")
        for axis, metric, title, ylabel in (
            (axes[0], "recall_at_1", "已知正例排到第1名的比例", "Recall@1"),
            (axes[1], "mean_reciprocal_rank", "已知正例的平均倒数排名", "MRR"),
        ):
            for seed in (0, 1):
                values = [value(model, seed, metric) for model in MODEL_ORDER]
                axis.bar(x + (seed - 0.5) * width, values, width, label=f"seed{seed}", color=colors[seed])
            axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=18, ha="right")
            axis.set_ylim(0, 1.06)
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.legend(frameon=False)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)

        fold_x = np.arange(6)
        for model, color, marker in (
            ("coordinate_anchor", "#64748B", "D"),
            ("identity_true_p_residual", "#0F766E", "o"),
            ("identity_wrong_p_residual", "#D97706", "s"),
        ):
            values = [
                next(row["recall_at_1"] for row in folds if row["model"] == model and row["seed"] == 0 and row["fold"] == fold)
                for fold in range(6)
            ]
            axes[2].plot(fold_x, values, marker=marker, linewidth=2.0, color=color, label=MODEL_LABELS[model])
        axes[2].set_xticks(fold_x, [f"折{fold}" for fold in range(6)])
        axes[2].set_ylim(0.65, 1.02)
        axes[2].set_ylabel("seed0 每折 Recall@1")
        axes[2].set_title("配对模型的逐折差异", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, fontsize=8.4)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        true0 = next(row for row in aggregates if row["model"] == "identity_true_p_residual" and row["seed"] == 0)
        true1 = next(row for row in aggregates if row["model"] == "identity_true_p_residual" and row["seed"] == 1)
        figure.text(0.065, 0.176, f"真实P残差：seed0 Recall@1={true0['recall_at_1']:.3f} / MRR={true0['mean_reciprocal_rank']:.3f}；seed1 Recall@1={true1['recall_at_1']:.3f} / MRR={true1['mean_reciprocal_rank']:.3f}。", ha="left", va="bottom", fontsize=9.6, color="#334155")
        decision_text = {
            "innovation2_present_r9_identity_true_p_residual_attributed": "通过：真实P残差稳定超过坐标锚点和错误P控制。",
            "innovation2_present_r9_identity_residual_capacity_only": "暂缓：残差容量有效，但真实P没有超过错误P控制。",
            "innovation2_present_r9_coordinate_identity_anchor_remains_best": "停止拓扑分支：坐标身份锚点仍是最佳模型。",
            "innovation2_present_r9_identity_topology_residual_protocol_invalid": "失败：来源、配对模型、fold或指标协议无效。",
        }[gate["decision"]]
        figure.text(0.065, 0.108, f"裁决：{decision_text}", ha="left", va="bottom", fontsize=9.8, color="#334155")
        figure.text(0.065, 0.045, "证据范围：公开ATM独立轮密钥语料内部的本地PU残差归因；不是新九轮关系、PRESENT-80密钥调度、区分器、攻击、远程规模或SOTA。", ha="left", va="bottom", fontsize=9.0, color="#526070")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
