from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROW_LABELS = {
    "pair_local_true_seed0": "pair-local\n正确P-layer",
    "pair_triangle_true_seed0": "triangle\n正确P-layer",
    "pair_local_corrupted_seed0": "pair-local\n错误P-layer",
    "pair_triangle_corrupted_seed0": "triangle\n错误P-layer",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E44 PRESENT pair-state topology attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.history.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    render_pair_state_attribution(summary, history, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_pair_state_attribution(
    summary: dict[str, Any], history: list[dict[str, str]], output: Path
) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    trained = [row for row in rows if row["training_performed"]]
    decisions = {
        "innovation2_present_pair_state_topology_attributed": (
            "候选与正确P-layer归因同时过门；下一步在同一矩阵运行seed1。"
        ),
        "innovation2_present_pair_state_candidate_not_ready": (
            "pair-state未超过冻结AUC门；停止增加容量并审计ANF交互复杂度。"
        ),
        "innovation2_present_pair_state_topology_not_attributed": (
            "预测信号存在，但正确P-layer未稳定领先公平错误拓扑。"
        ),
        "innovation2_present_pair_state_attribution_protocol_invalid": (
            "source、模型、控制、metric或训练协议无效。"
        ),
    }
    colors = {
        "pair_local_true_seed0": "#0F766E",
        "pair_triangle_true_seed0": "#2563EB",
        "pair_local_corrupted_seed0": "#D97706",
        "pair_triangle_corrupted_seed0": "#D97706",
    }
    maximum_epoch = max((int(row["epoch"]) for row in history), default=0)

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
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.28
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E44：64-bit pair-state是否在严格PRESENT四轮标签上使用正确拓扑",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "数据固定为E43 checkerboard：训练400/400、验证118/118；structure组互斥，一元边际AUC=0.5。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            f"左图查看{maximum_epoch}轮学习过程；右图比较正确P-layer候选与同processor公平错误P-layer控制。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        all_auc: list[float] = []
        for row_id in sorted({row["row_id"] for row in history}):
            row_history = [row for row in history if row["row_id"] == row_id]
            epochs = [int(row["epoch"]) for row in row_history]
            aucs = [float(row["validation_auc"]) for row in row_history]
            all_auc.extend(aucs)
            axes[0].plot(
                epochs,
                aucs,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                color=colors.get(row_id, "#64748B"),
                label=ROW_LABELS.get(row_id, row_id).replace("\n", " "),
            )
        axes[0].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[0].axhline(0.6, color="#DC2626", linestyle="--", linewidth=1.3)
        lower = min([0.48, *all_auc]) - 0.025
        upper = max([0.62, *all_auc]) + 0.025
        axes[0].set_ylim(max(0.0, lower), min(1.0, upper))
        axes[0].set_xlabel("训练轮次")
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("验证曲线（纵轴按实际范围放大）", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="best")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        bar_rows = [
            next(row for row in rows if row["row_id"] == "unary_marginal_baseline"),
            *trained,
        ]
        x = np.arange(len(bar_rows))
        values = [float(row["validation_auc"]) for row in bar_rows]
        labels = [
            "一元边际\n基线"
            if row["row_id"] == "unary_marginal_baseline"
            else ROW_LABELS.get(row["row_id"], row["row_id"])
            for row in bar_rows
        ]
        bar_colors = [
            "#94A3B8"
            if row["row_id"] == "unary_marginal_baseline"
            else colors.get(row["row_id"], "#64748B")
            for row in bar_rows
        ]
        axes[1].bar(x, values, color=bar_colors, width=0.62)
        for index, value in enumerate(values):
            axes[1].text(index, value + 0.018, f"{value:.3f}", ha="center")
        axes[1].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.4, max(0.68, max(values) + 0.08))
        axes[1].set_ylabel("最佳验证 AUC")
        axes[1].set_title("候选与拓扑控制", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.178,
            f"胜出处理器：{metrics['selected_processor']}；真拓扑-一元基线={metrics['best_true_minus_unary']:+.3f}；真拓扑-错误拓扑={metrics['best_true_minus_corrupted']:+.3f}。",
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
            "证据范围：PRESENT-80四轮、本地seed0、小数据神经readiness与拓扑归因；不是高轮或远程规模结果。",
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
