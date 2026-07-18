from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E96 post-E95 architecture portfolio boundary."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_architecture_portfolio(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_architecture_portfolio(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    rows = metrics["architecture_rows"]
    category_order = (
        "formal_confirmed",
        "label_ready_but_unattributed",
        "provider_missing",
        "mechanism_only_closed",
        "closed",
        "deferred_no_budget",
    )
    category_labels = (
        "正式确认",
        "标签就绪但未归因",
        "缺严格标签提供器",
        "机制边界关闭",
        "已关闭",
        "暂缓无预算",
    )
    counts = [metrics["evidence_class_counts"][name] for name in category_order]
    short_names = (
        "PRESENT/GIFT\nr3-only",
        "RECTANGLE\nCube-Lattice",
        "Mask-Query\nHypergraph",
        "活动维度\n条件算子",
        "RECTANGLE\nRow/原算子",
        "SKINNY\n残差",
        "PRESENT/GIFT\n共享算子",
        "通用\nTransformer/GNN",
    )
    stage_matrix = np.asarray(
        [
            [row["label_ready"], row["mechanism_ready"], row["formal_neural"]]
            for row in rows
        ],
        dtype=np.int8,
    )
    decisions = {
        "innovation2_architecture_portfolio_converged_no_new_training_budget": (
            "当前没有可立即训练的新架构；停止枚举，转严格provider研究或论文收束。"
        ),
        "innovation2_architecture_portfolio_new_candidate_ready": (
            "出现通过全部前置门的新候选；只允许预注册排名第一的候选。"
        ),
        "innovation2_architecture_portfolio_protocol_invalid": (
            "冻结来源或候选分类不一致；必须先修复证据综合。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.1,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.4, 9.3))
        figure.subplots_adjust(left=0.06, right=0.975, top=0.70, bottom=0.30, wspace=0.39)
        figure.text(
            0.06,
            0.958,
            "创新2 E96：现在哪些神经结构真正还有实验资格",
            ha="left",
            va="top",
            fontsize=15.4,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.06,
            0.898,
            "重放E69/E70/E80/E84/E86/E90/E92-E95共10个冻结gate；只做证据与训练预算裁决。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.06,
            0.850,
            "网络名称不能替代严格标签、独立机制和错误关系控制；本实验不训练模型，也不启动远程GPU。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        y = np.arange(len(category_order))
        category_colors = ("#0F766E", "#2563EB", "#D97706", "#7C3AED", "#DC2626", "#64748B")
        axes[0].barh(y, counts, color=category_colors)
        for index, value in enumerate(counts):
            axes[0].text(value + 0.05, index, str(value), va="center")
        axes[0].set_yticks(y, category_labels)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, max(counts) + 0.6)
        axes[0].set_xlabel("候选数量")
        axes[0].set_title("八类候选的证据归属", loc="left", fontweight="bold")
        axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        cmap = ListedColormap(("#FEE2E2", "#0F766E"))
        axes[1].imshow(stage_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks(np.arange(3), ("严格标签", "独立机制", "正式神经"))
        axes[1].set_yticks(np.arange(len(rows)), short_names)
        for row_index in range(len(rows)):
            for column in range(3):
                value = int(stage_matrix[row_index, column])
                axes[1].text(
                    column,
                    row_index,
                    "通过" if value else "未过",
                    ha="center",
                    va="center",
                    color="#FFFFFF" if value else "#991B1B",
                    fontsize=8.1,
                )
        axes[1].set_title("每个结构到哪一道门", loc="left", fontweight="bold")
        axes[1].tick_params(axis="both", length=0)

        budget_labels = ("立即训练", "仅provider研究", "保留正式方法", "关闭/暂缓")
        budget_values = (
            metrics["immediately_trainable_candidate_count"],
            metrics["provider_missing_candidate_count"],
            metrics["formal_method_family_count"],
            len(rows)
            - metrics["provider_missing_candidate_count"]
            - metrics["formal_method_family_count"],
        )
        budget_colors = ("#DC2626", "#D97706", "#0F766E", "#64748B")
        budget_x = np.arange(4)
        axes[2].bar(budget_x, budget_values, color=budget_colors)
        for index, value in enumerate(budget_values):
            axes[2].text(index, value + 0.12, str(value), ha="center")
        axes[2].set_xticks(budget_x, budget_labels, rotation=18, ha="right")
        axes[2].set_ylim(0, max(budget_values) + 1.0)
        axes[2].set_ylabel("候选数量")
        axes[2].set_title("下一阶段预算分配", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.06,
            0.190,
            (
                f"正式方法：{metrics['formal_method_family_count']}个家族、"
                f"{metrics['formal_real_spn_count']}个真实SPN；"
                f"可立即训练新候选：{metrics['immediately_trainable_candidate_count']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.06,
            0.114,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.06,
            0.050,
            "证据范围：创新2方法与架构预算边界；不是新神经收益、高轮区分器、攻击或SOTA。",
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
