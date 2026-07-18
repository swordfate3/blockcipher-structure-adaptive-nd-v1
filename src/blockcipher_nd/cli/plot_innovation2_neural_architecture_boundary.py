from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E93 Innovation 2 architecture boundary synthesis."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_architecture_boundary(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_architecture_boundary(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = gate["metrics"]["architecture_rows"]
    formal = next(
        row for row in rows if row["route"] == "separate_r3_only_profile_operator"
    )
    formal_margins = (formal["primary_margin"], formal["secondary_margin"])
    diagnostic_routes = (
        "rectangle_untyped_r3_only_operator",
        "rectangle_row_typed_representation",
        "rectangle_row_typed_shift_operator",
        "shared_topology_parameterized_operator",
        "skinny_true_ridge_sparse_residual",
    )
    route_labels = (
        "RECTANGLE\n原算子",
        "RECTANGLE\nrow机制",
        "RECTANGLE\nrow神经",
        "共享参数\nGIFT质量",
        "SKINNY\n残差",
    )
    by_route = {row["route"]: row for row in rows}
    diagnostic_margins = [by_route[route]["primary_margin"] for route in diagnostic_routes]
    counts = Counter(row["evidence_class"] for row in rows)
    class_names = ("formal_confirmed", "mechanism_only", "closed", "deferred")
    class_labels = ("正式确认", "机制/诊断", "关闭", "等待证据")
    class_counts = [counts[name] for name in class_names]
    decisions = {
        "innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed": (
            "正式方法仍为PRESENT/GIFT独立算子；停止当前benchmark网络枚举。"
        ),
        "innovation2_architecture_boundary_synthesis_protocol_invalid": (
            "冻结来源或证据分级不一致；必须先修复综合协议。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.3,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.06, right=0.975, top=0.70, bottom=0.29, wspace=0.38
        )
        figure.text(
            0.06,
            0.955,
            "创新2 E93：哪些神经结构已经成立，哪些路线必须停止",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.06,
            0.895,
            "重放PRESENT、GIFT、SKINNY、共享算子与RECTANGLE七个冻结裁决；本实验不训练网络。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.06,
            0.848,
            "柱高只表示各路线内部的控制margin或证据条数，不用于跨密码比较绝对AUC。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(2)
        axes[0].bar(x, formal_margins, color=("#0F766E", "#2563EB"), width=0.58)
        for index, value in enumerate(formal_margins):
            axes[0].text(index, value + 0.006, f"{value:+.4f}", ha="center")
        axes[0].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].set_xticks(x, ("PRESENT\nmean true-wrong", "GIFT\nmean true-wrong"))
        axes[0].set_ylim(0.0, max(formal_margins) + 0.05)
        axes[0].set_ylabel("双seed平均验证AUC margin")
        axes[0].set_title("正式确认的方法", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        dx = np.arange(len(diagnostic_routes))
        colors = ("#94A3B8", "#7C3AED", "#8B5CF6", "#D97706", "#64748B")
        axes[1].bar(dx, diagnostic_margins, color=colors, width=0.64)
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(diagnostic_margins):
            offset = 0.004 if value >= 0 else -0.012
            axes[1].text(index, value + offset, f"{value:+.4f}", ha="center")
        axes[1].set_xticks(dx, route_labels)
        axes[1].set_ylim(
            min(-0.08, min(diagnostic_margins) - 0.03),
            max(0.08, max(diagnostic_margins) + 0.04),
        )
        axes[1].set_ylabel("各路线内部关键margin")
        axes[1].set_title("诊断与关闭边界", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        cx = np.arange(4)
        axes[2].bar(
            cx,
            class_counts,
            color=("#0F766E", "#7C3AED", "#DC2626", "#94A3B8"),
            width=0.64,
        )
        for index, value in enumerate(class_counts):
            axes[2].text(index, value + 0.08, str(value), ha="center")
        axes[2].set_xticks(cx, class_labels)
        axes[2].set_ylim(0, max(class_counts) + 0.8)
        axes[2].set_ylabel("结构路线数量")
        axes[2].set_title("证据等级分布", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.06,
            0.183,
            "重新开放训练的条件：新sound标签、独立任务机制，或同容量预训练前真实拓扑margin至少+0.03。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.06,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.06,
            0.053,
            "证据范围：创新2四种真实SPN的神经结构边界综合；不是新神经收益、高轮攻击或SOTA。",
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
