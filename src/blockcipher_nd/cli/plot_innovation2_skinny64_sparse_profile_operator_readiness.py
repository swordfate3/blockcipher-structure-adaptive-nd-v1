from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E83 SKINNY sparse profile-operator readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_sparse_profile_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_sparse_profile_readiness(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in gate["metrics"]["rows"]}
    ridges = gate["metrics"]["ridges"]
    decisions = {
        "innovation2_skinny64_sparse_profile_readiness_passed": (
            "真实稀疏线性图的确定性与两轮神经门均通过；可预注册30轮seed0归因。"
        ),
        "innovation2_skinny64_sparse_profile_topology_not_attributed": (
            "公平ridge未归因真实线性图；停止当前稀疏算子。"
        ),
        "innovation2_skinny64_sparse_profile_readiness_not_passed": (
            "两轮真实图模型未同时超过控制；保留E82标签，停止正式训练。"
        ),
        "innovation2_skinny64_sparse_profile_readiness_protocol_invalid": (
            "E82来源、稀疏图、参数公平、等变或训练协议无效。"
        ),
    }
    neural_auc = [
        rows[mode]["validation_auc"] for mode in ("independent", "corrupted", "true")
    ]
    ridge_auc = [
        ridges[name]["validation_auc"]
        for name in ("local13", "corrupted_sparse39", "true_sparse39")
    ]
    margins = [
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
        gate["metrics"]["true_sparse_ridge_minus_local"],
        gate["metrics"]["true_sparse_ridge_minus_corrupted"],
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.3))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E83：真实SKINNY稀疏线性图能否提升五轮平衡谱预测",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "输入仅为目标五轮前的r4节点平衡谱特征；三行均为13维、4,795参数、两轮训练。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "真实图按ShiftRows+MixColumns聚合1至3个前驱；错误图保持128边、逐节点入度和bit lane。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        names = ("独立node", "错误稀疏图", "真实稀疏图")
        colors = ("#94A3B8", "#2563EB", "#0F766E")
        x = np.arange(3)
        axes[0].bar(x, neural_auc, color=colors)
        for index, value in enumerate(neural_auc):
            axes[0].text(index, value + 0.015, f"{value:.3f}", ha="center")
        axes[0].axhline(0.65, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, names)
        axes[0].set_ylim(0.4, max(0.82, max(neural_auc) + 0.08))
        axes[0].set_ylabel("validation AUC")
        axes[0].set_title("两轮神经readiness", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        axes[1].bar(x, ridge_auc, color=colors)
        for index, value in enumerate(ridge_auc):
            axes[1].text(index, value + 0.015, f"{value:.3f}", ha="center")
        axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[1].set_xticks(x, ("local13", "错误图展开", "真实图展开"))
        axes[1].set_ylim(0.4, max(0.92, max(ridge_auc) + 0.08))
        axes[1].set_ylabel("validation AUC")
        axes[1].set_title("train-only公平ridge", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_names = ("神经-独立", "神经-错误图", "ridge-local", "ridge-错误图")
        margin_x = np.arange(4)
        margin_colors = ["#0F766E" if value >= 0.03 else "#DC2626" for value in margins]
        axes[2].bar(margin_x, margins, color=margin_colors)
        for index, value in enumerate(margins):
            offset = 0.008 if value >= 0 else -0.018
            axes[2].text(index, value + offset, f"{value:+.3f}", ha="center")
        axes[2].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[2].axhline(0.0, color="#64748B", linewidth=1.0)
        axes[2].set_xticks(margin_x, margin_names, rotation=12)
        axes[2].set_ylim(
            min(-0.08, min(margins) - 0.05), max(0.25, max(margins) + 0.06)
        )
        axes[2].set_ylabel("validation AUC差值")
        axes[2].set_title("真实图归因margin", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        graph = gate["metrics"]["contract"]["linear_graph"]
        figure.text(
            0.065,
            0.188,
            (
                f"线性图契约：真实/错误图均{graph['true_edge_count']}条边；"
                f"真实图入度分布1/2/3 = {graph['true_degree_histogram']['1']}/"
                f"{graph['true_degree_histogram']['2']}/{graph['true_degree_histogram']['3']}个节点。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.116,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：SKINNY-64/64五轮严格标签的两轮本地结构readiness；不是正式增益、高轮攻击或SOTA。",
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
