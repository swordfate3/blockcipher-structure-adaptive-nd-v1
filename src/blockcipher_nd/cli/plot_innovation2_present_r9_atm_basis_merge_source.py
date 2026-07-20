from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E98-A ATM source audit evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_basis_merge_audit(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_basis_merge_audit(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    files = summary["file_ranks"]
    dependencies = summary["dependencies"]
    splits = summary["split_coverage"]
    metrics = gate["metrics"]
    file_labels = [row["split"] for row in files]
    serialized = [row["serialized_basis_elements"] for row in files]
    ranks = [row["recomputed_rank"] for row in files]
    split_labels = [row["split"] for row in splits]
    split_state = [1 if row["evidence_state"] == "published_result" else 0 for row in splits]

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
            "创新2 E98-A：九轮ATM的470为什么重算只有468维",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.897,
            "8个公开split各自都是满秩基底；秩亏发生在跨split合并，恢复出两条可逐坐标复算的GF(2)依赖。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.849,
            "冻结公开merge源码调用row_reduce()但未接收返回矩阵；保存的470等于去重关系数，不等于当前重算秩。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(files))
        width = 0.38
        axes[0].bar(x - width / 2, serialized, width, label="序列化元素", color="#64748B")
        axes[0].bar(x + width / 2, ranks, width, label="重算秩", color="#0F766E")
        axes[0].set_xticks(x, file_labels, rotation=28, ha="right")
        axes[0].set_ylabel("每个split的维度")
        axes[0].set_title("单文件基底没有秩亏", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        merge_labels = ("作者保存计数", "去重关系", "正确GF(2)秩")
        merge_values = (
            metrics["data_analysis_saved_merge_count"],
            metrics["deduplicated_relations"],
            metrics["recomputed_union_rank"],
        )
        merge_colors = ("#D97706", "#64748B", "#0F766E")
        mx = np.arange(3)
        axes[1].bar(mx, merge_values, color=merge_colors)
        for index, value in enumerate(merge_values):
            axes[1].text(index, value + 1.0, str(value), ha="center", fontsize=9.5)
        axes[1].set_xticks(mx, merge_labels, rotation=18, ha="right")
        axes[1].set_ylim(455, 474)
        axes[1].set_ylabel("合并后的数量/秩")
        axes[1].set_title("470是计数，468是重算秩", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].text(
            0.02,
            0.91,
            "依赖基：3成员×1，4成员×1",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            color="#7C2D12",
            fontsize=9.2,
        )

        sx = np.arange(len(splits))
        colors = ["#0F766E" if value else "#DC2626" for value in split_state]
        axes[2].bar(sx, split_state, color=colors)
        axes[2].set_xticks(sx, split_labels, rotation=28, ha="right")
        axes[2].set_yticks((0, 1), ("无公开结果", "公开结果"))
        axes[2].set_ylim(0, 1.25)
        axes[2].set_title("声明9个split，公开8个", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        missing_index = split_labels.index("3-3-3")
        axes[2].text(
            missing_index,
            0.08,
            "缺失",
            ha="center",
            va="bottom",
            color="#991B1B",
            fontweight="bold",
        )

        figure.text(
            0.065,
            0.182,
            (
                f"结论：8个单文件基底均满秩；合并后{metrics['deduplicated_relations']}个不同关系只张成"
                f"{metrics['recomputed_union_rank']}维空间，nullity={metrics['union_nullity']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.065,
            0.108,
            "裁决：来源差异已解释，但E98仍只有24条严格文件留出正例；不训练E99，不生成缺失split。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.065,
            0.045,
            "证据范围：冻结公开源码与结果的可执行合并审计；不是九轮神经结果、PRESENT-80调度验证、区分器、攻击或SOTA。",
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
