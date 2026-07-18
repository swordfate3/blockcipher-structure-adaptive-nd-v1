from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E61 ATM support Phase A.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--supports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.supports.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    render_atm_multicoordinate_support_phase_a(summary, rows, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_atm_multicoordinate_support_phase_a(
    summary: dict[str, Any], rows: Sequence[dict[str, Any]], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.69, bottom=0.28, wspace=0.43
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E61-A：PRESENT两轮多坐标GF(2)消去的完整支撑门",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "冻结池：输入cell 0的16个monomial × 输出cell 0的15个非空monomial，共240坐标。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "本地60秒硬预算；完整odd key-monomial支撑只作标签证书，不作为未来神经输入。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        coverage_values = [
            int(metrics["planned_coordinates"]),
            int(metrics["completed_coordinates"]),
            int(metrics["exact_coordinates"]),
        ]
        axes[0].bar(
            np.arange(3),
            coverage_values,
            color=["#94A3B8", "#0F766E", "#2563EB"],
            width=0.62,
        )
        axes[0].set_xticks(np.arange(3), ["计划", "落盘", "exact"])
        axes[0].set_ylabel("坐标数量")
        axes[0].set_title("60秒只落盘8/240", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(coverage_values):
            axes[0].text(index, value + 4, str(value), ha="center", fontweight="bold")

        query_indices = [int(row["query_index"]) for row in rows]
        support_sizes = [
            row["support"].get("nonzero_support_size") for row in rows
        ]
        exact_x = [
            index for index, size in zip(query_indices, support_sizes) if size is not None
        ]
        exact_y = [int(size) for size in support_sizes if size is not None]
        axes[1].bar(exact_x, exact_y, color="#7C3AED", width=0.72)
        unknown_x = [
            index for index, size in zip(query_indices, support_sizes) if size is None
        ]
        if unknown_x:
            axes[1].scatter(
                unknown_x,
                [3000] * len(unknown_x),
                marker="x",
                s=64,
                linewidths=2,
                color="#DC2626",
                label="trail cap unknown",
                zorder=3,
            )
            axes[1].legend(loc="upper left", frameon=False)
        axes[1].set_yscale("log")
        axes[1].set_xticks(query_indices, [f"v={index + 1}" for index in query_indices])
        axes[1].set_xlabel("u=0时的输出monomial")
        axes[1].set_ylabel("非零odd key项数量（log）")
        axes[1].set_title("二/三次输出项发生支撑爆炸", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        gate_labels = ["来源有效", "完成>=64", "负类atom>=16", "消去关系>=4", "开放训练"]
        gate_values = [
            all(gate["source_checks"].values()),
            gate["readiness_checks"]["completed_coordinates_at_least_64"],
            gate["readiness_checks"]["exact_key_dependent_supports_at_least_16"],
            gate["readiness_checks"]["low_weight_positive_relations_at_least_4"],
            gate["next_action"]["training"],
        ]
        gate_colors = ["#0F766E" if value else "#94A3B8" for value in gate_values]
        axes[2].barh(np.arange(5), [1] * 5, color=gate_colors, height=0.58)
        axes[2].set_yticks(np.arange(5), gate_labels)
        axes[2].set_xlim(0, 1.12)
        axes[2].set_xticks([])
        axes[2].invert_yaxis()
        axes[2].set_title("完整支撑路线未过门", loc="left", fontweight="bold")
        for index, value in enumerate(gate_values):
            axes[2].text(
                0.5,
                index,
                "是" if value else "否",
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
            )

        figure.text(
            0.07,
            0.17,
            "裁决：完整key-polynomial支撑导出不可扩展；不提高cap、不转远程、不训练RCCA。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：停止完整支撑矩阵，改审计不展开全部项的relation级证书或更换可执行严格标签源。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：两轮、192个独立轮密钥变量；不是PRESENT-80主密钥标签、攻击或神经结果。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
