from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E103 real ATM runtime evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_real_atm_compatibility(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_real_atm_compatibility(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
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
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 9.2))
        figure.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.29, wspace=0.36)
        figure.text(
            0.065,
            0.958,
            "创新2 E103：真实ATM运行时与断点恢复兼容性门",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.897,
            "真实对象是单个PRESENT 4-bit S-box的三轮(1,1,1)切片：官方构模、真实Avec、Manager cache和2个worker。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.849,
            "它不含64-bit P-layer或PRESENT-80密钥编排；本图是运行时兼容证据，不是PRESENT九轮新关系。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        time_labels = ("bitset构建", "模型构造", "官方搜索", "runner恢复")
        times = (
            metrics["bitset_build_seconds"],
            metrics["model_build_seconds"],
            metrics["official_anchor_seconds"],
            metrics["runner_resume_seconds"],
        )
        tx = np.arange(len(times))
        axes[0].bar(tx, times, color=("#2563EB", "#7C3AED", "#D97706", "#0F766E"), width=0.64)
        axes[0].set_xticks(tx, time_labels, rotation=18, ha="right")
        axes[0].set_ylabel("墙钟时间（秒）")
        axes[0].set_title("完整兼容性门在硬上限内完成", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(times):
            axes[0].text(index, value + max(times) * 0.025, f"{value:.2f}", ha="center", fontweight="bold")
        axes[0].text(
            0.97,
            0.94,
            f"总计 {metrics['total_seconds']:.2f}s / 硬上限 {metrics['hard_cap_seconds']}s",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=8.7,
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5, "alpha": 0.92},
        )

        call_labels = ("官方锚点", "中断后runner")
        outer_calls = (
            metrics["official_candidate_calls"],
            metrics["runner_candidate_calls"],
        )
        cx = np.arange(2)
        axes[1].bar(cx, outer_calls, color=("#2563EB", "#0F766E"), width=0.58)
        axes[1].set_xticks(cx, call_labels)
        axes[1].set_ylabel("外层Avec候选调用")
        axes[1].set_title("多进程预取透明计数，不伪装零开销", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(outer_calls):
            axes[1].text(index, value + 0.25, str(value), ha="center", fontweight="bold")
        axes[1].text(
            0.03,
            0.96,
            f"中断时worker已调用 {metrics['runner_calls_at_interrupt']} 个；父进程持久化并复用 {metrics['runner_reused_candidates']} 个",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#475569",
            wrap=True,
        )

        relation_labels = ("官方关系数", "runner关系数", "官方秩", "runner秩")
        relation_values = (
            metrics["official_relations"],
            metrics["runner_relations"],
            metrics["official_rank"],
            metrics["runner_rank"],
        )
        rx = np.arange(len(relation_values))
        relation_colors = ("#2563EB", "#0F766E", "#7C3AED", "#D97706")
        axes[2].scatter(
            rx,
            relation_values,
            s=88,
            facecolors="white",
            edgecolors=relation_colors,
            linewidths=2.2,
            zorder=3,
        )
        axes[2].set_xticks(rx, relation_labels, rotation=18, ha="right")
        axes[2].set_ylabel("关系数 / GF(2)秩")
        axes[2].set_ylim(-0.05, 1)
        axes[2].set_yticks((0, 1))
        axes[2].set_title("两边同为空空间：只证明兼容，不是发现", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(relation_values):
            axes[2].text(index, 0.10, str(value), ha="center", fontweight="bold")
        axes[2].text(
            0.03,
            0.96,
            f"内部oracle活动：官方 {metrics['official_internal_oracle_call_sum']:,} / runner {metrics['runner_internal_oracle_call_sum']:,}",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#475569",
        )

        figure.text(
            0.065,
            0.176,
            "门控：冻结来源5/5、环境7/7、真实运行14/14通过；官方与runner的GF(2)空间、秩和singleton集合一致。",
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.065,
            0.108,
            "裁决：通过真实ATM兼容性门；只开放E104的R9 (3,3,3)单split计划，尚未启动R9或R10搜索。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.065,
            0.045,
            "证据范围：本地4-bit S-box切片与独立4-bit轮密钥；不是64-bit PRESENT、PRESENT-80、九轮新关系、区分器、攻击或SOTA。",
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
