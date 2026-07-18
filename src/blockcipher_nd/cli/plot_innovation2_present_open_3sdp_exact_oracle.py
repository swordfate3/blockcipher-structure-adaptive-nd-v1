from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E53-A PRESENT exact-ANF and cancellation readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_open_3sdp_exact_oracle(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_open_3sdp_exact_oracle(
    summary: dict[str, Any], output: Path
) -> None:
    metrics = summary["metrics"]
    gate = summary["gate"]
    round_metrics = metrics["round_monomial_metrics"]
    fixture_counts = metrics["fixture_counts_by_round"]
    transition = metrics["transition"]
    decisions = {
        "innovation2_present_r5_open_3sdp_exact_oracle_ready": (
            "一、二轮 exact oracle 通过；下一步实现 GLPK trail 奇偶枚举器。"
        ),
        "innovation2_present_r5_open_3sdp_exact_oracle_invalid": (
            "exact ANF、bit order、fixture、mask XOR 或反例协议无效。"
        ),
        "innovation2_present_r5_open_3sdp_cancellation_control_failed": (
            "trail 奇偶未复现 S-box exact ANF，停止 provider 实现。"
        ),
        "innovation2_present_r5_open_3sdp_glpk_runtime_not_ready": (
            "exact oracle 正确，但 Sage/GLPK runtime 未就绪。"
        ),
    }

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
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.4))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.25, wspace=0.38
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E53-A：开放 3SDP 提供者的 exact-ANF 与 GF(2) 消去校准",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "完整保留 PRESENT-80 的 64 个 plaintext 变量和 80 个 key 变量；不是固定 key 或零 offset 近似。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "本图只裁决一、二轮校准 oracle；尚未执行五轮 16×64 子集，也没有训练神经网络。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        rounds = sorted(round_metrics, key=int)
        totals = [round_metrics[rounds_key]["total"] for rounds_key in rounds]
        x = np.arange(len(rounds))
        axes[0].bar(x, totals, color=["#0F766E", "#2563EB"], width=0.58)
        for index, value in enumerate(totals):
            axes[0].text(
                index,
                value * 1.28,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axes[0].set_yscale("log")
        axes[0].set_ylim(500, max(totals) * 4.0)
        axes[0].set_xticks(x, [f"PRESENT {rounds_key}轮" for rounds_key in rounds])
        axes[0].set_ylabel("完整输出 ANF 单项式总数（log）")
        axes[0].set_title("exact oracle 实际规模", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")
        axes[0].text(
            0.02,
            0.94,
            f"生成耗时：{summary['snapshot_seconds']:.2f} 秒",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            color="#334155",
            bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 2.0},
        )

        width = 0.24
        positions = np.arange(len(rounds))
        positives = [fixture_counts[key]["positive"] for key in rounds]
        negatives = [fixture_counts[key]["negative"] for key in rounds]
        multi = [fixture_counts[key]["multi_mask"] for key in rounds]
        axes[1].bar(
            positions - width,
            positives,
            width,
            color="#0F766E",
            label="严格正类",
        )
        axes[1].bar(
            positions,
            negatives,
            width,
            color="#2563EB",
            label="具体反例负类",
        )
        axes[1].bar(
            positions + width,
            multi,
            width,
            color="#D97706",
            label="多 bit mask",
        )
        for offset, values in ((-width, positives), (0, negatives), (width, multi)):
            for index, value in enumerate(values):
                axes[1].text(
                    index + offset,
                    value + 0.25,
                    str(value),
                    ha="center",
                    va="bottom",
                )
        axes[1].set_xticks(positions, [f"{rounds_key}轮" for rounds_key in rounds])
        axes[1].set_ylim(0, 10.5)
        axes[1].set_ylabel("校准 fixture 数量")
        axes[1].set_title("全 key / 全 offset 标签 fixture", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper center", fontsize=8.4)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        transition_names = ["存在任意 trail", "奇数 trail\n真实系数", "偶数抵消\n误报"]
        transition_values = [
            transition["existence_transitions"],
            transition["odd_parity_transitions"],
            transition["existence_only_false_positives"],
        ]
        colors = ["#94A3B8", "#0F766E", "#DC2626"]
        axes[2].bar(np.arange(3), transition_values, color=colors, width=0.62)
        for index, value in enumerate(transition_values):
            axes[2].text(
                index,
                value + 4,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axes[2].set_xticks(np.arange(3), transition_names)
        axes[2].set_ylim(0, max(transition_values) * 1.22)
        axes[2].set_ylabel("PRESENT S-box monomial transition 数量")
        axes[2].set_title("为什么不能只判断 trail 存在", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[2].text(
            0.02,
            0.95,
            f"最大偶数抵消路径数：{transition['maximum_cancelled_trail_count']}",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            color="#B91C1C",
            fontsize=8.8,
        )

        figure.text(
            0.065,
            0.145,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        figure.text(
            0.065,
            0.085,
            "推进门：GLPK 枚举器必须逐项复现 exact oracle；通过前禁止五轮子集、神经训练和远程 GPU。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.035,
            "证据范围：PRESENT-80 一、二轮代数校准；不是五轮标签结果、区分器、密码攻击或 SOTA。",
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
