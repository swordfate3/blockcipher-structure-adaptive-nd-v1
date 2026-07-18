from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E57 generalized-relation precursor boundary."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_precursor_boundary(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_precursor_boundary(summary: dict[str, Any], output: Path) -> None:
    metrics = summary["metrics"]
    gate = summary["gate"]
    decisions = {
        "innovation2_present_r9_generalized_relation_scalar_witness_infeasible": (
            "最小precursor relation已需2^60量级明文；关闭直接标量求常数与negative witness。"
        ),
        "innovation2_present_r9_generalized_relation_scalar_witness_ready": (
            "标量复杂度在冻结cap内，可进入常数与witness计算。"
        ),
        "innovation2_present_r9_generalized_relation_precursor_protocol_invalid": (
            "来源或precursor基语义无效。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.70, bottom=0.27, wspace=0.42
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E57：PRESENT九轮广义relation的precursor标量求值是否可执行",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "正确输入基：πu(x)=1[x≤u]，支持大小为2^wt(u)；不是普通monomial x^u。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "本审计只计算复杂度下界，没有枚举2^60明文、没有生成常数/负类，也没有训练网络。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        histogram = metrics["input_weight_histogram"]
        weights = [int(key) for key in sorted(histogram, key=int)]
        counts = [int(histogram[str(weight)]) for weight in weights]
        positions = np.arange(len(weights))
        axes[0].bar(positions, counts, color="#0F766E", width=0.62)
        for index, count in enumerate(counts):
            axes[0].text(index, count + 5, str(count), ha="center", va="bottom", fontweight="bold")
        axes[0].set_xticks(positions, [f"wt(u)={weight}" for weight in weights])
        axes[0].set_ylabel("公开basis坐标数")
        axes[0].set_title("输入precursor重量集中在60--63", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        cost_labels = ["最小relation", "中位relation", "最大relation", "本地标量cap"]
        costs = [
            int(metrics["minimum_precursor_plaintexts_per_relation_key"]),
            int(metrics["median_precursor_plaintexts_per_relation_key"]),
            int(metrics["maximum_precursor_plaintexts_per_relation_key"]),
            int(metrics["maximum_scalar_plaintexts"]),
        ]
        cost_log2 = [math.log2(value) for value in costs]
        cost_positions = np.arange(len(costs))
        axes[1].bar(
            cost_positions,
            cost_log2,
            color=["#D97706", "#D97706", "#D97706", "#64748B"],
            width=0.64,
        )
        axes[1].set_xticks(cost_positions, cost_labels, rotation=15, ha="right")
        axes[1].set_ylabel("每relation、每key明文数的log2")
        axes[1].set_title("正确precursor成本远超2^24门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(costs):
            axes[1].text(
                index,
                cost_log2[index] + 1,
                f"2^{value.bit_length() - 1}" if value and value & (value - 1) == 0 else f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        mapping_labels = ["错误x^u估算\n最小", "正确πu估算\n最小", "两key witness\n最小", "标量cap"]
        mapping_values = [
            int(metrics["minimum_wrong_monomial_plaintexts_per_relation_key"]),
            int(metrics["minimum_precursor_plaintexts_per_relation_key"]),
            int(metrics["minimum_two_key_witness_plaintexts"]),
            int(metrics["maximum_scalar_plaintexts"]),
        ]
        mapping_log2 = [math.log2(value) for value in mapping_values]
        mapping_positions = np.arange(len(mapping_values))
        axes[2].bar(
            mapping_positions,
            mapping_log2,
            color=["#94A3B8", "#D97706", "#DC2626", "#64748B"],
            width=0.64,
        )
        axes[2].set_xticks(mapping_positions, mapping_labels)
        axes[2].set_ylabel("明文数的log2")
        axes[2].set_title("wrong-basis把复杂度方向算反", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, exponent in enumerate(mapping_log2):
            axes[2].text(
                index,
                exponent + 0.8,
                f"2^{int(exponent)}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        figure.text(
            0.07,
            0.165,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.105,
            "下一步：只评估可执行的algebraic/SAT constant与key-dependence provider；不转远程机械枚举。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：ATM precursor-basis标量数据复杂度；不是relation常数、PRESENT-80负类、神经结果或攻击。",
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
