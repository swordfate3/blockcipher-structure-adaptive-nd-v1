from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E94 RECTANGLE nested-cube monotonic label readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_nested_cube_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_nested_cube_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    dimensions = ("d7", "d8", "d9")
    direct = metrics["direct_counts"]
    matched = metrics["matched_dimension_metrics"]
    unary = metrics["matched_unary_baselines"]
    decisions = {
        "innovation2_rectangle80_nested_cube_monotonic_labels_ready": (
            "严格嵌套标签、单调性、宽度和反捷径门通过；下一步只做无训练机制审计。"
        ),
        "innovation2_rectangle80_nested_cube_monotonic_labels_not_ready": (
            "标签宽度或匹配容量不足；关闭当前嵌套cube神经路线。"
        ),
        "innovation2_rectangle80_nested_cube_monotonic_protocol_invalid": (
            "E88重放、嵌套关系、单调closure或反例语义无效；不解释结果。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.7))
        figure.subplots_adjust(left=0.06, right=0.975, top=0.70, bottom=0.27, wspace=0.34)
        figure.text(
            0.06,
            0.955,
            "创新2 E94：RECTANGLE-80四轮的7/8/9-bit嵌套cube标签是否可用",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.06,
            0.895,
            "192条严格嵌套链，每条链查询64个输出bit；本实验只验证标签与数学单调性，不训练网络。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.06,
            0.847,
            "正类由ANF support证明或由子cube单调继承；负类必须保留具体80-bit key与inactive offset反例。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.24
        colors = ("#0F766E", "#2563EB", "#94A3B8")
        for offset, (key, label, color) in enumerate(
            zip(("positive", "negative", "unknown"), ("可证明平衡", "反例非平衡", "未知"), colors)
        ):
            values = [direct[name][key] for name in dimensions]
            axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axes[0].set_xticks(x, ("7-bit", "8-bit", "9-bit"))
        axes[0].set_ylabel("chain × output bit 数量")
        axes[0].set_title("三个维度的原始严格标签", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, fontsize=8.5)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        prevalence = [metrics["closed_positive_prevalence"][name] for name in dimensions]
        axes[1].plot(x, prevalence, marker="o", linewidth=2.2, color="#0F766E")
        for index, value in enumerate(prevalence):
            axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")
        axes[1].set_xticks(x, ("7-bit", "8-bit", "9-bit"))
        axes[1].set_ylim(0, 1.08)
        axes[1].set_ylabel("resolved标签中的平衡比例")
        axes[1].set_title("平衡性随活动维度单调增加", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        violations = metrics["monotonicity"]
        violation_total = sum(
            violations[key]
            for key in (
                "d7_positive_to_d8_negative",
                "d8_positive_to_d9_negative",
                "d7_positive_to_d9_negative",
            )
        )
        axes[1].text(
            0.02,
            0.05,
            f"正类→超集负类冲突：{violation_total}",
            transform=axes[1].transAxes,
            color="#334155",
        )

        train_positive = [matched[name]["train"]["positive"] for name in dimensions]
        validation_positive = [matched[name]["validation"]["positive"] for name in dimensions]
        axes[2].bar(x - width / 2, train_positive, width, color="#2563EB", label="训练每类")
        axes[2].bar(x + width / 2, validation_positive, width, color="#D97706", label="验证每类")
        for index, value in enumerate(train_positive):
            axes[2].text(index - width / 2, value + max(train_positive) * 0.025, str(value), ha="center")
        for index, value in enumerate(validation_positive):
            axes[2].text(index + width / 2, value + max(train_positive) * 0.025, str(value), ha="center")
        axes[2].set_xticks(x, ("7-bit", "8-bit", "9-bit"))
        axes[2].set_ylabel("匹配后单类标签数量")
        axes[2].set_title("chain-disjoint匹配容量", loc="left", fontweight="bold")
        axes[2].legend(frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.06,
            0.178,
            (
                f"最强一元捷径 AUC={unary['strongest_auc']:.3f}（门槛≤0.650）；"
                f"跨维度状态变化链={metrics['transition_chains']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.06,
            0.108,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.06,
            0.048,
            "证据范围：RECTANGLE-80最终版四轮相邻cube维度标签门；不是神经收益、高轮区分器、攻击或SOTA。",
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
