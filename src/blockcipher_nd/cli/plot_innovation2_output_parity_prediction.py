from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Innovation 2 OP1 output parity prediction readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_output_parity_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_output_parity_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["model"]: row for row in summary["trained_rows"]}
    parity_values = (
        rows["full_output_mlp"]["derived_parity_accuracy"],
        rows["direct_parity_mlp"]["parity_accuracy"],
        rows["direct_parity_label_shuffle"]["parity_accuracy"],
    )
    parity_auc = (
        rows["full_output_mlp"]["derived_parity_macro_auc"],
        rows["direct_parity_mlp"]["parity_macro_auc"],
        rows["direct_parity_label_shuffle"]["parity_macro_auc"],
    )
    labels = ("完整输出\n派生parity", "直接预测\nparity", "训练标签打乱\nparity")
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
        figure, axes = plt.subplots(1, 2, figsize=(14.8, 8.2))
        figure.subplots_adjust(left=0.08, right=0.975, top=0.70, bottom=0.29, wspace=0.28)
        figure.text(
            0.08,
            0.955,
            "创新2 OP1：固定密钥下直接预测密文输出parity",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "输入是未见明文的64个bit；标签是同一秘密密钥下真实密文的输出值或4-bit异或值。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.08,
            0.844,
            "这里没有真假样本：0/1是密文parity本身，不是real-vs-random类别。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        x = np.arange(3)
        colors = ("#64748B", "#0F766E", "#B91C1C")
        axes[0].bar(x, parity_values, width=0.62, color=colors)
        axes[1].bar(x, parity_auc, width=0.62, color=colors)
        for axis, values, title, ylabel in (
            (axes[0], parity_values, "测试集parity准确率", "Accuracy"),
            (axes[1], parity_auc, "测试集parity宏平均AUC", "Macro AUC"),
        ):
            axis.axhline(0.5, color="#334155", linestyle="--", linewidth=1.1, label="随机基线0.5")
            axis.set_xticks(x, labels)
            lower = min(0.45, min(values) - 0.04)
            upper = max(0.60, max(values) + 0.08)
            axis.set_ylim(lower, min(1.02, upper))
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            axis.legend(frameon=False, loc="upper left")
            for index, value in enumerate(values):
                axis.text(index, value + 0.012, f"{value:.3f}", ha="center", fontweight="bold")
        metrics = gate["metrics"]
        figure.text(
            0.08,
            0.185,
            (
                f"完整输出逐bit准确率={metrics['full_bit_accuracy']:.3f}；"
                f"直接parity-派生parity={metrics['direct_minus_derived_parity']:+.3f}；"
                f"直接parity-标签打乱={metrics['direct_minus_shuffled_parity']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        decision_text = (
            "固定密钥输出预测通路通过；连续nibble parity无信号，下一步只开放一轮mask几何校准。"
            if gate["status"] == "pass"
            else "固定密钥输出预测协议无效；只修数据或指标，不解释模型表现。"
        )
        figure.text(
            0.08,
            0.112,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.052,
            "证据边界：PRESENT-80一轮、本地小规模就绪门；不是高轮结果，也未证明直接parity优于完整输出。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
