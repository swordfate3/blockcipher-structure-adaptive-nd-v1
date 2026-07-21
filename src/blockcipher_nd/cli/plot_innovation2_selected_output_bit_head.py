from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "full64_mlp_true_output",
    "selected8_mlp_true_output",
    "selected8_mlp_label_shuffle",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OP11 selected output-bit head confirmation."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_selected_output_bit_head(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_selected_output_bit_head(
    summary: dict[str, Any], output: Path
) -> None:
    selected_bits = summary["metadata"]["selected_msb_indices"]
    rows = {
        (row["model"], int(row["msb_index"])): row for row in summary["bit_rows"]
    }
    gate = summary["gate"]
    x = np.arange(len(selected_bits))
    labels = [f"bit {bit}" for bit in selected_bits]
    colors = ("#2563EB", "#0F766E", "#B91C1C")
    model_labels = ("完整64输出MLP", "专用八输出MLP", "八输出标签打乱MLP")
    auc_values = {
        model: [rows[(model, bit)]["auc"] for bit in selected_bits]
        for model in MODEL_ORDER
    }
    margin_values = {
        model: [
            rows[(model, bit)]["accuracy_minus_majority"] for bit in selected_bits
        ]
        for model in MODEL_ORDER
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
        figure, axes = plt.subplots(2, 2, figsize=(15.8, 10.3))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.79,
            bottom=0.14,
            hspace=0.44,
            wspace=0.25,
        )
        figure.text(
            0.075,
            0.965,
            "创新2 OP11：PRESENT三轮固定八输出bit的独立密钥确认",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.918,
            "八个MSB-first位置由seed0预注册；当前使用第二把固定未知密钥，不在本次结果中重新选位置。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.075,
            0.882,
            "比较完整64输出MLP、专用八输出MLP及其架构匹配标签打乱控制；标签均为真实密文输出值。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        mode = summary["metadata"]["mode"]
        figure.text(
            0.075,
            0.846,
            (
                "当前为64条训练/64条测试的本地实现门，数值不作性能结论。"
                if mode == "smoke"
                else "当前为2^17条训练、2^16条测试的独立固定密钥确认结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )

        width = 0.25
        auc_axis = axes[0, 0]
        for offset, model, label, color in zip(
            (-width, 0.0, width), MODEL_ORDER, model_labels, colors, strict=True
        ):
            auc_axis.bar(
                x + offset,
                auc_values[model],
                width=width,
                color=color,
                label=label,
            )
        auc_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.51,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.1,
            label="确认门0.510",
        )
        all_auc = [value for values in auc_values.values() for value in values]
        auc_axis.set_ylim(
            max(0.0, min(all_auc + [0.5]) - 0.015),
            min(1.0, max(all_auc + [0.51]) + 0.08),
        )
        auc_axis.set_xticks(x, labels)
        auc_axis.set_ylabel("AUC")
        auc_axis.set_title("八个预注册输出bit的AUC", loc="left", fontweight="bold")
        auc_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        auc_axis.legend(frameon=False, ncol=2, loc="upper left")

        margin_axis = axes[0, 1]
        for model, label, color in zip(MODEL_ORDER, model_labels, colors, strict=True):
            margin_axis.plot(
                x,
                margin_values[model],
                marker="o",
                linewidth=1.5,
                label=label,
                color=color,
            )
        margin_axis.axhline(0.0, color="#475569", linestyle="--", linewidth=1.0)
        margin_axis.axhline(
            0.005,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.1,
            label="确认门+0.005",
        )
        all_margins = [
            value for values in margin_values.values() for value in values
        ]
        margin_axis.set_ylim(
            min(all_margins + [0.0]) - 0.02,
            max(all_margins + [0.005]) + 0.05,
        )
        margin_axis.set_xticks(x, labels)
        margin_axis.set_ylabel("Accuracy - majority")
        margin_axis.set_title(
            "逐bit准确率超过多数类基线的幅度",
            loc="left",
            fontweight="bold",
        )
        margin_axis.grid(color="#E5E7EB", linewidth=0.7)
        margin_axis.legend(frameon=False, ncol=2, loc="upper left")

        mean_axis = axes[1, 0]
        means = [float(np.mean(auc_values[model])) for model in MODEL_ORDER]
        mean_axis.bar(np.arange(3), means, width=0.62, color=colors)
        mean_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        mean_axis.set_xticks(np.arange(3), model_labels)
        mean_axis.set_ylabel("八位置平均AUC")
        mean_axis.set_title("模型平均表现与专用头增益", loc="left", fontweight="bold")
        low = max(0.0, min(means + [0.5]) - 0.015)
        high = min(1.0, max(means + [0.5]) + 0.02)
        mean_axis.set_ylim(low, high)
        mean_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        for index, value in enumerate(means):
            mean_axis.text(
                index,
                value + (high - low) * 0.04,
                f"{value:.4f}",
                ha="center",
                fontweight="bold",
            )

        confirm_axis = axes[1, 1]
        confirmations = gate["bit_confirmation"]
        selected_auc = [row["selected8_auc"] for row in confirmations]
        shuffle_auc = [row["selected8_shuffle_auc"] for row in confirmations]
        confirm_axis.bar(
            x - 0.18,
            selected_auc,
            width=0.36,
            color="#0F766E",
            label="专用八输出MLP",
        )
        confirm_axis.bar(
            x + 0.18,
            shuffle_auc,
            width=0.36,
            color="#B91C1C",
            label="架构匹配标签打乱",
        )
        confirm_axis.axhline(
            0.51,
            color="#475569",
            linestyle="--",
            linewidth=1.0,
            label="AUC确认门0.510",
        )
        confirm_axis.set_xticks(x, labels)
        confirm_axis.set_ylabel("AUC")
        confirm_axis.set_title(
            "独立密钥逐bit确认与匹配控制",
            loc="left",
            fontweight="bold",
        )
        values = selected_auc + shuffle_auc + [0.51]
        confirm_axis.set_ylim(
            max(0.0, min(values) - 0.015), min(1.0, max(values) + 0.08)
        )
        confirm_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        confirm_axis.legend(frameon=False, loc="upper left")

        metrics = gate["metrics"]
        decision_text = {
            "innovation2_selected8_independent_key_local_smoke_passed": (
                "固定八输出独立密钥实现门通过"
            ),
            "innovation2_selected8_cross_key_and_dedicated_head_supported": (
                "八输出位置跨密钥成立，且专用头优于完整输出anchor"
            ),
            "innovation2_selected8_cross_key_supported_without_head_gain": (
                "八输出位置跨密钥成立，但专用头未超过完整输出anchor"
            ),
            "innovation2_selected8_not_cross_key_supported": (
                "固定八输出位置未通过独立密钥确认"
            ),
            "innovation2_selected8_independent_key_protocol_invalid": "实验协议无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.075,
            0.068,
            (
                f"裁决：{decision_text}；确认={metrics['confirmed_count']}/8；"
                f"专用头-完整输出平均AUC={metrics['mean_selected8_auc_minus_full64']:+.4f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.075,
            0.028,
            "证据边界：第二固定密钥PRESENT三轮八个预注册真实输出bit；不是完整密文恢复、广泛跨密钥统计或样本分类。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
