from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("mlp", "lstm", "rescnn", "transformer", "present_spn")
MODEL_LABELS = {
    "mlp": "MLP锚点",
    "lstm": "六层LSTM",
    "rescnn": "一维残差CNN",
    "transformer": "Transformer",
    "present_spn": "PRESENT结构网络",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPA1 selected-output architecture screen."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_selected_output_architecture(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_selected_output_architecture(
    summary: dict[str, Any], output: Path
) -> None:
    bits = [int(bit) for bit in summary["metadata"]["selected_msb_indices"]]
    rows = {
        (str(row["architecture"]), int(row["msb_index"])): row
        for row in summary["bit_rows"]
    }
    auc = np.asarray(
        [[float(rows[(model, bit)]["auc"]) for bit in bits] for model in MODEL_ORDER]
    )
    accuracy_margin = np.asarray(
        [
            [float(rows[(model, bit)]["accuracy_minus_majority"]) for bit in bits]
            for model in MODEL_ORDER
        ]
    )
    means = auc.mean(axis=1)
    gains = means - means[0]
    mode = summary["metadata"]["mode"]
    gate = summary["gate"]

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
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.5))
        figure.subplots_adjust(
            left=0.09,
            right=0.965,
            top=0.79,
            bottom=0.13,
            hspace=0.48,
            wspace=0.28,
        )
        figure.text(
            0.09,
            0.96,
            "创新2 OPA1：PRESENT三轮固定八输出bit的多架构发现屏",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.09,
            0.915,
            "输入是未见明文的64个bit；目标是八个预注册真实密文bit，不是真假样本分类。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.09,
            0.878,
            "五个模型使用同一第三把固定未知密钥、同一数据和训练预算；本阶段只筛候选，不作最终架构结论。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.09,
            0.841,
            (
                "当前为64条训练/64条测试、1 epoch的本地实现门，数值不作性能结论。"
                if mode == "smoke"
                else "当前为131072条训练、65536条测试、100 epochs的第三密钥发现结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )

        heat_axis = axes[0, 0]
        max_deviation = max(0.015, float(np.max(np.abs(auc - 0.5))))
        image = heat_axis.imshow(
            auc,
            cmap="RdYlBu",
            vmin=0.5 - max_deviation,
            vmax=0.5 + max_deviation,
            aspect="auto",
        )
        heat_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        heat_axis.set_yticks(
            np.arange(len(MODEL_ORDER)),
            [MODEL_LABELS[model] for model in MODEL_ORDER],
        )
        heat_axis.set_title("逐bit AUC热力图（颜色围绕0.5对称）", loc="left", fontweight="bold")
        for row_index in range(auc.shape[0]):
            for column_index in range(auc.shape[1]):
                heat_axis.text(
                    column_index,
                    row_index,
                    f"{auc[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#111827",
                )
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.035)
        colorbar.set_label("AUC")

        mean_axis = axes[0, 1]
        colors = ["#334155", "#2563EB", "#0F766E", "#7C3AED", "#B45309"]
        x = np.arange(len(MODEL_ORDER))
        mean_axis.bar(x, means, width=0.62, color=colors)
        mean_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        mean_axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=15)
        mean_axis.set_ylabel("八位置平均AUC")
        mean_axis.set_title("同预算模型平均表现", loc="left", fontweight="bold")
        low = max(0.0, float(min(means.min(), 0.5) - 0.015))
        high = min(1.0, float(max(means.max(), 0.5) + 0.025))
        mean_axis.set_ylim(low, high)
        mean_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        for index, value in enumerate(means):
            mean_axis.text(
                index,
                value + (high - low) * 0.035,
                f"{value:.4f}",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
            )

        gain_axis = axes[1, 0]
        non_mlp_x = np.arange(len(MODEL_ORDER) - 1)
        gain_colors = ["#2563EB" if value >= 0 else "#B91C1C" for value in gains[1:]]
        gain_axis.bar(non_mlp_x, gains[1:], width=0.62, color=gain_colors)
        gain_axis.axhline(0.0, color="#475569", linewidth=1.0)
        gain_axis.axhline(0.003, color="#B45309", linestyle=":", linewidth=1.2)
        gain_axis.set_xticks(
            non_mlp_x,
            [MODEL_LABELS[model] for model in MODEL_ORDER[1:]],
            rotation=12,
        )
        gain_axis.set_ylabel("平均AUC - MLP平均AUC")
        gain_axis.set_title("候选相对MLP锚点的增益", loc="left", fontweight="bold")
        gain_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        span = max(0.008, float(np.max(np.abs(gains[1:]))) + 0.004)
        gain_axis.set_ylim(-span, span)
        for index, value in enumerate(gains[1:]):
            gain_axis.text(
                index,
                value + (0.0006 if value >= 0 else -0.0012),
                f"{value:+.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.8,
            )

        margin_axis = axes[1, 1]
        margin_means = accuracy_margin.mean(axis=1)
        margin_axis.bar(x, margin_means, width=0.62, color=colors)
        margin_axis.axhline(0.0, color="#475569", linewidth=1.0)
        margin_axis.axhline(0.005, color="#B45309", linestyle=":", linewidth=1.2)
        margin_axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=15)
        margin_axis.set_ylabel("平均准确率 - 多数类基线")
        margin_axis.set_title("预测正确率相对平凡基线", loc="left", fontweight="bold")
        margin_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

        selected = gate["metrics"]["selected_candidate_for_phase_b"]
        selected_text = MODEL_LABELS[selected] if selected else "无候选"
        decision_text = {
            "innovation2_selected8_architecture_screen_local_smoke_passed": "本地五模型实现门通过",
            "innovation2_selected8_architecture_candidate_requires_confirmation": "发现候选，必须进入第四密钥匹配shuffle确认",
            "innovation2_selected8_mlp_anchor_retained_after_screen": "没有非MLP候选通过预注册增益门，保留MLP",
            "innovation2_selected8_architecture_protocol_invalid": "实验协议或产物无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.09,
            0.063,
            f"裁决：{decision_text}；Phase B候选：{selected_text}。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.09,
            0.027,
            "证据边界：PRESENT三轮、第三把固定密钥、八个预注册输出bit的架构发现；没有匹配shuffle，不能宣称架构最终胜出。",
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
