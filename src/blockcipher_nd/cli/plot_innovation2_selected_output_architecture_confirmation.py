from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURE_LABELS = {
    "lstm": "六层LSTM",
    "rescnn": "位置保持ResCNN",
    "transformer": "Transformer",
    "present_spn": "PRESENT结构网络",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPA2 matched-control confirmation."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_architecture_confirmation(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_architecture_confirmation(summary: dict[str, Any], output: Path) -> None:
    candidate = str(summary["metadata"]["candidate_architecture"])
    candidate_label = ARCHITECTURE_LABELS[candidate]
    bits = [int(bit) for bit in summary["metadata"]["selected_msb_indices"]]
    model_order = (
        "selected8_mlp_true_output",
        "selected8_mlp_label_shuffle",
        f"selected8_{candidate}_true_output",
        f"selected8_{candidate}_label_shuffle",
    )
    model_labels = (
        "MLP真实输出",
        "MLP标签打乱",
        f"{candidate_label}真实输出",
        f"{candidate_label}标签打乱",
    )
    rows = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in summary["bit_rows"]
    }
    auc = np.asarray(
        [[float(rows[(model, bit)]["auc"]) for bit in bits] for model in model_order]
    )
    accuracy_margin = np.asarray(
        [
            [float(rows[(model, bit)]["accuracy_minus_majority"]) for bit in bits]
            for model in model_order
        ]
    )
    means = auc.mean(axis=1)
    true_minus_shuffle = np.asarray((means[0] - means[1], means[2] - means[3]))
    candidate_minus_mlp = auc[2] - auc[0]
    gate = summary["gate"]
    mode = summary["metadata"]["mode"]

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
            left=0.105,
            right=0.965,
            top=0.79,
            bottom=0.13,
            hspace=0.48,
            wspace=0.29,
        )
        figure.text(
            0.105,
            0.96,
            f"创新2 OPA2：{candidate_label}与MLP的第四密钥匹配控制确认",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.105,
            0.915,
            "候选由OPA1正式gate唯一指定；八个输出位置、模型结构和训练预算均未重新选择。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.878,
            "四行分别为候选/MLP的真实输出与同架构标签打乱；测试标签始终是真实密文输出bit。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.841,
            (
                "当前为64条训练/64条测试、1 epoch的本地实现门，数值不作性能结论。"
                if mode == "smoke"
                else "当前为131072条训练、65536条测试、100 epochs的第四固定密钥确认结果。"
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
        heat_axis.set_yticks(np.arange(4), model_labels)
        heat_axis.set_title("逐bit AUC与匹配打乱控制", loc="left", fontweight="bold")
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
        colors = ("#334155", "#B91C1C", "#0F766E", "#D97706")
        x = np.arange(4)
        mean_axis.bar(x, means, width=0.62, color=colors)
        mean_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        mean_axis.set_xticks(x, model_labels, rotation=12)
        mean_axis.set_ylabel("八位置平均AUC")
        mean_axis.set_title("真实输出与标签打乱的平均表现", loc="left", fontweight="bold")
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

        control_axis = axes[1, 0]
        control_axis.bar(
            np.arange(2),
            true_minus_shuffle,
            width=0.58,
            color=("#334155", "#0F766E"),
        )
        control_axis.axhline(0.0, color="#475569", linewidth=1.0)
        control_axis.axhline(0.005, color="#B45309", linestyle=":", linewidth=1.2)
        control_axis.set_xticks(np.arange(2), ("MLP", candidate_label))
        control_axis.set_ylabel("真实输出平均AUC - 匹配shuffle平均AUC")
        control_axis.set_title("架构匹配控制归因", loc="left", fontweight="bold")
        control_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        span = max(0.01, float(np.max(np.abs(true_minus_shuffle))) + 0.006)
        control_axis.set_ylim(-span, span)
        for index, value in enumerate(true_minus_shuffle):
            control_axis.text(
                index,
                value + (0.0007 if value >= 0 else -0.0014),
                f"{value:+.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontweight="bold",
            )

        gain_axis = axes[1, 1]
        gain_colors = ["#0F766E" if value >= 0 else "#B91C1C" for value in candidate_minus_mlp]
        gain_axis.bar(np.arange(len(bits)), candidate_minus_mlp, color=gain_colors)
        gain_axis.axhline(0.0, color="#475569", linewidth=1.0)
        gain_axis.axhline(0.002, color="#B45309", linestyle=":", linewidth=1.2)
        gain_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        gain_axis.set_ylabel(f"{candidate_label}真实AUC - MLP真实AUC")
        gain_axis.set_title("候选相对MLP的逐bit增益", loc="left", fontweight="bold")
        gain_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        gain_span = max(0.01, float(np.max(np.abs(candidate_minus_mlp))) + 0.006)
        gain_axis.set_ylim(-gain_span, gain_span)

        decision_text = {
            "innovation2_selected8_architecture_confirmation_local_smoke_passed": "第四密钥四行确认实现门通过",
            "innovation2_selected8_architecture_priority_independently_confirmed": f"{candidate_label}通过独立密钥与匹配控制确认",
            "innovation2_selected8_mlp_retained_after_architecture_confirmation": "候选未通过独立确认，保留MLP",
            "innovation2_selected8_architecture_confirmation_protocol_invalid": "实验协议或产物无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.105,
            0.063,
            f"裁决：{decision_text}；候选-MLP平均AUC={gate['metrics']['mean_candidate_minus_mlp_auc']:+.4f}。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.105,
            0.027,
            "证据边界：PRESENT三轮、第四把固定密钥、八个预注册真实输出bit；不是四轮证据、完整密文恢复或样本分类。",
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
