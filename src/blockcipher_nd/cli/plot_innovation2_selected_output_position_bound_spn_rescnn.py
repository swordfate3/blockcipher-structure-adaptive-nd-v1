from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "selected8_global_head_rescnn_anchor_true_output",
    "selected8_position_head_rescnn_no_p_true_output",
    "selected8_position_head_spn_rescnn_exact_p_true_output",
    "selected8_position_head_spn_rescnn_wrong_p_true_output",
    "selected8_position_head_spn_rescnn_exact_p_label_shuffle",
)
MODEL_LABELS = (
    "全局头\nResCNN",
    "位置头\n无P",
    "位置头\n真实P",
    "位置头\n错误P",
    "位置头\n标签打乱",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPD1 position-bound SPN-ResCNN."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_position_bound_spn_rescnn(
        json.loads(args.summary.read_text(encoding="utf-8")),
        args.output,
    )
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_position_bound_spn_rescnn(
    summary: dict[str, Any],
    output: Path,
) -> None:
    bits = [int(bit) for bit in summary["metadata"]["selected_msb_indices"]]
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row for row in summary["bit_rows"]
    }
    auc = np.asarray(
        [[float(indexed[(model, bit)]["auc"]) for bit in bits] for model in MODEL_ORDER]
    )
    means = auc.mean(axis=1)
    deltas = np.vstack(
        (auc[2] - auc[0], auc[2] - auc[1], auc[2] - auc[3], auc[2] - auc[4])
    )
    mode = str(summary["metadata"]["mode"])
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
        figure = plt.figure(figsize=(16.5, 9.6))
        figure.subplots_adjust(
            left=0.105,
            right=0.97,
            top=0.70,
            bottom=0.18,
            wspace=0.27,
        )
        grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.12), wspace=0.27)
        heat_axis = figure.add_subplot(grid[0, 0])
        delta_grid = grid[0, 1].subgridspec(4, 1, hspace=0.55)
        delta_axes = [figure.add_subplot(delta_grid[index, 0]) for index in range(4)]
        figure.text(
            0.105,
            0.955,
            "创新2 OPD1：PRESENT三轮位置绑定 SPN-ResCNN 输出预测",
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.105,
            0.900,
            "问题：把可吸收最后P重排的全局输出头换成参数匹配的位置绑定head，能否恢复真实P的可归因增益？",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.850,
            "控制：全局头ResCNN、无P位置头、真实P位置头、错误P位置头、真实P位置头标签打乱。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.800,
            (
                "当前是64条训练/64条测试、1 epoch本地实现门；随机小样本AUC不作性能结论。"
                if mode == "smoke"
                else "当前是131072条训练、65536条测试、100 epochs的第八固定密钥正式归因结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )

        deviation = max(0.02, float(np.max(np.abs(auc - 0.5))))
        image = heat_axis.imshow(
            auc,
            cmap="RdYlBu",
            vmin=0.5 - deviation,
            vmax=0.5 + deviation,
            aspect="auto",
        )
        heat_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        heat_axis.set_yticks(np.arange(5), MODEL_LABELS)
        heat_axis.set_title("各输出位AUC", loc="left", fontweight="bold")
        for row in range(5):
            for column in range(len(bits)):
                heat_axis.text(
                    column,
                    row,
                    f"{auc[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=(
                        "#FFFFFF"
                        if abs(float(auc[row, column]) - 0.5) / deviation >= 0.72
                        else "#111827"
                    ),
                )
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.035)
        colorbar.set_label("AUC")

        positions = np.arange(len(bits))
        colors = ("#0F766E", "#0891B2", "#2563EB", "#7C3AED")
        labels = (
            "真实P位置头 - 全局头ResCNN（逐bit门 >= +0.005）",
            "真实P位置头 - 无P位置头（逐bit门 >= +0.005）",
            "真实P位置头 - 错误P位置头（逐bit门 >= +0.015）",
            "真实P位置头 - 标签打乱（逐bit门 >= +0.015）",
        )
        thresholds = (0.005, 0.005, 0.015, 0.015)
        for index, (axis, row, color, label, threshold) in enumerate(
            zip(delta_axes, deltas, colors, labels, thresholds, strict=True)
        ):
            axis.bar(positions, row, width=0.62, color=color)
            axis.axhline(0.0, color="#475569", linewidth=0.9)
            axis.axhline(
                threshold,
                color="#DC2626",
                linewidth=0.9,
                linestyle="--",
            )
            axis.set_title(label, loc="left", fontsize=8.9, fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            span = max(0.025, threshold * 2.0, float(np.max(np.abs(row))) * 1.25)
            axis.set_ylim(-span, span)
            axis.set_xticks(
                positions,
                [f"bit {bit}" for bit in bits] if index == 3 else [""] * len(bits),
            )
            for position, value in zip(positions, row, strict=True):
                offset = span * 0.055
                axis.text(
                    position,
                    value + (offset if value >= 0 else -offset),
                    f"{value:+.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7.1,
                    color="#111827",
                )
        delta_axes[1].set_ylabel("AUC差值")

        decision_text = {
            "innovation2_position_bound_spn_rescnn_local_readiness_passed": (
                "五行模型、位置头可识别性、数据和产物实现门通过；本地小样本不作性能裁决"
            ),
            "innovation2_position_bound_spn_rescnn_requires_confirmation": (
                "真实P位置头超过全部锚点与控制；进入独立密钥确认"
            ),
            "innovation2_position_bound_spn_rescnn_not_supported": (
                "真实P位置头未通过同预算锚点与控制门；保留全局头ResCNN"
            ),
            "innovation2_position_bound_spn_rescnn_protocol_invalid": (
                "输出头、模型、数据、控制、训练或产物协议无效"
            ),
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.105,
            0.092,
            (
                f"裁决：{decision_text}；平均AUC为 全局头={means[0]:.4f}、无P位置头={means[1]:.4f}、"
                f"真实P={means[2]:.4f}、错误P={means[3]:.4f}、标签打乱={means[4]:.4f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.105,
            0.045,
            "证据边界：PRESENT三轮、八个预注册真实输出bit；不是四轮、完整密文恢复、真假样本分类或SOTA结果。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
