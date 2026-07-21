from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "present_spn_exact_p_true_output",
    "present_spn_identity_p_true_output",
    "present_spn_wrong_p_true_output",
)
MODEL_LABELS = ("真实P-layer", "Identity P-layer", "固定错误P-layer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPA3 PRESENT topology attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_topology_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_topology_attribution(summary: dict[str, Any], output: Path) -> None:
    bits = [int(bit) for bit in summary["metadata"]["selected_msb_indices"]]
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in summary["bit_rows"]
    }
    auc = np.asarray(
        [[float(indexed[(model, bit)]["auc"]) for bit in bits] for model in MODEL_ORDER]
    )
    means = auc.mean(axis=1)
    exact_minus_identity = auc[0] - auc[1]
    exact_minus_wrong = auc[0] - auc[2]
    gate = summary["gate"]
    mode = str(summary["metadata"]["mode"])

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
            "创新2 OPA3：PRESENT三轮真实P-layer拓扑归因",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.105,
            0.915,
            "三组网络参数、局部nibble混合、初始化和训练数据相同；唯一变量是无参数bit排列。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.878,
            "Identity不扩散；固定错误P仍为双射但与真实PRESENT映射64/64位置不同。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.105,
            0.841,
            (
                "当前为64条训练/64条测试、1 epoch本地实现门，数值不作性能结论。"
                if mode == "smoke"
                else "当前为131072条训练、65536条测试、100 epochs的第四固定密钥归因结果。"
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
        heat_axis.set_yticks(np.arange(3), MODEL_LABELS)
        heat_axis.set_title("逐bit AUC", loc="left", fontweight="bold")
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
        colors = ("#0F766E", "#475569", "#B45309")
        mean_axis.bar(np.arange(3), means, width=0.62, color=colors)
        mean_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        mean_axis.set_xticks(np.arange(3), MODEL_LABELS, rotation=8)
        mean_axis.set_ylabel("八位置平均AUC")
        mean_axis.set_title("真实拓扑与两个同容量控制", loc="left", fontweight="bold")
        low = max(0.0, float(min(means.min(), 0.5) - 0.02))
        high = min(1.0, float(max(means.max(), 0.5) + 0.04))
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

        identity_axis = axes[1, 0]
        identity_axis.bar(np.arange(len(bits)), exact_minus_identity, color="#0F766E")
        identity_axis.axhline(0.0, color="#475569", linewidth=1.0)
        identity_axis.axhline(0.02, color="#B45309", linestyle=":", linewidth=1.2)
        identity_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        identity_axis.set_ylabel("真实P AUC - Identity P AUC")
        identity_axis.set_title("跨nibble扩散归因", loc="left", fontweight="bold")
        identity_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        identity_span = max(0.03, float(np.max(np.abs(exact_minus_identity))) + 0.01)
        identity_axis.set_ylim(-identity_span, identity_span)

        wrong_axis = axes[1, 1]
        wrong_axis.bar(np.arange(len(bits)), exact_minus_wrong, color="#2563EB")
        wrong_axis.axhline(0.0, color="#475569", linewidth=1.0)
        wrong_axis.axhline(0.02, color="#B45309", linestyle=":", linewidth=1.2)
        wrong_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        wrong_axis.set_ylabel("真实P AUC - 固定错误P AUC")
        wrong_axis.set_title("精确PRESENT连线归因", loc="left", fontweight="bold")
        wrong_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        wrong_span = max(0.03, float(np.max(np.abs(exact_minus_wrong))) + 0.01)
        wrong_axis.set_ylim(-wrong_span, wrong_span)

        decision_text = {
            "innovation2_selected8_topology_attribution_local_smoke_passed": "三种P-layer同容量归因实现门通过",
            "innovation2_selected8_present_topology_independently_attributed": "真实PRESENT拓扑通过独立归因",
            "innovation2_selected8_present_topology_not_attributed": "真实PRESENT拓扑未超过同容量控制",
            "innovation2_selected8_topology_attribution_protocol_invalid": "实验协议或产物无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.105,
            0.063,
            (
                f"裁决：{decision_text}；真实P减最佳控制平均AUC="
                f"{gate['metrics']['exact_minus_best_control_mean_auc']:+.4f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.105,
            0.027,
            "证据边界：PRESENT三轮、第四固定密钥、八个预注册真实输出bit；不是四轮证据、完整密文恢复或样本分类。",
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
