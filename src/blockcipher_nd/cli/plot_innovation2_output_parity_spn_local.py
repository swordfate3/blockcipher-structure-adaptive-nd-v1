from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Innovation 2 OP6 PRESENT r3 SPN-local parity readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_spn_local_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_spn_local_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    bit_role = "wrong_p_macro_auc" in metrics
    control_name = "wrong_p" if bit_role else "shuffled_p"
    accuracy = (
        metrics["mlp_accuracy"],
        metrics["true_p_accuracy"],
        metrics[f"{control_name}_accuracy"],
        metrics["label_shuffle_accuracy"],
    )
    auc = (
        metrics["mlp_macro_auc"],
        metrics["true_p_macro_auc"],
        metrics[f"{control_name}_macro_auc"],
        metrics["label_shuffle_macro_auc"],
    )
    labels = (
        (
            "全连接MLP\n真实输出",
            "精确bit-role\n真实P层",
            "精确bit-role\n错误P层",
            "真实P层网络\n训练标签打乱",
        )
        if bit_role
        else (
            "全连接MLP\n真实输出",
            "SPN局部网络\n真实P层",
            "SPN局部网络\n错误P层",
            "真实P层网络\n训练标签打乱",
        )
    )
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.4))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.69, bottom=0.31, wspace=0.27
        )
        figure.text(
            0.075,
            0.955,
            (
                "创新2 OP7：PRESENT三轮真实密文输出parity的精确bit-role路由门"
                if bit_role
                else "创新2 OP6：PRESENT三轮真实密文输出parity的SPN局部网络就绪门"
            ),
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.892,
            "输入仍只有未见明文；标签仍是同一固定秘密密钥下真实三轮密文的结构对齐四位异或值。",
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        figure.text(
            0.075,
            0.844,
            "只比较网络表示：同预算MLP、真实P层、错误P层和训练标签打乱；没有真假或平衡类别。",
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        x = np.arange(4)
        colors = ("#64748B", "#0F766E", "#B45309", "#B91C1C")
        for axis, values, title, ylabel in (
            (axes[0], accuracy, "测试集逐parity准确率", "Accuracy"),
            (axes[1], auc, "测试集parity宏平均AUC", "Macro AUC"),
        ):
            axis.bar(x, values, width=0.64, color=colors)
            axis.axhline(
                0.5,
                color="#334155",
                linestyle="--",
                linewidth=1.1,
                label="随机基线 0.5",
            )
            axis.set_xticks(x, labels)
            lower = min(0.44, min(values) - 0.06)
            upper = max(0.62, max(values) + 0.09)
            axis.set_ylim(lower, min(1.02, upper))
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            axis.legend(frameon=False, loc="upper left")
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + 0.014,
                    f"{value:.3f}",
                    ha="center",
                    fontweight="bold",
                )
        figure.text(
            0.075,
            0.202,
            (
                f"真实P-MLP={metrics['true_minus_mlp_macro_auc']:+.3f}；"
                f"真实P-错误P={metrics[f'true_minus_{control_name}_macro_auc']:+.3f}；"
                f"真实P-标签打乱={metrics['true_minus_label_shuffle_macro_auc']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        if gate["decision"] in {
            "innovation2_output_parity_present_r3_spn_local_attributed",
            "innovation2_output_parity_present_r3_bit_role_attributed",
        }:
            decision_text = "真实P层局部表示通过归因门；下一步只做独立固定密钥复验。"
        elif gate["decision"] in {
            "innovation2_output_parity_present_r3_spn_local_generic_gain_only",
            "innovation2_output_parity_present_r3_bit_role_generic_gain_only",
        }:
            decision_text = "只有通用局部表示收益；先审计精确bit-role路由，不扩轮。"
        elif gate["status"] == "hold":
            decision_text = (
                "精确bit-role网络仍未过门；转确定性依赖锥与布尔函数难度审计。"
                if bit_role
                else "nibble邻接网络未恢复三轮信号；转精确bit-level SPN路由。"
            )
        else:
            decision_text = "输出、token顺序、拓扑控制或训练协议无效；只修协议。"
        figure.text(
            0.075,
            0.122,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.075,
            0.054,
            "证据边界：PRESENT-80三轮、单固定密钥、本地表示就绪门；不是高轮攻击、论文复现或SOTA。",
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
