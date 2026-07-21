from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Innovation 2 OP2 output parity mask-geometry calibration."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_output_parity_mask_geometry(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_output_parity_mask_geometry(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["model"]: row for row in summary["trained_rows"]}
    models = (
        "contiguous_parity_mlp",
        "aligned_parity_mlp",
        "aligned_parity_label_shuffle",
    )
    accuracy = tuple(rows[model]["test_accuracy"] for model in models)
    auc = tuple(rows[model]["test_macro_auc"] for model in models)
    labels = (
        "连续四位\n真实输出parity",
        "同一S-box对齐\n真实输出parity",
        "对齐parity\n训练标签打乱",
    )
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
        figure.subplots_adjust(
            left=0.08, right=0.975, top=0.69, bottom=0.30, wspace=0.28
        )
        figure.text(
            0.08,
            0.955,
            "创新2 OP2：真实密文输出parity的mask几何校准",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "输入始终是未见明文；比较连续四个密文位与同一末轮S-box经P层后的四个输出位置。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.08,
            0.844,
            "0/1标签来自该明文的真实密文异或值，没有real-vs-random或平衡/不平衡样本类别。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        x = np.arange(3)
        colors = ("#64748B", "#0F766E", "#B91C1C")
        for axis, values, title, ylabel in (
            (axes[0], accuracy, "测试集逐parity准确率", "Accuracy"),
            (axes[1], auc, "测试集parity宏平均AUC", "Macro AUC"),
        ):
            axis.bar(x, values, width=0.62, color=colors)
            axis.axhline(
                0.5,
                color="#334155",
                linestyle="--",
                linewidth=1.1,
                label="随机基线 0.5",
            )
            axis.set_xticks(x, labels)
            lower = min(0.44, min(values) - 0.07)
            upper = max(0.62, max(values) + 0.10)
            axis.set_ylim(lower, min(1.02, upper))
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            axis.legend(frameon=False, loc="upper left")
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + 0.016,
                    f"{value:.3f}",
                    ha="center",
                    fontweight="bold",
                )
        metrics = gate["metrics"]
        figure.text(
            0.08,
            0.192,
            (
                f"对齐AUC-连续AUC={metrics['aligned_minus_contiguous_macro_auc']:+.3f}；"
                f"对齐AUC-打乱AUC={metrics['aligned_minus_shuffled_macro_auc']:+.3f}；"
                f"完整密文逐bit AUC={metrics['full_bit_macro_auc']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        if gate["decision"] == "innovation2_output_parity_mask_geometry_supported":
            decision_text = "S-box/P层对齐mask信号通过；下一步只做独立固定密钥复验。"
        elif gate["status"] == "hold":
            decision_text = (
                "对齐mask未过归因门；停止加样本、epoch、密钥和轮数，转论文协议审计。"
            )
        else:
            decision_text = "输出预测或配对mask协议无效；只修协议，不解释神经指标。"
        figure.text(
            0.08,
            0.115,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.052,
            "证据边界：PRESENT-80一轮、单固定密钥、本地小规模；不是高轮攻击、论文复现或SOTA结果。",
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
