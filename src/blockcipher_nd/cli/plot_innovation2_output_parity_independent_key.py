from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Innovation 2 OP3 independent fixed-key confirmation."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_output_parity_independent_key(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_output_parity_independent_key(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    seed_metrics = (gate["metrics"]["seed0"], gate["metrics"]["seed1"])
    values = tuple(
        (
            metrics["contiguous_parity_macro_auc"],
            metrics["aligned_parity_macro_auc"],
            metrics["shuffled_aligned_parity_macro_auc"],
        )
        for metrics in seed_metrics
    )
    labels = (
        "连续四位\n真实parity",
        "S-box/P层对齐\n真实parity",
        "对齐parity\n标签打乱",
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
            "创新2 OP3：结构对齐密文输出parity的独立固定密钥确认",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "两次运行使用不同固定秘密密钥和零重合明文；网络输入仍只有未见明文。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.08,
            0.844,
            "0/1是每条明文对应的真实密文四位置异或输出，不是真假或平衡类别。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        x = np.arange(3)
        colors = ("#64748B", "#0F766E", "#B91C1C")
        for seed, (axis, seed_values) in enumerate(zip(axes, values, strict=True)):
            axis.bar(x, seed_values, width=0.62, color=colors)
            axis.axhline(
                0.5,
                color="#334155",
                linestyle="--",
                linewidth=1.1,
                label="随机基线 0.5",
            )
            axis.set_xticks(x, labels)
            axis.set_ylim(0.42, 1.03)
            axis.set_ylabel("Macro AUC")
            axis.set_title(
                f"固定密钥 seed{seed}：测试集宏平均AUC",
                loc="left",
                fontweight="bold",
            )
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            axis.legend(frameon=False, loc="upper left")
            for index, value in enumerate(seed_values):
                axis.text(
                    index,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    fontweight="bold",
                )
        metrics = gate["metrics"]
        figure.text(
            0.08,
            0.192,
            (
                f"两把密钥对齐parity最低AUC={metrics['minimum_aligned_parity_macro_auc']:.3f}；"
                f"平均AUC={metrics['mean_aligned_parity_macro_auc']:.3f}；"
                f"跨密钥极差={metrics['aligned_parity_macro_auc_range']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        if gate["decision"] == (
            "innovation2_output_parity_mask_geometry_two_key_confirmed"
        ):
            decision_text = "双固定密钥确认通过；下一步只把PRESENT轮数从一轮改为两轮。"
        elif gate["status"] == "hold":
            decision_text = "独立密钥未确认；停止扩轮与扩规模，转输出预测论文协议审计。"
        else:
            decision_text = "anchor、密钥、明文独立性或输出预测协议无效；只修协议。"
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
            "证据边界：PRESENT-80一轮、两把固定密钥、本地小规模；不是高轮攻击、论文复现或SOTA。",
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
