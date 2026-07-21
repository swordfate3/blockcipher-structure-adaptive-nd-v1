from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OP9 Kimura-LSTM output prediction."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_kimura_lstm_output_prediction(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_kimura_lstm_output_prediction(
    summary: dict[str, Any], output: Path
) -> None:
    rows = {row["model"]: row for row in summary["trained_rows"]}
    gate = summary["gate"]
    names = (
        "kimura_lstm_true_output",
        "matched_mlp_true_output",
        "kimura_lstm_label_shuffle",
    )
    labels = ("Kimura式LSTM\n真实输出", "参数量匹配MLP\n真实输出", "Kimura式LSTM\n训练标签打乱")
    colors = ("#0F766E", "#2563EB", "#B91C1C")
    bit_match = [rows[name]["test_bit_match"] for name in names]
    auc = [rows[name]["test_macro_auc"] for name in names]
    exact_counts = [rows[name]["test_exact_match_count"] for name in names]
    parameters = [rows[name]["parameters"] / 1_000_000 for name in names]
    mode = summary["metadata"]["mode"]
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
        figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.2))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.77,
            bottom=0.18,
            hspace=0.48,
            wspace=0.26,
        )
        figure.text(
            0.075,
            0.965,
            "创新2 OP9：PRESENT三轮完整64-bit真实密文输出预测",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.916,
            "输入是未见明文；输出是同一固定未知密钥下的完整真实密文，不存在真假样本类别。",
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        figure.text(
            0.075,
            0.881,
            (
                "当前为本地实现门，数字不作性能结论。"
                if mode == "smoke"
                else "远程单固定密钥论文族校准；不是Kimura的100密钥完整复现。"
            ),
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        x = np.arange(3)
        for axis, values, title, ylabel, baseline in (
            (axes[0, 0], bit_match, "逐bit输出值匹配率", "Bit match", 0.5),
            (axes[0, 1], auc, "逐bit宏平均AUC（支持指标）", "Macro AUC", 0.5),
        ):
            axis.bar(x, values, width=0.62, color=colors)
            axis.axhline(
                baseline,
                color="#334155",
                linestyle="--",
                linewidth=1.1,
                label="随机基线0.5",
            )
            lower = max(0.0, min(values + [baseline]) - 0.025)
            upper = min(1.0, max(values + [baseline]) + 0.04)
            if upper - lower < 0.08:
                center = (upper + lower) / 2
                lower = max(0.0, center - 0.04)
                upper = min(1.0, center + 0.04)
            axis.set_ylim(lower, upper)
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            axis.legend(frameon=False, loc="upper left")
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + (upper - lower) * 0.035,
                    f"{value:.4f}",
                    ha="center",
                    fontweight="bold",
                )
        for axis, values, title, ylabel, formatter in (
            (
                axes[1, 0],
                exact_counts,
                "完整64-bit密文精确命中数（论文主指标）",
                "Exact matches",
                lambda value: str(int(value)),
            ),
            (
                axes[1, 1],
                parameters,
                "模型参数量校准",
                "Million parameters",
                lambda value: f"{value:.3f}M",
            ),
        ):
            axis.bar(x, values, width=0.62, color=colors)
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            top = max(values) if max(values) > 0 else 1
            axis.set_ylim(0, top * 1.22)
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + top * 0.035,
                    formatter(value),
                    ha="center",
                    fontweight="bold",
                )
        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.108,
            (
                f"LSTM-打乱：bit match={metrics['lstm_minus_shuffled_bit_match']:+.4f}；"
                f"macro AUC={metrics['lstm_minus_shuffled_macro_auc']:+.4f}；"
                f"LSTM-匹配MLP bit match={metrics['lstm_minus_matched_mlp_bit_match']:+.4f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        decision_text = (
            "本地实现与证据协议通过，只开放推送提交后的远程论文规模单密钥校准。"
            if gate["decision"].endswith("local_smoke_passed")
            else (
                "单密钥论文族校准通过，下一步只做第二固定密钥确认。"
                if gate["status"] == "pass"
                else "单密钥论文族校准未通过，停止机械扩数据、epoch、层数和轮数。"
            )
        )
        figure.text(
            0.075,
            0.066,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.075,
            0.027,
            "证据边界：固定密钥PRESENT三轮真实输出预测；AUC只是支持项，不替换完整输出exact-match。",
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
