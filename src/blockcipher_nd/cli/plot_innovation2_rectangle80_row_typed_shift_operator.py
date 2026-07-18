from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E92 RECTANGLE Row-Typed Shift Operator readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_row_typed_operator(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_row_typed_operator(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = (
        "untyped_true",
        "row_typed_true",
        "row_typed_corrupted",
        "wrong_row_true",
    )
    labels = ("无类型\n真实P", "row类型\n真实P", "row类型\n错误P", "错误row\n真实P")
    training = [rows[mode]["train_auc"] for mode in modes]
    validation = [rows[mode]["validation_auc"] for mode in modes]
    margins = (
        gate["metrics"]["typed_true_minus_untyped"],
        gate["metrics"]["typed_true_minus_corrupted"],
        gate["metrics"]["typed_true_minus_wrong_row"],
        gate["metrics"]["typed_true_minus_typed_ridge"],
    )
    decisions = {
        "innovation2_rectangle80_row_typed_shift_operator_readiness_passed": (
            "参数零增量的row类型算子通过全部两轮门；可进入冻结30轮seed0。"
        ),
        "innovation2_rectangle80_row_typed_shift_operator_not_ready": (
            "row类型算子未同时超过无类型、错误P、错误row和ridge；关闭该路线。"
        ),
        "innovation2_rectangle80_row_typed_shift_operator_protocol_invalid": (
            "来源、通道置换、参数公平或训练协议无效；必须先修复。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.29, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E92：参数零增量的RECTANGLE行类型消息算子是否有效",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "四行均为4,795参数和两轮seed0；row类型只控制P层前驱隐藏通道的固定循环置换。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "比较真实row、错误row、真实P和错误P；E91 typed ridge是确定性防退化锚点。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(4)
        width = 0.34
        axes[0].bar(x - width / 2, training, width, color="#94A3B8", label="训练AUC")
        axes[0].bar(x + width / 2, validation, width, color="#2563EB", label="验证AUC")
        for index, value in enumerate(validation):
            axes[0].text(index + width / 2, value + 0.009, f"{value:.4f}", ha="center")
        axes[0].axhline(0.65, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(
            gate["metrics"]["e91_typed_true_ridge_auc"],
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="E91 typed ridge",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.45, 1.02)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("四行参数配平神经矩阵", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=3)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(4)
        margin_labels = (
            "typed true -\nuntyped true",
            "typed true -\ntyped错误P",
            "typed true -\n错误row",
            "typed true -\ntyped ridge",
        )
        axes[1].bar(
            margin_x,
            margins,
            color=("#0F766E", "#2563EB", "#7C3AED", "#D97706"),
            width=0.62,
        )
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.01, color="#D97706", linestyle=":", linewidth=1.2)
        axes[1].axhline(-0.03, color="#64748B", linestyle="-.", linewidth=1.1)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            axes[1].text(index, value + 0.004, f"{value:+.4f}", ha="center")
        axes[1].set_xticks(margin_x, margin_labels)
        axes[1].set_ylim(
            min(-0.10, min(margins) - 0.04),
            max(0.15, max(margins) + 0.05),
        )
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("row类型神经归因门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = summary["contract"]
        figure.text(
            0.075,
            0.183,
            (
                f"四行参数量={next(iter(contract['parameter_counts'].values()))}；"
                f"cell重标号误差={contract['cell_relabel_max_abs_error']:.2e}；"
                "没有row embedding参数。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：RECTANGLE-80四轮row类型算子的两轮本地就绪实验；不是正式收益、7轮复现或SOTA。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
