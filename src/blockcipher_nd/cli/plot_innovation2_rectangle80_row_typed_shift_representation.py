from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E91 RECTANGLE row-typed representation audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_row_typed_audit(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_row_typed_audit(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    reports = gate["metrics"]["ridges"]
    variants = (
        "untyped_true",
        "untyped_corrupted",
        "typed_true",
        "typed_corrupted",
        "wrong_row_typed_true",
    )
    labels = (
        "无类型\n真实P",
        "无类型\n错误P",
        "row类型\n真实P",
        "row类型\n错误P",
        "错误row类型\n真实P",
    )
    aucs = [reports[name]["validation_auc"] for name in variants]
    margins = (
        gate["metrics"]["typed_true_minus_untyped_true"],
        gate["metrics"]["typed_true_minus_typed_corrupted"],
        gate["metrics"]["typed_true_minus_wrong_row_typed"],
    )
    decisions = {
        "innovation2_rectangle80_row_typed_representation_ready": (
            "row类型信息通过三项机制门；可设计容量配平的Row-Typed Shift Operator。"
        ),
        "innovation2_rectangle80_row_typed_representation_not_ready": (
            "row类型信息未同时超过锚点和错误控制；不训练新网络。"
        ),
        "innovation2_rectangle80_row_typed_representation_protocol_invalid": (
            "E88/E90来源或row-typed表示协议无效；必须先修复。"
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
            "创新2 E91：保留RECTANGLE行类型能否解释真实ShiftRow优势",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "不训练神经网络；比较同一E88标签上的无类型39维与row-typed 117维确定性ridge。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "错误P保持相同维数；错误row控制按结构打乱row语义但不读取标签。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(variants))
        axes[0].bar(
            x,
            aucs,
            color=("#94A3B8", "#D97706", "#0F766E", "#2563EB", "#7C3AED"),
            width=0.68,
        )
        for index, value in enumerate(aucs):
            axes[0].text(index, value + 0.008, f"{value:.4f}", ha="center")
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.55, min(1.02, max(aucs) + 0.08))
        axes[0].set_ylabel("structure-disjoint验证AUC")
        axes[0].set_title("五个冻结确定性表示", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(3)
        margin_labels = (
            "row真实P -\n无类型真实P",
            "row真实P -\nrow错误P",
            "row真实P -\n错误row真实P",
        )
        axes[1].bar(
            margin_x,
            margins,
            color=("#0F766E", "#2563EB", "#7C3AED"),
            width=0.62,
        )
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.01, color="#D97706", linestyle=":", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            axes[1].text(index, value + 0.004, f"{value:+.4f}", ha="center")
        axes[1].set_xticks(margin_x, margin_labels)
        axes[1].set_ylim(
            min(-0.06, min(margins) - 0.03),
            max(0.12, max(margins) + 0.04),
        )
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("row类型新增信息门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.183,
            "真实row类型和错误row类型均为117维；所有ridge只用训练集标准化，lambda=1e-3。",
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
            "证据范围：RECTANGLE-80四轮row类型表示机制审计；不是神经收益、7轮复现、高轮攻击或SOTA。",
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
