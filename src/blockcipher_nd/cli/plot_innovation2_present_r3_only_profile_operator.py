from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E73 r3-only readiness.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_r3_only_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_r3_only_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立node", "错误P", "正确P")
    training = [rows[mode]["train_auc"] for mode in modes]
    validation = [rows[mode]["validation_auc"] for mode in modes]
    margins = (
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
    )
    decisions = {
        "innovation2_present_r3_only_profile_readiness_passed": (
            "r3-only绝对与拓扑门通过；进入冻结30轮seed0。"
        ),
        "innovation2_present_r3_only_profile_readiness_not_passed": (
            "r3-only质量或拓扑增益未过门；保留完整39维E68。"
        ),
        "innovation2_present_r3_only_profile_protocol_invalid": (
            "source、r3切片、参数公平或训练协议无效。"
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
        figure, axes = plt.subplots(1, 2, figsize=(14.8, 8.0))
        figure.subplots_adjust(
            left=0.08, right=0.975, top=0.70, bottom=0.28, wspace=0.32
        )
        figure.text(
            0.08,
            0.955,
            "创新2 E73：只用第3轮前缀能否保留PRESENT平衡谱结构增益",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "删除r1/r2共26维输入，复用E68的两步共享node/P-layer消息算子和E65严格标签。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.845,
            "同参数三行比较独立node、错误P和正确P；两轮只作为正式训练前的readiness门。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.34
        axes[0].bar(x - width / 2, training, width, label="训练AUC", color="#94A3B8")
        axes[0].bar(x + width / 2, validation, width, label="验证AUC", color="#2563EB")
        for index, value in enumerate(validation):
            axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.75, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("r3-only两轮readiness", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_labels = ("正确P - 独立node", "正确P - 错误P")
        axes[1].bar(np.arange(2), margins, color=("#0F766E", "#2563EB"), width=0.58)
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            axes[1].text(index, value + 0.006, f"{value:+.3f}", ha="center")
        lower = min(-0.05, min(margins) - 0.04)
        upper = max(0.15, max(margins) + 0.05)
        axes[1].set_ylim(lower, upper)
        axes[1].set_xticks(np.arange(2), margin_labels)
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("正确P-layer归因门", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = summary["contract"]
        figure.text(
            0.08,
            0.183,
            (
                f"参数量={next(iter(contract['parameter_counts'].values()))}，"
                f"相对E68={contract['parameter_ratio_to_e68']:.3f}；"
                f"cell重标号误差={contract['cell_relabel_max_abs_error']:.2e}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.08,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.053,
            "证据范围：PRESENT-80四轮、8-bit活动cube的两轮本地readiness；不是高轮、跨维度、攻击或SOTA。",
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
