from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E71 RR-PGPO readiness.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_round_recurrent_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_round_recurrent_readiness(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["mode"]: row for row in summary["trained_rows"]}
    modes = (
        "wrong_order_true_P",
        "true_order_corrupted_P",
        "true_order_true_P",
    )
    labels = ("反向轮序\n正确P", "正确轮序\n错误P", "正确轮序\n正确P")
    validation = [rows[mode]["validation_auc"] for mode in modes]
    training = [rows[mode]["train_auc"] for mode in modes]
    margins = (
        gate["metrics"]["candidate_minus_wrong_order"],
        gate["metrics"]["candidate_minus_corrupted"],
    )
    decisions = {
        "innovation2_present_round_recurrent_readiness_passed": (
            "两轮readiness通过；进入冻结30轮seed0归因。"
        ),
        "innovation2_present_round_recurrent_readiness_not_passed": (
            "轮序或拓扑增益未过门；停止RR-PGPO。"
        ),
        "innovation2_present_round_recurrent_protocol_invalid": (
            "source、等变性、参数公平或训练协议无效。"
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
            "创新2 E71：显式轮序的PRESENT平衡谱算子是否值得正式训练",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "同一E65严格标签、同一参数预算；只比较正确轮序、错误轮序和错误P-layer。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.845,
            "每次按r1→r2→r3读取13维ANF前缀，用共享GRU节点更新与共享S/P消息块输出64个logit。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.34
        axes[0].bar(x - width / 2, training, width, label="训练AUC", color="#94A3B8")
        axes[0].bar(
            x + width / 2,
            validation,
            width,
            label="验证AUC",
            color="#2563EB",
        )
        for index, value in enumerate(validation):
            axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.70, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("两轮readiness表现", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, ncol=2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_labels = ("正确轮序 - 反向轮序", "正确P - 错误P")
        axes[1].bar(np.arange(2), margins, color=("#0F766E", "#2563EB"), width=0.58)
        axes[1].axhline(0.02, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for index, value in enumerate(margins):
            axes[1].text(index, value + 0.006, f"{value:+.3f}", ha="center")
        lower = min(-0.05, min(margins) - 0.04)
        upper = max(0.12, max(margins) + 0.05)
        axes[1].set_ylim(lower, upper)
        axes[1].set_xticks(np.arange(2), margin_labels)
        axes[1].set_ylabel("验证AUC差值")
        axes[1].set_title("结构归因门", loc="left", fontweight="bold")
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
