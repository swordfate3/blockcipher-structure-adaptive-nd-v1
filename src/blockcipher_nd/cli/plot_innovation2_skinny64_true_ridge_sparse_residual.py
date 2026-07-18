from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E84 SKINNY true-ridge sparse residual readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_true_ridge_residual(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_true_ridge_residual(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = gate["metrics"]["rows"]
    anchor = next(row for row in rows if not row["training_performed"])
    trained = {row["relation_mode"]: row for row in rows if row["training_performed"]}
    decisions = {
        "innovation2_skinny64_true_ridge_residual_readiness_passed": (
            "真实图残差超过ridge与两类控制；可预注册30轮seed0正式归因。"
        ),
        "innovation2_skinny64_true_ridge_residual_not_ready": (
            "神经残差未超过强ridge和控制；SKINNY神经搜索在此收束。"
        ),
        "innovation2_skinny64_true_ridge_residual_protocol_invalid": (
            "来源、ridge、零残差、冻结buffer、图或训练协议无效。"
        ),
    }
    names = ("ridge锚点", "独立残差", "错误图残差", "真实图残差")
    aucs = [
        anchor["validation_auc"],
        trained["independent"]["validation_auc"],
        trained["corrupted"]["validation_auc"],
        trained["true"]["validation_auc"],
    ]
    margins = [
        gate["metrics"]["true_minus_ridge"],
        gate["metrics"]["true_minus_independent"],
        gate["metrics"]["true_minus_corrupted"],
    ]
    train_auc = [trained[mode]["train_auc"] for mode in ("independent", "corrupted", "true")]
    validation_auc = [
        trained[mode]["validation_auc"] for mode in ("independent", "corrupted", "true")
    ]
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
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.3))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E84：冻结真实拓扑ridge后，稀疏神经残差是否仍有增益",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "四行共享同一个train-only真实图ridge；三条神经行只改变残差关系，均训练两轮。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "残差head从零开始且被限制在±0.25；epoch0严格等于ridge，不能用弱初始化制造假增益。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(4)
        colors = ("#D97706", "#94A3B8", "#2563EB", "#0F766E")
        axes[0].bar(x, aucs, color=colors)
        for index, value in enumerate(aucs):
            axes[0].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].set_xticks(x, names, rotation=10)
        axes[0].set_ylim(0.65, max(0.95, max(aucs) + 0.06))
        axes[0].set_ylabel("validation AUC")
        axes[0].set_title("冻结base与三条残差", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_names = ("真实-ridge", "真实-独立", "真实-错误图")
        margin_x = np.arange(3)
        margin_colors = ["#0F766E" if value >= threshold else "#DC2626" for value, threshold in zip(margins, (0.02, 0.03, 0.03), strict=True)]
        axes[1].bar(margin_x, margins, color=margin_colors)
        for index, value in enumerate(margins):
            offset = 0.006 if value >= 0 else -0.014
            axes[1].text(index, value + offset, f"{value:+.4f}", ha="center")
        axes[1].axhline(
            0.02,
            color="#D97706",
            linestyle=":",
            linewidth=1.2,
            label="ridge门 0.02",
        )
        axes[1].axhline(
            0.03,
            color="#DC2626",
            linestyle="--",
            linewidth=1.2,
            label="控制门 0.03",
        )
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        axes[1].set_xticks(margin_x, margin_names)
        axes[1].set_ylim(
            min(-0.02, min(margins) - 0.012),
            max(0.06, max(margins) + 0.015),
        )
        axes[1].set_ylabel("validation AUC差值")
        axes[1].set_title("神经残差推进门", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, fontsize=8.4, loc="upper right")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        relation_names = ("独立", "错误图", "真实图")
        relation_x = np.arange(3)
        width = 0.34
        axes[2].bar(relation_x - width / 2, train_auc, width, label="train", color="#94A3B8")
        axes[2].bar(relation_x + width / 2, validation_auc, width, label="validation", color="#0F766E")
        for index, value in enumerate(validation_auc):
            axes[2].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[2].set_xticks(relation_x, relation_names)
        axes[2].set_ylim(0.65, max(0.95, max(train_auc + validation_auc) + 0.06))
        axes[2].set_ylabel("AUC")
        axes[2].set_title("训练与验证一致性", loc="left", fontweight="bold")
        axes[2].legend(frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = gate["metrics"]["contract"]
        figure.text(
            0.065,
            0.188,
            (
                f"零残差最大误差={max(contract['zero_residual_max_abs_errors'].values()):.1e}；"
                f"ridge buffer冻结；三条神经行均{next(iter(contract['parameter_counts'].values())):,}参数。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.116,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：SKINNY五轮严格标签的两轮本地ridge引导残差readiness；不是正式增益、高轮攻击或SOTA。",
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
