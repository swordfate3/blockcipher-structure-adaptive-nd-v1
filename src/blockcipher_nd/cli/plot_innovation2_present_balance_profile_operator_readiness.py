from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E66 prefix-guided profile operator readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_profile_operator_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_profile_operator_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in summary["trained_rows"]}
    modes = ("independent", "corrupted", "true")
    labels = ("独立node\n容量控制", "错误P\nprofile mixer", "正确P\nprofile mixer")
    train_auc = [float(rows[mode]["train_auc"]) for mode in modes]
    validation_auc = [float(rows[mode]["validation_auc"]) for mode in modes]
    contract = summary["contract"]
    decisions = {
        "innovation2_present_profile_operator_readiness_passed": (
            "实现与两轮学习门通过；下一步另建30轮seed0正式归因。"
        ),
        "innovation2_present_profile_operator_optimization_not_ready": (
            "安全前缀的两轮学习未就绪；停止正式训练并检查优化。"
        ),
        "innovation2_present_profile_operator_protocol_invalid": (
            "source、等变性、masked loss、参数或训练协议无效。"
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
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E66：prefix引导的逐节点PRESENT平衡谱算子是否训练就绪",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "每个活动结构一次输出64个logit，只在E65观察坐标计算masked BCE；两轮结果仅作readiness。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "三行模型参数完全相同；唯一变化是独立node、正确P关系或fair-corrupted P关系。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        width = 0.35
        axes[0].bar(x - width / 2, train_auc, width, label="训练", color="#94A3B8")
        axes[0].bar(
            x + width / 2, validation_auc, width, label="验证", color="#2563EB"
        )
        for index, value in enumerate(train_auc):
            axes[0].text(index - width / 2, value + 0.012, f"{value:.3f}", ha="center")
        for index, value in enumerate(validation_auc):
            axes[0].text(index + width / 2, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.4, max(0.85, max(train_auc + validation_auc) + 0.07))
        axes[0].set_ylabel("观察坐标 AUC")
        axes[0].set_title("两轮短训练只检查防退化", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract_labels = (
            "masked loss\n误差",
            "cell重标号\n误差",
            "true/corrupt\nlogit差",
        )
        contract_values = (
            max(float(contract["masked_loss_explicit_max_abs_error"]), 1e-12),
            max(float(contract["cell_relabel_max_abs_error"]), 1e-12),
            float(contract["true_corrupted_logit_max_abs_difference"]),
        )
        cx = np.arange(3)
        axes[1].bar(cx, contract_values, color=["#0F766E", "#7C3AED", "#D97706"])
        axes[1].set_yscale("log")
        axes[1].set_xticks(cx, contract_labels)
        axes[1].set_ylabel("绝对值（log）")
        axes[1].set_title("实现contract：误差低、关系差异非零", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(contract_values):
            axes[1].text(index, value * 1.35, f"{value:.2e}", ha="center", fontweight="bold")

        figure.text(
            0.075,
            0.178,
            (
                f"参数量={rows['true']['parameter_count']}（三行一致）；"
                f"true-independent={gate['metrics']['true_minus_independent']:+.3f}，"
                f"true-corrupted={gate['metrics']['true_minus_corrupted']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：PRESENT-80四轮严格unit平衡谱的本地两轮readiness；不是正式性能、高轮结论或新攻击。",
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
