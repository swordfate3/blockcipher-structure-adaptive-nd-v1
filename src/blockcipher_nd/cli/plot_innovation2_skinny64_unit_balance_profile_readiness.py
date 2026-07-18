from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E81 SKINNY-64/64 r4 unit-profile label readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_skinny64_unit_balance_profile(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_skinny64_unit_balance_profile(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    next_action = str(gate.get("next_action", {}).get("action", ""))
    if next_action == "stop r4 expansion and audit an r5 label-distribution transition":
        not_ready_decision = (
            "四轮原始标签过度偏向平衡；不扩到192结构，下一步审计五轮标签分布。"
        )
    elif "192-structure packing" in next_action:
        not_ready_decision = "原始标签够宽但matching不足；先审计192结构packing，不训练网络。"
    else:
        not_ready_decision = "标签宽度或反捷径门未通过；停止当前标签路线，不训练网络。"
    split = metrics["matched_split_metrics"]
    raw_counts = (
        metrics["raw_positive"],
        metrics["raw_negative"],
        metrics["raw_unknown"],
    )
    decisions = {
        "innovation2_skinny64_unit_balance_profile_ready": (
            "严格标签、宽度和反捷径门通过；下一步只开放两轮本地三行readiness。"
        ),
        "innovation2_skinny64_unit_balance_profile_not_ready": not_ready_decision,
        "innovation2_skinny64_unit_balance_profile_protocol_invalid": (
            "SKINNY坐标、向量化、support或反例协议无效；必须先修复。"
        ),
    }
    colors = ("#0F766E", "#2563EB", "#94A3B8")
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
        figure, axes = plt.subplots(1, 3, figsize=(15.6, 8.2))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.27, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E81：SKINNY-64四轮的64位积分平衡谱能否形成严格训练标签",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "复用GIFT E74的96个8活动bit输入cube、16个见证密钥、每结构8个固定偏移、split和门槛。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "正类由活动变量ANF支撑缺失证明；负类由真实64位主密钥经轮密钥调度后的固定偏移反例证明。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        axes[0].bar(x, raw_counts, color=colors)
        raw_max = max(1, max(raw_counts))
        for index, value in enumerate(raw_counts):
            axes[0].text(index, value + raw_max * 0.035, f"{value:,}", ha="center")
        axes[0].set_xticks(x, ("可证明平衡", "反例非平衡", "未知"))
        axes[0].set_ylim(0, raw_max * 1.18)
        axes[0].set_ylabel("结构 × 输出bit 数量")
        axes[0].set_title("原始三态标签atlas", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        split_x = np.arange(2)
        width = 0.34
        positives = [split[name]["positive"] for name in ("train", "validation")]
        negatives = [split[name]["negative"] for name in ("train", "validation")]
        split_max = max(1, max(positives + negatives))
        axes[1].bar(
            split_x - width / 2, positives, width, color="#0F766E", label="平衡"
        )
        axes[1].bar(
            split_x + width / 2, negatives, width, color="#2563EB", label="非平衡"
        )
        for index, value in enumerate(positives):
            axes[1].text(
                index - width / 2, value + split_max * 0.04, str(value), ha="center"
            )
        for index, value in enumerate(negatives):
            axes[1].text(
                index + width / 2, value + split_max * 0.04, str(value), ha="center"
            )
        axes[1].set_xticks(split_x, ("训练", "验证"))
        axes[1].set_ylim(0, split_max * 1.24)
        axes[1].set_ylabel("checkerboard标签数量")
        axes[1].set_title("行列同时正负配平", loc="left", fontweight="bold")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        unary = metrics["matched_marginal_baselines"]
        names = ("全局", "输出bit", "活动bit", "最强")
        auc = [
            unary[name]
            for name in ("global", "output_bit", "active_bit", "strongest_auc")
        ]
        auc_x = np.arange(4)
        axes[2].bar(
            auc_x, auc, color=("#94A3B8", "#D97706", "#7C3AED", "#0F766E")
        )
        for index, value in enumerate(auc):
            axes[2].text(index, value + 0.018, f"{value:.3f}", ha="center")
        axes[2].axhline(0.65, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[2].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.1)
        axes[2].set_xticks(auc_x, names)
        axes[2].set_ylim(0.4, max(0.78, max(auc) + 0.08))
        axes[2].set_ylabel("structure-disjoint验证 AUC")
        axes[2].set_title("一元捷径检查", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.065,
            0.178,
            (
                f"matched覆盖：训练{split['train']['structures']}个结构，"
                f"验证{split['validation']['structures']}个结构；"
                f"验证覆盖{split['validation']['output_bits']}个输出bit。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：SKINNY-64/64四轮严格标签与matching readiness；不是神经性能、高轮、攻击或SOTA。",
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
