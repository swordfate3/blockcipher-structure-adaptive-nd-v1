from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E87 RECTANGLE-80 strict unit-profile readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_rectangle80_unit_profile(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_rectangle80_unit_profile(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    metadata = summary.get("metadata", {})
    experiment = metadata.get("experiment", "e87")
    structure_count = metadata.get("config", {}).get("structure_count")
    if structure_count is None:
        structure_count = metrics.get("raw_rows", 6144) // 64
    split = metrics["matched_split_metrics"]
    raw_counts = (
        metrics["raw_positive"],
        metrics["raw_negative"],
        metrics["raw_unknown"],
    )
    decisions = {
        "innovation2_rectangle80_unit_profile_ready": (
            "严格标签、matching和反捷径门通过；下一步扩展到192结构，仍不训练网络。"
        ),
        "innovation2_rectangle80_unit_profile_raw_labels_not_ready": (
            "四轮原始正负标签未进入过渡区；下一步只把轮数改为五轮。"
        ),
        "innovation2_rectangle80_unit_profile_matching_not_ready": (
            "原始标签足够但checkerboard容量不足；下一步只扩到192结构。"
        ),
        "innovation2_rectangle80_unit_profile_protocol_invalid": (
            "最终版规范、行序、向量化、support或反例协议无效；必须先修复。"
        ),
        "innovation2_rectangle80_unit_profile_expansion_ready": (
            "192结构容量复核通过；下一步只做本地、仅使用第3轮前缀的三行神经网络就绪实验。"
        ),
        "innovation2_rectangle80_unit_profile_expansion_not_ready": (
            "192结构仍未稳定保持宽度或反捷径门；关闭当前RECTANGLE四轮神经路线。"
        ),
        "innovation2_rectangle80_unit_profile_expansion_protocol_invalid": (
            "E87锚点重放或RECTANGLE标签协议无效；不解释扩结构结果。"
        ),
    }
    title = (
        "创新2 E88：RECTANGLE-80四轮扩到192结构后能否开放神经网络实验"
        if experiment == "e88"
        else "创新2 E87：RECTANGLE-80四轮能否形成严格输出平衡谱标签"
    )
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
        figure, axes = plt.subplots(1, 3, figsize=(15.6, 8.3))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.36
        )
        figure.text(
            0.065,
            0.955,
            title,
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            (
                f"{structure_count}个8维输入cube，每个结构查询64个输出bit；"
                "当前只审计标签，不训练神经网络。"
            ),
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "正类由ANF support缺失证明；负类由具体80-bit key与inactive offset的XOR=1反例证明。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        colors = ("#0F766E", "#2563EB", "#94A3B8")
        axes[0].bar(x, raw_counts, color=colors)
        raw_pad = max(raw_counts) * 0.035 if max(raw_counts) else 1
        for index, value in enumerate(raw_counts):
            axes[0].text(index, value + raw_pad, f"{value:,}", ha="center")
        axes[0].set_xticks(x, ("可证明平衡", "反例非平衡", "未知"))
        axes[0].set_ylim(0, max(1, max(raw_counts) * 1.18))
        axes[0].set_ylabel("结构 × 输出bit 数量")
        axes[0].set_title("原始三态标签atlas", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        split_x = np.arange(2)
        width = 0.34
        positives = [split[name]["positive"] for name in ("train", "validation")]
        negatives = [split[name]["negative"] for name in ("train", "validation")]
        axes[1].bar(
            split_x - width / 2, positives, width, color="#0F766E", label="平衡"
        )
        axes[1].bar(
            split_x + width / 2, negatives, width, color="#2563EB", label="非平衡"
        )
        value_pad = max(positives + negatives) * 0.04 if max(positives + negatives) else 1
        for index, value in enumerate(positives):
            axes[1].text(index - width / 2, value + value_pad, str(value), ha="center")
        for index, value in enumerate(negatives):
            axes[1].text(index + width / 2, value + value_pad, str(value), ha="center")
        axes[1].set_xticks(split_x, ("训练", "验证"))
        axes[1].set_ylim(0, max(1, max(positives + negatives) * 1.24))
        axes[1].set_ylabel("checkerboard标签数量")
        axes[1].set_title("行列同时正负平衡", loc="left", fontweight="bold")
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
            0.184,
            (
                f"matched覆盖：训练{split['train']['structures']}个结构；"
                f"验证{split['validation']['structures']}个结构、"
                f"{split['validation']['output_bits']}个输出bit。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            (
                "证据范围：RECTANGLE-80最终版四轮严格标签与matching容量readiness；"
                "不是7轮论文复现、神经性能、高轮攻击或SOTA。"
            ),
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
