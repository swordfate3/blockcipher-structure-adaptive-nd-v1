from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = ("位置规则", "坐标集合网 seed0", "坐标集合网 seed1")
COLORS = ("#64748B", "#0F766E", "#D97706")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E105 PRESENT r9 split333 source-heldout ranking evidence."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_source_heldout_ranking(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_source_heldout_ranking(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    result_rows = summary["result_rows"]
    anchor = next(row for row in result_rows if row["model"].startswith("absolute_position"))
    neural = [
        next(row for row in result_rows if row["seed"] == seed)
        for seed in (0, 1)
    ]
    rows = (anchor, *neural)

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
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 9.2))
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.70,
            bottom=0.29,
            wspace=0.34,
        )
        figure.text(
            0.065,
            0.958,
            "创新2 E105：PRESENT九轮缺失(3,3,3)来源留出排序",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.897,
            "模型只在ATM公开八个九轮split上训练；新生成的(3,3,3)关系仅用于一次冻结评估，不更新权重。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.849,
            "每个池包含一条已验证正关系及其同步旋转未标注候选；未标注候选不是严格负类。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(MODEL_LABELS))
        width = 0.34
        recall1 = [float(row["recall_at_1"]) for row in rows]
        recall5 = [float(row["recall_at_5"]) for row in rows]
        axes[0].bar(x - width / 2, recall1, width, color="#94A3B8", label="Recall@1")
        axes[0].bar(x + width / 2, recall5, width, color="#475569", label="Recall@5")
        axes[0].axhline(0.50, color="#DC2626", linestyle="--", linewidth=1.2, label="Recall@5最低线0.50")
        axes[0].set_xticks(x, MODEL_LABELS, rotation=18, ha="right")
        axes[0].set_ylim(0, 1.06)
        axes[0].set_ylabel("比例")
        axes[0].set_title("正关系进入前1名或前5名的比例", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, fontsize=8.2)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        mrr = [float(row["mean_reciprocal_rank"]) for row in rows]
        axes[1].bar(x, mrr, color=COLORS, width=0.58)
        axes[1].axhline(0.40, color="#DC2626", linestyle="--", linewidth=1.2, label="MRR最低线0.40")
        axes[1].set_xticks(x, MODEL_LABELS, rotation=18, ha="right")
        axes[1].set_ylim(0, 1.06)
        axes[1].set_ylabel("MRR")
        axes[1].set_title("正关系平均排名质量", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, fontsize=8.2)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        gain_x = np.arange(2)
        recall_gain = [row["recall_at_5"] - anchor["recall_at_5"] for row in neural]
        mrr_gain = [row["mean_reciprocal_rank"] - anchor["mean_reciprocal_rank"] for row in neural]
        axes[2].bar(gain_x - width / 2, recall_gain, width, color="#0F766E", label="Recall@5增益")
        axes[2].bar(gain_x + width / 2, mrr_gain, width, color="#D97706", label="MRR增益")
        axes[2].axhline(0.20, color="#0F766E", linestyle="--", linewidth=1.2, alpha=0.75, label="Recall门槛+0.20")
        axes[2].axhline(0.15, color="#D97706", linestyle=":", linewidth=1.4, alpha=0.85, label="MRR门槛+0.15")
        lower = min(0.0, min((*recall_gain, *mrr_gain)) - 0.05)
        upper = max(0.25, max((*recall_gain, *mrr_gain)) + 0.08)
        axes[2].set_ylim(lower, upper)
        axes[2].set_xticks(gain_x, ("seed0", "seed1"))
        axes[2].set_ylabel("相对位置规则的绝对增益")
        axes[2].set_title("冻结模型是否超过确定性位置规则", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, fontsize=8.0)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        audit = summary.get("audit", {})
        relation_count = int(audit.get("heldout_relations", 0))
        minimum_candidates = int(audit.get("minimum_unlabeled_per_pool", 0))
        figure.text(
            0.065,
            0.176,
            f"评估宽度：{relation_count}条新关系；每池最少{minimum_candidates}个未标注候选；训练/反向传播/权重更新均为0。",
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        decision_text = {
            "innovation2_present_r9_split333_source_heldout_signal_confirmed": "通过：公开语料坐标排序信号在新来源中确认。",
            "innovation2_present_r9_split333_source_heldout_diagnostic_only": "诊断：关系或候选宽度不足，不作来源泛化结论。",
            "innovation2_present_r9_split333_source_shift_not_confirmed": "暂缓：冻结模型未通过来源迁移门，停止当前坐标身份路线。",
            "innovation2_present_r9_split333_source_heldout_protocol_invalid": "失败：来源、候选或冻结权重协议无效。",
        }.get(gate["decision"], f"未识别裁决：{gate['decision']}")
        figure.text(
            0.065,
            0.108,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.065,
            0.045,
            "证据范围：独立轮密钥PRESENT九轮ATM关系的正例-未标注排序；不是二分类、PRESENT-80、区分器、攻击、论文公开结果或SOTA。",
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
