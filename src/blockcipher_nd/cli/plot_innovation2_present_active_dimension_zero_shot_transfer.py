from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E70 active-dimension zero-shot transfer."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_active_dimension_transfer(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_active_dimension_transfer(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    dimensions = gate["metrics"]["dimensions"]
    labels = ("4-bit活动cube", "12-bit活动cube")
    raw_positive = [dimensions[key]["raw_positive"] for key in ("4", "12")]
    raw_negative = [dimensions[key]["raw_negative"] for key in ("4", "12")]
    raw_unknown = [dimensions[key]["raw_unknown"] for key in ("4", "12")]
    labels_ready = gate["decision"] not in {
        "innovation2_present_active_dimension_transfer_labels_not_ready",
        "innovation2_present_active_dimension_transfer_protocol_invalid",
    }
    decisions = {
        "innovation2_present_active_dimension_zero_shot_confirmed": (
            "严格标签与零样本门通过；允许统一dimension-conditioned算子。"
        ),
        "innovation2_present_active_dimension_zero_shot_not_confirmed": (
            "零样本结构增益未确认；保留E68的8-bit域内结果。"
        ),
        "innovation2_present_active_dimension_transfer_labels_not_ready": (
            "4/12-bit严格标签宽度不足；不解释迁移AUC。"
        ),
        "innovation2_present_active_dimension_transfer_protocol_invalid": (
            "source、前缀兼容或checkpoint协议无效。"
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
            "创新2 E70：8-bit训练的PRESENT平衡谱算子能否跨活动维度迁移",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "为4-bit和12-bit活动cube生成严格unit标签，直接读取E67/E68双seed checkpoint，不重新训练。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "39维前缀按2^dimension归一化，degree>=8折叠；8-bit fixture必须与E65逐值一致。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(2)
        axes[0].bar(x, raw_positive, label="positive证书", color="#2563EB")
        axes[0].bar(x, raw_negative, bottom=raw_positive, label="negative反例", color="#D97706")
        bottoms = np.asarray(raw_positive) + np.asarray(raw_negative)
        axes[0].bar(x, raw_unknown, bottom=bottoms, label="unknown", color="#94A3B8")
        for index in range(2):
            if raw_positive[index]:
                axes[0].text(
                    index,
                    raw_positive[index] / 2,
                    str(raw_positive[index]),
                    ha="center",
                )
            if raw_negative[index]:
                axes[0].text(
                    index,
                    raw_positive[index] + raw_negative[index] / 2,
                    str(raw_negative[index]),
                    ha="center",
                )
            if raw_unknown[index]:
                axes[0].text(
                    index,
                    bottoms[index] + raw_unknown[index] / 2,
                    str(raw_unknown[index]),
                    ha="center",
                )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("16个结构 × 64个输出比特")
        axes[0].set_title("严格标签宽度", loc="left", fontweight="bold")
        axes[0].legend(
            frameon=False,
            ncol=3,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
        )
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        if labels_ready:
            mean_auc = {}
            for dimension in ("4", "12"):
                transfer = dimensions[dimension]["transfer"]
                mean_auc[dimension] = {
                    mode: (
                        transfer[f"seed0_{mode}_auc"]
                        + transfer[f"seed1_{mode}_auc"]
                    )
                    / 2.0
                    for mode in ("independent", "corrupted", "true")
                }
                mean_auc[dimension]["ridge"] = transfer["e65_ridge_auc"]
            modes = ("ridge", "independent", "corrupted", "true")
            mode_labels = ("ANF ridge", "独立node", "错误P", "正确P")
            colors = ("#0F766E", "#94A3B8", "#D97706", "#2563EB")
            width = 0.19
            for mode_index, (mode, mode_label, color) in enumerate(
                zip(modes, mode_labels, colors, strict=True)
            ):
                offset = (mode_index - 1.5) * width
                values = [mean_auc[key][mode] for key in ("4", "12")]
                axes[1].bar(x + offset, values, width, label=mode_label, color=color)
                for index, value in enumerate(values):
                    axes[1].text(
                        index + offset,
                        value + 0.012,
                        f"{value:.3f}",
                        ha="center",
                        fontsize=8.4,
                    )
            axes[1].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.2)
            axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
            axes[1].set_ylim(0.35, 1.04)
            axes[1].set_ylabel("双seed平均零样本 AUC")
            axes[1].set_title("checkpoint零样本迁移", loc="left", fontweight="bold")
            axes[1].legend(frameon=False, ncol=2)
            deltas = gate["metrics"]["mean_deltas"]
            evidence_line = (
                f"四个dimension-seed组合mean增益：独立node="
                f"{deltas['true_minus_independent']:+.3f}，错误P="
                f"{deltas['true_minus_corrupted']:+.3f}，ANF ridge="
                f"{deltas['true_minus_ridge']:+.3f}。"
            )
        else:
            completed = [dimensions[key]["completed_structures"] for key in ("4", "12")]
            axes[1].bar(x, completed, width=0.58, color=("#0F766E", "#D97706"))
            for index, value in enumerate(completed):
                axes[1].text(index, value + 0.45, f"{value}/16", ha="center")
            cap_events = dimensions["12"].get("provider_cap_events", [])
            if cap_events:
                event = cap_events[0]
                axes[1].text(
                    1,
                    2.2,
                    (
                        f"第四轮候选组合 {event['candidate_count']:,}\n"
                        f"> 冻结上限 {event['cap']:,}"
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=9.0,
                    color="#92400E",
                )
            axes[1].set_ylim(0, 18)
            axes[1].set_ylabel("完成严格标签计算的结构数")
            axes[1].set_title("标签提供器完成度", loc="left", fontweight="bold")
            evidence_line = (
                "严格标签门未通过：不展示、不计算也不解释空匹配集合的AUC占位值。"
            )
        axes[1].set_xticks(x, labels)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.178,
            evidence_line,
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
            "证据范围：PRESENT-80四轮严格unit标签的无训练跨维度审计；不是高轮结论、新攻击或SOTA。",
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
