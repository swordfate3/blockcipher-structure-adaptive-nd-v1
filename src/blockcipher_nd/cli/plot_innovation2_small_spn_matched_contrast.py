from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E32b matched-contrast readjudication."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_matched_contrast_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_matched_contrast_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    split_metrics = summary["split_metrics"]
    baselines = summary["marginal_baselines"]
    split_names = ["train", "unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["训练拓扑块", "未见S-box", "未见P-layer", "双轴未见"]
    positive = [split_metrics[name]["positive"] for name in split_names]
    negative = [split_metrics[name]["negative"] for name in split_names]
    heldout = ["unseen_sbox", "unseen_player", "dual_unseen"]
    heldout_labels = ["未见S-box", "未见P-layer", "双轴未见"]
    raw = [metrics["raw_strongest_marginal_auc"][name] for name in heldout]
    matched = [baselines[name]["strongest_auc"] for name in heldout]
    decisions = {
        "innovation2_small_spn_matched_contrast_ready": "训练内contrast消除ID捷径；可准备E33网络比较。",
        "innovation2_small_spn_matched_contrast_still_shortcut_dominated": "matched数据仍被ID边际解释；停止当前benchmark。",
        "innovation2_small_spn_matched_contrast_too_narrow": "matched数据宽度不足；不得读取heldout放宽选择。",
        "innovation2_small_spn_matched_contrast_protocol_invalid": "来源、train-only选择或索引协议无效。",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.3,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.8, 8.2), gridspec_kw={"width_ratios": [1, 1.05]})
        figure.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.24, wspace=0.26)
        figure.suptitle(
            "创新2 E32b：训练内matched-contrast重裁决",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=15.2,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.885,
            "只用9个训练拓扑选择positive count为1..8的(round, structure, mask) cell；不读取heldout标签，不改变原始0/1 target。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.837,
            "比较原始E32与matched数据的同一ID边际AUC；目标是确认网络未来必须读取cipher topology，而不是记忆通用ID。",
            ha="left",
            va="top",
            fontsize=9.5,
            color="#526070",
        )
        x = np.arange(len(split_names))
        width = 0.36
        axes[0].bar(x - width / 2, positive, width, color="#16A34A", label="正类")
        axes[0].bar(x + width / 2, negative, width, color="#DC2626", label="负类")
        axes[0].set_xticks(x, split_labels, rotation=15, ha="right")
        axes[0].set_ylabel("matched标签数量")
        axes[0].set_title("589个selected base cell的split宽度", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False)
        for index, value in enumerate(positive):
            axes[0].text(index - width / 2, value, str(value), ha="center", va="bottom", fontsize=8.5)
        for index, value in enumerate(negative):
            axes[0].text(index + width / 2, value, str(value), ha="center", va="bottom", fontsize=8.5)

        y = np.arange(len(heldout))
        axes[1].scatter(raw, y - 0.10, color="#DC2626", s=62, label="原始E32最强ID边际")
        axes[1].scatter(matched, y + 0.10, color="#0F766E", s=62, label="matched最强ID边际")
        for index in range(len(heldout)):
            axes[1].plot([matched[index], raw[index]], [index + 0.10, index - 0.10], color="#CBD5E1", linewidth=1.2)
        axes[1].axvline(0.75, color="#DC2626", linestyle="--", linewidth=1.1, label="双轴停止线 0.75")
        axes[1].axvline(0.80, color="#D97706", linestyle=":", linewidth=1.2, label="单轴停止线 0.80")
        axes[1].set_yticks(y, heldout_labels)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0.45, 1.01)
        axes[1].set_xlabel("strongest train-derived ID marginal AUC")
        axes[1].set_title("train-only选择前后的组外捷径", loc="left", fontweight="bold", pad=12)
        axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
        figure.text(
            0.08,
            0.108,
            (
                f"selected base cells={metrics['selected_base_cells']}，总label rows={metrics['selected_total_label_rows']}，"
                f"不同16-topology模式={metrics['distinct_topology_label_patterns']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.08,
            0.064,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.08,
            0.020,
            "证据范围：冻结16-bit合成SPN精确标签的train-only重裁决；不是实际密码高轮结果、heldout调参或神经性能。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
