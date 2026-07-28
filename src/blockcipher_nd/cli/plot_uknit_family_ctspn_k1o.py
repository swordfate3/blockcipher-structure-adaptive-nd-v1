from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    AUC_FLOOR,
    CANDIDATE_VIEW,
    INVARIANT_VIEW,
    LABEL_SHUFFLE_VIEW,
    NO_SBOX_VIEW,
    RAW_VIEW,
    VIEW_NAMES,
    WRONG_SBOX_VIEW,
)


VIEW_LABELS = {
    RAW_VIEW: "仅原始密文",
    CANDIDATE_VIEW: "精确五阶段\n保留位置",
    NO_SBOX_VIEW: "不执行逆S盒",
    WRONG_SBOX_VIEW: "错误S盒归属",
    INVARIANT_VIEW: "精确五阶段\n汇聚位置",
    LABEL_SHUFFLE_VIEW: "打乱训练标签",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-O deterministic signal-audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1o_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1o_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"0", "1"}:
        raise ValueError("K1-O plot requires both seed summaries")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.8))
        figure.subplots_adjust(
            left=0.07,
            right=0.96,
            top=0.78,
            bottom=0.12,
            hspace=0.53,
            wspace=0.23,
        )
        figure.suptitle(
            "创新1：uKNIT 五轮的精确局部状态里是否真的存在可学习信号",
            x=0.05,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.905,
            "固定同一批数据，用闭式 Fisher/LDA 比较原始密文、正确/错误 S 盒、位置汇聚和标签打乱；全程不训练神经网络。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.85,
            _decision_text(gate),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        for column, split in enumerate(("same_key_fresh", "cross_key_validation")):
            _plot_auc_panel(axes[0, column], seed_results, split)
            _plot_margin_panel(axes[1, column], seed_results, split)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 10.8,
        "language": "zh-CN",
        "panels": 4,
        "title_explains_run": True,
        "uses_point_comparison_instead_of_overlapping_curves": True,
    }


def _plot_auc_panel(
    axis: plt.Axes,
    seed_results: Mapping[str, Any],
    split: str,
) -> None:
    x = list(range(len(VIEW_NAMES)))
    colors = ("#0F766E", "#2563EB")
    markers = ("o", "s")
    all_values: list[float] = []
    for seed, color, marker in zip(("0", "1"), colors, markers, strict=True):
        summary = seed_results[seed][split]
        aucs = _view_aucs(summary)
        values = [aucs[view] for view in VIEW_NAMES]
        all_values.extend(values)
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=1.4,
            markersize=6,
            label=f"seed{seed}",
        )
        for index, value in enumerate(values):
            label_offset = (-9, 8) if seed == "0" else (9, -14)
            axis.annotate(
                f"{value:.3f}",
                (index, value),
                xytext=label_offset,
                textcoords="offset points",
                ha="right" if seed == "0" else "left",
                fontsize=7.4,
                color=color,
            )
    lower = min(0.49, min(all_values) - 0.015)
    upper = max(0.57, max(all_values) + 0.018)
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, [VIEW_LABELS[view] for view in VIEW_NAMES])
    axis.tick_params(axis="x", labelsize=8.2)
    axis.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle=(0, (3, 3)))
    axis.axhline(AUC_FLOOR, color="#047857", linewidth=1.2, linestyle=(0, (5, 3)))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_ylabel("AUC（局部放大）")
    axis.set_title(
        f"{_split_label(split)}：六种统计视图的区分能力",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=2, loc="upper right")


def _plot_margin_panel(
    axis: plt.Axes,
    seed_results: Mapping[str, Any],
    split: str,
) -> None:
    keys = (
        "exact_minus_raw",
        "exact_minus_no_sbox",
        "exact_minus_wrong_sbox",
        "exact_minus_invariant",
        "exact_minus_label_shuffle",
    )
    labels = (
        "相对原始",
        "相对无S盒",
        "相对错误S盒",
        "相对位置汇聚",
        "相对标签打乱",
    )
    thresholds = (0.010, 0.005, 0.005, 0.010, 0.030)
    x = list(range(len(keys)))
    width = 0.34
    for offset, seed, color in ((-width / 2, "0", "#0F766E"), (width / 2, "1", "#2563EB")):
        values = [float(seed_results[seed][split][key]) for key in keys]
        bars = axis.bar(
            [position + offset for position in x],
            values,
            width=width,
            color=color,
            alpha=0.88,
            label=f"seed{seed}",
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.4, color=color)
    axis.scatter(x, thresholds, marker="_", s=330, linewidths=2.2, color="#047857", label="前进门槛")
    all_values = [
        float(seed_results[seed][split][key])
        for seed in ("0", "1")
        for key in keys
    ]
    span = max(0.04, max(all_values + list(thresholds)) - min(all_values))
    axis.set_ylim(
        min(-0.02, min(all_values) - 0.18 * span),
        max(0.04, max(all_values + list(thresholds)) + 0.24 * span),
    )
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    axis.set_xticks(x, labels)
    axis.tick_params(axis="x", labelsize=8.8)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_ylabel("精确位置 AUC - 控制 AUC")
    axis.set_title(
        f"{_split_label(split)}：正确局部状态的净优势",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.2)


def _view_aucs(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        RAW_VIEW: float(summary["raw_auc"]),
        CANDIDATE_VIEW: float(summary["exact_auc"]),
        NO_SBOX_VIEW: float(summary["no_sbox_auc"]),
        WRONG_SBOX_VIEW: float(summary["wrong_sbox_auc"]),
        INVARIANT_VIEW: float(summary["invariant_auc"]),
        LABEL_SHUFFLE_VIEW: float(summary["label_shuffle_auc"]),
    }


def _split_label(split: str) -> str:
    return "同一密钥的新样本" if split == "same_key_fresh" else "更换密钥的新样本"


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("position_preserving_signal_supported"):
        return "裁决：精确局部状态信号成立，而且保留 cell 位置很重要；下一步只改为位置保持的 cell-stage 头。"
    if decision.endswith("invariant_stage_signal_supported"):
        return "裁决：精确阶段统计信号成立，但绝对 cell 位置没有额外贡献；下一步使用更小的阶段直方图分支。"
    if decision.endswith("signal_without_sbox_identifiability"):
        return "裁决：存在统计信号，但正确 S 盒没有领先错误/无 S 盒；先定位原始或线性阶段，不再堆 S 盒条件。"
    if decision.endswith("current_differential_signal_not_supported"):
        return "裁决：当前 uKNIT 五轮输入差分在精确局部统计中仍无稳定信号；停止改网络，先更换或审计差分。"
    if decision.endswith("partial_state_signal_unstable"):
        return "裁决：不同 seed 或数据划分不一致；先按密钥和阶段定位不稳定来源，不扩大规模。"
    return "裁决：协议无效；只修复失败的来源、特征或评分不变量后原样重跑。"


def _decision_color(status: str) -> str:
    if status == "pass":
        return "#047857"
    if status == "hold":
        return "#B45309"
    return "#B91C1C"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1o_svg"]
