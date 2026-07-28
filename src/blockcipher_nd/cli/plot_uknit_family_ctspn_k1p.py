from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1p import (
    AUC_FLOOR,
    LABEL_SHUFFLE_MARGIN,
    RAW_MARGIN,
)


SPLIT_LABELS = {
    "same_key_fresh": "同一密钥的新样本",
    "cross_key_validation": "更换密钥的新样本",
}
SEED_STYLES = {
    "0": ("#0F766E", "o"),
    "1": ("#C2410C", "s"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-P round-calibration chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1p_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1p_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    rounds = gate.get("round_results", {})
    if set(rounds) != {"3", "4", "5"}:
        raise ValueError("K1-P plot requires r3, r4, and r5 summaries")
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
            left=0.075,
            right=0.965,
            top=0.77,
            bottom=0.11,
            hspace=0.48,
            wspace=0.23,
        )
        figure.suptitle(
            "创新1：uKNIT 的 0x40 输入差分从第几轮开始失去稳定信号",
            x=0.05,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.905,
            "固定四对密文、两颗 seed 和严格负样本，只把加密轮数从 3 轮改到 5 轮；5 轮数据直接复用 K1-O。",
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
        for column, split in enumerate(SPLIT_LABELS):
            _plot_auc_panel(axes[0, column], rounds, split)
            _plot_margin_panel(axes[1, column], rounds, split)
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
        "rounds_are_explicit": True,
        "uses_local_axis_ranges": True,
        "seed_labels_offset_separately": True,
    }


def _plot_auc_panel(
    axis: plt.Axes,
    round_results: Mapping[str, Any],
    split: str,
) -> None:
    x = [3, 4, 5]
    all_values: list[float] = []
    for seed, (color, marker) in SEED_STYLES.items():
        values = [
            float(round_results[str(rounds)][seed][split]["exact_auc"])
            for rounds in x
        ]
        all_values.extend(values)
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=7,
            label=f"seed{seed}",
        )
        for rounds, value in zip(x, values, strict=True):
            offset = (-8, 9) if seed == "0" else (8, -15)
            axis.annotate(
                f"{value:.3f}",
                (rounds, value),
                xytext=offset,
                textcoords="offset points",
                ha="right" if seed == "0" else "left",
                fontsize=8.0,
                color=color,
            )
    lower = min(0.485, min(all_values) - 0.018)
    upper = max(0.575, max(all_values) + 0.022)
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, ["3轮", "4轮", "5轮"])
    axis.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle=(0, (3, 3)))
    axis.axhline(
        AUC_FLOOR,
        color="#047857",
        linewidth=1.2,
        linestyle=(0, (5, 3)),
        label="有效信号门槛 0.550",
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_ylabel("精确五阶段 AUC（局部放大）")
    axis.set_title(
        f"{SPLIT_LABELS[split]}：正确局部状态是否可分",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5)


def _plot_margin_panel(
    axis: plt.Axes,
    round_results: Mapping[str, Any],
    split: str,
) -> None:
    x = [3, 4, 5]
    metrics = (
        ("exact_minus_raw", "超过原始密文", "#2563EB", "o", RAW_MARGIN),
        (
            "exact_minus_label_shuffle",
            "超过标签打乱",
            "#7C3AED",
            "s",
            LABEL_SHUFFLE_MARGIN,
        ),
    )
    all_values: list[float] = []
    for seed_index, seed in enumerate(SEED_STYLES):
        for metric_index, (key, label, color, marker, threshold) in enumerate(metrics):
            values = [
                float(round_results[str(rounds)][seed][split][key])
                for rounds in x
            ]
            all_values.extend(values)
            linestyle = "-" if seed == "0" else "--"
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6,
                label=f"{label} · seed{seed}",
            )
            for rounds, value in zip(x, values, strict=True):
                direction = 1 if (seed_index + metric_index) % 2 == 0 else -1
                axis.annotate(
                    f"{value:+.3f}",
                    (rounds, value),
                    xytext=(0, 8 * direction),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if direction > 0 else "top",
                    fontsize=7.2,
                    color=color,
                )
            axis.axhline(
                threshold,
                color=color,
                linewidth=0.9,
                alpha=0.45,
                linestyle=(0, (2, 3)),
            )
    lower = min(-0.04, min(all_values) - 0.025)
    upper = max(0.065, max(all_values) + 0.03)
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, ["3轮", "4轮", "5轮"])
    axis.axhline(0.0, color="#6B7280", linewidth=1)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_ylabel("正确结构的 AUC 净优势")
    axis.set_title(
        f"{SPLIT_LABELS[split]}：是否真正超过两个控制",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=2, loc="best", fontsize=8.0)


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("lower_round_signal_supported_r5_loss_boundary"):
        return "结论：3轮和4轮都有稳定信号，5轮是当前 0x40 差分的首个失效锚点。"
    if decision.endswith("r3_signal_supported_boundary_before_r4"):
        return "结论：3轮有稳定信号，但4轮已经失效；应先为4轮重新寻找差分。"
    if decision.endswith("current_difference_unresolved_from_r3"):
        return "结论：从3轮开始就没有稳定信号；先检查 0x40 的位置、位序和数据构造。"
    if decision.endswith("nonmonotonic_round_or_split_instability"):
        return "结论：轮数表现不单调；先检查轮调用、位序、密钥和运行时窗口是否对齐。"
    if decision.endswith("lower_round_signal_or_attribution_unstable"):
        return "结论：部分样本有信号但控制门槛不稳定；暂不支持继续改网络。"
    return "结论：实验协议无效；修复失败的不变量后按原计划重跑。"


def _decision_color(status: str) -> str:
    if status == "pass":
        return "#047857"
    if status == "hold":
        return "#B45309"
    return "#B91C1C"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1p_svg"]
