from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E36 small-SPN topology-label identifiability audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_topology_identifiability_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_topology_identifiability_svg(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    counts = gate["metrics"]["counts"]
    thresholds = gate["thresholds"]
    per_round = gate["metrics"]["per_round"]
    decisions = {
        "innovation2_small_spn_topology_labels_identifiable": "P-layer条件标签达到百级组外宽度；先扩充独立拓扑族并重建benchmark。",
        "innovation2_small_spn_topology_labels_not_identifiable": "P-layer条件标签宽度不足；停止当前合成标签路线。",
        "innovation2_small_spn_topology_label_audit_protocol_invalid": "标签来源、shape、variant顺序或train-only选择无效。",
    }
    metric_keys = [
        "train_p_sensitive_any_s",
        "train_p_sensitive_all_s",
        "heldout_p3_novel_any_s",
        "heldout_p3_novel_all_s",
        "dual_p_effect_cells",
        "train_interaction_cells",
        "full_interaction_cells",
    ]
    metric_labels = [
        "训练P敏感\n至少一个S",
        "训练P敏感\n全部S",
        "P3新变化\n至少一个S",
        "P3新变化\n全部S",
        "dual P效应",
        "训练S×P\ninteraction",
        "完整S×P\ninteraction",
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
        figure, axes = plt.subplots(1, 2, figsize=(14.5, 7.6))
        figure.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.22, wspace=0.25)
        figure.text(
            0.08,
            0.955,
            "创新2 E36：精确标签是否包含可识别的P-layer条件信号",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.895,
            "直接审计4个S-box × 4个P-layer的589个train-only matched cell；不训练网络，不修改selected mask。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.850,
            "绿色柱为实际cell数，红色短线为预注册最低宽度；所有宽度门同时通过才允许重建benchmark。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        values = np.asarray([counts[key] for key in metric_keys])
        minimums = np.asarray([thresholds[key] for key in metric_keys])
        x = np.arange(len(metric_keys))
        axes[0].bar(x, values, color="#0F766E", width=0.66, label="实际cell数")
        axes[0].scatter(
            x,
            minimums,
            marker="_",
            s=240,
            linewidths=2.2,
            color="#DC2626",
            label="最低宽度门",
            zorder=3,
        )
        for index, value in enumerate(values):
            axes[0].text(index, value + 10, str(int(value)), ha="center", va="bottom", fontsize=8.8)
        axes[0].set_xticks(x, metric_labels)
        axes[0].set_ylim(0, 650)
        axes[0].set_ylabel("base cell数量（共589）")
        axes[0].set_title("P敏感性与S×P交互宽度", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)

        round_labels = [f"{row['round_index'] + 2}轮" for row in per_round]
        round_x = np.arange(len(per_round))
        width = 0.20
        series = [
            ("selected cell", "selected_cells", "#64748B"),
            ("训练P敏感", "train_p_sensitive_any_s", "#2563EB"),
            ("dual P效应", "dual_p_effect_cells", "#D97706"),
            ("完整interaction", "full_interaction_cells", "#0F766E"),
        ]
        for series_index, (label, key, color) in enumerate(series):
            offset = (series_index - 1.5) * width
            axes[1].bar(
                round_x + offset,
                [row[key] for row in per_round],
                width,
                label=label,
                color=color,
            )
        axes[1].set_xticks(round_x, round_labels)
        axes[1].set_ylim(0, 420)
        axes[1].set_ylabel("base cell数量")
        axes[1].set_title("按轮数分解标签宽度", loc="left", fontweight="bold", pad=12)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, ncol=2, fontsize=8.5)

        figure.text(
            0.08,
            0.135,
            "dual P效应子集：{}个，目标正类{}、负类{}；训练只包含3个独立P-layer拓扑。".format(
                counts["dual_p_effect_cells"],
                counts["dual_p_effect_positive"],
                counts["dual_p_effect_negative"],
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.08,
            0.090,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.08,
            0.045,
            "证据范围：16-bit合成SPN精确标签可识别性；不是神经结果、真实密码攻击或高轮积分证明。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
