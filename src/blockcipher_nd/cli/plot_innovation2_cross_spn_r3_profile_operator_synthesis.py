from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E80 cross-SPN r3-only method synthesis."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_method_synthesis(summary, args.output)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


def render_method_synthesis(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    ciphers = gate["metrics"]["ciphers"]
    skinny = gate["metrics"]["skinny"]
    decisions = {
        "innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready": (
            "双密码方法证据通过；SKINNY严格标签尚未就绪，下一步先做E81标签门。"
        ),
        "innovation2_cross_spn_r3_profile_method_confirmed_third_spn_ready": (
            "双密码方法证据通过，第三SPN标签已就绪，可运行冻结三行readiness。"
        ),
        "innovation2_cross_spn_r3_profile_method_not_confirmed": (
            "PRESENT或GIFT逐seed归因未同时通过，不提升为跨SPN方法证据。"
        ),
        "innovation2_cross_spn_method_synthesis_protocol_invalid": (
            "冻结来源、hash、训练契约或历史裁决不一致。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(16.0, 8.7),
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.12]},
        )
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.24, wspace=0.31
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E80：同一r3-only平衡谱算子是否在两个真实SPN上获得独立证据",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "PRESENT与GIFT分别在各自严格标签上重新训练；只比较每个密码内部的真实P与控制。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.text(
            0.065,
            0.848,
            "两套标签的结构数和可观测边不同，禁止把两个面板的AUC高低解释为跨密码性能排名。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#475569",
        )
        figure.legend(
            handles=[
                Patch(facecolor="#94A3B8", label="独立node"),
                Patch(facecolor="#2563EB", label="错误P"),
                Patch(facecolor="#0F766E", label="真实P"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.065, 0.79),
            frameon=False,
            ncol=3,
            fontsize=8.8,
        )

        for axis, row in zip(axes[:2], ciphers, strict=True):
            _plot_cipher_panel(axis, row)

        stages = [
            ("r7论文kernel复现", skinny["r7_kernel_reproduced"]),
            ("r8论文kernel复现", skinny["r8_kernel_reproduced"]),
            ("r8相邻pair标签宽度", skinny["r8_adjacent_pair_ready"]),
            ("r8底行pair闭合", skinny["r8_bottom_row_ready"]),
            ("r7单cell标签宽度", skinny["r7_single_cell_ready"]),
            ("同级strict profile标签", skinny["strict_profile_labels_ready"]),
        ]
        y = np.arange(len(stages))[::-1]
        colors = ["#0F766E" if passed else "#CBD5E1" for _, passed in stages]
        axes[2].barh(y, [1.0] * len(stages), color=colors, height=0.58)
        for position, (label, passed) in zip(y, stages, strict=True):
            axes[2].text(
                0.03,
                position,
                label,
                ha="left",
                va="center",
                color="#FFFFFF" if passed else "#334155",
                fontsize=8.8,
                fontweight="bold" if passed else "normal",
            )
            axes[2].text(
                1.04,
                position,
                "通过" if passed else "未就绪",
                ha="left",
                va="center",
                color="#0F766E" if passed else "#64748B",
                fontsize=8.8,
            )
        axes[2].set_xlim(0.0, 1.32)
        axes[2].set_ylim(-0.8, len(stages) - 0.2)
        axes[2].set_yticks([])
        axes[2].set_xticks([])
        axes[2].set_title(
            "第三密码：SKINNY标签条件",
            loc="left",
            fontweight="bold",
            pad=12,
        )
        for spine in axes[2].spines.values():
            spine.set_visible(False)
        axes[2].text(
            0.0,
            -0.18,
            "论文kernel复现不等于训练标签就绪；当前ready label family = 0。",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#526070",
            wrap=True,
        )

        figure.text(
            0.065,
            0.165,
            (
                "共享方法契约：64个输出节点、每节点13维r3特征、hidden 32、"
                "2步共享消息传递、4,795参数、30轮、seed0/1。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.103,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.047,
            "证据范围：两个真实SPN的四轮方法归因；不是零样本迁移、高轮区分器、攻击、远程规模或SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _plot_cipher_panel(axis: Any, row: dict[str, Any]) -> None:
    seeds = ("seed0", "seed1")
    per_seed = row["per_seed"]
    independent = [per_seed[seed]["independent_auc"] for seed in seeds]
    corrupted = [per_seed[seed]["corrupted_auc"] for seed in seeds]
    true = [per_seed[seed]["true_auc"] for seed in seeds]
    x = np.arange(2)
    width = 0.23
    axis.bar(x - width, independent, width, label="独立node", color="#94A3B8")
    axis.bar(x, corrupted, width, label="错误P", color="#2563EB")
    axis.bar(x + width, true, width, label="真实P", color="#0F766E")
    for index, value in enumerate(true):
        axis.text(index + width, value + 0.012, f"{value:.3f}", ha="center", fontsize=8.6)
    axis.axhline(0.5, color="#64748B", linestyle=":", linewidth=1.0)
    axis.set_xticks(x, ("seed0", "seed1"))
    axis.set_ylim(0.45, 1.03)
    axis.set_ylabel("validation AUC")
    axis.set_title(
        f"{row['cipher']}四轮\n{row['structures']}结构 / {row['observed_matched_edges']}条可观测边",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.text(
        0.0,
        -0.18,
        (
            f"平均真实P-独立node {row['mean_true_minus_independent']:+.3f}；"
            f"平均真实P-错误P {row['mean_true_minus_corrupted']:+.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#526070",
    )


if __name__ == "__main__":
    raise SystemExit(main())
