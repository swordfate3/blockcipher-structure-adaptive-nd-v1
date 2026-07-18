from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E86 PRESENT/GIFT shared profile seed0 attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_shared_profile_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_shared_profile_attribution(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in gate["metrics"]["rows"]}
    modes = ("independent", "corrupted", "true")
    mode_names = ("独立关系", "错误拓扑", "真实拓扑")
    colors = ("#94A3B8", "#2563EB", "#0F766E")
    anchors = {"present": 0.9455555555555556, "gift": 0.913111342351717}
    aucs = {
        cipher: [rows[mode][f"{cipher}_validation_auc"] for mode in modes]
        for cipher in ("present", "gift")
    }
    margins = [
        gate["metrics"]["present_true_minus_independent"],
        gate["metrics"]["present_true_minus_corrupted"],
        gate["metrics"]["gift_true_minus_independent"],
        gate["metrics"]["gift_true_minus_corrupted"],
    ]
    margin_names = (
        "PRESENT\n真实-独立",
        "PRESENT\n真实-错误",
        "GIFT\n真实-独立",
        "GIFT\n真实-错误",
    )
    decisions = {
        "innovation2_shared_profile_operator_seed0_attributed": (
            "共享模型在两个密码上均保留正式质量与拓扑增益；下一步运行完全相同的seed1。"
        ),
        "innovation2_shared_profile_operator_quality_not_retained": (
            "至少一个密码未保留独立模型质量；保留E73/E79并关闭共享参数分支。"
        ),
        "innovation2_shared_profile_operator_topology_not_attributed": (
            "至少一个密码未超过独立或错误拓扑；保留E73/E79并关闭共享参数分支。"
        ),
        "innovation2_shared_profile_operator_attribution_protocol_invalid": (
            "E85/锚点来源、30轮schedule、动态拓扑或产物协议无效。"
        ),
    }
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
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.5))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.29, wspace=0.34
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E86：共享拓扑算子30轮seed0能否保持双密码正式质量",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "同一套4,795参数联合训练PRESENT与GIFT；三行共享数据、初始化、630次更新和optimizer。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "每个密码分别对齐其独立30轮seed0锚点；不能用另一密码的高分补偿失败。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        for axis, cipher, title in zip(
            axes[:2],
            ("present", "gift"),
            ("PRESENT-80四轮seed0", "GIFT-64四轮seed0"),
            strict=True,
        ):
            x = np.arange(3)
            axis.bar(x, aucs[cipher], color=colors)
            for index, value in enumerate(aucs[cipher]):
                axis.text(index, value + 0.014, f"{value:.3f}", ha="center")
            axis.axhline(
                anchors[cipher],
                color="#D97706",
                linestyle="--",
                linewidth=1.2,
                label=f"独立30轮锚点 {anchors[cipher]:.3f}",
            )
            axis.axhline(
                anchors[cipher] - 0.03,
                color="#DC2626",
                linestyle=":",
                linewidth=1.1,
                label=f"质量下限 {anchors[cipher] - 0.03:.3f}",
            )
            axis.set_xticks(x, mode_names, rotation=8)
            axis.set_ylim(0.45, 1.0)
            axis.set_ylabel("validation AUC")
            axis.set_title(title, loc="left", fontweight="bold")
            axis.legend(frameon=False, fontsize=8.0, loc="upper left")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)

        x = np.arange(4)
        margin_colors = [
            "#0F766E" if value >= 0.03 else "#DC2626" for value in margins
        ]
        axes[2].bar(x, margins, color=margin_colors)
        for index, value in enumerate(margins):
            offset = 0.012 if value >= 0 else -0.025
            axes[2].text(index, value + offset, f"{value:+.3f}", ha="center")
        axes[2].axhline(
            0.03,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label="拓扑门 +0.03",
        )
        axes[2].axhline(0.0, color="#64748B", linewidth=1.0)
        axes[2].set_xticks(x, margin_names)
        axes[2].set_ylim(
            min(-0.08, min(margins) - 0.04),
            max(0.40, max(margins) + 0.06),
        )
        axes[2].set_ylabel("真实拓扑 AUC差值")
        axes[2].set_title("逐密码正式归因门", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, fontsize=8.2, loc="upper right")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.065,
            0.196,
            (
                f"共享模型={gate['metrics']['shared_parameter_count']:,}参数；"
                f"两套独立模型={gate['metrics']['separate_parameter_count']:,}参数；"
                f"macro true={gate['metrics']['macro_true_auc']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.121,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.053,
            "证据范围：PRESENT/GIFT严格四轮profile的共享参数30轮seed0归因；不是零样本、未见密码、高轮攻击或SOTA。",
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
