from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E85 PRESENT/GIFT shared profile readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_shared_profile_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_shared_profile_readiness(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = {row["relation_mode"]: row for row in gate["metrics"]["rows"]}
    modes = ("independent", "corrupted", "true")
    mode_names = ("独立关系", "错误拓扑", "真实拓扑")
    colors = ("#94A3B8", "#2563EB", "#0F766E")
    aucs = {
        cipher: [rows[mode][f"{cipher}_validation_auc"] for mode in modes]
        for cipher in ("present", "gift")
    }
    anchors = {"present": 0.8341666666666666, "gift": 0.7606659729448492}
    margins = [
        gate["metrics"]["present_true_minus_independent"],
        gate["metrics"]["present_true_minus_corrupted"],
        gate["metrics"]["gift_true_minus_independent"],
        gate["metrics"]["gift_true_minus_corrupted"],
    ]
    margin_names = ("PRESENT\n真实-独立", "PRESENT\n真实-错误", "GIFT\n真实-独立", "GIFT\n真实-错误")
    decisions = {
        "innovation2_shared_profile_operator_readiness_passed": (
            "单一共享算子在两个密码上均超过独立与错误拓扑；可进入E86固定30轮seed0。"
        ),
        "innovation2_shared_profile_operator_readiness_not_passed": (
            "共享参数未同时保留两密码质量与拓扑增益；保留E73/E79两套独立模型。"
        ),
        "innovation2_shared_profile_operator_protocol_invalid": (
            "来源、运行时拓扑、共享参数、公平预算或等变协议无效。"
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
            "创新2 E85：一套共享神经算子能否同时处理PRESENT与GIFT拓扑",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "三行均只有4,795参数、训练两轮；同一模型按运行时P-layer切换密码，不使用cipher ID或专属head。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "每轮完整消费PRESENT 7个batch与GIFT 14个batch，并按密码归一loss；只改变真实、错误或独立关系。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        for axis, cipher, title in zip(
            axes[:2],
            ("present", "gift"),
            ("PRESENT-80四轮验证", "GIFT-64四轮验证"),
            strict=True,
        ):
            x = np.arange(3)
            axis.bar(x, aucs[cipher], color=colors)
            for index, value in enumerate(aucs[cipher]):
                axis.text(index, value + 0.018, f"{value:.3f}", ha="center")
            axis.axhline(
                anchors[cipher],
                color="#D97706",
                linestyle="--",
                linewidth=1.2,
                label=f"独立readiness锚点 {anchors[cipher]:.3f}",
            )
            axis.set_xticks(x, mode_names, rotation=8)
            axis.set_ylim(0.45, 1.0)
            axis.set_ylabel("validation AUC")
            axis.set_title(title, loc="left", fontweight="bold")
            axis.legend(frameon=False, fontsize=8.2, loc="upper left")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)

        x = np.arange(4)
        margin_colors = ["#0F766E" if value >= 0.03 else "#DC2626" for value in margins]
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
        axes[2].set_ylim(min(-0.08, min(margins) - 0.04), max(0.35, max(margins) + 0.06))
        axes[2].set_ylabel("真实拓扑 AUC差值")
        axes[2].set_title("逐密码拓扑归因门", loc="left", fontweight="bold")
        axes[2].legend(frameon=False, fontsize=8.2, loc="upper right")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        contract = gate["metrics"]["contract"]
        figure.text(
            0.065,
            0.196,
            (
                f"共享参数={next(iter(contract['parameter_counts'].values())):,}；"
                f"两密码macro true AUC={gate['metrics']['macro_true_auc']:.3f}；"
                "P-layer不在state_dict中，无cipher ID/adapter/专属head。"
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
            "证据范围：两套严格四轮unit-balance profile的两轮本地共享参数readiness；不是零样本迁移、高轮攻击或SOTA。",
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
