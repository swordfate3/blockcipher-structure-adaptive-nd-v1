from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E65 PRESENT unit-output balance profile readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_unit_balance_profile(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_unit_balance_profile(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    reports = summary["reports"]
    split = metrics["split_metrics"]
    families = ("static_set", "corrupted_topology", "true_topology", "anf_prefix")
    family_labels = ("静态set", "错误P可达", "正确P可达", "ANF 1–3轮前缀")
    auc = [float(reports[name]["validation_auc"]) for name in families]
    decisions = {
        "innovation2_present_unit_balance_profile_topology_ready": (
            "正确拓扑路线过门；下一步测试共享轮次、逐节点输出的profile operator。"
        ),
        "innovation2_present_unit_balance_profile_prefix_ready": (
            "ANF前缀路线过门；下一步测试prefix引导的逐节点profile operator。"
        ),
        "innovation2_present_unit_balance_profile_signal_not_ready": (
            "非平凡信号不足；停止单位输出谱神经结构。"
        ),
        "innovation2_present_unit_balance_profile_shortcut_dominated": (
            "行列边际仍可解释标签；禁止训练profile网络。"
        ),
        "innovation2_present_unit_balance_profile_too_narrow": (
            "单位输出谱宽度不足；禁止训练profile网络。"
        ),
        "innovation2_present_unit_balance_profile_protocol_invalid": (
            "E43重放、谱重排、特征或split协议无效。"
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
            "创新2 E65：一次预测PRESENT四轮的单位输出积分平衡谱是否可行",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "把E43严格unit-mask标签重排为每个活动结构的64维masked profile；不生成新标签、不训练网络。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "checkerboard保证每个被选结构和输出bit正负平衡；validation输出bit全部在train出现。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        labels = ("训练\n正类", "训练\n负类", "验证\n正类", "验证\n负类")
        counts = (
            split["train"]["positive"],
            split["train"]["negative"],
            split["validation"]["positive"],
            split["validation"]["negative"],
        )
        x = np.arange(4)
        axes[0].bar(x, counts, color=["#2563EB", "#D97706", "#0F766E", "#7C3AED"])
        for index, value in enumerate(counts):
            axes[0].text(index, value + 4, str(value), ha="center", fontweight="bold")
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0, max(counts) * 1.22)
        axes[0].set_ylabel("已观察profile坐标数")
        axes[0].set_title(
            f"{metrics['train_structures']}个训练结构 / {metrics['validation_structures']}个验证结构",
            loc="left",
            fontweight="bold",
        )
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        ax = np.arange(len(auc))
        axes[1].bar(ax, auc, color=["#94A3B8", "#D97706", "#2563EB", "#0F766E"])
        for index, value in enumerate(auc):
            axes[1].text(index, value + 0.014, f"{value:.3f}", ha="center", fontweight="bold")
        axes[1].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.1)
        axes[1].set_xticks(ax, family_labels)
        axes[1].set_ylim(0.4, max(0.78, max(auc) + 0.07))
        axes[1].set_ylabel("structure-disjoint验证 AUC")
        axes[1].set_title("同一profile行上的确定性架构路由", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.178,
            (
                f"输出覆盖：train={metrics['train_outputs']}，validation={metrics['validation_outputs']}，"
                f"共享={metrics['shared_outputs']}；边际最强AUC="
                f"{metrics['marginal_baselines']['strongest_auc']:.3f}。"
            ),
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
            "证据范围：PRESENT-80四轮严格unit标签的多输出重排与路由；不是神经性能、高轮结论或新攻击。",
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
