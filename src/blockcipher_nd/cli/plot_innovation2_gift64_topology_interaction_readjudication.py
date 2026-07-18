from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E77 GIFT topology-interaction readjudication."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_topology_interaction(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_topology_interaction(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    ridges = gate["metrics"]["ridges"]
    counterfactuals = gate["metrics"]["checkpoint_counterfactuals"]
    variants = (
        "local",
        "corrupted_shift1",
        "corrupted_shift2",
        "corrupted_shift3",
        "true",
    )
    labels = ("本节点", "错误P\nshift1", "错误P\nshift2", "错误P\nshift3", "真实P")
    ridge_auc = [ridges[name]["validation_auc"] for name in variants]
    inference_variants = variants[1:]
    inference_labels = labels[1:]
    inference_auc = [counterfactuals[name]["auc"] for name in inference_variants]
    decisions = {
        "innovation2_gift64_topology_interaction_gate_repaired": (
            "拓扑展开ridge与同权重反事实均归因到真实GIFT P；可另立正式训练计划。"
        ),
        "innovation2_gift64_topology_interaction_not_confirmed": (
            "确定性或同权重拓扑归因未确认；关闭GIFT r3-only。"
        ),
        "innovation2_gift64_topology_interaction_protocol_invalid": (
            "E75/E76重放、拓扑变体、ridge或checkpoint协议无效。"
        ),
    }
    colors = ("#94A3B8", "#D97706", "#7C3AED", "#2563EB", "#0F766E")
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.1))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.28, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E77：GIFT-64拓扑交互信号能否通过公平基线与同权重反事实",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "左图给ridge同样的本节点、同cell和P前驱信息；右图冻结E76参数，只替换P-layer。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "三个错误P保持64位置换与4-bit cell/lane结构；本实验不训练新模型、不修改E76 gate。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(variants))
        axes[0].bar(x, ridge_auc, color=colors, width=0.70)
        for index, value in enumerate(ridge_auc):
            axes[0].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.35, 1.02)
        axes[0].set_ylabel("structure-disjoint验证 AUC")
        axes[0].set_title("信息范围对齐的确定性ridge", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        ix = np.arange(len(inference_variants))
        axes[1].bar(ix, inference_auc, color=colors[1:], width=0.68)
        for index, value in enumerate(inference_auc):
            axes[1].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[1].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[1].set_xticks(ix, inference_labels)
        axes[1].set_ylim(0.35, 1.02)
        axes[1].set_ylabel("冻结checkpoint验证 AUC")
        axes[1].set_title("同一套可学习参数，只替换P-layer", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.183,
            (
                f"真实P ridge相对本节点={gate['metrics']['true_minus_local_ridge']:+.3f}，"
                f"相对最强错误P={gate['metrics']['true_minus_max_corrupted_ridge']:+.3f}；"
                f"同权重反事实margin={gate['metrics']['true_minus_max_corrupted_inference']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：GIFT-64四轮无新训练拓扑归因；不是正式神经收益、高轮、跨密码、攻击或SOTA。",
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
