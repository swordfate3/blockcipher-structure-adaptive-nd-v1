from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E72 round-slice audit.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_round_slice_direction(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_round_slice_direction(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    rounds = ("r1", "r2", "r3")
    ridge = [metrics["ridge_auc"][round_name] for round_name in rounds]
    seed0 = [
        metrics["checkpoints"]["seed0"]["ablation_drop"][round_name]
        for round_name in rounds
    ]
    seed1 = [
        metrics["checkpoints"]["seed1"]["ablation_drop"][round_name]
        for round_name in rounds
    ]
    direction_not_confirmed = (
        "r3稳定主导，但与反向轮序优势矛盾；保留E68并停止轮递归路线。"
        if metrics.get("consensus") and metrics.get("dominant_round") == "r3"
        else "主导轮或效应不一致；保留E68并停止轮递归路线。"
    )
    decisions = {
        "innovation2_present_early_round_skip_candidate_ready": (
            "r1主导在ridge和双seed中一致；可设计同预算early-round skip。"
        ),
        "innovation2_present_round_direction_not_confirmed": direction_not_confirmed,
        "innovation2_present_round_slice_protocol_invalid": (
            "E65/E67/E68重放或切片中和协议无效。"
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
        figure, axes = plt.subplots(1, 2, figsize=(14.8, 8.0))
        figure.subplots_adjust(
            left=0.08, right=0.975, top=0.70, bottom=0.28, wspace=0.32
        )
        figure.text(
            0.08,
            0.955,
            "创新2 E72：E71反向轮序高分由哪个前缀轮驱动",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.08,
            0.892,
            "不训练新网络：比较单轮train-only ridge，并中和E67/E68 checkpoint的r1/r2/r3切片。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.845,
            "切片用50个训练结构的逐output均值替换；AUC下降越大，冻结模型对该轮越依赖。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(3)
        axes[0].bar(x, ridge, color=("#0F766E", "#D97706", "#2563EB"), width=0.62)
        for index, value in enumerate(ridge):
            axes[0].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.0)
        axes[0].set_xticks(x, ("第1轮前缀", "第2轮前缀", "第3轮前缀"))
        axes[0].set_ylim(0.35, 1.04)
        axes[0].set_ylabel("验证AUC")
        axes[0].set_title("单轮确定性ridge", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        width = 0.34
        axes[1].bar(x - width / 2, seed0, width, label="seed0", color="#64748B")
        axes[1].bar(x + width / 2, seed1, width, label="seed1", color="#2563EB")
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        for offset, values in ((-width / 2, seed0), (width / 2, seed1)):
            for index, value in enumerate(values):
                axes[1].text(index + offset, value + 0.006, f"{value:+.3f}", ha="center", fontsize=8.8)
        lower = min(-0.08, min(seed0 + seed1) - 0.04)
        upper = max(0.15, max(seed0 + seed1) + 0.05)
        axes[1].set_ylim(lower, upper)
        axes[1].set_xticks(x, ("中和r1", "中和r2", "中和r3"))
        axes[1].set_ylabel("完整AUC - 中和后AUC")
        axes[1].set_title("双seed checkpoint切片依赖", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, ncol=2)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.08,
            0.183,
            (
                f"ridge排序={' > '.join(metrics['ridge_order'])}；"
                f"seed0排序={' > '.join(metrics['checkpoint_orders']['seed0'])}；"
                f"seed1排序={' > '.join(metrics['checkpoint_orders']['seed1'])}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.08,
            0.113,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.053,
            "证据范围：PRESENT-80四轮、8-bit活动cube的无训练特征归因；不是新网络、高轮、攻击或SOTA。",
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
