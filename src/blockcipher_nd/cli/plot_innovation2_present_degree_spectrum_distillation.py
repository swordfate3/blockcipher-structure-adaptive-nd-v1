from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROW_LABELS = {
    "mspn_spectrum_true_seed0": "正确P + 真谱蒸馏",
    "mspn_spectrum_target_shuffle_seed0": "正确P + target打乱",
    "mspn_spectrum_corrupted_seed0": "错误P + 自洽谱",
}
ROW_COLORS = {
    "mspn_spectrum_true_seed0": "#0F766E",
    "mspn_spectrum_target_shuffle_seed0": "#A855F7",
    "mspn_spectrum_corrupted_seed0": "#D97706",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E49 PRESENT degree-spectrum distillation readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.history.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    render_degree_spectrum_distillation(summary, history, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_degree_spectrum_distillation(
    summary: dict[str, Any], history: list[dict[str, str]], output: Path
) -> None:
    gate = summary["gate"]
    rows = {
        row["row_id"]: row
        for row in summary["rows"]
        if row["training_performed"]
    }
    decisions = {
        "innovation2_present_degree_spectrum_readiness_passed": (
            "中间degree谱可组外学习且balance未退化；允许另建E50正式计划。"
        ),
        "innovation2_present_degree_spectrum_not_learned": (
            "真谱没有明显优于target打乱；停止证书传播神经路线。"
        ),
        "innovation2_present_degree_spectrum_balance_degenerated": (
            "中间谱可学但balance已退化；只允许预注册0.10 loss-scale审计。"
        ),
        "innovation2_present_degree_spectrum_protocol_invalid": (
            "source、teacher、泄漏、模型、控制或训练协议无效。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.4, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.30
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E49：MSPN中间状态能否学会1–3轮degree spectrum",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "同一E43 structure-disjoint split；共享13维辅助头只参与训练loss，不作为balance head输入。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "本图是2轮本地readiness；target打乱排除普通正则化，错误P使用自洽错误transport谱。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        row_ids = tuple(ROW_LABELS)
        x = np.arange(len(row_ids))
        labels = ("正确P\n真谱蒸馏", "正确P\ntarget打乱", "错误P\n自洽谱")
        colors = [ROW_COLORS[row_id] for row_id in row_ids]
        auc_values = [float(rows[row_id]["validation_auc"]) for row_id in row_ids]
        mse_values = [
            float(rows[row_id]["validation_teacher_normalized_mse"])
            for row_id in row_ids
        ]
        axes[0].bar(x, auc_values, color=colors, width=0.60)
        for index, value in enumerate(auc_values):
            axes[0].text(index, value + 0.004, f"{value:.3f}", ha="center")
        axes[0].axhline(
            0.48,
            color="#DC2626",
            linestyle="--",
            linewidth=1.2,
            label="readiness门 0.48",
        )
        axes[0].axhline(
            0.5186727951738006,
            color="#2563EB",
            linestyle=":",
            linewidth=1.2,
            label="E47 label-only 0.519",
        )
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.40, 0.55)
        axes[0].set_ylabel("验证 balance AUC")
        axes[0].set_title(
            "2轮readiness主任务（线为冻结门与旧锚点）",
            loc="left",
            fontweight="bold",
        )
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        axes[1].bar(x, mse_values, color=colors, width=0.60)
        for index, value in enumerate(mse_values):
            axes[1].text(index, value + 0.008, f"{value:.3f}", ha="center")
        axes[1].axhline(
            0.90,
            color="#DC2626",
            linestyle="--",
            linewidth=1.2,
            label="可学门 0.90",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.65, 0.95)
        axes[1].set_ylabel("验证 teacher 标准化 MSE（局部放大，越低越好）")
        axes[1].set_title(
            "中间谱组外可学性（截断纵轴）", loc="left", fontweight="bold"
        )
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.178,
            f"真谱MSE={metrics['distilled_true_validation_normalized_mse']:.3f}；打乱MSE={metrics['target_shuffle_validation_normalized_mse']:.3f}；差值={metrics['true_minus_shuffle_normalized_mse']:+.3f}。",
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
            "证据范围：PRESENT-80四轮中间degree谱蒸馏的2轮本地readiness；不是有效预测、高轮结论或新攻击。",
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
