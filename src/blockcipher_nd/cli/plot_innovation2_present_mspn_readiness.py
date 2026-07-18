from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROW_LABELS = {
    "mspn_true_seed0": "MSPN 正确P-layer",
    "mspn_corrupted_seed0": "MSPN 错误P-layer",
    "mspn_label_shuffle_seed0": "MSPN 标签打乱",
}
ROW_COLORS = {
    "mspn_true_seed0": "#0F766E",
    "mspn_corrupted_seed0": "#D97706",
    "mspn_label_shuffle_seed0": "#64748B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E46 PRESENT MSPN readiness.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.history.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    render_mspn_readiness(summary, history, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_mspn_readiness(
    summary: dict[str, Any], history: list[dict[str, str]], output: Path
) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    trained = [row for row in rows if row["training_performed"]]
    decisions = {
        "innovation2_present_mspn_readiness_passed": (
            "MSPN实现与短训练readiness通过；下一步建立30轮seed0正式归因计划。"
        ),
        "innovation2_present_mspn_readiness_failed": (
            "MSPN等变、有限性、参数、source或训练控制未通过；先修实现。"
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.29
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E46：Monomial Support Propagation Network实现是否可进入正式训练",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "MSPN按PRESENT S-box真实ANF项共享传播4步；输入仅含active bits、mask和P-layer。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "本图是2轮readiness，不用smoke AUC宣称性能；E44/E45锚点仅用于说明后续正式目标。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        all_auc: list[float] = []
        for row_id in sorted({row["row_id"] for row in history}):
            row_history = [row for row in history if row["row_id"] == row_id]
            epochs = [int(row["epoch"]) for row in row_history]
            aucs = [float(row["validation_auc"]) for row in row_history]
            all_auc.extend(aucs)
            axes[0].plot(
                epochs,
                aucs,
                marker="o",
                linewidth=1.8,
                color=ROW_COLORS[row_id],
                label=ROW_LABELS[row_id],
            )
        axes[0].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[0].set_xticks((1, 2))
        axes[0].set_ylim(
            max(0.3, min([0.48, *all_auc]) - 0.04),
            min(0.75, max([0.55, *all_auc]) + 0.05),
        )
        axes[0].set_xlabel("训练轮次")
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("MSPN短训练流程控制", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="best")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        bar_rows = rows
        x = np.arange(len(bar_rows))
        values = [float(row["validation_auc"]) for row in bar_rows]
        labels = [
            "E45 ANF前缀\nridge"
            if row["row_id"] == "e45_anf_prefix_ridge_anchor"
            else "E44 triangle\n30轮"
            if row["row_id"] == "e44_triangle_anchor"
            else ROW_LABELS[row["row_id"]].replace(" ", "\n", 1)
            for row in bar_rows
        ]
        colors = [
            "#2563EB"
            if row["row_id"] == "e45_anf_prefix_ridge_anchor"
            else "#94A3B8"
            if row["row_id"] == "e44_triangle_anchor"
            else ROW_COLORS[row["row_id"]]
            for row in bar_rows
        ]
        axes[1].bar(x, values, color=colors, width=0.62)
        for index, value in enumerate(values):
            axes[1].text(index, value + 0.016, f"{value:.3f}", ha="center")
        axes[1].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.35, max(0.75, max(values) + 0.07))
        axes[1].set_ylabel("验证 AUC")
        axes[1].set_title("锚点与2轮readiness（不可作性能排名）", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.178,
            f"参数量：{metrics['parameter_count']:,}（E44的{metrics['parameter_ratio_to_e44']:.2f}倍）；cell重标号误差={metrics['cell_relabeling_max_abs_logit_error']:.2e}。",
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
            "证据范围：PRESENT-80四轮MSPN实现与2轮本地readiness；不是有效预测、高轮结论或远程规模结果。",
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
