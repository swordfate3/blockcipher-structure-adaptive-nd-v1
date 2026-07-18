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
    parser = argparse.ArgumentParser(description="Render E47 PRESENT MSPN attribution.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.history.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    render_mspn_attribution(summary, history, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_mspn_attribution(
    summary: dict[str, Any], history: list[dict[str, str]], output: Path
) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    decisions = {
        "innovation2_present_mspn_topology_attributed": (
            "MSPN候选与正确P-layer归因全部过门；下一步同矩阵运行seed1。"
        ),
        "innovation2_present_mspn_candidate_not_ready": (
            "MSPN未过候选门；停止增加容量并审计压缩support state的信息损失。"
        ),
        "innovation2_present_mspn_topology_not_attributed": (
            "MSPN候选信号存在，但未归因到正确P-layer transport。"
        ),
        "innovation2_present_mspn_attribution_protocol_invalid": (
            "source、MSPN、控制、metric或训练协议无效。"
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
            "创新2 E47：MSPN能否在严格PRESENT四轮标签上超过pair-state并使用正确拓扑",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "固定E43 checkerboard、MSPN hidden32/degree9、30轮seed0；正确/错误P-layer与shuffle同预算。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "E45 prefix ridge是确定性目标锚点；MSPN不读取prefix特征或最终证书oracle。",
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
                markersize=3.0,
                linewidth=1.7,
                color=ROW_COLORS[row_id],
                label=ROW_LABELS[row_id],
            )
        axes[0].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[0].axhline(0.62, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[0].set_ylim(
            max(0.3, min([0.46, *all_auc]) - 0.035),
            min(0.82, max([0.65, *all_auc]) + 0.045),
        )
        axes[0].set_xlabel("训练轮次")
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("30轮验证曲线（纵轴按实际范围放大）", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="best")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        x = np.arange(len(rows))
        values = [float(row["validation_auc"]) for row in rows]
        labels = [
            "E45 ANF前缀\nridge"
            if row["row_id"] == "e45_anf_prefix_ridge_anchor"
            else "E44 triangle\n30轮"
            if row["row_id"] == "e44_triangle_anchor"
            else ROW_LABELS[row["row_id"]].replace(" ", "\n", 1)
            for row in rows
        ]
        colors = [
            "#2563EB"
            if row["row_id"] == "e45_anf_prefix_ridge_anchor"
            else "#94A3B8"
            if row["row_id"] == "e44_triangle_anchor"
            else ROW_COLORS[row["row_id"]]
            for row in rows
        ]
        axes[1].bar(x, values, color=colors, width=0.62)
        for index, value in enumerate(values):
            axes[1].text(index, value + 0.017, f"{value:.3f}", ha="center")
        axes[1].axhline(0.62, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[1].axhline(0.5, color="#64748B", linestyle=":", linewidth=1.1)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0.35, max(0.78, max(values) + 0.07))
        axes[1].set_ylabel("最佳验证 AUC")
        axes[1].set_title("候选、控制与冻结锚点", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.178,
            f"MSPN真拓扑-E44={metrics['mspn_true_minus_e44']:+.3f}；真拓扑-错误拓扑={metrics['mspn_true_minus_corrupted']:+.3f}；真拓扑-prefix={metrics['mspn_true_minus_prefix']:+.3f}。",
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
            "证据范围：PRESENT-80四轮、本地seed0严格标签上的MSPN候选与拓扑归因；不是高轮或远程规模结果。",
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
