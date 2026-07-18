from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROW_IDS = (
    "e45_anf_prefix_ridge_anchor",
    "cgpr_prefix_only_seed0",
    "cgpr_pair_true_seed0",
    "cgpr_pair_corrupted_seed0",
)
ROW_LABELS = (
    "E45\nridge",
    "prefix-only\n残差",
    "正确P\npair残差",
    "错误P\npair残差",
)
ROW_COLORS = ("#2563EB", "#64748B", "#0F766E", "#D97706")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E51 PRESENT CGPR formal attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_cgpr_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_cgpr_attribution(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    row_by_id = {row["row_id"]: row for row in summary["rows"]}
    rows = [row_by_id[row_id] for row_id in ROW_IDS]
    decisions = {
        "innovation2_present_cgpr_topology_attributed": (
            "正确P残差同时超过ridge、prefix-only和错误P；允许本地seed1。"
        ),
        "innovation2_present_cgpr_candidate_not_ready": (
            "正确P残差未过候选门；停止CGPR和E43四轮新网络枚举。"
        ),
        "innovation2_present_cgpr_pair_residual_not_attributed": (
            "正确P未超过prefix-only；pair-state残差没有独立贡献。"
        ),
        "innovation2_present_cgpr_topology_not_attributed": (
            "正确P未超过错误P；撤回P-layer残差归因。"
        ),
        "innovation2_present_cgpr_attribution_protocol_invalid": (
            "source、ridge、模型、控制或30轮训练协议无效。"
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
            "创新2 E51：证书引导pair-state残差能否通过正式候选与拓扑归因",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "同一E43 structure-disjoint split、30轮seed0；ridge冻结，三行残差模型同预算。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "正式候选必须超过0.70与ridge；pair贡献和正确P贡献分别接受独立控制。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(rows))
        validation = [float(row["validation_auc"]) for row in rows]
        axes[0].bar(x, validation, color=ROW_COLORS, width=0.60)
        for index, value in enumerate(validation):
            axes[0].text(index, value + 0.008, f"{value:.3f}", ha="center")
        axes[0].axhline(
            0.70,
            color="#DC2626",
            linestyle="--",
            linewidth=1.2,
            label="候选门 0.70",
        )
        axes[0].set_xticks(x, ROW_LABELS)
        axes[0].set_ylim(0.35, max(0.80, max(validation) + 0.08))
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("正式validation结果", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        trained = rows[1:]
        tx = np.arange(len(trained))
        width = 0.34
        train_auc = [float(row["train_auc"]) for row in trained]
        validation_auc = [float(row["validation_auc"]) for row in trained]
        axes[1].bar(
            tx - width / 2,
            train_auc,
            width=width,
            color="#94A3B8",
            label="训练 AUC",
        )
        axes[1].bar(
            tx + width / 2,
            validation_auc,
            width=width,
            color=ROW_COLORS[1:],
            label="验证 AUC",
        )
        for index, (train_value, validation_value) in enumerate(
            zip(train_auc, validation_auc, strict=True)
        ):
            axes[1].text(
                index - width / 2,
                train_value + 0.008,
                f"{train_value:.3f}",
                ha="center",
                fontsize=8.4,
            )
            axes[1].text(
                index + width / 2,
                validation_value + 0.008,
                f"{validation_value:.3f}",
                ha="center",
                fontsize=8.4,
            )
        axes[1].set_xticks(tx, ROW_LABELS[1:])
        axes[1].set_ylim(
            0.35, max(0.90, max([*train_auc, *validation_auc]) + 0.08)
        )
        axes[1].set_ylabel("AUC")
        axes[1].set_title("训练/验证泛化差距", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.178,
            f"正确P-ridge={metrics['true_minus_ridge']:+.3f}；正确P-prefix-only={metrics['true_minus_prefix_residual']:+.3f}；正确P-错误P={metrics['true_minus_corrupted']:+.3f}。",
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
            "证据范围：PRESENT-80四轮CGPR本地seed0正式归因；不是高轮积分区分器、新攻击、远程规模结果或SOTA。",
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
