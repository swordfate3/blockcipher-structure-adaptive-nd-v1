from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E63 DeepSets/RCCA results.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_small_spn_rcca(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_small_spn_rcca(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    contract = summary["contract"]
    mode = summary["metadata"]["mode"]
    labels = [_display_name(row) for row in rows]
    colors = [
        "#0F766E" if row["model_name"] == "deepsets" else
        "#7C3AED" if row["label_mode"] == "shuffled" else
        "#D97706" if row["topology_mode"] == "corrupted" else "#2563EB"
        for row in rows
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.69, bottom=0.30, wspace=0.43
        )
        title_suffix = "训练readiness" if mode == "smoke" else "正式双seed筛选"
        figure.text(
            0.07,
            0.955,
            f"创新2 E63：小型SPN DeepSets与RCCA {title_suffix}",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "标签来自E62全256主密钥严格relation；checkpoint只看训练拓扑的validation relation。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "RCCA只增加coordinate bit-query与cipher graph node的对齐cross-attention。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        dual = [row["dual_unseen_auc"] for row in rows]
        x = np.arange(len(rows))
        axes[0].bar(x, dual, color=colors, width=0.62)
        axes[0].axhline(
            gate["thresholds"]["dual_marginal_anchor"],
            color="#DC2626",
            linestyle="--",
            linewidth=1.5,
            label="E62最强边际",
        )
        axes[0].set_xticks(x, labels, rotation=28, ha="right")
        axes[0].set_ylim(0.42, 1.0)
        axes[0].set_ylabel("双重未见拓扑 AUC")
        axes[0].set_title("候选与必要控制", loc="left", fontweight="bold")
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(dual):
            axes[0].text(index, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")

        split_values = np.asarray(
            [
                [
                    row["unseen_sbox_auc"],
                    row["unseen_player_auc"],
                    row["dual_unseen_auc"],
                ]
                for row in rows
            ]
        )
        image = axes[1].imshow(split_values, vmin=0.45, vmax=1.0, cmap="viridis", aspect="auto")
        axes[1].set_xticks(np.arange(3), ["未见S", "未见P", "双重未见"])
        axes[1].set_yticks(np.arange(len(rows)), labels)
        axes[1].set_title("组外泛化全景", loc="left", fontweight="bold")
        for row_index in range(len(rows)):
            for column_index in range(3):
                value = split_values[row_index, column_index]
                axes[1].text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="#FFFFFF" if value < 0.78 else "#0F172A",
                    fontweight="bold",
                )
        figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="AUC")

        parameter_names = ["DeepSets", "RCCA"]
        parameter_values = [
            contract["parameter_counts"]["deepsets"],
            contract["parameter_counts"]["rcca"],
        ]
        axes[2].bar(
            np.arange(2), parameter_values, color=["#0F766E", "#2563EB"], width=0.58
        )
        axes[2].set_xticks(np.arange(2), parameter_names)
        axes[2].set_ylabel("可训练参数")
        axes[2].set_title(
            f"同量级参数（比值{contract['parameter_ratio']:.3f}）",
            loc="left",
            fontweight="bold",
        )
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(parameter_values):
            axes[2].text(index, value + 1800, f"{value:,}", ha="center", fontweight="bold")

        if mode == "smoke":
            verdict = "readiness通过；当前8-epoch AUC只验证流程，进入冻结40-epoch双seed正式矩阵。"
        elif gate["status"] == "pass":
            verdict = "正式筛选通过；进入paired wrong-P双seed拓扑归因。"
        else:
            verdict = "正式筛选未过门；关闭RCCA，不增加模型或训练预算。"
        figure.text(
            0.07,
            0.19,
            f"裁决：{verdict}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.125,
            f"协议：hidden{summary['metadata']['hidden_dim']} / {summary['metadata']['epochs']} epochs / "
            f"{summary['metadata']['fit_relations']} fit + {summary['metadata']['validation_relations']} validation relations。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.075,
            f"不变量：token交换 {contract['relation_swap_max_logit_error']:.2e} / "
            f"cell重标号 {contract['cell_relabel_max_logit_error']:.2e} / "
            f"true-wrong-P {contract['true_corrupted_max_logit_delta']:.2e}。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.025,
            "证据范围：16-bit合成SPN全密钥relation预测；不是PRESENT/GIFT结果、攻击或SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _display_name(row: dict[str, Any]) -> str:
    if row["label_mode"] == "shuffled":
        return f"RCCA 标签打乱 s{row['seed']}"
    if row["model_name"] == "deepsets":
        return f"DeepSets s{row['seed']}"
    if row["topology_mode"] == "corrupted":
        return f"RCCA wrong-P s{row['seed']}"
    return f"RCCA true-P s{row['seed']}"


if __name__ == "__main__":
    raise SystemExit(main())
