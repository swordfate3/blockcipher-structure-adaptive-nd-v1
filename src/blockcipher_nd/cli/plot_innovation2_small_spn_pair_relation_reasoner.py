from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ID_BASELINE = [0.6881980369450377, 0.6487534140097411, 0.6843931010265274]
GRAPHGPS_MEAN = [0.8970503177432455, 0.6740733504004107, 0.6416819230939694]
CETT_MEAN = [0.902192577657096, 0.6510466302030222, 0.6296566195848206]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E39 small-SPN pair-relation reasoner results."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_pair_relation_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_pair_relation_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    decisions = {
        "innovation2_small_spn_pair_relation_readiness_passed": (
            "pair关系、等变性与训练流程就绪；自动运行两seed筛选。"
        ),
        "innovation2_small_spn_pair_relation_candidate_screened": (
            "SPN-PRR稳定超过ID门；进入公平错误拓扑归因。"
        ),
        "innovation2_small_spn_pair_relation_reasoner_not_ready": (
            "SPN-PRR未稳定超过ID门；停止pair模型机械扩展。"
        ),
        "innovation2_small_spn_pair_relation_not_attributed": (
            "label-shuffle控制异常；暂不解释候选。"
        ),
        "innovation2_small_spn_pair_relation_protocol_invalid": (
            "pair初始化、triangle update、等变性或训练协议无效。"
        ),
    }
    true_rows = [
        row
        for row in rows
        if row["topology_mode"] == "true" and row["label_mode"] == "true"
    ]
    split_keys = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双重未见"]
    colors = ["#2563EB", "#D97706", "#0F766E"]
    pair_mean = [
        float(np.mean([row[f"{split}_auc"] for row in true_rows]))
        for split in split_keys
    ]
    model_values = [ID_BASELINE, GRAPHGPS_MEAN, CETT_MEAN, pair_mean]
    model_labels = ["ID边际", "GraphGPS", "CETT", "SPN-PRR"]
    shuffle_rows = [row for row in rows if row["label_mode"] == "shuffled"]
    shuffle_dual = shuffle_rows[0]["dual_unseen_auc"] if shuffle_rows else None
    parameter_count = true_rows[0]["parameter_count"] if true_rows else None

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.7,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.71, bottom=0.25, wspace=0.25
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E39：有向bit-pair路径组合能否解决新P-layer外推",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "SPN-PRR维护16×16关系状态，并用共享triangle update组合 i→k→j 路径。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "任务与E37数据保持不变：根据密码结构、轮数、积分输入结构和输出mask预测XOR平衡。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(model_values))
        width = 0.22
        for split_index, (label, color) in enumerate(
            zip(split_labels, colors, strict=True)
        ):
            offset = (split_index - 1) * width
            values = [model[split_index] for model in model_values]
            axes[0].bar(x + offset, values, width, color=color, label=label)
            for model_index, value in enumerate(values):
                axes[0].text(
                    model_index + offset,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.1,
                )
        axes[0].set_xticks(x, model_labels)
        axes[0].set_ylim(0.45, 1.02)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("同一benchmark的组外AUC", loc="left", fontweight="bold", pad=42)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=3,
            fontsize=8.7,
        )

        seed_values = [row["dual_unseen_auc"] for row in true_rows]
        seed_x = np.arange(len(seed_values))
        axes[1].bar(
            seed_x,
            seed_values,
            width=0.52,
            color=["#2563EB", "#D97706"][: len(seed_values)],
        )
        for index, value in enumerate(seed_values):
            axes[1].text(
                index,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
            )
        axes[1].axhline(
            ID_BASELINE[2],
            color="#DC2626",
            linewidth=1.5,
            linestyle="--",
            label="dual ID边际",
        )
        axes[1].axhline(
            ID_BASELINE[2] + 0.03,
            color="#7C3AED",
            linewidth=1.5,
            linestyle=":",
            label="进入公平控制门",
        )
        axes[1].set_xticks(seed_x, [f"seed{row['seed']}" for row in true_rows])
        axes[1].set_ylim(0.45, 1.02)
        axes[1].set_ylabel("双重未见拓扑AUC")
        axes[1].set_title("SPN-PRR的seed稳定性", loc="left", fontweight="bold", pad=42)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.7,
        )

        shuffle_text = "未运行" if shuffle_dual is None else f"{shuffle_dual:.3f}"
        parameter_text = "未知" if parameter_count is None else str(parameter_count)
        figure.text(
            0.075,
            0.165,
            f"label-shuffle dual AUC：{shuffle_text}；模型参数：{parameter_text}（GraphGPS锚点297409）。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.105,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.052,
            "证据范围：16-bit合成SPN结构条件积分标签预测；不是实际密码高轮结果、攻击或SOTA。",
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
