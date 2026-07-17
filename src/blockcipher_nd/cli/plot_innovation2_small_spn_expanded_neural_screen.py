from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E38 expanded-topology GraphGPS/CETT screen."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_expanded_neural_screen_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_expanded_neural_screen_svg(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    rows = summary["rows"]
    baseline = gate.get("metrics", {}).get(
        "baseline_auc",
        {
            "unseen_sbox": 0.6881980369450377,
            "unseen_player": 0.6487534140097411,
            "dual_unseen": 0.6843931010265274,
        },
    )
    decisions = {
        "innovation2_small_spn_expanded_neural_screen_readiness_passed": (
            "readiness通过；自动运行冻结的两seed Phase A。"
        ),
        "innovation2_small_spn_expanded_neural_candidate_screened": (
            "至少一个候选过ID门；只对最强候选运行公平错误拓扑归因。"
        ),
        "innovation2_small_spn_expanded_neural_screen_not_ready": (
            "两种候选均未稳定过ID门；停止GraphGPS/CETT机械扩展。"
        ),
        "innovation2_small_spn_expanded_neural_screen_not_attributed": (
            "label-shuffle控制异常；修复前不选择候选。"
        ),
        "innovation2_small_spn_expanded_neural_screen_protocol_invalid": (
            "来源、split、等变性、checkpoint或metric协议无效。"
        ),
    }
    true_rows = [row for row in rows if row["label_mode"] == "true"]
    model_order = ["graphgps", "cett"]
    model_labels = ["Cell-equivariant\nGraphGPS", "Cipher Edge-Token\nTransformer"]
    splits = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双重未见"]
    colors = ["#2563EB", "#D97706", "#0F766E"]
    means = {
        model: [
            float(
                np.mean(
                    [
                        row[f"{split}_auc"]
                        for row in true_rows
                        if row["model_name"] == model
                    ]
                )
            )
            for split in splits
        ]
        for model in model_order
    }
    seed_dual = {
        model: [
            row["dual_unseen_auc"]
            for row in true_rows
            if row["model_name"] == model
        ]
        for model in model_order
    }
    shuffle_rows = [row for row in rows if row["label_mode"] == "shuffled"]
    shuffle_dual = shuffle_rows[0]["dual_unseen_auc"] if shuffle_rows else None

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
            "创新2 E38：12个训练P-layer下，GraphGPS与边Token网络谁能组外泛化",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "任务：根据SPN结构、轮数、积分输入结构和输出mask，预测输出线性积分是否平衡。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "同预算：hidden64、3层、40 epochs、seed0/1；虚线为E37 train-only确定性ID边际。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(model_order))
        width = 0.22
        for split_index, (split, label, color) in enumerate(
            zip(splits, split_labels, colors, strict=True)
        ):
            offset = (split_index - 1) * width
            values = [means[model][split_index] for model in model_order]
            axes[0].bar(x + offset, values, width, color=color, label=label)
            for item_index, value in enumerate(values):
                axes[0].text(
                    item_index + offset,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                )
        axes[0].set_xticks(x, model_labels)
        axes[0].set_ylim(0.45, 1.02)
        axes[0].set_ylabel("AUC")
        axes[0].set_title("两seed平均组外AUC", loc="left", fontweight="bold", pad=42)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=3,
            fontsize=8.7,
        )

        dual_x = np.arange(2)
        seed_width = 0.32
        for seed_index in range(max(len(values) for values in seed_dual.values())):
            values = [
                seed_dual[model][seed_index]
                if seed_index < len(seed_dual[model])
                else np.nan
                for model in model_order
            ]
            offset = (seed_index - 0.5) * seed_width
            axes[1].bar(
                dual_x + offset,
                values,
                seed_width,
                color=["#2563EB", "#D97706"][seed_index],
                label=f"seed{seed_index}",
            )
            for item_index, value in enumerate(values):
                if np.isfinite(value):
                    axes[1].text(
                        item_index + offset,
                        value + 0.012,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8.8,
                    )
        axes[1].axhline(
            baseline["dual_unseen"],
            color="#DC2626",
            linewidth=1.5,
            linestyle="--",
            label="dual ID边际",
        )
        axes[1].axhline(
            baseline["dual_unseen"] + 0.03,
            color="#7C3AED",
            linewidth=1.5,
            linestyle=":",
            label="dual进入控制门",
        )
        axes[1].set_xticks(dual_x, model_labels)
        axes[1].set_ylim(0.45, 1.02)
        axes[1].set_ylabel("双重未见拓扑AUC")
        axes[1].set_title("双重未见的seed稳定性", loc="left", fontweight="bold", pad=42)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            fontsize=8.6,
        )

        shuffle_text = "未运行" if shuffle_dual is None else f"{shuffle_dual:.3f}"
        selected = gate.get("metrics", {}).get("selected_candidate")
        figure.text(
            0.075,
            0.165,
            (
                f"label-shuffle dual AUC：{shuffle_text}；"
                f"Phase B候选：{selected or '无'}。"
            ),
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
