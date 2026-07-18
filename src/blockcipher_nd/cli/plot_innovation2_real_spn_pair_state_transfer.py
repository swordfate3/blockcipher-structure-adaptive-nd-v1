from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABELS = {
    "present_r7_context": "PRESENT r7\ncontext",
    "skinny_r7_single_cell": "SKINNY r7\nsingle-cell",
    "skinny_r8_adjacent_pair": "SKINNY r8\nadjacent-pair",
    "skinny_r8_bottom_row_pair": "SKINNY r8\nbottom-row",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E42 real-SPN pair-state transfer readiness."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_real_spn_transfer_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_real_spn_transfer_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    label_rows = summary["label_rows"]
    model_rows = summary["model_rows"]
    decisions = {
        "innovation2_real_spn_pair_state_transfer_ready": (
            "真实标签族与64-bit模型均就绪；可冻结本地seed0训练矩阵。"
        ),
        "innovation2_real_spn_pair_state_label_bank_not_ready": (
            "64-bit模型就绪，但真实标签库未过门；禁止神经训练。"
        ),
        "innovation2_real_spn_pair_state_model_not_ready": (
            "真实标签或64-bit模型契约不足；先修模型readiness。"
        ),
        "innovation2_real_spn_pair_state_transfer_protocol_invalid": (
            "来源、标签聚合、64-bit模型或metric协议无效。"
        ),
    }
    processors = ("local", "triangle")
    success_counts = [
        sum(row["success"] for row in model_rows if row["processor_mode"] == mode)
        for mode in processors
    ]
    total_counts = [
        sum(row["processor_mode"] == mode for row in model_rows) for mode in processors
    ]
    target_pass = {
        row["processor_mode"]: row["success"]
        for row in model_rows
        if row["hidden_dim"] == 32 and row["batch_size"] == 4
    }
    max_rss_gib = max(
        row["peak_process_rss_bytes"] for row in model_rows if row["success"]
    ) / (1024**3)

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
            left=0.075, right=0.975, top=0.71, bottom=0.26, wspace=0.27
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E42：真实SPN标签与64-bit pair-state是否可进入训练",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "左：PRESENT/SKINNY真实kernel标签的8项readiness；右：64×64 pair-state前向/反向矩阵。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "本阶段只审计现有机器产物与模型内存，不训练、不连接远程GPU。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(label_rows))
        passed = [row["passed_checks"] for row in label_rows]
        required = [row["required_checks"] for row in label_rows]
        colors = ["#0F766E" if row["train_ready"] else "#D97706" for row in label_rows]
        axes[0].bar(x, passed, width=0.58, color=colors)
        for index, (value, total) in enumerate(zip(passed, required, strict=True)):
            axes[0].text(
                index,
                value + 0.16,
                f"{value}/{total}",
                ha="center",
                va="bottom",
                fontsize=9.0,
            )
        axes[0].axhline(8, color="#DC2626", linestyle="--", linewidth=1.4)
        axes[0].set_xticks(
            x,
            [FAMILY_LABELS.get(row["family_id"], row["family_id"]) for row in label_rows],
        )
        axes[0].set_ylim(0, 9.2)
        axes[0].set_ylabel("通过的readiness检查数")
        axes[0].set_title("真实标签族：全部8项才开放训练", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        px = np.arange(len(processors))
        axes[1].bar(px, success_counts, width=0.52, color=["#2563EB", "#0F766E"])
        for index, (value, total) in enumerate(
            zip(success_counts, total_counts, strict=True)
        ):
            target = "通过" if target_pass.get(processors[index], False) else "失败"
            axes[1].text(
                index,
                value + 0.22,
                f"{value}/{total}\nh32,b4 {target}",
                ha="center",
                va="bottom",
                fontsize=9.0,
            )
        axes[1].axhline(12, color="#DC2626", linestyle="--", linewidth=1.4)
        axes[1].set_xticks(px, ["pair-local", "triangle"])
        axes[1].set_ylim(0, 14.2)
        axes[1].set_ylabel("成功的前向/反向配置数")
        axes[1].set_title("64-bit模型：hidden×batch共12项", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        ready_families = sum(row["train_ready"] for row in label_rows)
        model_success = sum(row["success"] for row in model_rows)
        figure.text(
            0.075,
            0.174,
            f"标签就绪：{ready_families}/{len(label_rows)}族；模型成功：{model_success}/{len(model_rows)}项；进程峰值RSS：{max_rss_gib:.3f} GiB。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.108,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.052,
            "证据范围：现有经验kernel标签库与模型readiness；不是神经性能、真实密码攻击或SOTA。",
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
