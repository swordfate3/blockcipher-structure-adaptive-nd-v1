from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E56 generalized-integral relation contract audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_generalized_relation_contract(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_generalized_relation_contract(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = summary["metrics"]
    key_model = summary["source_contract"]["key_model"]
    generalized = gate["generalized_relation_checks"]
    original = gate["original_target_checks"]
    decisions = {
        "innovation2_generalized_relation_label_contract_not_ready": (
            "广义relation正类存在，但key model、严格负类和可用拆分未过门；神经训练保持关闭。"
        ),
        "innovation2_generalized_relation_extension_ready": (
            "广义relation扩展标签就绪，但不能写成原linear-mask balance任务。"
        ),
        "innovation2_generalized_relation_original_target_ready": (
            "广义relation与原目标映射均通过，可先构建标签atlas。"
        ),
        "innovation2_generalized_relation_contract_protocol_invalid": (
            "来源版本、安全解析或源码契约无效。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.1,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.2, 8.5))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.70, bottom=0.27, wspace=0.45
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E56：公开PRESENT九轮广义积分relation能否成为可信神经标签",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "必须区分：relation的和与key无关、常数是否等于0、以及真实PRESENT-80 linear-mask是否XOR平衡。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "源码key model：每轮使用独立64-bit局部轮密钥；没有PRESENT-80的80-bit主密钥调度。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        widths = [
            int(metrics["deduplicated_relations"]),
            int(metrics["relations_common_to_all_files"]),
            int(metrics["relations_unique_to_one_file"]),
            int(metrics["proven_key_dependent_negative_relations"]),
        ]
        width_labels = ["去重正relation", "8文件共同relation", "单文件独有relation", "严格负relation"]
        width_colors = ["#0F766E", "#D97706", "#64748B", "#DC2626"]
        positions = np.arange(len(widths))
        axes[0].bar(positions, widths, color=width_colors, width=0.64)
        axes[0].axhline(
            int(metrics["minimum_negative_relations"]),
            color="#DC2626",
            linestyle="--",
            linewidth=1.3,
            label="负类门 256",
        )
        for index, value in enumerate(widths):
            axes[0].text(index, value + 8, str(value), ha="center", va="bottom", fontweight="bold")
        axes[0].set_xticks(positions, width_labels, rotation=18, ha="right")
        axes[0].set_ylim(0, max(max(widths), 256) * 1.22)
        axes[0].set_ylabel("去重relation数量")
        axes[0].set_title("正类够宽，但严格负类为0", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.3)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        file_rows = summary["relation_overlap"]["files"]
        file_positions = np.arange(len(file_rows))
        totals = [int(row["relations"]) for row in file_rows]
        shared = [int(row["common_to_all_files"]) for row in file_rows]
        axes[1].bar(file_positions, totals, color="#94A3B8", label="文件basis总数")
        axes[1].bar(file_positions, shared, color="#D97706", label="8文件共同relation")
        axes[1].set_xticks(file_positions, [f"F{index}" for index in range(1, 9)])
        axes[1].set_ylabel("relation数量")
        axes[1].set_title("按公开文件拆分会泄漏共同relation", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper left", fontsize=8.3)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        check_labels = [
            "正类membership\n定义明确",
            "正类宽度\n>=256",
            "严格负类\n>=256",
            "负类witness\n可复验",
            "文件拆分\nrelation互斥",
            "真实PRESENT-80\nkey schedule",
            "常数0/1\n已知",
            "原目标映射\n完整",
        ]
        check_values = [
            generalized["positive_membership_semantics_defined"],
            generalized["deduplicated_positive_width_at_least_256"],
            generalized["proven_negative_width_at_least_256"],
            generalized["negative_witnesses_replayable"],
            generalized["public_file_split_relation_disjoint"],
            original["actual_present80_master_key_schedule"],
            original["constant_zero_or_one_known"],
            original["original_linear_mask_balance_mapping_complete"],
        ]
        check_positions = np.arange(len(check_values))
        axes[2].barh(
            check_positions,
            [int(value) for value in check_values],
            color=["#0F766E" if value else "#DC2626" for value in check_values],
            height=0.66,
        )
        axes[2].set_xlim(0, 1.15)
        axes[2].set_xticks([0, 1], ["失败", "通过"])
        axes[2].set_yticks(check_positions, check_labels)
        axes[2].invert_yaxis()
        axes[2].set_title("标签契约逐项门", loc="left", fontweight="bold")
        axes[2].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.07,
            0.165,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.105,
            "下一步：不使用random unknown补负类；先获得真实PRESENT-80 key schedule下可复验的key-dependent negatives。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：公开ATM commit的九轮relation标签契约；不是PRESENT-80平衡标签、神经结果、攻击或SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
