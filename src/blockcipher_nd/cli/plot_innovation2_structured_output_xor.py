from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MASK_LABELS = (
    "同S盒 0^32",
    "同S盒 2^34",
    "同S盒 8^40",
    "同S盒 10^42",
    "同角色4位 A",
    "同角色4位 B",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OP12 PRESENT r4 structured output-XOR gate."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_structured_output_xor(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_structured_output_xor(
    summary: dict[str, Any], output: Path
) -> None:
    rows = summary["result_rows"]
    gate = summary["gate"]
    structured = [
        row for row in rows if row["model"] == "structured6_mlp_true_xor"
    ]
    geometry = [row for row in rows if row["model"] == "geometry6_mlp_true_xor"]
    shuffled = [
        row for row in rows if row["model"] == "structured6_mlp_label_shuffle"
    ]
    selected = [
        row for row in rows if row["model"] == "selected8_mlp_true_output"
    ]
    x = np.arange(6)
    colors = {
        "structured": "#0F766E",
        "geometry": "#2563EB",
        "shuffle": "#B91C1C",
        "derived": "#7C3AED",
        "component": "#B45309",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(15.8, 10.4))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.80,
            bottom=0.14,
            hspace=0.47,
            wspace=0.25,
        )
        figure.text(
            0.075,
            0.965,
            "创新2 OP12：PRESENT四轮多输出bit结构化XOR预测",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.918,
            "输入是未见明文；标签是同一明文真实四轮密文中预注册多个位置的XOR值，不是样本分类。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.075,
            0.882,
            "直接结构化XOR必须同时超过同重量几何控制、匹配标签打乱、单bit派生parity和最佳组成bit。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        mode = summary["metadata"]["mode"]
        figure.text(
            0.075,
            0.846,
            (
                "当前为64条训练/64条测试的本地实现门，数值不作性能结论。"
                if mode == "smoke"
                else "当前为2^17条训练、2^16条测试、第二固定密钥的四轮正式结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )

        auc_axis = axes[0, 0]
        width = 0.25
        auc_axis.bar(
            x - width,
            [row["auc"] for row in structured],
            width,
            color=colors["structured"],
            label="直接结构化XOR",
        )
        auc_axis.bar(
            x,
            [row["auc"] for row in geometry],
            width,
            color=colors["geometry"],
            label="同重量几何控制",
        )
        auc_axis.bar(
            x + width,
            [row["auc"] for row in shuffled],
            width,
            color=colors["shuffle"],
            label="结构化XOR标签打乱",
        )
        auc_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.51,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.1,
            label="AUC门0.510",
        )
        auc_values = [
            float(row["auc"]) for row in structured + geometry + shuffled
        ]
        auc_axis.set_ylim(
            max(0.0, min(auc_values + [0.5]) - 0.02),
            min(1.0, max(auc_values + [0.51]) + 0.07),
        )
        auc_axis.set_xticks(x, MASK_LABELS, rotation=12, ha="right")
        auc_axis.set_ylabel("AUC")
        auc_axis.set_title(
            "六个预注册XOR目标及控制的AUC", loc="left", fontweight="bold"
        )
        auc_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        auc_axis.legend(frameon=False, ncol=2, loc="upper left")

        margin_axis = axes[0, 1]
        margin_specs = (
            ("auc_minus_geometry", "直接 - 几何控制", colors["geometry"]),
            ("auc_minus_shuffle", "直接 - 标签打乱", colors["shuffle"]),
            ("auc_minus_derived", "直接 - 单bit派生", colors["derived"]),
            (
                "auc_minus_best_component",
                "直接 - 最佳组成bit",
                colors["component"],
            ),
        )
        for field, label, color in margin_specs:
            margin_axis.plot(
                x,
                [row[field] for row in structured],
                marker="o",
                linewidth=1.5,
                label=label,
                color=color,
            )
        margin_axis.axhline(0.0, color="#475569", linestyle="--", linewidth=1.0)
        margin_axis.axhline(
            0.005,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.1,
            label="主要差值门+0.005",
        )
        margin_values = [
            float(row[field]) for field, _, _ in margin_specs for row in structured
        ]
        margin_axis.set_ylim(
            min(margin_values + [0.0]) - 0.025,
            max(margin_values + [0.005]) + 0.09,
        )
        margin_axis.set_xticks(x, MASK_LABELS, rotation=12, ha="right")
        margin_axis.set_ylabel("AUC差值")
        margin_axis.set_title(
            "直接XOR相对四类基线的增益", loc="left", fontweight="bold"
        )
        margin_axis.grid(color="#E5E7EB", linewidth=0.7)
        margin_axis.legend(frameon=False, ncol=2, loc="upper left")

        bit_axis = axes[1, 0]
        bit_x = np.arange(len(selected))
        bit_axis.bar(
            bit_x,
            [row["auc"] for row in selected],
            width=0.62,
            color="#2563EB",
        )
        bit_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        bit_axis.axhline(0.51, color="#B91C1C", linestyle=":", linewidth=1.1)
        bit_auc = [float(row["auc"]) for row in selected]
        bit_axis.set_ylim(
            max(0.0, min(bit_auc + [0.5]) - 0.02),
            min(1.0, max(bit_auc + [0.51]) + 0.035),
        )
        bit_axis.set_xticks(
            bit_x, [f"bit {row['msb_index']}" for row in selected]
        )
        bit_axis.set_ylabel("AUC")
        bit_axis.set_title(
            "四轮八个单bit专用头anchor", loc="left", fontweight="bold"
        )
        bit_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

        family_axis = axes[1, 1]
        family_names = ("同S盒双bit", "同角色四bit")
        family_rows = (structured[:4], structured[4:])
        family_x = np.arange(2)
        family_width = 0.2
        summaries = (
            ("直接结构化", "auc", colors["structured"]),
            ("几何控制", "geometry_auc", colors["geometry"]),
            ("单bit派生", "derived_auc", colors["derived"]),
            ("最佳组成bit", "best_component_auc", colors["component"]),
        )
        for offset, (label, field, color) in zip(
            (-1.5, -0.5, 0.5, 1.5), summaries, strict=True
        ):
            family_axis.bar(
                family_x + offset * family_width,
                [float(np.mean([row[field] for row in group])) for group in family_rows],
                family_width,
                color=color,
                label=label,
            )
        family_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        family_values = [
            float(np.mean([row[field] for row in group]))
            for _, field, _ in summaries
            for group in family_rows
        ]
        family_axis.set_ylim(
            max(0.0, min(family_values + [0.5]) - 0.04),
            min(1.0, max(family_values + [0.5]) + 0.15),
        )
        family_axis.set_xticks(family_x, family_names)
        family_axis.set_ylabel("家族平均AUC")
        family_axis.set_title(
            "结构家族与强基线的平均表现", loc="left", fontweight="bold"
        )
        family_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        family_axis.legend(frameon=False, ncol=2, loc="upper left")

        decision_text = {
            "innovation2_r4_structured_xor_local_smoke_passed": (
                "四轮结构化XOR实现门通过"
            ),
            "innovation2_r4_structured_xor_supported": (
                "至少一个结构家族通过，进入另一固定密钥确认"
            ),
            "innovation2_r4_structured_xor_not_supported": (
                "四轮结构化XOR未过门，停止机械扩展"
            ),
            "innovation2_r4_structured_xor_protocol_invalid": "实验协议无效",
        }.get(gate["decision"], gate["decision"])
        metrics = gate["metrics"]
        figure.text(
            0.075,
            0.067,
            (
                f"裁决：{decision_text}；通过mask={metrics['passed_mask_count']}/6；"
                f"双bit家族={metrics['pair_family_pass_count']}/4，"
                f"四bit家族={metrics['role4_family_pass_count']}/2。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.075,
            0.027,
            "证据边界：单固定密钥PRESENT四轮真实输出XOR预测；不是样本分类、积分平衡、跨密钥确认、五轮结果或SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
