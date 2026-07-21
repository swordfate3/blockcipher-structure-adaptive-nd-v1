from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "selected8_present_spn_anchor_exact_p_true_output",
    "selected8_topology_bottleneck_exact_p_true_output",
    "selected8_topology_bottleneck_wrong_p_true_output",
    "selected8_topology_bottleneck_exact_p_label_shuffle",
)
MODEL_LABELS = (
    "原SPN锚点\n真实P",
    "瓶颈候选\n真实P",
    "瓶颈控制\n错误P",
    "瓶颈控制\n标签打乱",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPB1 topology-bottleneck output prediction."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_topology_bottleneck(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_topology_bottleneck(summary: dict[str, Any], output: Path) -> None:
    bits = [int(bit) for bit in summary["metadata"]["selected_msb_indices"]]
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in summary["bit_rows"]
    }
    auc = np.asarray(
        [
            [float(indexed[(model, bit)]["auc"]) for bit in bits]
            for model in MODEL_ORDER
        ]
    )
    means = auc.mean(axis=1)
    candidate_minus_anchor = auc[1] - auc[0]
    candidate_minus_wrong = auc[1] - auc[2]
    candidate_minus_shuffle = auc[1] - auc[3]
    gate = summary["gate"]
    mode = str(summary["metadata"]["mode"])

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
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.5))
        figure.subplots_adjust(
            left=0.11,
            right=0.965,
            top=0.785,
            bottom=0.145,
            hspace=0.50,
            wspace=0.30,
        )
        figure.text(
            0.11,
            0.958,
            "创新2 OPB1：PRESENT三轮拓扑瓶颈真实输出预测",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.11,
            0.912,
            "候选删除每位置完整189维向量，只保留每轮64个标量条件乘共享方向；输出仍是八个真实密文bit。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.11,
            0.873,
            "四行共享数据和训练预算：原SPN锚点、瓶颈真实P、瓶颈错误P、瓶颈标签打乱。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.11,
            0.834,
            (
                "当前为64条训练/64条测试、1 epoch本地实现门；图中数值不作性能结论。"
                if mode == "smoke"
                else "当前为131072条训练、65536条测试、100 epochs的第五固定密钥正式结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )

        heat_axis = axes[0, 0]
        max_deviation = max(0.02, float(np.max(np.abs(auc - 0.5))))
        image = heat_axis.imshow(
            auc,
            cmap="RdYlBu",
            vmin=0.5 - max_deviation,
            vmax=0.5 + max_deviation,
            aspect="auto",
        )
        heat_axis.set_xticks(np.arange(len(bits)), [f"bit {bit}" for bit in bits])
        heat_axis.set_yticks(np.arange(4), MODEL_LABELS)
        heat_axis.set_title("逐bit AUC", loc="left", fontweight="bold")
        for row_index in range(auc.shape[0]):
            for column_index in range(auc.shape[1]):
                value = auc[row_index, column_index]
                heat_axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=(
                        "#FFFFFF"
                        if abs(float(value) - 0.5) / max_deviation >= 0.72
                        else "#111827"
                    ),
                )
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.035)
        colorbar.set_label("AUC")

        mean_axis = axes[0, 1]
        colors = ("#475569", "#0F766E", "#B45309", "#7C3AED")
        mean_axis.bar(np.arange(4), means, width=0.62, color=colors)
        mean_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        mean_axis.set_xticks(np.arange(4), MODEL_LABELS)
        mean_axis.set_ylabel("八位置平均AUC")
        mean_axis.set_title("候选、锚点与必要控制", loc="left", fontweight="bold")
        low = max(0.0, float(min(means.min(), 0.5) - 0.025))
        high = min(1.0, float(max(means.max(), 0.5) + 0.05))
        if high - low < 0.08:
            low = max(0.0, low - 0.04)
            high = min(1.0, high + 0.04)
        mean_axis.set_ylim(low, high)
        mean_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        span = high - low
        for index, value in enumerate(means):
            inside = value + span * 0.055 >= high
            mean_axis.text(
                index,
                value + span * (-0.04 if inside else 0.04),
                f"{value:.4f}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=8.6,
                fontweight="bold",
                color="#FFFFFF" if inside else "#111827",
            )

        control_axis = axes[1, 0]
        positions = np.arange(len(bits))
        width = 0.36
        control_axis.bar(
            positions - width / 2,
            candidate_minus_wrong,
            width=width,
            color="#2563EB",
            label="候选真实P - 候选错误P",
        )
        control_axis.bar(
            positions + width / 2,
            candidate_minus_shuffle,
            width=width,
            color="#7C3AED",
            label="候选真实P - 标签打乱",
        )
        control_axis.axhline(0.0, color="#475569", linewidth=1.0)
        control_axis.axhline(0.02, color="#B45309", linestyle=":", linewidth=1.2)
        control_axis.set_xticks(positions, [f"bit {bit}" for bit in bits])
        control_axis.set_ylabel("AUC差值")
        control_axis.set_title("拓扑与标签归因", loc="left", fontweight="bold")
        control_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        control_axis.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.3)
        control_span = max(
            0.05,
            float(
                max(
                    np.max(np.abs(candidate_minus_wrong)),
                    np.max(np.abs(candidate_minus_shuffle)),
                )
                + 0.02
            ),
        )
        control_axis.set_ylim(-control_span, control_span)

        anchor_axis = axes[1, 1]
        anchor_axis.bar(positions, candidate_minus_anchor, color="#0F766E")
        anchor_axis.axhline(0.0, color="#475569", linewidth=1.0)
        anchor_axis.axhline(-0.10, color="#B45309", linestyle=":", linewidth=1.2)
        anchor_axis.set_xticks(positions, [f"bit {bit}" for bit in bits])
        anchor_axis.set_ylabel("候选AUC - 原SPN锚点AUC")
        anchor_axis.set_title("结构瓶颈的性能代价", loc="left", fontweight="bold")
        anchor_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        anchor_span = max(0.12, float(np.max(np.abs(candidate_minus_anchor)) + 0.03))
        anchor_axis.set_ylim(-anchor_span, anchor_span)

        decision_text = {
            "innovation2_topology_bottleneck_local_smoke_passed": "拓扑瓶颈四行实现门通过",
            "innovation2_topology_bottleneck_ready_for_independent_confirmation": "候选兼顾拓扑归因与输出预测能力",
            "innovation2_topology_bottleneck_attributed_with_performance_cost": "拓扑可归因但候选存在性能代价",
            "innovation2_topology_bottleneck_not_attributed": "候选未通过拓扑归因门",
            "innovation2_topology_bottleneck_protocol_invalid": "实验协议或产物无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.11,
            0.072,
            (
                f"裁决：{decision_text}；候选减错误P平均AUC="
                f"{gate['metrics']['candidate_minus_wrong_mean_auc']:+.4f}，"
                f"候选减标签打乱={gate['metrics']['candidate_minus_shuffle_mean_auc']:+.4f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.11,
            0.030,
            "证据边界：PRESENT三轮、第五固定密钥、八个预注册真实输出bit；不是四轮、完整密文恢复或样本分类证据。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
