from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CHECK_LABELS = {
    "candidate_auc_at_least_0_520": "候选AUC达到0.520",
    "candidate_auc_at_least_0_510": "候选AUC达到0.510",
    "accuracy_margin_at_least_0_005": "准确率超过多数基线0.005",
    "candidate_minus_shuffle_at_least_0_010": "候选比标签打乱高0.010",
    "candidate_minus_shuffle_at_least_0_005": "候选比标签打乱高0.005",
    "candidate_minus_geometry_at_least_0_005": "候选比右移控制高0.005",
    "candidate_minus_derived_at_least_0_005": "候选比单bit派生高0.005",
    "candidate_minus_best_component_at_least_0_002": "候选比最佳组成bit高0.002",
    "candidate_minus_mlp_at_least_0_002": "候选比同目标MLP高0.002",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OPE1 selected-eight output parity prediction."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_selected8_parity(
        json.loads(args.summary.read_text(encoding="utf-8")),
        args.output,
    )
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_selected8_parity(summary: dict[str, Any], output: Path) -> None:
    rows = summary["result_rows"]
    gate = summary["gate"]
    metadata = summary["metadata"]
    candidate = next(
        row
        for row in rows
        if row["model"] == "selected8_parity_rescnn_true_output"
    )
    selected = [
        row for row in rows if row["model"] == "selected8_mlp_true_output"
    ]
    auc_labels = (
        "直接ResCNN",
        "同目标MLP",
        "右移mask控制",
        "标签打乱",
        "单bit派生",
        "最佳组成bit",
    )
    auc_values = np.asarray(
        [
            candidate["auc"],
            candidate["mlp_auc"],
            candidate["geometry_auc"],
            candidate["shuffle_auc"],
            candidate["derived_auc"],
            candidate["best_component_auc"],
        ],
        dtype=np.float64,
    )
    delta_labels = ("-MLP", "-右移控制", "-标签打乱", "-派生", "-最佳bit")
    delta_values = np.asarray(
        [
            candidate["auc_minus_mlp"],
            candidate["auc_minus_geometry"],
            candidate["auc_minus_shuffle"],
            candidate["auc_minus_derived"],
            candidate["auc_minus_best_component"],
        ],
        dtype=np.float64,
    )
    rounds = int(metadata["config"]["rounds"])
    mode = str(metadata["mode"])
    colors = ("#0F766E", "#2563EB", "#0891B2", "#B91C1C", "#7C3AED", "#B45309")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(15.8, 9.7))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.76,
            bottom=0.15,
            hspace=0.47,
            wspace=0.24,
        )
        figure.text(
            0.075,
            0.955,
            f"创新2 OPE1：PRESENT{rounds}轮八个密文bit全异或为一个bit的输出预测",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.905,
            "标签 z=C[0]⊕C[2]⊕C[8]⊕C[10]⊕C[32]⊕C[34]⊕C[40]⊕C[42]；z是同一密文的真实输出值，不是样本分类。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.075,
            0.860,
            "直接ResCNN必须超过同目标MLP、右移一位同重量mask、匹配标签打乱、单bit派生parity和最佳组成bit。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.075,
            0.815,
            (
                "当前为64条训练/64条测试、1 epoch本地实现门；数值不作性能结论。"
                if mode == "smoke"
                else "当前为4096条训练/4096条测试、10 epochs本地可行性诊断；不是正式规模结论。"
            ),
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )

        auc_axis = axes[0, 0]
        auc_x = np.arange(len(auc_labels))
        auc_axis.bar(auc_x, auc_values, color=colors, width=0.65)
        auc_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        auc_axis.axhline(
            0.52 if rounds == 3 else 0.51,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.1,
        )
        auc_span = max(0.025, float(np.max(np.abs(auc_values - 0.5))) * 1.35)
        auc_axis.set_ylim(max(0.0, 0.5 - auc_span), min(1.0, 0.5 + auc_span))
        auc_axis.set_xticks(auc_x, auc_labels, rotation=14, ha="right")
        auc_axis.set_ylabel("AUC")
        auc_axis.set_title("同一目标与必要控制", loc="left", fontweight="bold")
        auc_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        for x, value in zip(auc_x, auc_values, strict=True):
            auc_axis.text(
                x,
                value + auc_span * 0.035,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

        delta_axis = axes[0, 1]
        delta_x = np.arange(len(delta_labels))
        delta_colors = colors[1:]
        delta_axis.bar(delta_x, delta_values, color=delta_colors, width=0.62)
        delta_axis.axhline(0.0, color="#475569", linewidth=1.0)
        delta_axis.axhline(0.005, color="#B91C1C", linestyle=":", linewidth=1.1)
        delta_span = max(0.015, float(np.max(np.abs(delta_values))) * 1.35)
        delta_axis.set_ylim(-delta_span, delta_span)
        delta_axis.set_xticks(delta_x, delta_labels)
        delta_axis.set_ylabel("候选AUC - 基线AUC")
        delta_axis.set_title("直接ResCNN相对基线的差值", loc="left", fontweight="bold")
        delta_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        for x, value in zip(delta_x, delta_values, strict=True):
            delta_axis.text(
                x,
                value + (delta_span * 0.035 if value >= 0 else -delta_span * 0.035),
                f"{value:+.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.0,
            )

        bit_axis = axes[1, 0]
        bit_x = np.arange(len(selected))
        bit_auc = np.asarray([row["auc"] for row in selected], dtype=np.float64)
        bit_axis.bar(bit_x, bit_auc, color="#2563EB", width=0.62)
        bit_axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
        bit_span = max(0.025, float(np.max(np.abs(bit_auc - 0.5))) * 1.35)
        bit_axis.set_ylim(max(0.0, 0.5 - bit_span), min(1.0, 0.5 + bit_span))
        bit_axis.set_xticks(
            bit_x, [f"bit {int(row['msb_index'])}" for row in selected]
        )
        bit_axis.set_ylabel("AUC")
        bit_axis.set_title("八个组成bit的MLP基线", loc="left", fontweight="bold")
        bit_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

        gate_axis = axes[1, 1]
        checks = (
            gate["metrics"]["r3_calibration_checks"]
            if rounds == 3
            else gate["metrics"]["r4_feasibility_checks"]
        )
        check_labels = [CHECK_LABELS.get(label, label) for label in checks]
        check_values = np.asarray([bool(value) for value in checks.values()], dtype=int)
        check_y = np.arange(len(check_labels))
        gate_axis.barh(
            check_y,
            np.ones(len(check_labels)),
            color=["#0F766E" if value else "#DC2626" for value in check_values],
            height=0.58,
        )
        gate_axis.set_xlim(0.0, 1.0)
        gate_axis.set_xticks([])
        gate_axis.set_yticks(check_y, check_labels)
        gate_axis.invert_yaxis()
        gate_axis.set_title("冻结门逐项结果", loc="left", fontweight="bold")
        for y, value in zip(check_y, check_values, strict=True):
            gate_axis.text(
                0.5,
                y,
                "通过" if value else "未通过",
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
            )

        decision_text = {
            "innovation2_selected8_parity_local_readiness_passed": "实现与产物门通过",
            "innovation2_selected8_parity_r3_calibrated": "三轮同目标校准通过",
            "innovation2_selected8_parity_r3_not_calibrated": "三轮同目标校准未通过",
            "innovation2_selected8_parity_r4_local_supported": "四轮本地可行性门通过",
            "innovation2_selected8_parity_r4_not_supported": "四轮本地可行性门未通过",
            "innovation2_selected8_parity_protocol_invalid": "实验协议无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.075,
            0.075,
            (
                f"裁决：{decision_text}；直接ResCNN AUC={float(candidate['auc']):.6f}，"
                f"accuracy-majority={float(candidate['accuracy_minus_majority']):+.6f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.075,
            0.035,
            "证据边界：单固定密钥、本地诊断规模、真实密文输出parity；不是积分平衡、真假样本、正式高轮结论或SOTA。",
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
