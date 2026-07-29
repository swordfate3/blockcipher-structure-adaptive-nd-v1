from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    RESULT_CONDITIONS,
)


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC 五轮",
    "midori64": "Midori-64 四轮",
    "dialga128": "Dialga-128 四轮",
}
CONDITION_LABELS = {
    "correct_operator": "正确扩散算子",
    "same_summary_corrupted_operator": "同摘要错误算子",
    "cross_cipher_operator": "其他密码算子",
    "identity_operator": "恒等算子",
    "label_shuffled_correct_operator": "打乱训练标签",
}
CIPHER_COLORS = {
    "uknit64": "#0F766E",
    "midori64": "#2563EB",
    "dialga128": "#B45309",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BH exact GF(2) response audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bh_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bh_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = _ordered_panels(gate.get("panels", []))
    if len(panels) != 12:
        raise ValueError("K1-BH plot requires twelve replica/cipher/split panels")
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(18, 12))
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.79,
            bottom=0.145,
            hspace=0.58,
            wspace=0.23,
        )
        figure.suptitle(
            "创新1 K1-BH：正确的 GF(2) 扩散算子是否留下独有的标签信号",
            x=0.045,
            y=0.965,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.045,
            0.91,
            "对同一批4-pair密文直接执行精确逆线性变换；Fisher只在正确算子的训练特征上拟合，错误算子不得重新拟合。",
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.045,
            0.855,
            _decision_text(gate),
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        _render_correct_auc(axes[0, 0], panels)
        _render_condition_summary(axes[0, 1], panels)
        _render_topology_margins(axes[1, 0], panels)
        _render_control_margins(axes[1, 1], panels)
        figure.text(
            0.045,
            0.028,
            "这是本地确定性机制审计，不是神经网络准确率、正式规模、密码攻击、任意SPN泛化或SOTA结果。",
            ha="left",
            fontsize=9.7,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 12.0,
        "language": "zh-CN",
        "panels": 4,
        "result_panels": 12,
        "uses_scatter_and_margin_views_instead_of_overlapping_curves": True,
        "title_explains_cipher_rounds_and_mechanism": True,
        "status_from_gate": gate.get("status"),
    }


def _render_correct_auc(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    positions = np.arange(len(panels))
    for cipher in CIPHER_LABELS:
        selected = [
            (index, float(row["correct_auc"]))
            for index, row in enumerate(panels)
            if row["cipher_key"] == cipher
        ]
        axis.scatter(
            [item[0] for item in selected],
            [item[1] for item in selected],
            color=CIPHER_COLORS[cipher],
            s=55,
            label=CIPHER_LABELS[cipher],
            zorder=3,
        )
    axis.axhline(0.55, color="#047857", linestyle="--", linewidth=1.5, label="信号门槛 0.55")
    values = [float(row["correct_auc"]) for row in panels]
    axis.set_ylim(min(0.48, min(values) - 0.02), max(0.58, max(values) + 0.025))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("正确算子 AUC（局部放大）")
    axis.set_title("正确算子本身是否稳定保留标签信号", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)


def _render_condition_summary(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
        "correct_operator": "correct_auc",
        "same_summary_corrupted_operator": "same_summary_wrong_auc",
        "cross_cipher_operator": "cross_cipher_wrong_auc",
        "identity_operator": "identity_auc",
        "label_shuffled_correct_operator": "label_shuffle_auc",
    }
    medians = []
    lower = []
    upper = []
    for condition in RESULT_CONDITIONS:
        values = np.asarray([float(row[fields[condition]]) for row in panels])
        median = float(np.median(values))
        medians.append(median)
        lower.append(median - float(values.min()))
        upper.append(float(values.max()) - median)
    positions = np.arange(len(fields))
    colors = ("#0F766E", "#B91C1C", "#7C3AED", "#6B7280", "#2563EB")
    axis.bar(positions, medians, color=colors, width=0.68, alpha=0.9)
    axis.errorbar(
        positions,
        medians,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#111827",
        capsize=4,
        linewidth=1.1,
    )
    for position, median in zip(positions, medians, strict=True):
        axis.annotate(
            f"{median:.3f}",
            (position, median),
            xytext=(9, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#374151",
        )
    axis.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle=(0, (3, 3)))
    axis.set_xticks(
        positions,
        [CONDITION_LABELS[condition] for condition in RESULT_CONDITIONS],
        rotation=18,
        ha="right",
    )
    minimum = min(float(np.median([float(row[field]) for row in panels])) for field in fields.values())
    maximum = max(float(row[field]) for field in fields.values() for row in panels)
    axis.set_ylim(min(0.47, minimum - 0.025), max(0.58, maximum + 0.055))
    axis.set_ylabel("十二个面板的 AUC 中位数（须结合误差线）")
    axis.set_title("同一评分器下，正确与错误算子的总体差别", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)


def _render_topology_margins(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    positions = np.arange(len(panels))
    series = (
        ("correct_minus_same_summary_wrong", "正确 - 同摘要错误", "#B91C1C", "o"),
        ("correct_minus_cross_cipher_wrong", "正确 - 其他密码算子", "#7C3AED", "s"),
    )
    all_values: list[float] = []
    for field, label, color, marker in series:
        values = [float(row[field]) for row in panels]
        all_values.extend(values)
        axis.scatter(
            positions,
            values,
            color=color,
            marker=marker,
            s=48,
            label=label,
            zorder=3,
        )
    axis.axhline(0.01, color="#047857", linestyle="--", linewidth=1.5, label="通过门槛 +0.01")
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    span = max(0.03, max(all_values + [0.01]) - min(all_values))
    axis.set_ylim(min(-0.02, min(all_values) - 0.18 * span), max(0.025, max(all_values) + 0.2 * span))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("正确算子 AUC - 错误算子 AUC")
    axis.set_title("核心裁决：正确拓扑是否在每个面板都独有", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)


def _render_control_margins(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    positions = np.arange(len(panels))
    identity = [float(row["correct_minus_identity"]) for row in panels]
    shuffle = [float(row["correct_minus_label_shuffle"]) for row in panels]
    axis.scatter(positions, identity, color="#6B7280", marker="D", s=44, label="正确 - 恒等（门槛 +0.01）")
    axis.scatter(positions, shuffle, color="#2563EB", marker="^", s=52, label="正确 - 标签打乱（门槛 +0.03）")
    axis.axhline(0.01, color="#6B7280", linestyle=(0, (3, 3)), linewidth=1.2)
    axis.axhline(0.03, color="#2563EB", linestyle="--", linewidth=1.3)
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    values = identity + shuffle
    span = max(0.04, max(values + [0.03]) - min(values))
    axis.set_ylim(min(-0.02, min(values) - 0.18 * span), max(0.045, max(values) + 0.2 * span))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("正确算子 AUC - 控制 AUC")
    axis.set_title("排除原始密文捷径与标签偶然相关", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)


def _ordered_panels(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["replica"]),
            list(CIPHER_LABELS).index(str(row["cipher_key"])),
            list(("same_key_fresh", "cross_key_validation")).index(str(row["split"])),
        ),
    )


def _panel_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    short = {"uknit64": "uKNIT", "midori64": "Midori", "dialga128": "Dialga"}
    return [
        f"{short[str(row['cipher_key'])]} R{int(row['replica'])}\n"
        f"{'同钥' if row['split'] == 'same_key_fresh' else '跨钥'}"
        for row in rows
    ]


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("exact_operator_topology_signal_supported"):
        return "裁决：直接 GF(2) 变换能稳定识别正确拓扑；下一步设计共享的位置保持状态残差，不增加数据或pair数。"
    if decision.endswith("predictive_but_not_topology_identifying"):
        return "裁决：精确响应可以预测，但不能稳定认出正确拓扑；先审计错误算子等价性，不训练新网络。"
    if decision.endswith("exact_operator_signal_unstable"):
        return "裁决：独立bit均值不能保留uKNIT信号；下一步只测试运行时cell的4-bit联合类别响应，不重复扫差分位置。"
    if decision.endswith("shuffle_attribution_not_supported"):
        return "裁决：标签打乱控制没有通过；先修复评分归因，当前结果不能作为结构信号。"
    return "裁决：协议无效；只修复失败的来源、GF(2)实现、评分器复用或产物绑定后原样重跑。"


def _decision_color(status: str) -> str:
    if status == "pass":
        return "#047857"
    if status == "hold":
        return "#B45309"
    return "#B91C1C"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1bh_svg"]
