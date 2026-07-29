from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC 五轮",
    "midori64": "Midori-64 四轮",
    "dialga128": "Dialga-128 四轮",
}
CIPHER_COLORS = {
    "uknit64": "#0F766E",
    "midori64": "#2563EB",
    "dialga128": "#B45309",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BJ multi-shuffle null audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bj_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bj_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = _ordered_panels(gate.get("panels", []))
    if len(panels) != 12:
        raise ValueError("K1-BJ plot requires twelve replica/cipher/split panels")
    if any(len(panel.get("null_strengths", [])) != 31 for panel in panels):
        raise ValueError("K1-BJ plot requires 31 null strengths per panel")
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
            "创新1 K1-BJ：正确 cell 联合信号能否超过 31 次标签打乱零分布",
            x=0.045,
            y=0.965,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.045,
            0.91,
            "完全复用K1-BI的4-pair数据和特征；正确与打乱统一比较 |AUC-0.5|，避免把反向AUC误判为随机。",
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
        _render_empirical_p(axes[0, 1], panels)
        _render_null_distribution(axes[1, 0], panels)
        _render_q95_margin(axes[1, 1], panels)
        figure.text(
            0.045,
            0.028,
            "这是本地零神经参数的统计归因审计，不是正式训练、密码攻击、任意SPN泛化或SOTA结果。",
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
        "null_permutations_per_panel": 31,
        "orientation_invariant_statistic_visible": True,
        "title_explains_cipher_rounds_and_null_mechanism": True,
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
    axis.axhline(0.55, color="#047857", linestyle="--", linewidth=1.5, label="uKNIT边界 0.55")
    values = [float(row["correct_auc"]) for row in panels]
    axis.set_ylim(min(0.45, min(values) - 0.02), max(0.58, max(values) + 0.025))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("K1-BI 正确算子 AUC（精确重放）")
    axis.set_title("先确认原始信号和 uKNIT 线性边界", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)


def _render_empirical_p(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    positions = np.arange(len(panels))
    for cipher in CIPHER_LABELS:
        selected = [
            (index, float(row["empirical_p"]))
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
    axis.axhline(0.05, color="#047857", linestyle="--", linewidth=1.5, label="归因门槛 p=0.05")
    values = [float(row["empirical_p"]) for row in panels]
    axis.set_ylim(0.0, max(0.12, max(values) + 0.06))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("经验 p 值（越低越不像标签偶然相关）")
    axis.set_title("正确强度在31次打乱中是否仍属异常", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right", fontsize=8.5)


def _render_null_distribution(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    positions = np.arange(len(panels))
    distributions = [
        [float(value) for value in row["null_strengths"]] for row in panels
    ]
    boxes = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111827", "linewidth": 1.2},
        boxprops={"facecolor": "#CBD5E1", "edgecolor": "#64748B"},
        whiskerprops={"color": "#64748B"},
        capprops={"color": "#64748B"},
        flierprops={
            "marker": ".",
            "markerfacecolor": "#94A3B8",
            "markeredgecolor": "#94A3B8",
            "markersize": 3,
        },
    )
    boxes["boxes"][0].set_label("31次标签打乱的 |AUC-0.5|")
    correct = [float(row["correct_strength"]) for row in panels]
    axis.scatter(
        positions,
        correct,
        color="#DC2626",
        marker="D",
        s=45,
        label="正确标签的 |AUC-0.5|",
        zorder=4,
    )
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("方向不变强度 |AUC - 0.5|")
    axis.set_title("正确信号与完整打乱零分布直接对照", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)


def _render_q95_margin(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    positions = np.arange(len(panels))
    for cipher in CIPHER_LABELS:
        selected = [
            (index, float(row["correct_strength_minus_null_q95"]))
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
    axis.axhline(0.10, color="#047857", linestyle="--", linewidth=1.5, label="Midori/Dialga门槛 +0.10")
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    values = [float(row["correct_strength_minus_null_q95"]) for row in panels]
    span = max(0.05, max(values + [0.10]) - min(values))
    axis.set_ylim(min(-0.03, min(values) - 0.15 * span), max(0.14, max(values) + 0.18 * span))
    axis.set_xticks(positions, _panel_labels(panels), rotation=27, ha="right")
    axis.set_ylabel("正确强度 - 打乱零分布95%分位")
    axis.set_title("结构信号领先随机方向效应多少", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)


def _ordered_panels(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["replica"]),
            list(CIPHER_LABELS).index(str(row["cipher_key"])),
            list(("same_key_fresh", "cross_key_validation")).index(
                str(row["split"])
            ),
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
    if decision.endswith("linear_transport_boundary_confirmed"):
        return "裁决：Midori/Dialga信号显著超过31次打乱，但uKNIT仍低于0.55；停止纯线性路线，转S盒感知cell原语。"
    if decision.endswith("null_attribution_not_supported"):
        return "裁决：多次打乱仍不能支持Midori/Dialga归因；固定特征，继续审计Fisher与零分布机制。"
    if decision.endswith("uknit_boundary_not_confirmed"):
        return "裁决：uKNIT重放意外越过0.55；先核对K1-BI来源和重放，暂不设计网络。"
    return "裁决：协议无效；只修复来源、重放、置换或产物完整性后原样重跑。"


def _decision_color(status: str) -> str:
    if status == "pass":
        return "#047857"
    if status == "hold":
        return "#B45309"
    return "#B91C1C"


__all__ = ["main", "render_k1bj_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
