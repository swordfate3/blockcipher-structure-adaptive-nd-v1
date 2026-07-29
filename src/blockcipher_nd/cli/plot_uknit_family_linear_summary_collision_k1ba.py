from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC",
    "midori64": "Midori64",
    "dialga128": "Dialga-128",
}
SHORT_CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}
SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BA linear-summary collision chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--collisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    collisions = json.loads(args.collisions.read_text(encoding="utf-8"))
    report = render_k1ba_svg(gate, collisions["collision_rows"], args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ba_svg(
    gate: Mapping[str, Any],
    collision_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    collisions = _ordered_collisions(collision_rows)
    panels = _ordered_panels(gate["panel_results"])
    cross_key = [panel for panel in panels if panel["split"] == "cross_key_validation"]
    collision_result = gate["collision_results"]
    scalar_result = gate["scalar_rank_results"]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 12.0))
        figure.subplots_adjust(
            left=0.08,
            right=0.975,
            top=0.735,
            bottom=0.16,
            hspace=0.63,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 K1-BA：18维统计摘要无法标识真实线性拓扑",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "零训练冻结审计：uKNIT-BC 5轮、Midori64 4轮、Dialga-128 4轮；"
                "K1-AZ两个epoch-9检查点、原12个新样本面板、每样本4对密文。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                f"三种密码的错误GF(2)算子至少改变 {collision_result['minimum_matrix_hamming_fraction'] * 100:.2f}% "
                f"矩阵位，但18维摘要最大差异仍为 {collision_result['maximum_summary_delta']:.1f}；"
                "12/12面板输出逐位相同。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.830,
            (
                "跨密码粗摘要能改变边门，但概率排序几乎不动："
                f"最小边门变化 {scalar_result['minimum_edge_gate_delta']:.6f}，"
                f"最小排序相关 {scalar_result['minimum_probability_spearman']:.6f}，"
                f"最大AUC变化 {scalar_result['maximum_auc_delta_abs']:.6f}。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.786,
            "裁决：问题首先在结构输入。当前统计摘要没有实际 source→target 连线，无法承担拓扑适配。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_matrix_collision(axes[0, 0], collisions)
        _render_active_dimensions(axes[0, 1], collisions)
        _render_gate_response(axes[1, 0], cross_key)
        _render_auc_response(axes[1, 1], panels)

        figure.text(
            0.05,
            0.052,
            (
                "下一步 K1-BB：设计共享的位置保持线性算子 token 编码器，直接读取实际源位、目标位和轮位置；"
                "先做零训练正确/损坏算子可分性 readiness，再决定是否训练。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.019,
            "当前证据仅是2048/class本地机制审计，不是正式规模、攻击结果、任意SPN泛化或SOTA证据。",
            ha="left",
            fontsize=9.8,
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
        "result_panels": len(panels),
        "collision_ciphers": len(collisions),
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _ordered_collisions(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_cipher = {str(row["cipher_key"]): row for row in rows}
    if set(by_cipher) != set(CIPHER_LABELS):
        raise ValueError("K1-BA plot requires three collision rows")
    return [by_cipher[cipher] for cipher in CIPHER_LABELS]


def _ordered_panels(
    panels: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    ordered = []
    for replica in (0, 1):
        for cipher in CIPHER_LABELS:
            for split in ("same_key_fresh", "cross_key_validation"):
                ordered.append(panels[f"replica{replica}_{cipher}_{split}"])
    if len(ordered) != 12:
        raise ValueError("K1-BA plot requires twelve panels")
    return ordered


def _render_matrix_collision(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [CIPHER_LABELS[str(row["cipher_key"])] for row in rows]
    values = [float(row["matrix_hamming_fraction"]) * 100.0 for row in rows]
    positions = np.arange(len(rows))
    bars = axis.bar(positions, values, width=0.58, color=["#2563EB", "#0F766E", "#B45309"])
    axis.axhline(0.1, color="#7C3AED", linestyle="--", linewidth=1.3, label="有效变化门槛 0.1%")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("损坏算子与正确算子的矩阵位差异（%）")
    axis.set_title("算子确实改变，但18维摘要差异仍为0.0", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.22,
            f"{value:.2f}%\n摘要Δ=0.0",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
        )
    axis.set_ylim(0.0, max(values) + 1.5)
    axis.legend(loc="upper right", frameon=False, fontsize=8.8)


def _render_active_dimensions(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [CIPHER_LABELS[str(row["cipher_key"])] for row in rows]
    values = [int(row["cross_cipher_active_linear_dimension_count"]) for row in rows]
    positions = np.arange(len(rows))
    bars = axis.bar(positions, values, width=0.58, color=["#DC2626", "#0F766E", "#2563EB"])
    axis.axhline(18, color="#64748B", linestyle="--", linewidth=1.2, label="摘要总宽度 18")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("相对跨密码错配发生变化的摘要维度数")
    axis.set_title("uKNIT→Midori：18维中只有1维发生变化", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value}/18",
            ha="center",
            fontweight="bold",
        )
    axis.set_ylim(0.0, 20.0)
    axis.legend(loc="upper right", frameon=False, fontsize=8.8)


def _render_gate_response(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        f"{SHORT_CIPHER_LABELS[str(panel['cipher_key'])]} R{panel['replica']}"
        for panel in panels
    ]
    cross = [float(panel["cross_cipher_edge_gate_delta"]) for panel in panels]
    collision = [float(panel["collision_edge_gate_delta"]) for panel in panels]
    positions = np.arange(len(panels))
    width = 0.36
    axis.bar(positions - width / 2, cross, width, label="跨密码粗摘要", color="#B45309")
    axis.bar(positions + width / 2, collision, width, label="同摘要损坏拓扑", color="#DC2626")
    axis.axhline(0.0005, color="#7C3AED", linestyle="--", linewidth=1.3, label="响应门槛 0.0005")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("相对正确描述符的边门绝对变化")
    axis.set_title("门只响应粗统计；同摘要错误拓扑变化严格为0", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        fontsize=8.3,
        ncol=3,
    )


def _render_auc_response(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        f"{SHORT_CIPHER_LABELS[str(panel['cipher_key'])]} R{panel['replica']} · "
        f"{SPLIT_LABELS[str(panel['split'])]}"
        for panel in panels
    ]
    values = [abs(float(panel["cross_cipher_auc_delta"])) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    axis.barh(y, values, height=0.60, color="#2563EB")
    axis.axvline(0.001, color="#DC2626", linestyle="--", linewidth=1.4, label="语义优势门槛 0.001")
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.00108)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_xlabel("正确与跨密码线性摘要的 AUC 差异绝对值")
    axis.set_title("边门虽变化，12/12面板AUC仍近乎不动", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ba_svg"]
