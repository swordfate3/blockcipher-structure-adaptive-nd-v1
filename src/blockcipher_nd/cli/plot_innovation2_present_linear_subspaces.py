from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E30 PRESENT-r7 linear-subspace kernel diversity."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    render_linear_subspace_svg(rows, gate, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


def render_linear_subspace_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output: Path
) -> None:
    anchors = [row for row in rows if row.get("role") == "coordinate_anchor"]
    random_rows = [row for row in rows if row.get("role") == "random_orientation"]
    if len(anchors) != 4 or not random_rows:
        raise ValueError("E30 plot requires four anchors and random orientations")
    decisions = {
        "innovation2_present_linear_subspace_readiness_passed": (
            "实现就绪；运行冻结的32-orientation、128-key审计。"
        ),
        "innovation2_present_linear_subspace_kernel_family_ready": (
            "一般线性子空间形成足够宽的稳定kernel族；进入E31捷径审计。"
        ),
        "innovation2_present_linear_subspace_kernel_family_too_sparse": (
            "稳定kernel或签名仍不足；停止当前随机orientation benchmark。"
        ),
        "innovation2_present_linear_subspace_protocol_invalid": (
            "RREF、密钥、缓存、PRESENT向量化或GF(2)协议无效。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(
            1, 2, figsize=(16.2, 8.5), gridspec_kw={"width_ratios": [1, 4]}
        )
        figure.subplots_adjust(left=0.07, right=0.975, top=0.72, bottom=0.27, wspace=0.16)
        figure.suptitle(
            "创新2 E30：PRESENT-80 7轮16维线性子空间输出kernel多样性",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.2,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.895,
            (
                "每个结构完整遍历2¹⁶个plaintext；64把discovery与64把validation密钥形成输出XOR矩阵。"
                "纵轴是64-bit输出mask空间的kernel维数。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.legend(
            handles=[
                plt.Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=7)
                for color, label in (
                    ("#2563EB", "discovery nullity"),
                    ("#D97706", "validation nullity"),
                    ("#DC2626", "joint nullity（跨两半稳定）"),
                )
            ],
            loc="upper left",
            bbox_to_anchor=(0.07, 0.82),
            ncol=3,
            frameon=False,
            fontsize=9.0,
        )
        for axis, panel_rows, title in (
            (axes[0], anchors, "同预算坐标子空间锚点"),
            (axes[1], random_rows, "RREF去重的随机orientation"),
        ):
            x = np.arange(len(panel_rows), dtype=np.float64)
            for offset, key, color in (
                (-0.18, "discovery_nullity", "#2563EB"),
                (0.0, "validation_nullity", "#D97706"),
                (0.18, "joint_nullity", "#DC2626"),
            ):
                values = [int(row[key]) for row in panel_rows]
                axis.scatter(x + offset, values, s=46, color=color, edgecolor="#FFFFFF", linewidth=0.6)
            axis.axhline(0, color="#64748B", linewidth=1.0)
            axis.set_title(title, loc="left", fontweight="bold", pad=12)
            axis.set_xticks(x, [str(row["structure_id"]).replace("coordinate_", "C").replace("random_", "R") for row in panel_rows], rotation=55, ha="right")
            axis.set_xlabel("输入子空间ID")
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].set_ylabel("输出kernel维数（nullity）")
        maximum = max(int(row["discovery_nullity"]) for row in rows)
        for axis in axes:
            axis.set_ylim(-0.8, max(4.0, maximum + 2.0))
        metrics = gate.get("metrics", {})
        figure.text(
            0.07,
            0.115,
            (
                f"随机orientation非平凡joint kernel={metrics.get('random_nontrivial_joint_kernel_count', 'NA')}，"
                f"不同非零签名={metrics.get('distinct_nonzero_joint_kernel_signatures', 'NA')}，"
                f"平均half-retention={metrics.get('mean_half_intersection_retention', 'NA')}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.07,
            0.065,
            f"裁决：{decisions.get(str(gate.get('decision')), str(gate.get('decision')))}",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.07,
            0.020,
            "证据范围：固定PRESENT-80 7轮、16维线性子空间与64+64经验密钥；不是神经训练、全密钥证明或affine-space积分首创声明。",
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
