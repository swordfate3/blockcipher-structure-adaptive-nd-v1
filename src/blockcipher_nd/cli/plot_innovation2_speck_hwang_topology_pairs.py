from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FAMILY_LAYOUT = (
    ("ror7_add_aligned", "真实拓扑：y_i 与 x_(i+7) 固定", "#2563EB"),
    ("offset_minus_one", "错位控制：y_i 与 x_(i+6) 固定", "#64748B"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E27-N SPECK topology pair results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    render_topology_pair_svg(rows, gate, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


def render_topology_pair_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output: Path
) -> None:
    by_key = {(str(row["family"]), int(row["lane"])): row for row in rows}
    if len(by_key) != 32:
        raise ValueError("E27-N plot requires 16 lanes in both families")
    decisions = {
        "innovation2_speck_topology_aligned_family": "真实ROR7-to-addition对齐形成稳定位置族；进入多mask标签门。",
        "innovation2_speck_topology_pair_not_specific": "真实对齐没有超过错位控制；停止offset扫描。",
        "innovation2_speck_topology_pair_too_narrow": "真实对齐仅一个稳定lane；标签族仍过窄。",
        "innovation2_speck_topology_pair_no_signal": "真实对齐没有64-key稳定lane；停止SPECK固定pair路线。",
        "innovation2_speck_topology_pair_protocol_invalid": "映射、缓存、密钥、计时或GF(2)协议无效。",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.1), sharey=True)
        figure.subplots_adjust(left=0.07, right=0.975, top=0.70, bottom=0.25, wspace=0.18)
        figure.suptitle(
            "创新2 E27-N：SPECK ROR7模加对齐与错位控制",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.895,
            "每个pair固定两个跨word bit为00，其余30 bit完整遍历2³⁰明文；纵轴为8把筛选密钥中的目标mask平衡次数。",
            ha="left",
            va="top",
            fontsize=10.1,
            color="#526070",
        )
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=color,
                label=label,
                markersize=7,
            )
            for color, label in (
                ("#94A3B8", "8-key筛选失败"),
                ("#2563EB", "筛选命中但未进入64-key验证"),
                ("#D97706", "进入64-key验证但未稳定"),
                ("#DC2626", "64-key稳定位置"),
            )
        ]
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.82),
            ncol=4,
            frameon=False,
            fontsize=9.0,
        )
        for axis, (family, title, base_color) in zip(axes, FAMILY_LAYOUT):
            ordered = [by_key[(family, lane)] for lane in range(16)]
            values = np.asarray([int(row["screen_balanced_keys"]) for row in ordered])
            colors = [
                "#DC2626" if row.get("stable_positive")
                else "#D97706" if row.get("validation_selected")
                else base_color if row.get("screen_pass")
                else "#94A3B8"
                for row in ordered
            ]
            x = np.arange(16, dtype=np.float64)
            axis.plot(x, values, color="#CBD5E1", linewidth=1.2, zorder=1)
            axis.scatter(x, values, c=colors, s=60, edgecolor="#FFFFFF", linewidth=0.8, zorder=3)
            axis.axhline(8, color="#059669", linestyle="--", linewidth=1.4, zorder=0)
            axis.set_title(title, loc="left", fontweight="bold", pad=12)
            axis.set_xlabel("模加低word lane i（每点对应一个跨word固定pair）")
            axis.set_xticks(x, [str(lane) for lane in range(16)])
            axis.set_ylim(-0.45, 8.9)
            axis.set_yticks(range(9))
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
        axes[0].set_ylabel("8把筛选密钥中的平衡次数")
        metrics = gate.get("metrics", {})
        figure.text(
            0.07,
            0.105,
            (
                f"真实拓扑screen命中={metrics.get('true_screen_hits', 'NA')}，"
                f"错位控制命中={metrics.get('control_screen_hits', 'NA')}，"
                f"差值={metrics.get('screen_hit_delta', 'NA')}；"
                f"64-key稳定数={metrics.get('true_stable_count', 'NA')}/"
                f"{metrics.get('control_stable_count', 'NA')}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.07,
            0.018,
            "证据范围：同一组Phase C密钥；每组16个pair先做8-key筛选，每组最多4个候选补到64-key。这不是神经训练或全密钥证明。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.050,
            f"裁决：{decisions.get(str(gate.get('decision')), str(gate.get('decision')))}",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
