from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Innovation 2 E27 SPECK fixed-position audit."
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
    render_position_family_svg(rows, gate, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


def render_position_family_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    by_start = {int(row["position_start"]): row for row in rows}
    expected = tuple(range(15)) + tuple(range(16, 31))
    if tuple(sorted(by_start)) != expected:
        raise ValueError("E27 plot requires all 30 preregistered adjacent pairs")
    decisions = {
        "innovation2_speck_hwang_position_family_advance": (
            "位置族通过：正负标签数量和跨 word 覆盖满足门槛；下一步做组外捷径审计。"
        ),
        "innovation2_speck_hwang_position_family_narrow": (
            "位置族过窄：暂停训练，评估非相邻或旋转等价结构族。"
        ),
        "innovation2_speck_hwang_position_family_anchor_only": (
            "仅论文锚点稳定：停止当前 SPECK 位置标签路线。"
        ),
        "innovation2_speck_hwang_position_family_protocol_invalid": (
            "协议无效：baseline、缓存、映射、计时或本地重算未通过。"
        ),
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
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 8.2), sharey=True)
        figure.subplots_adjust(left=0.07, right=0.975, top=0.71, bottom=0.25, wspace=0.17)
        figure.suptitle(
            "创新2 E27：SPECK32/64 固定位置与 7 轮输出平衡关系",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.895,
            (
                "每个点表示两个相邻固定 bit；其余30 bit完整遍历2³⁰明文。纵轴是8把筛选密钥中，"
                "输出掩码0x02050204保持XOR平衡的密钥数。"
            ),
            ha="left",
            va="top",
            fontsize=10.2,
            color="#526070",
        )
        legend_handles = None
        legend_labels = None
        for axis, starts, title in (
            (axes[0], tuple(range(15)), "低 16-bit word：固定位置 0–15"),
            (axes[1], tuple(range(16, 31)), "高 16-bit word：固定位置 16–31"),
        ):
            ordered = [by_start[start] for start in starts]
            x = np.arange(len(starts), dtype=np.float64)
            y = np.asarray([int(row["screen_balanced_keys"]) for row in ordered])
            colors = [
                "#DC2626" if bool(row.get("stable_positive"))
                else "#D97706" if bool(row.get("validation_selected"))
                else "#2563EB" if bool(row.get("screen_pass"))
                else "#94A3B8"
                for row in ordered
            ]
            axis.plot(x, y, color="#CBD5E1", linewidth=1.2, zorder=1)
            axis.scatter(x, y, c=colors, s=62, edgecolor="#FFFFFF", linewidth=0.8, zorder=3)
            for index, (start, row) in enumerate(zip(starts, ordered)):
                if start == 5:
                    axis.annotate(
                        "论文正锚点 {5,6}",
                        (index, y[index]),
                        xytext=(0, 16),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8.5,
                        color="#991B1B",
                    )
                elif start == 0:
                    axis.annotate(
                        "位置负控制 {0,1}",
                        (index, y[index]),
                        xytext=(5, 16),
                        textcoords="offset points",
                        ha="left",
                        fontsize=8.5,
                        color="#475569",
                    )
            axis.axhline(8, color="#059669", linestyle="--", linewidth=1.4, zorder=0)
            axis.set_title(title, loc="left", fontweight="bold", pad=12)
            axis.set_xlabel("固定相邻 bit pair")
            axis.set_xticks(x, [f"{start},{start + 1}" for start in starts], rotation=55, ha="right")
            axis.set_ylim(-0.45, 9.25)
            axis.set_yticks(range(9))
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
            if axis is axes[0]:
                axis.set_ylabel("8把筛选密钥中的平衡次数")
                handles = [
                    plt.Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=7)
                    for color, label in (
                        ("#94A3B8", "筛选失败"),
                        ("#2563EB", "筛选命中但未进入64-key验证"),
                        ("#D97706", "进入64-key验证但未稳定"),
                        ("#DC2626", "64-key稳定正位置"),
                    )
                ]
                legend_handles = handles
                legend_labels = [handle.get_label() for handle in handles]
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.815),
            ncol=4,
            frameon=False,
            fontsize=9.0,
        )
        metrics = gate.get("metrics", {})
        figure.text(
            0.07,
            0.105,
            (
                f"筛选候选={metrics.get('screen_candidate_count', 'NA')}，"
                f"64-key稳定正位置={metrics.get('stable_positive_count', 'NA')}，"
                f"采样负位置={metrics.get('sampled_negative_count', 'NA')}。 "
                f"裁决：{decisions.get(str(gate.get('decision')), str(gate.get('decision')))}"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.07,
            0.045,
            (
                "证据范围：同一组Phase C密钥；28个新位置先8-key精确筛选，最多8个候选补到64-key。"
                "这不是神经训练、全密钥证明或论文规模复现。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
