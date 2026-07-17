from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BASELINE_LABELS = {
    "global_rate": "全局正例率",
    "position_identity_marginal": "位置身份边际",
    "position_weight_marginal": "位置重量边际",
    "mask_identity_marginal": "mask身份边际",
    "mask_weight_marginal": "mask重量边际",
    "position_mask_additive": "位置+mask加性",
    "position_mask_bitwise_linear": "32+32位模式线性",
    "position_disjoint_bitwise": "位置组外位模式",
    "mask_disjoint_bitwise": "mask组外位模式",
    "dual_disjoint_bitwise": "位置+mask双轴组外",
    "label_shuffle_dual_disjoint": "标签打乱双轴组外",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the E28 SPECK position-mask label-width audit."
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
    render_position_label_svg(rows, gate, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


def render_position_label_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    decisions = {
        "innovation2_speck_position_label_grid_advance": (
            "标签宽度与组外捷径门通过；下一步做fresh-key稳定性和结构扩展。"
        ),
        "innovation2_speck_position_label_grid_shortcut_dominated": (
            "简单位置/mask位模式可泛化解释标签；停止当前标签表。"
        ),
        "innovation2_speck_position_label_grid_too_narrow": (
            "完整位置、flipping mask或签名不足；未进入神经训练。"
        ),
        "innovation2_speck_position_label_protocol_invalid": (
            "E27来源或kernel行无效；本结果不可裁决。"
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
        figure, axis = plt.subplots(figsize=(14.8, 8.4))
        figure.subplots_adjust(left=0.30, right=0.965, top=0.72, bottom=0.22)
        figure.suptitle(
            "创新2 E28：SPECK位置 × 输出mask标签宽度与捷径审计",
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
                "标签来自E27完整64-key joint kernel；候选仅使用basis与同kernel两两XOR，"
                "不新增加密、不训练神经网络。"
            ),
            ha="left",
            va="top",
            fontsize=10.1,
            color="#526070",
        )
        if rows:
            ordered = [row for row in rows if str(row["baseline"]) in BASELINE_LABELS]
            labels = [BASELINE_LABELS[str(row["baseline"])] for row in ordered]
            values = np.asarray([float(row["directional_auc"]) for row in ordered])
            colors = [
                "#7C3AED" if "shuffle" in str(row["baseline"])
                else "#DC2626" if value >= 0.75
                else "#2563EB"
                for row, value in zip(ordered, values)
            ]
            y = np.arange(len(ordered), dtype=np.float64)
            bars = axis.barh(y, values, color=colors, height=0.66, zorder=3)
            axis.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values],
                padding=4,
                fontsize=8.8,
                color="#334155",
            )
            axis.axvline(0.75, color="#D97706", linestyle="--", linewidth=1.5, zorder=1)
            axis.axvline(0.50, color="#94A3B8", linestyle=":", linewidth=1.2, zorder=1)
            axis.set_yticks(y, labels)
            axis.invert_yaxis()
            axis.set_xlim(0.45, 1.03)
            axis.set_xlabel("方向无关AUC = max(AUC, 1-AUC)；越接近0.5越难被简单捷径解释")
            axis.set_title(
                "边际、加性、位模式与组外基线",
                loc="left",
                fontweight="bold",
                pad=12,
            )
            axis.grid(axis="x", color="#E5E7EB", linewidth=0.8, zorder=0)
        else:
            axis.set_axis_off()
            axis.text(
                0.5,
                0.55,
                "标签宽度门未通过\n未生成捷径基线",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color="#475569",
                linespacing=1.5,
            )
        figure.text(
            0.07,
            0.105,
            (
                f"完整64-key位置={gate.get('full_evidence_positions', 'NA')}，"
                f"flipping mask={gate.get('flipping_masks', 'NA')}，"
                f"位置标签签名={gate.get('distinct_position_signatures', 'NA')}，"
                f"正例率={gate.get('positive_rate', 'NA')}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
