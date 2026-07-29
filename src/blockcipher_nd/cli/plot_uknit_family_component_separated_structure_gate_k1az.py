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
        description="Render the Chinese K1-AZ component-separated training chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1az_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1az_svg(
    gate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    panels = _ordered_panels(gate)
    cross_key = [panel for panel in panels if panel["split"] == "cross_key_validation"]
    macro = gate["macro_results"]
    mismatch = gate["mismatch_results"]
    harm_count = sum(float(panel["correct_minus_k1aw"]) < -0.005 for panel in panels)

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
            left=0.09,
            right=0.975,
            top=0.745,
            bottom=0.17,
            hspace=0.74,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 K1-AZ：分量隔离训练不稳定，未解决线性拓扑辨识",
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
                "本地同预算诊断：uKNIT-BC 5轮、Midori64 4轮、Dialga-128 4轮；"
                "每种密码2048/class、每样本4对密文、10 epochs、2个独立初始化。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "相对共享连接 K1-AW，跨密钥宏平均变化为 "
                f"{macro['replica0']['improvement']:+.4f}（副本0）和 "
                f"{macro['replica1']['improvement']:+.4f}（副本1）；"
                "两次训练方向不一致。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.833,
            (
                f"逐面板有 {harm_count}/12 项低于 -0.005 无伤害线；正确结构超过错配 "
                f"+0.001 的面板为：完整错配 {mismatch['full_mismatch']['passing_panels']}/12、"
                f"S盒错配 {mismatch['sbox_only_mismatch']['passing_panels']}/12、"
                f"线性层错配 {mismatch['linear_only_mismatch']['passing_panels']}/12。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.792,
            "裁决：暂缓。硬隔离既没有稳定保留旧模型性能，也没有让正确线性扩散描述符胜出。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_cross_key_anchor(axes[0, 0], cross_key)
        _render_macro_improvement(axes[0, 1], macro)
        _render_no_harm_margins(axes[1, 0], panels)
        _render_mismatch_margins(axes[1, 1], panels)

        figure.text(
            0.05,
            0.061,
            (
                "下一步：冻结两个 K1-AZ 检查点，逐维遮蔽结构摘要并重放残差响应，"
                "定位仍对错误描述符不变的维度；不增加 pairs、样本、epoch 或模型容量。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.026,
            "当前证据仅是2048/class本地诊断，不是正式规模、攻击结果、任意SPN泛化或SOTA证据。",
            ha="left",
            fontsize=9.9,
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
        "evaluation_rows": int(gate.get("evaluation_rows", len(panels) * 5)),
        "comparison_panels": len(panels),
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _ordered_panels(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = gate["panel_results"]
    panels = []
    for replica in (0, 1):
        for cipher in CIPHER_LABELS:
            for split in ("same_key_fresh", "cross_key_validation"):
                key = f"replica{replica}_{cipher}_{split}"
                row = dict(source[key])
                row.update(
                    {
                        "replica": replica,
                        "cipher_key": cipher,
                        "split": split,
                        "cross_key_label": f"{CIPHER_LABELS[cipher]} · 副本{replica}",
                        "full_label": (
                            f"{SHORT_CIPHER_LABELS[cipher]} R{replica} · "
                            f"{SPLIT_LABELS[split]}"
                        ),
                    }
                )
                panels.append(row)
    if len(panels) != 12:
        raise ValueError("K1-AZ plot requires twelve comparison panels")
    return panels


def _render_cross_key_anchor(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["cross_key_label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    height = 0.34
    candidate = np.asarray([float(panel["correct_auc"]) for panel in panels])
    anchors = np.asarray([float(panel["k1aw_anchor_auc"]) for panel in panels])
    axis.barh(y - height / 2, candidate, height, label="K1-AZ分量隔离", color="#2563EB")
    axis.barh(y + height / 2, anchors, height, label="K1-AW共享连接", color="#94A3B8")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    lower = max(0.5, min(float(candidate.min()), float(anchors.min())) - 0.025)
    axis.set_xlim(lower, 1.01)
    axis.set_xlabel(f"跨密钥 AUC（从{lower:.2f}开始，仅用于展开差异）")
    axis.set_title("同一预算下的三密码跨密钥结果", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=9.0,
        ncol=2,
    )


def _render_macro_improvement(axis: plt.Axes, macro: Mapping[str, Any]) -> None:
    labels = ["副本0", "副本1"]
    values = np.asarray(
        [float(macro[f"replica{index}"]["improvement"]) for index in range(2)]
    )
    positions = np.arange(2)
    colors = ["#0F766E" if value >= 0.0 else "#DC2626" for value in values]
    axis.bar(positions, values, width=0.56, color=colors)
    axis.axhline(0.0, color="#64748B", linewidth=1.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("K1-AZ - K1-AW 跨密钥宏平均 AUC")
    axis.set_title("宏平均保留门：仅副本1通过", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    padding = max(0.002, float(np.ptp(values)) * 0.18)
    axis.set_ylim(float(values.min()) - padding, float(values.max()) + padding)
    for index, value in enumerate(values):
        offset = 0.00055 if value >= 0 else -0.0012
        axis.text(index, value + offset, f"{value:+.4f}", ha="center", fontweight="bold")


def _render_no_harm_margins(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    values = np.asarray([float(panel["correct_minus_k1aw"]) for panel in panels])
    y = np.arange(len(panels), dtype=float)
    colors = np.where(values >= -0.005, "#0F766E", "#DC2626")
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(
        -0.005,
        color="#B45309",
        linestyle="--",
        linewidth=1.4,
        label="无伤害线 -0.005",
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    padding = max(0.004, float(np.ptp(values)) * 0.08)
    axis.set_xlim(float(values.min()) - padding, float(values.max()) + padding)
    axis.set_xlabel("正确结构 AUC - K1-AW AUC")
    axis.set_title("逐面板无伤害门：uKNIT副本0明显退步", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)


def _render_mismatch_margins(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    styles = (
        ("full_mismatch", "正确 - 完整错配", "#DC2626", "o", -0.18),
        ("sbox_only_mismatch", "正确 - S盒错配", "#B45309", "^", 0.0),
        ("linear_only_mismatch", "正确 - 线性层错配", "#2563EB", "s", 0.18),
    )
    all_values = []
    for condition, label, color, marker, offset in styles:
        values = [
            float(panel["correct_minus_condition"][condition]) for panel in panels
        ]
        all_values.extend(values)
        axis.scatter(
            values,
            y + offset,
            label=label,
            color=color,
            marker=marker,
            s=36,
            zorder=3,
        )
    axis.axvline(
        0.001,
        color="#7C3AED",
        linestyle="--",
        linewidth=1.4,
        label="通过线 +0.001",
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    span = max(all_values) - min(all_values)
    padding = max(0.0002, span * 0.08)
    axis.set_xlim(min(all_values) - padding, max(max(all_values) + padding, 0.00125))
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_xlabel("正确结构相对错误结构的 AUC 优势")
    axis.set_title("结构语义门：线性层错配 0/12 通过", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=8.0,
        ncol=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1az_svg"]
