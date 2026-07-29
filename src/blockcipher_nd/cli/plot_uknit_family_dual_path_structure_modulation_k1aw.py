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
CONDITIONS = {
    "correct_descriptor",
    "full_mismatch",
    "sbox_only_mismatch",
    "linear_only_mismatch",
    "dual_path_disabled",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AW dual-path training chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.controls)
    report = render_k1aw_svg(gate, rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1aw_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    panels = _collect_panels(rows)
    cross_key = [panel for panel in panels if panel["split"] == "cross_key_validation"]
    macro = gate["macro_results"]
    mismatch = gate["mismatch_results"]
    harm_panels = [panel for panel in panels if float(panel["anchor_margin"]) < -0.005]
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
            "创新1 双通道结构调制：宏平均提升，但尚未学稳正确结构语义",
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
                f"相对单通道 K1-AT，跨密钥宏平均分别提升 "
                f"{macro['replica0']['improvement']:+.4f} 和 "
                f"{macro['replica1']['improvement']:+.4f}；两次独立训练都通过宏平均保留门。"
            ),
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.833,
            (
                f"但有 {len(harm_panels)}/12 个面板低于 -0.005 无伤害线；正确结构超过错配 +0.001 的面板仅为："
                f"完整错配 {mismatch['full_mismatch']['passing_panels']}/12、"
                f"S盒错配 {mismatch['sbox_only_mismatch']['passing_panels']}/12、"
                f"线性层错配 {mismatch['linear_only_mismatch']['passing_panels']}/12。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.792,
            "裁决：暂缓扩样本和16 pairs；保留检查点，先查清两条结构通道是否学反或相互抵消。",
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
                "下一步：冻结这两个最佳检查点，做0次训练的通道方向审计；分别替换S盒摘要和"
                "线性层摘要，测 GF(2) 与S盒门控、logit及AUC变化方向。"
            ),
            ha="left",
            fontsize=10.4,
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
        "evaluation_rows": len(rows),
        "comparison_panels": len(panels),
        "status_from_gate": gate.get("status"),
        "auc_claim_present": True,
        "formal_scale_claim_present": False,
    }


def _collect_panels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    if len(rows) != 60 or len(grouped) != 12:
        raise ValueError("K1-AW plot requires 60 rows in 12 panels")
    if any(set(conditions) != CONDITIONS for conditions in grouped.values()):
        raise ValueError("K1-AW plot has an incomplete condition panel")

    panels = []
    cipher_order = {key: index for index, key in enumerate(CIPHER_LABELS)}
    split_order = {"same_key_fresh": 0, "cross_key_validation": 1}
    for (replica, cipher, split), conditions in grouped.items():
        correct = conditions["correct_descriptor"]
        correct_auc = float(correct["auc"])
        anchor_auc = float(correct["k1at_anchor_auc"])
        panels.append(
            {
                "replica": replica,
                "cipher_key": cipher,
                "split": split,
                "cross_key_label": f"{CIPHER_LABELS[cipher]} · 副本{replica}",
                "full_label": (
                    f"{SHORT_CIPHER_LABELS[cipher]} R{replica} · {SPLIT_LABELS[split]}"
                ),
                "correct_auc": correct_auc,
                "anchor_auc": anchor_auc,
                "anchor_margin": correct_auc - anchor_auc,
                "mismatch_margins": {
                    condition: correct_auc - float(conditions[condition]["auc"])
                    for condition in (
                        "full_mismatch",
                        "sbox_only_mismatch",
                        "linear_only_mismatch",
                    )
                },
            }
        )
    return sorted(
        panels,
        key=lambda panel: (
            cipher_order[str(panel["cipher_key"])],
            int(panel["replica"]),
            split_order[str(panel["split"])],
        ),
    )


def _render_cross_key_anchor(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["cross_key_label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    height = 0.34
    candidate = np.asarray([float(panel["correct_auc"]) for panel in panels])
    anchors = np.asarray([float(panel["anchor_auc"]) for panel in panels])
    axis.barh(y - height / 2, candidate, height, label="K1-AW双通道", color="#0F766E")
    axis.barh(y + height / 2, anchors, height, label="K1-AT单通道", color="#94A3B8")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.58, 1.01)
    axis.set_xlabel("跨密钥 AUC（从0.58开始，仅用于展开差异）")
    axis.set_title("三密码结果：提升主要来自 uKNIT", loc="left", fontweight="bold")
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
    values = np.asarray([float(macro[f"replica{index}"]["improvement"]) for index in range(2)])
    positions = np.arange(2)
    axis.bar(positions, values, width=0.56, color=["#0F766E", "#2563EB"])
    axis.axhline(0.0, color="#64748B", linewidth=1.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("K1-AW 跨密钥宏平均 AUC - K1-AT")
    axis.set_title("宏平均保留门：两次独立训练均通过", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(index, value + 0.00045, f"{value:+.4f}", ha="center", fontweight="bold")
    axis.set_ylim(-0.0015, max(values) + 0.0032)


def _render_no_harm_margins(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    values = np.asarray([float(panel["anchor_margin"]) for panel in panels])
    y = np.arange(len(panels), dtype=float)
    colors = np.where(values >= -0.005, "#0F766E", "#DC2626")
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(-0.005, color="#B45309", linestyle="--", linewidth=1.4, label="无伤害线 -0.005")
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    padding = max(0.004, (float(values.max()) - float(values.min())) * 0.08)
    axis.set_xlim(float(values.min()) - padding, float(values.max()) + padding)
    axis.set_xlabel("正确结构 AUC - K1-AT AUC")
    axis.set_title("逐面板无伤害门：Dialga副本0两项越线", loc="left", fontweight="bold")
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
        values = [float(panel["mismatch_margins"][condition]) for panel in panels]
        all_values.extend(values)
        axis.scatter(values, y + offset, label=label, color=color, marker=marker, s=36, zorder=3)
    axis.axvline(0.001, color="#7C3AED", linestyle="--", linewidth=1.4, label="通过线 +0.001")
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    span = max(all_values) - min(all_values)
    padding = max(0.0002, span * 0.08)
    axis.set_xlim(min(all_values) - padding, max(max(all_values) + padding, 0.00125))
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_xlabel("正确结构相对错误结构的 AUC 优势")
    axis.set_title("结构语义门：多数差值未达到千分位", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=8.0,
        ncol=2,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1aw_svg"]
