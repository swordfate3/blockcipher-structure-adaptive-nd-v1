from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC",
    "midori64": "Midori64",
    "dialga128": "Dialga-128",
}
SHORT_SPLITS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AQ inverse-norm result chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1aq_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1aq_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = _ordered_panels(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.3,
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
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 11.0))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.78,
            bottom=0.14,
            hspace=0.48,
            wspace=0.31,
        )
        figure.suptitle(
            "创新1 K1-AQ：固定逆范数缩放能否纠正三密码共享训练失衡",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.920,
            (
                "本地同预算诊断：每种密码2048/class、4 pairs、10 epochs、每副本1920个Adam步骤；"
                "只改变三密码固定损失系数。"
            ),
            ha="left",
            fontsize=11.1,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.868,
            (
                "结论：Midori 四个面板全部提升，但 uKNIT 与 Dialga 明显受损；"
                "正确 S盒和结构分支均为12/12，仍未通过逐密码不伤害门。"
            ),
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.821,
            "裁决：固定逆范数缩放暂缓并停止调系数；不增加 pairs/样本/epoch，不远程放大。",
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )

        _render_cross_key_auc(axes[0, 0], panels)
        _render_matched_deltas(axes[0, 1], panels)
        _render_semantic_margins(axes[1, 0], panels)
        _render_gate_counts(axes[1, 1], gate)

        figure.text(
            0.05,
            0.052,
            (
                "下一步：停止优化器权重路线，回到结构表示；优先解释为何 Midori 需要更强语义分支，"
                "而同一增强会破坏 uKNIT/Dialga 的共享基路径。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "本图是2048/class/cipher本地诊断，不是正式规模、攻击轮数或主流方法准确率对比。",
            ha="left",
            fontsize=10.1,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 11.0,
        "language": "zh-CN",
        "panels": 4,
        "comparison_panels": len(panels),
        "formal_scale_claim_present": False,
        "status_from_gate": gate.get("status"),
        "decision": gate.get("decision"),
    }


def _ordered_panels(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, values in gate.get("panel_results", {}).items():
        parts = key.split("_", 2)
        replica = int(parts[0].removeprefix("replica"))
        cipher = parts[1]
        split = parts[2]
        rows.append(
            {
                "replica": replica,
                "cipher": cipher,
                "split": split,
                "label": f"{CIPHER_LABELS[cipher]} R{replica} · {SHORT_SPLITS[split]}",
                **values,
            }
        )
    cipher_order = {cipher: index for index, cipher in enumerate(CIPHER_LABELS)}
    split_order = {split: index for index, split in enumerate(SHORT_SPLITS)}
    rows.sort(
        key=lambda row: (
            cipher_order[row["cipher"]],
            row["replica"],
            split_order[row["split"]],
        )
    )
    if len(rows) != 12:
        raise ValueError("K1-AQ plot requires exactly 12 comparison panels")
    return rows


def _cross_key_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [panel for panel in panels if panel["split"] == "cross_key_validation"]


def _render_cross_key_auc(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    rows = _cross_key_panels(panels)
    labels = [f"{CIPHER_LABELS[row['cipher']]} · R{row['replica']}" for row in rows]
    baseline = np.asarray([float(row["baseline_correct_auc"]) for row in rows])
    candidate = np.asarray([float(row["candidate_correct_auc"]) for row in rows])
    y = np.arange(len(rows), dtype=float)
    height = 0.34
    axis.barh(y - height / 2, baseline, height, color="#94A3B8", label="K1-AO 基线")
    axis.barh(y + height / 2, candidate, height, color="#0F766E", label="K1-AQ 逆范数")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.45, 1.02)
    axis.set_xlabel("跨密钥 AUC（从0.45开始，仅用于展开差异）")
    axis.set_title("跨密钥强度：Midori 上升，另外两者下降", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)


def _render_matched_deltas(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in panels]
    values = np.asarray([float(row["candidate_minus_baseline"]) for row in panels])
    colors = np.where(values >= 0.010, "#0F766E", np.where(values >= -0.010, "#2563EB", "#DC2626"))
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(-0.010, color="#B45309", linestyle="--", linewidth=1.3)
    axis.axvline(0.010, color="#166534", linestyle="--", linewidth=1.3)
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(-0.085, 0.09)
    axis.set_xlabel("K1-AQ 正确结构 AUC - K1-AO 正确结构 AUC")
    axis.set_title("逐面板变化：只改善 Midori，红色表示受损超门", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        -0.012,
        -0.65,
        "不伤害线 -0.010",
        color="#92400E",
        fontsize=8.8,
        ha="right",
    )
    axis.text(
        0.012,
        -0.65,
        "改善线 +0.010",
        color="#166534",
        fontsize=8.8,
        ha="left",
    )


def _render_semantic_margins(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in panels]
    wrong = np.asarray([float(row["correct_minus_wrong_sbox"]) for row in panels])
    branch = np.asarray([float(row["correct_minus_branch_off"]) for row in panels])
    y = np.arange(len(panels), dtype=float)
    axis.scatter(wrong, y - 0.13, color="#DC2626", s=38, label="正确 - 错误 S盒")
    axis.scatter(branch, y + 0.13, marker="s", color="#2563EB", s=34, label="正确 - 关闭分支")
    axis.axvline(0.005, color="#B45309", linestyle="--", linewidth=1.3)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.19)
    axis.set_xlabel("同一 K1-AQ 检查点的正确结构 AUC 优势")
    axis.set_title("结构语义：错误 S盒和关闭分支均为12/12", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)
    axis.text(0.006, -0.65, "门槛 +0.005", color="#92400E", fontsize=8.8)


def _render_gate_counts(axis: plt.Axes, gate: Mapping[str, Any]) -> None:
    labels = ("目标改善", "逐面板不伤害", "正确 S盒", "结构分支", "独立锚点保留")
    counts = np.asarray(
        [
            int(gate["target_improved_count"]) / 8,
            int(gate["no_harm_count"]) / 12,
            int(gate["semantic_pass_count"]) / 12,
            int(gate["branch_pass_count"]) / 12,
            int(gate["retention_pass_count"]) / 12,
        ]
    )
    requirements = np.asarray((6 / 8, 1.0, 1.0, 11 / 12, 1.0))
    raw_labels = (
        f"{gate['target_improved_count']}/8",
        f"{gate['no_harm_count']}/12",
        f"{gate['semantic_pass_count']}/12",
        f"{gate['branch_pass_count']}/12",
        f"{gate['retention_pass_count']}/12",
    )
    colors = np.where(counts >= requirements, "#0F766E", "#DC2626")
    y = np.arange(len(counts), dtype=float)
    axis.barh(y, counts, height=0.58, color=colors)
    axis.scatter(requirements, y, marker="|", s=220, linewidths=2.5, color="#111827")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.12)
    axis.set_xlabel("通过比例（黑色竖线是各项要求）")
    axis.set_title("裁决门：语义通过，但改善与不伤害失败", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        1.11,
        -0.58,
        "黑色竖线：预注册要求",
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="#111827",
    )
    for index, (value, label) in enumerate(zip(counts, raw_labels, strict=True)):
        axis.text(value + 0.018, y[index], label, va="center", fontsize=9.2)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1aq_svg"]
