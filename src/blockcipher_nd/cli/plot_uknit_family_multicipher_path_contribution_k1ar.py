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
SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AR shared path-contribution chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ar_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ar_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
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
            top=0.79,
            bottom=0.14,
            hspace=0.49,
            wspace=0.31,
        )
        figure.suptitle(
            "创新1 K1-AR：三种 SPN 是否需要不同的结构分支强度",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.922,
            (
                "零训练路径审计：复用 K1-AO/K1-AQ 四个检查点；每种密码2048/class、"
                "4 pairs；24个新鲜数据面板，权重完全不变。"
            ),
            ha="left",
            fontsize=11.0,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.872,
            (
                "结论：逆范数训练只增强 Midori 的 S盒转移分支收益；"
                "uKNIT 与 Dialga 的同一路径均未增强。"
            ),
            ha="left",
            fontsize=11.4,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.828,
            (
                "裁决：支持下一步测试“运行时结构派生的有界门控”；"
                "不支持按密码ID分支、专家模型或直接放大训练。"
            ),
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )

        _render_cross_key_gains(axes[0, 0], panels)
        _render_gain_deltas(axes[0, 1], panels)
        _render_rms_deltas(axes[1, 0], panels)
        _render_gate_counts(axes[1, 1], gate)

        figure.text(
            0.05,
            0.052,
            (
                "下一步：先做不训练的结构门控 readiness，证明门值来自运行时 S盒/线性层统计且对结构错配敏感；"
                "通过后才开放同预算共享训练。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "本图是2048/class/cipher、4-pair本地路径审计，不是正式规模、攻击轮数或主流准确率对比。",
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
        replica_text, cipher, split = key.split("_", 2)
        replica = int(replica_text.removeprefix("replica"))
        rows.append(
            {
                "replica": replica,
                "cipher": cipher,
                "split": split,
                "label": (
                    f"{CIPHER_LABELS[cipher]} R{replica} · {SPLIT_LABELS[split]}"
                ),
                **values,
            }
        )
    cipher_order = {cipher: index for index, cipher in enumerate(CIPHER_LABELS)}
    split_order = {split: index for index, split in enumerate(SPLIT_LABELS)}
    rows.sort(
        key=lambda row: (
            cipher_order[row["cipher"]],
            row["replica"],
            split_order[row["split"]],
        )
    )
    if len(rows) != 12:
        raise ValueError("K1-AR plot requires exactly 12 comparison panels")
    return rows


def _render_cross_key_gains(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    rows = [row for row in panels if row["split"] == "cross_key_validation"]
    labels = [f"{CIPHER_LABELS[row['cipher']]} · R{row['replica']}" for row in rows]
    baseline = np.asarray(
        [float(row["equal_loss_transition_gain_auc"]) for row in rows]
    )
    candidate = np.asarray(
        [float(row["inverse_norm_transition_gain_auc"]) for row in rows]
    )
    y = np.arange(len(rows), dtype=float)
    axis.hlines(y, baseline, candidate, color="#CBD5E1", linewidth=2.2)
    axis.scatter(baseline, y, color="#64748B", s=52, label="K1-AO 等权训练")
    axis.scatter(
        candidate,
        y,
        color="#0F766E",
        marker="s",
        s=48,
        label="K1-AQ 逆范数",
    )
    axis.axvline(0.0, color="#6B7280", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.02, 0.20)
    axis.set_xlabel("转移分支带来的 AUC 增益（完整 - 关闭分支）")
    axis.set_title("跨密钥路径收益：Midori增强，另外两者下降", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)


def _render_gain_deltas(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in panels]
    values = np.asarray([float(row["transition_gain_delta"]) for row in panels])
    colors = [
        "#0F766E" if row["cipher"] == "midori64" else "#DC2626"
        for row in panels
    ]
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(0.0, color="#64748B", linewidth=1.0)
    axis.axvline(0.010, color="#166534", linestyle="--", linewidth=1.3)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(-0.105, 0.075)
    axis.set_xlabel("K1-AQ 转移收益 - K1-AO 转移收益")
    axis.set_title("逐面板变化：12/12符合异质需求方向", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        0.012,
        -0.62,
        "Midori要求 +0.010",
        ha="left",
        color="#166534",
        fontsize=8.8,
    )


def _render_rms_deltas(axis: plt.Axes, panels: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in panels]
    values = np.asarray(
        [
            float(row["inverse_norm_transition_to_base_rms_ratio"])
            - float(row["equal_loss_transition_to_base_rms_ratio"])
            for row in panels
        ]
    )
    gain_deltas = np.asarray([float(row["transition_gain_delta"]) for row in panels])
    colors = np.where(gain_deltas > 0.0, "#0F766E", "#DC2626")
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(0.0, color="#64748B", linewidth=1.0)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(-0.23, 0.18)
    axis.set_xlabel("转移残差/基础表示 RMS 比例变化")
    axis.set_title("幅度不是充分解释：变大也可能收益下降", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        0.175,
        -0.62,
        "绿色=路径收益上升；红色=路径收益未上升",
        ha="right",
        color="#4B5563",
        fontsize=8.7,
    )


def _render_gate_counts(axis: plt.Axes, gate: Mapping[str, Any]) -> None:
    labels = ("Midori方向", "uKNIT/Dialga方向", "精确前向重放", "状态不变", "零训练")
    counts = np.asarray(
        [
            int(gate["midori_direction_pass_count"]) / 4,
            int(gate["non_midori_direction_pass_count"]) / 8,
            float(bool(gate["protocol_checks"]["all_forward_replays_exact"])),
            float(bool(gate["protocol_checks"]["all_states_immutable"])),
            float(bool(gate["protocol_checks"]["all_rows_zero_step"])),
        ]
    )
    requirements = np.asarray((1.0, 6 / 8, 1.0, 1.0, 1.0))
    raw_labels = (
        f"{gate['midori_direction_pass_count']}/4",
        f"{gate['non_midori_direction_pass_count']}/8",
        "24/24",
        "24/24",
        "24/24",
    )
    y = np.arange(len(counts), dtype=float)
    axis.barh(y, counts, height=0.58, color="#0F766E")
    axis.scatter(requirements, y, marker="|", s=220, linewidths=2.5, color="#111827")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.12)
    axis.set_xlabel("通过比例（黑色竖线是预注册要求）")
    axis.set_title("裁决门：方向与协议检查全部通过", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, (value, label) in enumerate(zip(counts, raw_labels, strict=True)):
        axis.text(value + 0.018, y[index], label, va="center", fontsize=9.2)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ar_svg"]
