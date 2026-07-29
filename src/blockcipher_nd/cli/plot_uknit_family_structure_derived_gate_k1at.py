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
    "descriptor_disabled",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AT structure-gated training chart."
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
    report = render_k1at_svg(gate, rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1at_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    panels = _collect_panels(rows)
    cross_key = [panel for panel in panels if panel["split"] == "cross_key_validation"]
    mismatch = gate.get("mismatch_results", {})
    macro = gate.get("macro_results", {})
    held = gate.get("status") != "pass"
    harm_panels = [panel for panel in panels if float(panel["anchor_margin"]) < -0.005]
    harm_ciphers = sorted(
        {SHORT_CIPHER_LABELS[str(panel["cipher_key"])] for panel in harm_panels}
    )
    harm_scope = ", ".join(harm_ciphers) if harm_ciphers else "无"
    macro0 = macro["replica0"]
    macro1 = macro["replica1"]
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
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 11.5))
        figure.subplots_adjust(
            left=0.085,
            right=0.97,
            top=0.76,
            bottom=0.135,
            hspace=0.57,
            wspace=0.32,
        )
        figure.suptitle(
            (
                "创新1 K1-AT：运行时结构门控没有稳定改善三密码共享模型"
                if held
                else "创新1 K1-AT：运行时结构门控通过本地同预算诊断"
            ),
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
            fontsize=10.8,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.877,
            (
                f"宏平均：副本0 {macro0['improvement']:+.4f}"
                f"（{'通过' if macro0['pass'] else '未通过'}），副本1 "
                f"{macro1['improvement']:+.4f}"
                f"（{'通过' if macro1['pass'] else '未通过'}）；"
                f"{harm_scope}共有{len(harm_panels)}个面板低于无伤害线。"
            ),
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.835,
            (
                "正确描述符超过 +0.001 的面板：完整错配 "
                f"{mismatch['full_mismatch']['passing_panels']}/12，S盒错配 "
                f"{mismatch['sbox_only_mismatch']['passing_panels']}/12，线性层错配 "
                f"{mismatch['linear_only_mismatch']['passing_panels']}/12。"
            ),
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )
        figure.text(
            0.05,
            0.795,
            (
                "裁决：当前34维结构汇总门控暂缓；这不是正式规模、攻击结果或任意SPN泛化证据。"
                if held
                else "裁决：通过本地诊断，仅开放远程准备审计；这不是正式规模或攻击结果。"
            ),
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_cross_key_anchor(axes[0, 0], cross_key)
        _render_effective_gates(axes[0, 1], cross_key)
        _render_no_harm_margins(axes[1, 0], panels)
        _render_mismatch_margins(axes[1, 1], panels)

        figure.text(
            0.05,
            0.055,
            (
                "下一步 K1-AU：冻结两个检查点做0次更新审计，检查结构汇总经过共享门控后是否"
                "只退化为一个近似标量，以及S盒/GF(2)分量的可识别性。"
                if held
                else "下一步：先完成65536/class/cipher远程缓存与恢复协议检查，再决定是否启动。"
            ),
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.023,
            "不增加pairs、样本、epoch、宽度或seed；不使用损失重加权、专家、MoE或远程GPU。",
            ha="left",
            fontsize=10.0,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)

    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 11.5,
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
        raise ValueError("K1-AT plot requires 60 rows in 12 panels")
    if any(set(conditions) != CONDITIONS for conditions in grouped.values()):
        raise ValueError("K1-AT plot has an incomplete condition panel")

    panels = []
    cipher_order = {key: index for index, key in enumerate(CIPHER_LABELS)}
    split_order = {"same_key_fresh": 0, "cross_key_validation": 1}
    for (replica, cipher, split), conditions in grouped.items():
        correct = conditions["correct_descriptor"]
        correct_auc = float(correct["auc"])
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
                "anchor_auc": float(correct["k1ao_anchor_auc"]),
                "anchor_margin": correct_auc - float(correct["k1ao_anchor_auc"]),
                "gate_values": {
                    condition: float(row["effective_transition_gate"])
                    for condition, row in conditions.items()
                },
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
    axis.barh(y - height / 2, candidate, height, label="K1-AT结构门控", color="#0F766E")
    axis.barh(y + height / 2, anchors, height, label="K1-AO同预算锚点", color="#94A3B8")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.56, 1.02)
    axis.set_xlabel("跨密钥 AUC（从0.56开始，仅用于展开差异）")
    axis.set_title("共享模型强度：候选与同预算锚点", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False, fontsize=9.0)
    for index, value in enumerate(candidate):
        axis.text(
            value + 0.006,
            y[index] - height / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.4,
        )


def _render_effective_gates(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    unique = [panel for panel in panels if panel["split"] == "cross_key_validation"]
    labels = [str(panel["cross_key_label"]) for panel in unique]
    y = np.arange(len(unique), dtype=float)
    styles = (
        ("correct_descriptor", "正确描述符", "#0F766E", "o", -0.20),
        ("full_mismatch", "完整错配", "#DC2626", "x", -0.10),
        ("sbox_only_mismatch", "S盒错配", "#B45309", "^", 0.0),
        ("linear_only_mismatch", "线性层错配", "#2563EB", "s", 0.10),
        ("descriptor_disabled", "关闭结构汇总", "#6B7280", "D", 0.20),
    )
    for condition, label, color, marker, offset in styles:
        axis.scatter(
            [float(panel["gate_values"][condition]) for panel in unique],
            y + offset,
            label=label,
            color=color,
            marker=marker,
            s=42,
            zorder=3,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.05, 0.225)
    axis.set_xlabel("有效转移门控值（同一检查点）")
    axis.set_title(
        "门控响应：描述符确实改变标量，但幅度接近", loc="left", fontweight="bold"
    )
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=8.4,
        ncol=3,
    )


def _render_no_harm_margins(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    values = np.asarray([float(panel["anchor_margin"]) for panel in panels])
    y = np.arange(len(panels), dtype=float)
    colors = np.where(values >= -0.005, "#0F766E", "#DC2626")
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(
        -0.005, color="#B45309", linestyle="--", linewidth=1.4, label="无伤害线 -0.005"
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    axis.set_xlim(-0.023, 0.037)
    axis.set_xlabel("K1-AT正确描述符 AUC - K1-AO AUC")
    harm_count = int(np.sum(values < -0.005))
    axis.set_title(
        f"逐面板无伤害门：{harm_count}项低于门槛",
        loc="left",
        fontweight="bold",
    )
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
    for condition, label, color, marker, offset in styles:
        axis.scatter(
            [float(panel["mismatch_margins"][condition]) for panel in panels],
            y + offset,
            label=label,
            color=color,
            marker=marker,
            s=35,
            zorder=3,
        )
    axis.axvline(
        0.001, color="#7C3AED", linestyle="--", linewidth=1.4, label="通过线 +0.001"
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.2)
    axis.invert_yaxis()
    axis.set_xlim(-0.00045, 0.0030)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.set_xlabel("正确描述符相对错配描述符的 AUC 优势")
    axis.set_title("结构语义门：多数差值停留在万分位", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="center right", frameon=False, fontsize=8.1, ncol=1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1at_svg"]
