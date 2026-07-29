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
CONTROL_LABELS = {
    "full_mismatch": "完整描述错配",
    "sbox_only_mismatch": "仅 S盒错配",
    "linear_only_mismatch": "仅线性层错配",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AS structure-derived gate chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--summaries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    report = render_k1as_svg(gate, rows, summaries, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1as_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if len(rows) != 24:
        raise ValueError("K1-AS plot requires exactly 24 readiness panels")
    summary_rows = summaries.get("rows", [])
    if len(summary_rows) != 3:
        raise ValueError("K1-AS plot requires three structure summaries")
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
            left=0.075,
            right=0.97,
            top=0.79,
            bottom=0.14,
            hspace=0.50,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-AS：运行时结构能否安全控制同一条 SPN 转移分支",
            x=0.045,
            y=0.975,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        figure.text(
            0.045,
            0.922,
            (
                "零训练 readiness：复用 K1-AO/K1-AQ 四个检查点；"
                "24个冻结面板，每面板检查32行；主干、分类头和数据均不改变。"
            ),
            ha="left",
            fontsize=11.0,
            color="#4B5563",
        )
        figure.text(
            0.045,
            0.870,
            (
                "结论：34维 S盒/GF(2) 摘要可驱动一套共享有界门控；"
                "关闭描述时旧 K1-AK logits 逐元素完全重放。"
            ),
            ha="left",
            fontsize=11.4,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.045,
            0.827,
            (
                "裁决：K1-AS readiness 通过，只开放本地 K1-AT 同预算训练；"
                "不开放16 pairs、远程放大、密码ID或专家模型。"
            ),
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )

        _render_summary_heatmap(axes[0, 0], summary_rows)
        _render_gate_offsets(axes[0, 1], rows)
        _render_minimum_gate_deltas(axes[1, 0], rows)
        _render_minimum_logit_deltas(axes[1, 1], rows)

        figure.text(
            0.045,
            0.052,
            (
                "下一步：K1-AT保持2048/class/cipher、4 pairs、10 epochs、replica0/1，"
                "仅比较结构门控与 K1-AO 等权共享锚点。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.045,
            0.022,
            "本图只证明实现与反事实控制可用，不包含新训练AUC，不是正式规模、攻击或主流方法对比。",
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
        "readiness_panels": len(rows),
        "formal_scale_claim_present": False,
        "status_from_gate": gate.get("status"),
        "decision": gate.get("decision"),
    }


def _render_summary_heatmap(axis: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    row_map = {str(row["cipher_key"]): row for row in rows}
    values = np.asarray(
        [row_map[cipher]["summary"] for cipher in CIPHER_LABELS], dtype=float
    )
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.axvline(15.5, color="#FFFFFF", linewidth=2.0)
    axis.set_yticks(np.arange(3), [CIPHER_LABELS[cipher] for cipher in CIPHER_LABELS])
    axis.set_xticks((0, 7, 15, 23, 33), ("1", "8", "16", "24", "34"))
    axis.set_xlabel("固定宽度结构摘要维度（1-16：S盒；17-34：GF(2)线性层）")
    axis.set_title("三种密码使用同一34维结构接口", loc="left", fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.028, pad=0.025)
    colorbar.set_label("归一化统计值")


def _render_gate_offsets(axis: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    first = {
        cipher: next(row for row in rows if row["cipher_key"] == cipher)
        for cipher in CIPHER_LABELS
    }
    controls = tuple(CONTROL_LABELS)
    x = np.arange(len(controls), dtype=float)
    width = 0.23
    colors = ("#0F766E", "#2563EB", "#B45309")
    for index, (cipher, color) in enumerate(zip(CIPHER_LABELS, colors, strict=True)):
        gate_values = first[cipher]["gate_values"]
        correct = float(gate_values["correct_descriptor"])
        offsets = [1000.0 * (float(gate_values[control]) - correct) for control in controls]
        axis.bar(
            x + (index - 1) * width,
            offsets,
            width=width,
            color=color,
            label=CIPHER_LABELS[cipher],
        )
    axis.axhline(0.0, color="#64748B", linewidth=1.0)
    axis.set_xticks(x, [CONTROL_LABELS[control] for control in controls])
    axis.set_ylabel("相对正确描述的门值变化 × 10⁻³")
    axis.set_title("错配描述改变同一个共享门，不切换专家", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncols=3,
        fontsize=8.8,
    )


def _render_minimum_gate_deltas(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    keys = (
        "full_mismatch_gate_delta",
        "sbox_only_mismatch_gate_delta",
        "linear_only_mismatch_gate_delta",
    )
    labels = tuple(CONTROL_LABELS.values())
    values = np.asarray([min(float(row[key]) for row in rows) for key in keys])
    bars = axis.bar(labels, values, color=("#0F766E", "#2563EB", "#B45309"))
    axis.axhline(1e-6, color="#991B1B", linestyle="--", linewidth=1.4, label="门槛 1×10⁻⁶")
    axis.set_yscale("log")
    axis.set_ylim(5e-7, max(values) * 3.5)
    axis.set_ylabel("24个面板中的最小绝对门值差（对数刻度）")
    axis.set_title("最弱门值差也超过预注册阈值", loc="left", fontweight="bold")
    axis.grid(axis="y", which="both", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.18,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )


def _render_minimum_logit_deltas(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    keys = (
        "full_mismatch_max_abs_logit_delta",
        "sbox_only_mismatch_max_abs_logit_delta",
        "linear_only_mismatch_max_abs_logit_delta",
    )
    labels = tuple(CONTROL_LABELS.values())
    values = np.asarray([min(float(row[key]) for row in rows) for key in keys])
    bars = axis.bar(labels, values, color=("#0F766E", "#2563EB", "#B45309"))
    axis.axhline(1e-8, color="#991B1B", linestyle="--", linewidth=1.4, label="门槛 1×10⁻⁸")
    axis.set_yscale("log")
    axis.set_ylim(3e-9, max(values) * 4.0)
    axis.set_ylabel("24个面板中的最小 logits 变化（对数刻度）")
    axis.set_title("结构错配能实际改变前向输出", loc="left", fontweight="bold")
    axis.grid(axis="y", which="both", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=False)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.22,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1as_svg"]
