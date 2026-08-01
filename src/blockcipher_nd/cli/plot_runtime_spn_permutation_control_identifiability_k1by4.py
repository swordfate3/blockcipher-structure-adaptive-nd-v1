from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


CONTROL_LABELS = (
    ("current", "现有错误目标绑定"),
    ("source_role", "cell 内 source-role 错接"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY4 permutation-control audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by4_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by4_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = list(gate.get("panels", []))
    if len(panels) != 8:
        raise ValueError("K1-BY4 plot requires eight seed/stage/tap panels")
    panels.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["execution_step"]),
            str(row["tap"]),
        )
    )
    labels = [_panel_label(row) for row in panels]
    change_rates = np.asarray(
        [
            [
                1.0 - float(row["current_multiset_equal_rate"]),
                1.0 - float(row["source_role_multiset_equal_rate"]),
            ]
            for row in panels
        ]
    )
    pooled_l1 = np.asarray(
        [
            [
                float(row["current_pooled_summary_l1"]),
                float(row["source_role_pooled_summary_l1"]),
            ]
            for row in panels
        ]
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 1, figsize=(16.8, 10.8))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.76,
            bottom=0.14,
            hspace=0.50,
        )
        figure.suptitle(
            "创新1 K1-BY4：PRESENT 置换控制可识别性审计",
            x=0.05,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.915,
            "不训练网络；读取 K1-BY3 冻结验证数据，检查错误结构在无位置池化前是否真的可区分。",
            ha="left",
            fontsize=11.3,
        )
        figure.text(
            0.05,
            0.865,
            "每个柱组依次表示 seed、逆执行阶段和观测点；越高表示错误控制越容易被结构表示发现。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _grouped_panel(
            axes[0],
            change_rates,
            labels,
            title="去掉 cell 顺序后，有多少样本的直方图多集合发生变化",
            ylabel="多集合变化率（1 - 完全相等率）",
            threshold=0.05,
            value_format="{:.3f}",
        )
        _grouped_panel(
            axes[1],
            pooled_l1,
            labels,
            title="K1-BY3 同类 mean/max 无位置汇总能看到多大差异",
            ylabel="归一化汇总 L1",
            threshold=0.0001,
            value_format="{:.5f}",
        )
        figure.text(
            0.05,
            0.055,
            _decision_text(gate),
            ha="left",
            fontsize=10.8,
            color=_decision_color(str(gate.get("status", ""))),
            fontweight="bold",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "run_id": gate.get("run_id"),
        "experiment": "K1-BY4",
        "panels": 2,
        "groups": 8,
        "controls": [name for name, _label in CONTROL_LABELS],
        "status": gate.get("status"),
    }


def _grouped_panel(
    axis: Any,
    values: np.ndarray,
    labels: list[str],
    *,
    title: str,
    ylabel: str,
    threshold: float,
    value_format: str,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = ("#64748B", "#0F766E")
    for index, (_name, control_label) in enumerate(CONTROL_LABELS):
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values[:, index],
            width,
            label=control_label,
            color=colors[index],
        )
        axis.bar_label(
            bars,
            labels=[value_format.format(value) for value in values[:, index]],
            padding=3,
            fontsize=8.2,
            rotation=90 if float(values.max()) < 0.01 else 0,
        )
    upper = max(threshold * 4.0, float(values.max()) * 1.28)
    axis.set_ylim(0.0, upper)
    axis.axhline(threshold, color="#B45309", linewidth=1.0, linestyle="--")
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=8.8)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=12.3, pad=10)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _panel_label(row: Mapping[str, Any]) -> str:
    tap = "逆线性" if row["tap"] == "inverse_linear" else "逆S盒后"
    return f"seed{row['seed']}\n第{int(row['execution_step']) + 1}步·{tap}"


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by4_learned_pooling_audit_required": (
            "裁决：现有错误绑定在确定性表示中已可识别；下一步定位学习网络在哪个池化阶段丢失差异。"
        ),
        "innovation1_runtime_spn_k1by4_source_role_control_preferred": (
            "裁决：完整 cell 搬移是弱控制；cell 内 source-role 错接更适合作为下一次同预算神经归因控制。"
        ),
        "innovation1_runtime_spn_k1by4_permutation_expert_hold": (
            "裁决：两种控制仍不足以稳定检验置换语义；暂停神经训练和远程扩样。"
        ),
        "innovation1_runtime_spn_k1by4_protocol_invalid": (
            "裁决：冻结来源、程序几何、直方图或产物协议不完整，本次数值不可解释。"
        ),
    }
    return labels.get(str(gate.get("decision", "")), str(gate.get("decision", "")))


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by4_svg"]
