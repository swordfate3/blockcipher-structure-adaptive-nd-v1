from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY5 affine endpoint control audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by5_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by5_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = sorted(
        gate.get("panels", []),
        key=lambda row: (
            int(row["seed"]),
            int(row["execution_step"]),
            str(row["tap"]),
        ),
    )
    if len(panels) != 8:
        raise ValueError("K1-BY5 plot requires eight seed/stage/tap panels")
    labels = [_panel_label(row) for row in panels]
    multiset_change = np.asarray(
        [float(row["multiset_change_rate"]) for row in panels]
    )
    pooled_l1 = np.asarray([float(row["pooled_summary_l1"]) for row in panels])
    ordered_l1 = np.asarray(
        [float(row["ordered_histogram_l1"]) for row in panels]
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(17.2, 8.8))
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.69,
            bottom=0.22,
            wspace=0.25,
        )
        figure.suptitle(
            "创新1 K1-BY5：PRESENT 全局仿射端点控制审计",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "不训练网络；用 u → (5u+1) mod 64 拆散每个 source cell，检查错误 P 层是否在每个阶段都可见。",
            ha="left",
            fontsize=11.2,
        )
        figure.text(
            0.05,
            0.835,
            "数据、seed、16 pairs、两阶段逆执行和门槛全部继承 K1-BY4；三幅图分别回答无位置、有位置和汇总差异。",
            ha="left",
            fontsize=10.4,
            color="#4B5563",
        )
        _bar_panel(
            axes[0],
            multiset_change,
            labels,
            title="去掉 cell 顺序后\n样本多集合变化率",
            ylabel="1 - 完全相等率",
            threshold=0.05,
            fmt="{:.3f}",
            color="#0F766E",
        )
        _bar_panel(
            axes[1],
            pooled_l1,
            labels,
            title="mean/max 无位置汇总\n归一化差异",
            ylabel="汇总 L1",
            threshold=0.0001,
            fmt="{:.5f}",
            color="#2563EB",
        )
        _bar_panel(
            axes[2],
            ordered_l1,
            labels,
            title="保留 cell 位置时\n直方图差异",
            ylabel="有序直方图 L1",
            threshold=None,
            fmt="{:.3f}",
            color="#7C3AED",
        )
        figure.text(
            0.05,
            0.075,
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
        "experiment": "K1-BY5",
        "panels": 3,
        "groups": 8,
        "status": gate.get("status"),
    }


def _bar_panel(
    axis: Any,
    values: np.ndarray,
    labels: list[str],
    *,
    title: str,
    ylabel: str,
    threshold: float | None,
    fmt: str,
    color: str,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    bars = axis.bar(positions, values, width=0.68, color=color)
    axis.bar_label(
        bars,
        labels=[fmt.format(value) for value in values],
        padding=3,
        fontsize=8.0,
        rotation=90 if float(values.max()) < 0.01 else 0,
    )
    if threshold is not None:
        axis.axhline(threshold, color="#B45309", linewidth=1.0, linestyle="--")
    upper = max(
        (threshold or 0.0) * 4.0,
        float(values.max()) * 1.30,
        0.001,
    )
    axis.set_ylim(0.0, upper)
    axis.set_xticks(positions, labels, rotation=55, ha="right")
    axis.tick_params(axis="x", labelsize=8.0)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=12.0, pad=10)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _panel_label(row: Mapping[str, Any]) -> str:
    tap = "逆线性" if row["tap"] == "inverse_linear" else "逆S盒后"
    return f"seed{row['seed']}·第{int(row['execution_step']) + 1}步·{tap}"


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by5_affine_endpoint_control_ready": (
            "裁决：仿射端点控制在两颗 seed、两个阶段和两个 tap 上全部可识别；允许进入同预算神经归因。"
        ),
        "innovation1_runtime_spn_k1by5_affine_endpoint_control_not_identifiable": (
            "裁决：仿射端点控制仍有不可识别 tap；停止继续搜索置换控制和扩大训练。"
        ),
        "innovation1_runtime_spn_k1by5_protocol_invalid": (
            "裁决：冻结来源、端点双射、程序几何、缓存或产物不完整，本次数值不可解释。"
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


__all__ = ["main", "render_k1by5_svg"]
