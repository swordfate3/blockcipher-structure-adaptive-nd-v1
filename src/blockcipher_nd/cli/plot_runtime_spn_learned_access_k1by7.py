from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_learned_access_audit_k1by7 import (
    TAPS,
)


TAP_LABELS = (
    "线性直方图",
    "置换专家输出",
    "单元融合",
    "阶段池化摘要",
    "分类前表示",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY7 learned-access audit."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by7_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by7_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = ("2", "3")
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != set(seeds):
        raise ValueError("K1-BY7 plot requires seed2 and seed3")
    correct = np.asarray(
        [
            [float(seed_results[seed]["taps"][tap]["correct_probe_auc"]) for tap in TAPS]
            for seed in seeds
        ]
    )
    affine = np.asarray(
        [
            [float(seed_results[seed]["taps"][tap]["affine_probe_auc"]) for tap in TAPS]
            for seed in seeds
        ]
    )
    margins = correct - affine

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(17.0, 9.2))
        figure.subplots_adjust(
            left=0.07,
            right=0.97,
            top=0.70,
            bottom=0.19,
            wspace=0.24,
        )
        figure.suptitle(
            "创新1 K1-BY7：正确 P 层语义在哪一步丢失",
            x=0.04,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.04,
            0.895,
            "冻结 K1-BY3/K1-BY6 检查点；偶数行确定均值差方向，奇数行独立计算内部探针 AUC。",
            ha="left",
            fontsize=11.3,
        )
        figure.text(
            0.04,
            0.842,
            "不重新训练网络；比较正确 PRESENT 程序与仿射错误端点程序在五个内部表示中的标签关联。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _auc_panel(axes[0], correct, affine, seeds)
        _margin_panel(axes[1], margins, seeds)
        figure.text(
            0.04,
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
        "panels": 2,
        "taps": list(TAPS),
        "seeds": [2, 3],
        "status": gate.get("status"),
    }


def _auc_panel(
    axis: Any,
    correct: np.ndarray,
    affine: np.ndarray,
    seeds: tuple[str, ...],
) -> None:
    positions = np.arange(len(TAPS), dtype=float)
    colors = ("#2563EB", "#D97706")
    for index, seed in enumerate(seeds):
        axis.plot(
            positions,
            correct[index],
            marker="o",
            linewidth=2.2,
            color=colors[index],
            label=f"seed{seed} 正确结构",
        )
        axis.plot(
            positions,
            affine[index],
            marker="s",
            linewidth=1.8,
            linestyle="--",
            color=colors[index],
            alpha=0.72,
            label=f"seed{seed} 仿射错误",
        )
    values = np.concatenate((correct.ravel(), affine.ravel()))
    axis.set_ylim(min(0.45, float(values.min()) - 0.04), max(0.60, float(values.max()) + 0.04))
    axis.axhline(0.5, color="#6B7280", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, TAP_LABELS, rotation=12)
    axis.set_ylabel("奇数行独立探针 AUC")
    axis.set_title("五个内部表示的可分性", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, fontsize=9.2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _margin_panel(axis: Any, margins: np.ndarray, seeds: tuple[str, ...]) -> None:
    positions = np.arange(len(TAPS), dtype=float)
    colors = ("#2563EB", "#D97706")
    for index, seed in enumerate(seeds):
        axis.plot(
            positions,
            margins[index],
            marker="o",
            linewidth=2.2,
            color=colors[index],
            label=f"seed{seed}",
        )
    bound = max(0.015, float(np.abs(margins).max()) * 1.3)
    axis.set_ylim(-bound, bound)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.2, linestyle=":")
    axis.set_xticks(positions, TAP_LABELS, rotation=12)
    axis.set_ylabel("正确探针 AUC - 仿射错误探针 AUC")
    axis.set_title("第一处低于 +0.005 的信息位置", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "invalid":
        return "裁决：来源、检查点、hook 或探针协议无效，本次内部指标不可解释。"
    loss = gate.get("first_loss_by_seed", {}).get("3")
    labels = dict(zip(TAPS, TAP_LABELS, strict=True))
    labels["final_classifier"] = "最终分类器"
    if loss is None:
        return "裁决：尚未定位 seed3 的结构优势丢失位置，保持模型路线暂缓。"
    return f"裁决：seed3 首次未达到正确结构优势门槛的位置是“{labels.get(loss, loss)}”。"


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_k1by7_svg"]
