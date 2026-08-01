from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ROWS = (
    ("correct_permutation_routing", "正确置换路由"),
    ("wrong_permutation_binding", "错误目标绑定"),
    ("no_compiler_conditioner", "不使用结构条件器"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY3 permutation-expert diagnostic."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by3_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by3_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"2", "3"}:
        raise ValueError("K1-BY3 plot requires seed2 and seed3 results")
    seeds = ("2", "3")
    aucs = np.asarray(
        [
            [float(seed_results[seed]["auc_by_condition"][condition]) for seed in seeds]
            for condition, _label in CONDITION_ROWS
        ]
    )
    controls = ("wrong_permutation_binding", "no_compiler_conditioner")
    margins = np.asarray(
        [
            [
                float(seed_results[seed]["correct_minus_control"][condition])
                for seed in seeds
            ]
            for condition in controls
        ]
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.6,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.8))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.68,
            bottom=0.16,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-BY3：PRESENT 第7轮置换专家诊断",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "同一套有序结构编译网络从 uKNIT 的 GF(2) 扩散切换到 PRESENT 的一对一 P 层。",
            ha="left",
            fontsize=11.5,
        )
        figure.text(
            0.05,
            0.835,
            "训练 2048/class，验证 1024/class，16 对输入；比较正确绑定、错误绑定和无结构条件器。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        _auc_panel(axes[0], aucs, seeds)
        _margin_panel(axes[1], margins, seeds)
        figure.text(
            0.05,
            0.075,
            _decision_text(gate),
            ha="left",
            fontsize=11.0,
            color=_decision_color(str(gate.get("status", ""))),
            fontweight="bold",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "run_id": gate.get("run_id"),
        "panels": 2,
        "seeds": [2, 3],
        "conditions": [name for name, _ in CONDITION_ROWS],
        "status": gate.get("status"),
    }


def _auc_panel(axis: Any, values: np.ndarray, seeds: tuple[str, ...]) -> None:
    positions = np.arange(len(CONDITION_ROWS), dtype=float)
    width = 0.34
    colors = ("#2563EB", "#D97706")
    for index, seed in enumerate(seeds):
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values[:, index],
            width,
            label=f"seed{seed}",
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9.5)
    axis.set_ylim(
        min(0.48, float(values.min()) - 0.03), max(0.60, float(values.max()) + 0.07)
    )
    axis.axhline(0.5, color="#6B7280", linewidth=1.0, linestyle="--")
    axis.axhline(0.55, color="#047857", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, [label for _name, label in CONDITION_ROWS])
    axis.tick_params(axis="x", rotation=10)
    axis.set_ylabel("独立验证集 AUC")
    axis.set_title("正确 P 层绑定是否产生可学习信号", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _margin_panel(axis: Any, values: np.ndarray, seeds: tuple[str, ...]) -> None:
    labels = ("相对错误绑定", "相对无条件器")
    positions = np.arange(len(labels), dtype=float)
    width = 0.34
    colors = ("#2563EB", "#D97706")
    for index, seed in enumerate(seeds):
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values[:, index],
            width,
            label=f"seed{seed}",
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%+.4f", padding=3, fontsize=9.5)
    bound = max(0.02, float(np.abs(values).max()) * 1.35)
    axis.set_ylim(-bound, bound)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("正确路由 AUC - 控制 AUC")
    axis.set_title("置换语义优势能否在两颗 seed 上复现", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by3_permutation_expert_supported": (
            "裁决：正确 PRESENT 置换路由在两颗 seed 上同时通过信号和控制优势门槛。"
        ),
        "innovation1_runtime_spn_k1by3_permutation_attribution_not_supported": (
            "裁决：信号存在，但正确绑定没有在两颗 seed 上稳定拉开全部控制。"
        ),
        "innovation1_runtime_spn_k1by3_permutation_signal_not_reproduced": (
            "裁决：正确置换路由未稳定复现最低信号，当前置换专家暂缓。"
        ),
        "innovation1_runtime_spn_k1by3_protocol_invalid": (
            "裁决：计划、缓存、模型或结果产物不完整，本次指标不可解释。"
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


__all__ = ["main", "render_k1by3_svg"]
