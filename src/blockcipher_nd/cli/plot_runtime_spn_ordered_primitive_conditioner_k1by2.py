from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ROWS = (
    ("correct_compiler_routing", "正确编译路由"),
    ("wrong_order_routing", "错误阶段顺序"),
    ("no_compiler_conditioner", "不使用结构条件器"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY2 fresh-seed confirmation."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by2_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by2_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"5", "6"}:
        raise ValueError("K1-BY2 plot requires seed5 and seed6 results")
    seeds = ("5", "6")
    aucs = np.asarray(
        [
            [float(seed_results[seed]["auc_by_condition"][condition]) for seed in seeds]
            for condition, _label in CONDITION_ROWS
        ]
    )
    controls = ("wrong_order_routing", "no_compiler_conditioner")
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
            "创新1 K1-BY2：uKNIT 第5轮可学习结构的新种子确认",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "换成未参与 K1-BY1 的 seed5/6 和新固定密钥；差分、16 对输入、网络和训练预算全部不变。",
            ha="left",
            fontsize=11.5,
        )
        figure.text(
            0.05,
            0.835,
            "训练 2048/class，跨密钥验证 1024/class；本图只回答结果能否跨新种子和密钥复现。",
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
        "seeds": [5, 6],
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
    axis.tick_params(axis="x", rotation=12)
    axis.set_ylabel("跨密钥验证 AUC")
    axis.set_title("新 seed 和新密钥下的 AUC", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _margin_panel(axis: Any, values: np.ndarray, seeds: tuple[str, ...]) -> None:
    labels = ("相对错误顺序", "相对无条件器")
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
    axis.set_title("正确路由优势是否在 fresh seed 上复现", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by2_fresh_seed_confirmed": (
            "裁决：正确结构路由在两颗 fresh seed 上再次通过信号与控制优势门槛。"
        ),
        "innovation1_runtime_spn_k1by2_fresh_seed_attribution_not_confirmed": (
            "裁决：信号存在，但相对控制的优势未在两颗 fresh seed 上全部复现。"
        ),
        "innovation1_runtime_spn_k1by2_seed_key_dependence_detected": (
            "裁决：正确路由未稳定复现，K1-BY1 可能依赖特定 seed 或密钥。"
        ),
        "innovation1_runtime_spn_k1by2_protocol_invalid": (
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


__all__ = ["main", "render_k1by2_svg"]
