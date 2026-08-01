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
    ("wrong_target_binding_routing", "错误目标绑定"),
    ("no_compiler_conditioner", "不使用结构条件器"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY1 primitive conditioner comparison."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by1_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by1_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"3", "4"}:
        raise ValueError("K1-BY1 plot requires seed3 and seed4 results")
    seeds = ("3", "4")
    aucs = np.asarray(
        [
            [float(seed_results[seed]["auc_by_condition"][condition]) for seed in seeds]
            for condition, _label in CONDITION_ROWS
        ]
    )
    controls = tuple(condition for condition, _ in CONDITION_ROWS[1:])
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
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.8))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.68,
            bottom=0.16,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-BY1：可学习密码结构能否帮助 uKNIT 第5轮区分",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "固定数据、差分位置、16 对密文、训练预算和网络参数量；只改变编译后的结构路由。",
            ha="left",
            fontsize=11.5,
        )
        figure.text(
            0.05,
            0.835,
            "训练 2048/class，跨密钥验证 1024/class，seed3/4；这是本地诊断，不是正式规模或跨密码迁移结果。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _grouped_auc_bars(axes[0], aucs, seeds)
        _grouped_margin_bars(axes[1], margins, seeds)
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
        "seeds": [3, 4],
        "conditions": [name for name, _ in CONDITION_ROWS],
        "status": gate.get("status"),
    }


def _grouped_auc_bars(axis: Any, values: np.ndarray, seeds: tuple[str, ...]) -> None:
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
    lower = min(0.48, float(values.min()) - 0.03)
    upper = max(0.60, float(values.max()) + 0.07)
    axis.set_ylim(lower, upper)
    axis.axhline(0.5, color="#6B7280", linewidth=1.0, linestyle="--")
    axis.axhline(0.55, color="#047857", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, [label for _name, label in CONDITION_ROWS])
    axis.tick_params(axis="x", rotation=14)
    axis.set_ylabel("跨密钥验证 AUC")
    axis.set_title("四种结构路由的 AUC（放大有效区间）", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _grouped_margin_bars(
    axis: Any,
    values: np.ndarray,
    seeds: tuple[str, ...],
) -> None:
    labels = ("相对错误顺序", "相对错误目标绑定", "相对无条件器")
    reordered = values[[0, 1, 2]]
    positions = np.arange(len(labels), dtype=float)
    width = 0.34
    colors = ("#2563EB", "#D97706")
    for index, seed in enumerate(seeds):
        bars = axis.bar(
            positions + (index - 0.5) * width,
            reordered[:, index],
            width,
            label=f"seed{seed}",
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%+.4f", padding=3, fontsize=9.5)
    bound = max(0.02, float(np.abs(reordered).max()) * 1.35)
    axis.set_ylim(-bound, bound)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.0, linestyle=":")
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", rotation=12)
    axis.set_ylabel("正确路由 AUC - 控制 AUC")
    axis.set_title("正确路由是否真正依赖顺序和绑定", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by1_compiler_conditioner_supported": (
            "裁决：正确编译路由在两颗 seed 上同时达到信号和全部控制优势门槛。"
        ),
        "innovation1_runtime_spn_k1by1_structure_attribution_not_supported": (
            "裁决：模型有信号，但正确路由没有稳定拉开全部控制，结构归因暂不成立。"
        ),
        "innovation1_runtime_spn_k1by1_conditioner_signal_not_reproduced": (
            "裁决：正确编译路由未在两颗 seed 上复现最低信号，当前条件器接口暂停。"
        ),
        "innovation1_runtime_spn_k1by1_protocol_invalid": (
            "裁决：计划、缓存、模型或训练产物不完整，本次指标不可解释。"
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


__all__ = ["main", "render_k1by1_svg"]
