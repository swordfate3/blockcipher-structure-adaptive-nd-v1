from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ROWS = (
    ("correct_auc", "正确 PRESENT P 层"),
    ("affine_wrong_endpoint_auc", "仿射错误端点"),
    ("no_conditioner_auc", "不使用结构条件器"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BY6 affine neural-attribution result."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1by6_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1by6_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"2", "3"}:
        raise ValueError("K1-BY6 plot requires seed2 and seed3 results")
    seeds = ("2", "3")
    aucs = np.asarray(
        [
            [float(seed_results[seed][key]) for seed in seeds]
            for key, _label in CONDITION_ROWS
        ]
    )
    margins = np.asarray(
        [float(seed_results[seed]["correct_minus_affine_auc"]) for seed in seeds]
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
            bottom=0.17,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-BY6：PRESENT 正确扩散结构神经归因",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "只训练两条可识别的仿射错误端点控制；正确结构和无结构结果沿用 K1-BY3。",
            ha="left",
            fontsize=11.5,
        )
        figure.text(
            0.05,
            0.835,
            "PRESENT-80 第7轮，训练 2048/class，验证 1024/class，16 对输入，10 epochs。",
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
        "conditions": [name for name, _label in CONDITION_ROWS],
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
        min(0.48, float(values.min()) - 0.03),
        max(0.60, float(values.max()) + 0.07),
    )
    axis.axhline(0.5, color="#6B7280", linewidth=1.0, linestyle="--")
    axis.set_xticks(positions, [label for _name, label in CONDITION_ROWS])
    axis.tick_params(axis="x", rotation=8)
    axis.set_ylabel("独立验证集 AUC")
    axis.set_title("正确扩散结构、错误结构与无结构对比", fontsize=12.5)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _margin_panel(axis: Any, values: np.ndarray, seeds: tuple[str, ...]) -> None:
    positions = np.arange(len(seeds), dtype=float)
    colors = ("#2563EB", "#D97706")
    bars = axis.bar(positions, values, width=0.55, color=colors)
    axis.bar_label(bars, fmt="%+.4f", padding=4, fontsize=10.5)
    bound = max(0.012, float(np.abs(values).max()) * 1.45)
    axis.set_ylim(-bound, bound)
    axis.axhline(0.0, color="#6B7280", linewidth=1.0)
    axis.axhline(0.005, color="#047857", linewidth=1.2, linestyle=":")
    axis.set_xticks(positions, [f"seed{seed}" for seed in seeds])
    axis.set_ylabel("正确结构 AUC - 仿射错误结构 AUC")
    axis.set_title("每颗 seed 是否超过 +0.005 归因门槛", fontsize=12.5)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_runtime_spn_k1by6_permutation_attribution_supported": (
            "裁决：两颗 seed 都稳定偏好正确 PRESENT P 层，可进入同预算 GIFT 验证。"
        ),
        "innovation1_runtime_spn_k1by6_permutation_attribution_not_supported": (
            "裁决：至少一颗 seed 未拉开可识别错误结构，下一步审计网络内部信息访问。"
        ),
        "innovation1_runtime_spn_k1by6_protocol_invalid": (
            "裁决：来源、缓存、设备、模型或结果不完整，本次指标不可解释。"
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


__all__ = ["main", "render_k1by6_svg"]
