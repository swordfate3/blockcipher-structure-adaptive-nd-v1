from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    AUC_FLOOR,
    EXPECTED_SEEDS,
    FRESH_SPLITS,
    NO_TOPOLOGY_MARGIN,
    SEMANTIC_MARGIN,
)


CONDITION_LABELS = (
    ("exact_composition", "正确 S盒 + 正确扩散"),
    ("wrong_sbox_semantics", "错误 S盒 + 正确扩散"),
    ("no_sbox_composition", "无 S盒 + 正确扩散"),
    ("no_topology", "无 S盒 + 无扩散拓扑"),
)
CONTROL_LABELS = (
    ("wrong_sbox_semantics", "正确结构 - 错误 S盒"),
    ("no_sbox_composition", "正确结构 - 无 S盒"),
    ("no_topology", "正确结构 - 无拓扑"),
)
SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥新样本",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-R neural attribution chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1r_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1r_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    expected_seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
    if set(seed_results) != expected_seed_keys:
        raise ValueError("K1-R plot requires seed3 and seed4 results")
    columns = [(str(seed), split) for seed in EXPECTED_SEEDS for split in FRESH_SPLITS]
    auc_values = np.asarray(
        [
            [
                _condition_auc(seed_results[seed][split], condition)
                for seed, split in columns
            ]
            for condition, _ in CONDITION_LABELS
        ],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [
                float(seed_results[seed][split][f"exact_minus_{condition}"])
                for seed, split in columns
            ]
            for condition, _ in CONTROL_LABELS
        ],
        dtype=float,
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
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
        figure, axes = plt.subplots(1, 2, figsize=(16, 8.6))
        figure.subplots_adjust(
            left=0.17,
            right=0.965,
            top=0.70,
            bottom=0.13,
            wspace=0.36,
        )
        figure.suptitle(
            "创新1 K1-R：uKNIT 第5轮换成强差分后，正确密码结构是否真正帮助神经网络",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.895,
            "固定 cell11 差分 0x0000400000000000、每样本4对密文和相同训练预算；四种结构分别独立训练10轮。",
            ha="left",
            fontsize=10.8,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.83,
            _decision_text(gate),
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.775,
            "左图看每种网络能否区分；右图只看正确结构比控制高多少，避免相近曲线难以辨认。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )

        _render_auc_heatmap(axes[0], auc_values, columns)
        _render_margin_heatmap(axes[1], margin_values, columns)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 8.6,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_run": True,
        "heatmaps_used_instead_of_overlapping_curves": True,
        "all_values_annotated_to_four_decimals": True,
        "all_seed_split_combinations_visible": True,
    }


def _render_auc_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[tuple[str, str]],
) -> None:
    lower = min(0.48, float(values.min()) - 0.02)
    upper = max(0.58, float(values.max()) + 0.02)
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=lower,
        vmax=upper,
    )
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(
        range(len(CONDITION_LABELS)),
        [label for _, label in CONDITION_LABELS],
    )
    axis.set_title(
        f"Fresh AUC（正确结构必须全部 ≥ {AUC_FLOOR:.3f}）",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight=("bold" if row == 0 and value >= AUC_FLOOR else "normal"),
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("AUC")


def _render_margin_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[tuple[str, str]],
) -> None:
    limit = max(0.02, float(np.max(np.abs(values))) + 0.01)
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(
        range(len(CONTROL_LABELS)),
        [label for _, label in CONTROL_LABELS],
    )
    axis.set_title(
        "结构归因净优势（S盒门槛 +0.005；无拓扑门槛 +0.010）",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    for row, (condition, _) in enumerate(CONTROL_LABELS):
        threshold = (
            NO_TOPOLOGY_MARGIN if condition == "no_topology" else SEMANTIC_MARGIN
        )
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight=("bold" if value >= threshold else "normal"),
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("正确结构 AUC - 控制 AUC")


def _condition_auc(result: Mapping[str, Any], condition: str) -> float:
    field = "exact_auc" if condition == "exact_composition" else f"{condition}_auc"
    return float(result[field])


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("cell11_neural_structure_attribution_supported"):
        return "结论：正确结构在两颗 seed、两种新样本上同时学到信号并稳定领先控制，可进入中等规模诊断。"
    if decision.endswith("cell11_signal_learned_structure_attribution_not_supported"):
        return "结论：神经网络学到了 cell11 信号，但正确结构未稳定领先控制；当前增益不能归因于密码结构。"
    if decision.endswith("cell11_key_specific_neural_attribution"):
        return "结论：同密钥成立但跨密钥不成立；当前结构主要学到密钥相关规律，不允许扩规模。"
    if decision.endswith("cell11_neural_signal_not_supported"):
        return "结论：Fisher 可见的强信号没有被当前网络稳定学到；瓶颈在表示或优化，而不是继续找差分位置。"
    return "协议无效：先修复缓存、checkpoint 或评估绑定，当前指标不能解释。"


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1r_svg"]
