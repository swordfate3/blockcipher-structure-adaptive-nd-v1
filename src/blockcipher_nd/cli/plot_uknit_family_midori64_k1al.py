from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    AUC_FLOOR,
    EXPECTED_SEEDS,
    FRESH_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_transition_causal_k1al import (
    BRANCH_MARGIN,
    SEMANTIC_MARGIN,
)


SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "新密钥新样本",
}
CONDITION_LABELS = (
    ("correct_runtime_auc", "正确 S盒 + 转移分支开启"),
    ("wrong_sbox_auc", "错误 S盒 + 同一检查点"),
    ("transition_branch_off_auc", "正确 S盒 + 转移分支关闭"),
)
MARGIN_LABELS = (
    ("correct_minus_wrong_sbox", "正确运行时 - 错误 S盒"),
    (
        "correct_minus_transition_branch_off",
        "正确运行时 - 关闭转移分支",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese Midori64 K1-AL causal-audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1al_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1al_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("K1-AL plot requires seed6 and seed7 results")
    columns = [(str(seed), split) for seed in EXPECTED_SEEDS for split in FRESH_SPLITS]
    auc_values = np.asarray(
        [
            [float(seed_results[seed][split][field]) for seed, split in columns]
            for field, _label in CONDITION_LABELS
        ],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [float(seed_results[seed][split][field]) for seed, split in columns]
            for field, _label in MARGIN_LABELS
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
            left=0.18,
            right=0.91,
            top=0.69,
            bottom=0.13,
            wspace=0.48,
        )
        figure.suptitle(
            "创新1 K1-AL：同一组 Midori64 权重是否真正使用正确 S盒和新转移分支",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.895,
            (
                "固定 K1-AK 正确结构最佳检查点与原始数据，全程零训练；只换错误 S盒或"
                "在前向计算中关闭转移分支。"
            ),
            ha="left",
            fontsize=10.8,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.825,
            _decision_text(gate),
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.765,
            (
                "左图比较三种推理条件的 fresh AUC；右图放大正确运行时相对两个干预的"
                "因果净优势，所有格子都必须达到 +0.005。"
            ),
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
        "same_checkpoint": True,
        "training_performed": False,
        "heatmaps_used_instead_of_overlapping_curves": True,
        "auc_values_annotated_to_four_decimals": True,
        "margin_values_annotated_to_five_decimals": True,
        "all_seed_split_combinations_visible": True,
    }


def _render_auc_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[tuple[str, str]],
) -> None:
    lower = min(0.52, float(values.min()) - 0.02)
    upper = max(0.68, float(values.max()) + 0.02)
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(
        range(len(CONDITION_LABELS)),
        [label for _field, label in CONDITION_LABELS],
    )
    axis.set_title(
        f"Fresh AUC（正确运行时必须全部 >= {AUC_FLOOR:.3f}）",
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
        range(len(MARGIN_LABELS)),
        [label for _field, label in MARGIN_LABELS],
    )
    axis.set_title(
        "同检查点因果净优势（两项门槛均为 +0.005）",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    thresholds = (SEMANTIC_MARGIN, BRANCH_MARGIN)
    for row, threshold in enumerate(thresholds):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.5f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight=("bold" if value >= threshold else "normal"),
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("AUC 差值")


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("transition_and_sbox_causal_use_supported"):
        return (
            "结论：同一组权重同时依赖正确 S盒和新转移分支；错误 S盒独立训练时学到的是替代捷径。"
        )
    if decision.endswith("transition_causal_sbox_identification_failed"):
        return "结论：新转移分支确实参与预测，但当前表示仍不能稳定辨认正确 S盒。"
    if decision.endswith("transition_branch_causal_use_failed"):
        return "结论：关闭新转移分支没有稳定损失，K1-AK 的提升不能归因于该分支。"
    return "协议无效：先修复源证据、检查点、数据、运行时或重放绑定，当前指标不能解释。"


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1al_svg"]
