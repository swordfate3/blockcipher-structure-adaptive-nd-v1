from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    EXPECTED_SEEDS,
    FRESH_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_semantic_contrast_k1am import (
    ANCHOR_RETENTION_MARGIN,
    BRANCH_MARGIN,
    INDEPENDENT_WRONG_MARGIN,
    ORIENTATION_MARGIN,
    SEMANTIC_MARGIN,
)


SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "新密钥新样本",
}
AUC_LABELS = (
    ("candidate_correct_auc", "成对训练：正确 S盒"),
    ("k1ak_correct_anchor_auc", "原 K1-AK：正确 S盒"),
    ("k1ak_independent_wrong_auc", "原 K1-AK：错误 S盒独立训练"),
    ("swapped_primary_auc", "交换方向：错误 S盒优先"),
)
MARGIN_LABELS = (
    ("candidate_minus_k1ak_correct", "候选 - 原正确模型", ANCHOR_RETENTION_MARGIN),
    (
        "candidate_minus_k1ak_wrong",
        "候选 - 错误 S盒独立模型",
        INDEPENDENT_WRONG_MARGIN,
    ),
    (
        "candidate_minus_swapped_primary",
        "候选 - 交换方向模型",
        ORIENTATION_MARGIN,
    ),
    (
        "candidate_minus_wrong_same_checkpoint",
        "同检查点：正确 - 错误 S盒",
        SEMANTIC_MARGIN,
    ),
    (
        "candidate_minus_transition_off",
        "同检查点：正确 - 关闭转移分支",
        BRANCH_MARGIN,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese Midori64 K1-AM semantic-contrast chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1am_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1am_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("K1-AM plot requires seed6 and seed7 results")
    columns = [(str(seed), split) for seed in EXPECTED_SEEDS for split in FRESH_SPLITS]
    auc_values = np.asarray(
        [
            [float(seed_results[seed][split][field]) for seed, split in columns]
            for field, _label in AUC_LABELS
        ],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [float(seed_results[seed][split][field]) for seed, split in columns]
            for field, _label, _threshold in MARGIN_LABELS
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
        figure, axes = plt.subplots(1, 2, figsize=(17, 9.2))
        figure.subplots_adjust(
            left=0.19,
            right=0.92,
            top=0.68,
            bottom=0.12,
            wspace=0.52,
        )
        figure.suptitle(
            "创新1 K1-AM：成对语义训练能否消除错误 S盒的替代解",
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
                "Midori64 第4轮、cell8 差分、每样本4对密文、2048/class、seed6/7；"
                "正确与错误 S盒共享同一组训练权重。"
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
            fontsize=11.1,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.755,
            (
                "左图比较候选与独立训练锚点；右图逐项检查预注册差值。"
                "带圈数值达到门槛，叉号表示失败。"
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
        "width_inches": 17.0,
        "height_inches": 9.2,
        "language": "zh-CN",
        "panels": 2,
        "fresh_seed_split_combinations": len(columns),
        "auc_rows": len(AUC_LABELS),
        "margin_rows": len(MARGIN_LABELS),
        "heatmaps_used_instead_of_overlapping_curves": True,
        "threshold_outcomes_visible": True,
    }


def _render_auc_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[tuple[str, str]],
) -> None:
    lower = min(0.62, float(values.min()) - 0.01)
    upper = max(0.68, float(values.max()) + 0.01)
    image = axis.imshow(values, cmap="YlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(range(len(AUC_LABELS)), [label for _field, label in AUC_LABELS])
    axis.set_title(
        "Fresh AUC：候选与独立训练锚点", loc="left", fontweight="bold", pad=14
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.4f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if row == 0 else "normal",
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
        [label for _field, label, _threshold in MARGIN_LABELS],
    )
    axis.set_title(
        "判定差值：必须逐 seed、逐 fresh 范围通过",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    for row, (_field, _label, threshold) in enumerate(MARGIN_LABELS):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            passed = value >= threshold
            axis.text(
                column,
                row,
                f"{'○' if passed else '×'} {value:+.4f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold" if not passed else "normal",
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("AUC 差值")


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("paired_semantic_contrast_supported"):
        return (
            "结论：正确方向稳定击败独立错误模型与交换方向模型，成对语义约束得到支持。"
        )
    if decision.endswith("semantic_preference_imposed_substitute_unresolved"):
        return (
            "结论：同检查点下正确 S盒确实更重要，但错误 S盒独立训练仍能找到替代解；"
            "不能放大 pair 数或样本量。"
        )
    if decision.endswith("semantic_contrast_destroys_anchor_signal"):
        return "结论：成对约束破坏了原 K1-AK 信号，应丢弃该训练目标。"
    if decision.endswith("semantic_contrast_causal_retention_failed"):
        return "结论：候选没有同时保留 S盒和转移分支的因果作用，应丢弃该目标。"
    return "协议无效：先修复数据、训练、检查点或评估绑定，当前数值不能解释。"


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1am_svg"]
