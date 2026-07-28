from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    EXPECTED_SEEDS,
    FRESH_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    ANCHOR_MARGIN,
    AUC_FLOOR,
    INVARIANT_MARGIN,
    WRONG_SBOX_MARGIN,
)


CONDITION_LABELS = (
    ("exact_position_histogram_residual", "正确结构 + 保留位置"),
    ("wrong_sbox_position_histogram_residual", "错误 S盒 + 保留位置"),
    ("invariant_histogram_residual", "正确结构 + 抹除位置"),
    ("current_k1r_exact_anchor", "旧 K1-R 正确结构"),
)
MARGIN_LABELS = (
    ("exact_minus_anchor", "旧 K1-R", ANCHOR_MARGIN),
    ("exact_minus_wrong_sbox", "错误 S盒", WRONG_SBOX_MARGIN),
    ("exact_minus_invariant", "位置抹除", INVARIANT_MARGIN),
)
SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥新样本",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-T position-residual chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1t_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1t_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("K1-T plot requires seed3 and seed4 results")
    columns = [(str(seed), split) for seed in EXPECTED_SEEDS for split in FRESH_SPLITS]
    auc_values = np.asarray(
        [
            [
                float(seed_results[seed][split][f"{condition}_auc"])
                for seed, split in columns
            ]
            for condition, _ in CONDITION_LABELS
        ],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [
                float(seed_results[seed][split][field])
                for seed, split in columns
            ]
            for field, _, _ in MARGIN_LABELS
        ],
        dtype=float,
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
        figure, axes = plt.subplots(1, 2, figsize=(16, 8.8))
        figure.subplots_adjust(
            left=0.17,
            right=0.91,
            top=0.70,
            bottom=0.13,
            wspace=0.39,
        )
        figure.suptitle(
            "创新1 K1-T：保留 uKNIT 原生位置统计后，神经网络能否学到第5轮信号",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.895,
            "固定 cell11 差分、每样本4对密文和相同训练预算；只增加一个有界的五阶段位置直方图残差。",
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
            "左图看新分支能否学到信号；右图看优势是否来自正确 S盒和原生 cell 位置，而非容量本身。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _heatmap(
            axes[0],
            auc_values,
            columns,
            labels=[label for _, label in CONDITION_LABELS],
            title=f"Fresh AUC（正确位置残差必须全部 ≥ {AUC_FLOOR:.3f}）",
            lower=min(0.45, float(auc_values.min()) - 0.03),
            upper=max(0.82, float(auc_values.max()) + 0.03),
            signed=False,
        )
        margin_limit = max(0.08, float(np.max(np.abs(margin_values))) + 0.03)
        _heatmap(
            axes[1],
            margin_values,
            columns,
            labels=[f"对比{label}\n门槛 +{threshold:.3f}" for _, label, threshold in MARGIN_LABELS],
            title="正确位置残差的归因净优势（正值表示候选更好）",
            lower=-margin_limit,
            upper=margin_limit,
            signed=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 8.8,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_run": True,
        "heatmaps_used_instead_of_overlapping_curves": True,
        "all_values_annotated_to_four_decimals": True,
        "all_seed_split_combinations_visible": True,
    }


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[tuple[str, str]],
    *,
    labels: list[str],
    title: str,
    lower: float,
    upper: float,
    signed: bool,
) -> None:
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}" if signed else f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=9.8,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_family_ctspn_k1t_deterministic_position_residual_supported": (
            "裁决：正确位置残差在两颗 seed 的两种新样本上均通过学习和归因门槛。"
        ),
        "innovation1_uknit_family_ctspn_k1t_signal_without_wrong_sbox_attribution": (
            "裁决：能学到信号，但正确 S盒没有稳定优于错误 S盒，暂不能归因。"
        ),
        "innovation1_uknit_family_ctspn_k1t_signal_without_position_necessity": (
            "裁决：能学到信号，但保留原生位置没有稳定优于位置抹除。"
        ),
        "innovation1_uknit_family_ctspn_k1t_key_specific_position_residual": (
            "裁决：同密钥新样本通过、跨密钥失败，当前分支仍依赖训练密钥。"
        ),
        "innovation1_uknit_family_ctspn_k1t_trainable_position_residual_not_supported": (
            "裁决：确定性统计很强，但当前随机初始化的可训练残差仍未稳定学到。"
        ),
        "innovation1_uknit_family_ctspn_k1t_protocol_invalid": (
            "裁决：来源、缓存、权重或产物绑定无效，本次指标不可解释。"
        ),
    }
    decision = str(gate.get("decision", ""))
    return labels.get(decision, f"裁决：{decision}")


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status, "#374151"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1t_svg"]
