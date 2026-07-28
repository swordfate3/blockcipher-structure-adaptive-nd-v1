from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import (
    AUC_FLOOR,
    EXPECTED_SEEDS,
    FRESH_SPLITS,
    LABEL_SHUFFLE_MARGIN,
    TAPS,
)


TAP_LABELS = {
    TAPS[0]: "T0 确定性五阶段位置直方图",
    TAPS[1]: "T1 位编码器输出（保留bit位置）",
    TAPS[2]: "T2 拓扑更新量（保留cell位置）",
    TAPS[3]: "T3 cell无序池化后表示",
}
SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥新样本",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-S representation-access chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1s_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1s_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("K1-S plot requires seed3 and seed4 results")
    columns = [(str(seed), split) for seed in EXPECTED_SEEDS for split in FRESH_SPLITS]
    auc_values = np.asarray(
        [
            [seed_results[seed][split][tap]["auc"] for seed, split in columns]
            for tap in TAPS
        ],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [
                seed_results[seed][split][tap]["minus_label_shuffle"]
                for seed, split in columns
            ]
            for tap in TAPS
        ],
        dtype=float,
    )
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.2,
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
            left=0.21,
            right=0.91,
            top=0.70,
            bottom=0.13,
            wspace=0.50,
        )
        figure.suptitle(
            "创新1 K1-S：uKNIT 第5轮强信号在神经网络哪一层消失",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.895,
            "复用 K1-Q 的 cell11 数据和 K1-R 的最佳权重；不训练，只逐层读取内部表示并用相同 Fisher 程序评分。",
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
            "左图看每层是否仍能读出真假样本；右图扣除标签打乱控制，排除高维表示的偶然拟合。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _heatmap(
            axes[0],
            auc_values,
            columns,
            title=f"逐层 Fresh AUC（可访问门槛 ≥ {AUC_FLOOR:.3f}）",
            cmap="RdYlGn",
            lower=min(0.45, float(auc_values.min()) - 0.03),
            upper=max(0.85, float(auc_values.max()) + 0.03),
            suffix="",
            ytick_labels=[TAP_LABELS[tap] for tap in TAPS],
        )
        limit = max(0.05, float(np.max(np.abs(margin_values))) + 0.03)
        _heatmap(
            axes[1],
            margin_values,
            columns,
            title=(
                "真实标签优势：本层 AUC - 标签打乱 AUC\n"
                f"（归因门槛 ≥ +{LABEL_SHUFFLE_MARGIN:.3f}）"
            ),
            cmap="RdYlGn",
            lower=-limit,
            upper=limit,
            suffix="",
            ytick_labels=["T0", "T1", "T2", "T3"],
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
    title: str,
    cmap: str,
    lower: float,
    upper: float,
    suffix: str,
    ytick_labels: list[str],
) -> None:
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(range(len(TAPS)), ytick_labels)
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}{suffix}" if lower < 0.0 else f"{value:.4f}{suffix}",
                ha="center",
                va="center",
                fontsize=9.8,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    labels = {
        "innovation1_uknit_family_ctspn_k1s_invariant_cell_pool_bottleneck_supported": (
            "裁决：T2 保留强信号而 T3 丢失，cell 无序池化是已定位瓶颈。"
        ),
        "innovation1_uknit_family_ctspn_k1s_downstream_residual_fusion_bottleneck_supported": (
            "裁决：T3 仍保留信号，问题位于残差投影、门控融合或最终分类头。"
        ),
        "innovation1_uknit_family_ctspn_k1s_cell_aggregation_or_update_bottleneck_supported": (
            "裁决：位编码后仍有信号，但 cell 聚合或拓扑更新阶段将其破坏。"
        ),
        "innovation1_uknit_family_ctspn_k1s_learned_representation_access_not_supported": (
            "裁决：T0 强信号重放成功，但当前学习表示均未稳定保留可访问信号。"
        ),
        "innovation1_uknit_family_ctspn_k1s_first_destructive_stage_ambiguous": (
            "裁决：逐层结果不满足单一瓶颈门槛，暂不修改网络或扩大样本。"
        ),
        "innovation1_uknit_family_ctspn_k1s_protocol_invalid": (
            "裁决：来源或重放校验失败，本次指标不可解释。"
        ),
    }
    return labels.get(decision, f"裁决：{decision}")


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status, "#374151"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1s_svg"]
