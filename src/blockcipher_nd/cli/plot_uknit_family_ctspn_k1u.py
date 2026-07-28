from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import EXPECTED_SEEDS
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u import (
    AUC_FLOOR,
    INVARIANT_MARGIN,
    WRONG_SBOX_MARGIN,
)


CONDITIONS = (
    ("exact_position_histogram_residual_auc", "正确结构 + 保留位置"),
    ("wrong_sbox_position_histogram_residual_auc", "错误 S盒 + 保留位置"),
    ("invariant_histogram_residual_auc", "正确结构 + 抹除位置"),
)
MARGINS = (
    ("exact_minus_wrong_sbox", "对比错误 S盒", WRONG_SBOX_MARGIN),
    ("exact_minus_invariant", "对比位置抹除", INVARIANT_MARGIN),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-U remote-medium chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1u_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1u_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    seeds = [str(seed) for seed in EXPECTED_SEEDS]
    if set(seed_results) != set(seeds):
        raise ValueError("K1-U plot requires seed3 and seed4 results")
    auc_values = np.asarray(
        [[float(seed_results[seed][field]) for seed in seeds] for field, _ in CONDITIONS],
        dtype=float,
    )
    margin_values = np.asarray(
        [[float(seed_results[seed][field]) for seed in seeds] for field, _, _ in MARGINS],
        dtype=float,
    )
    columns = [f"seed{seed}\n跨密钥验证" for seed in seeds]

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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.18,
            right=0.91,
            top=0.68,
            bottom=0.14,
            wspace=0.44,
        )
        figure.suptitle(
            "创新1 K1-U：uKNIT 第5轮位置残差在中型数据规模是否仍成立",
            x=0.05,
            y=0.95,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "每颗 seed 使用 65536/class 训练、32768/class 跨密钥验证、每样本4对密文和10轮训练；只扩大数据量。",
            ha="left",
            fontsize=10.8,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.815,
            _decision_text(gate),
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.755,
            "左图回答能否区分，右图回答优势是否确实依赖正确 S盒语义和原生 cell 位置。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _heatmap(
            axes[0],
            auc_values,
            columns,
            [label for _, label in CONDITIONS],
            title=f"跨密钥 AUC（正确位置残差门槛 ≥ {AUC_FLOOR:.3f}）",
            lower=min(0.45, float(auc_values.min()) - 0.03),
            upper=max(0.82, float(auc_values.max()) + 0.03),
            signed=False,
        )
        limit = max(0.08, float(np.max(np.abs(margin_values))) + 0.03)
        _heatmap(
            axes[1],
            margin_values,
            columns,
            [f"{label}\n门槛 +{threshold:.3f}" for _, label, threshold in MARGINS],
            title="正确位置残差的归因净优势（正值表示候选更好）",
            lower=-limit,
            upper=limit,
            signed=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 15.2,
        "height_inches": 8.2,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_run": True,
        "heatmaps_used_instead_of_overlapping_curves": True,
        "all_values_annotated_to_four_decimals": True,
        "both_seeds_visible": True,
    }


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    columns: list[str],
    labels: list[str],
    *,
    title: str,
    lower: float,
    upper: float,
    signed: bool,
) -> None:
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(range(len(columns)), columns)
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
                fontsize=10,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=8)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_family_ctspn_k1u_medium_position_residual_supported": (
            "裁决：两颗 seed 均通过中型规模的信号、正确 S盒和原生位置归因门。"
        ),
        "innovation1_uknit_family_ctspn_k1u_medium_signal_without_wrong_sbox_attribution": (
            "裁决：有信号，但正确 S盒没有稳定优于错误 S盒。"
        ),
        "innovation1_uknit_family_ctspn_k1u_medium_signal_without_position_necessity": (
            "裁决：有信号，但保留原生位置没有稳定优于位置抹除。"
        ),
        "innovation1_uknit_family_ctspn_k1u_medium_seed_key_instability": (
            "裁决：只有一颗 seed 通过，当前中型机制仍存在密钥或随机种子不稳定。"
        ),
        "innovation1_uknit_family_ctspn_k1u_medium_position_residual_not_supported": (
            "裁决：正确位置残差未在两颗 seed 上保持中型规模信号。"
        ),
        "innovation1_uknit_family_ctspn_k1u_protocol_invalid": (
            "裁决：来源、缓存、权重或结果绑定无效，本次指标不可解释。"
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


__all__ = ["main", "parse_args", "render_k1u_svg"]
