from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_trail_multibit_k1bn import (
    AUC_FLOOR,
    CANDIDATE_FAMILIES,
    CONFIRMATION_SEEDS,
    FAMILY_CELL_LOCAL,
    FRESH_SPLITS,
    LABEL_SHUFFLE_MARGIN,
    RAW_MARGIN,
    SELECTED_PER_FAMILY,
)


FAMILY_LABELS = {
    FAMILY_CELL_LOCAL: "单 cell 多 bit",
    "two_cell_low_spread": "双 cell 低扩散",
}
SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese uKNIT K1-BN gate.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bn_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bn_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    ranking = list(gate.get("selection", {}).get("ranking", []))
    if len(ranking) != 2 * SELECTED_PER_FAMILY:
        raise ValueError("K1-BN plot requires all 48 frozen discovery candidates")
    selected = [str(value) for value in gate["selection"]["selected_candidate_ids"]]
    confirmation = gate.get("confirmation_summary", {})

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 11.2))
        figure.subplots_adjust(
            left=0.08,
            right=0.96,
            top=0.78,
            bottom=0.10,
            hspace=0.50,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1：uKNIT 第6轮 DDT/轨迹引导多 bit 差分审判",
            x=0.05,
            y=0.965,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.91,
            "先按6轮最佳差分特征评分冻结两类各24个输入，再用严格密文数据检验；轨迹信息不进入网络。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.855,
            str(gate.get("decision_text_zh", "")),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )

        trail_values, exact_values, margin_values = _matrices(ranking)
        _heatmap(
            axes[0, 0],
            trail_values,
            title="候选冻结依据：6轮最佳特征的 log2 概率",
            value_format=".0f",
            threshold=None,
        )
        _heatmap(
            axes[0, 1],
            exact_values,
            title="发现阶段：每个候选的最差 fresh AUC",
            value_format=".3f",
            threshold=AUC_FLOOR,
        )
        _heatmap(
            axes[1, 0],
            margin_values,
            title="发现阶段：精确五阶段相对原始密文的最差优势",
            value_format="+.2f",
            threshold=RAW_MARGIN,
        )
        _confirmation_panel(axes[1, 1], selected, confirmation)

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 11.2,
        "language": "zh-CN",
        "panels": 4,
        "all_48_candidates_visible": True,
        "decision_matches_gate": True,
    }


def _matrices(
    ranking: list[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_family_rank = {
        (str(row["family"]), int(row["family_rank"])): row for row in ranking
    }
    trail = np.asarray(
        [
            [
                float(by_family_rank[(family, rank)]["trail_log2_probability"])
                for rank in range(1, SELECTED_PER_FAMILY + 1)
            ]
            for family in CANDIDATE_FAMILIES
        ]
    )
    exact = np.asarray(
        [
            [
                float(by_family_rank[(family, rank)]["minimum_fresh_exact_auc"])
                for rank in range(1, SELECTED_PER_FAMILY + 1)
            ]
            for family in CANDIDATE_FAMILIES
        ]
    )
    margin = np.asarray(
        [
            [
                float(
                    by_family_rank[(family, rank)][
                        "minimum_fresh_exact_minus_raw"
                    ]
                )
                for rank in range(1, SELECTED_PER_FAMILY + 1)
            ]
            for family in CANDIDATE_FAMILIES
        ]
    )
    return trail, exact, margin


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    value_format: str,
    threshold: float | None,
) -> None:
    axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=float(values.min()),
        vmax=float(values.max()) if values.max() > values.min() else values.min() + 1,
    )
    ticks = list(range(0, SELECTED_PER_FAMILY, 2))
    axis.set_xticks(ticks, [str(index + 1) for index in ticks])
    axis.set_yticks(
        range(len(CANDIDATE_FAMILIES)),
        [FAMILY_LABELS[family] for family in CANDIDATE_FAMILIES],
    )
    axis.set_xlabel("族内轨迹排名")
    axis.set_title(title, loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight=(
                    "bold" if threshold is not None and value >= threshold else "normal"
                ),
                color="#111827",
            )
    if threshold is not None:
        axis.text(
            0.0,
            -0.24,
            f"加粗表示达到单项门槛 {threshold:+.3f}；最终仍要求两个 fresh split 同时通过。",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4B5563",
        )
    else:
        axis.text(
            0.0,
            -0.24,
            "数值越接近0，最佳差分特征概率越高；这里只负责冻结候选，不是区分结果。",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4B5563",
        )
    axis.tick_params(length=0)


def _confirmation_panel(
    axis: plt.Axes,
    selected: list[str],
    confirmation: Mapping[str, Any],
) -> None:
    if not selected:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.57,
            "确认阶段未启动",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#6B7280",
        )
        axis.text(
            0.5,
            0.42,
            "48个轨迹优先多 bit 候选均未过发现门。",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6B7280",
        )
        return
    columns = [
        (seed, split) for seed in CONFIRMATION_SEEDS for split in FRESH_SPLITS
    ]
    values = np.asarray(
        [
            [
                float(confirmation[candidate_id][str(seed)][split]["exact_auc"])
                for seed, split in columns
            ]
            for candidate_id in selected
        ]
    )
    axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=min(0.48, float(values.min())),
        vmax=max(0.60, float(values.max())),
    )
    axis.set_xticks(
        range(len(columns)),
        [f"seed{seed}\n{SPLIT_LABELS[split]}" for seed, split in columns],
    )
    axis.set_yticks(range(len(selected)), selected)
    axis.set_title(
        "未见 seed 确认：绝对 AUC（还必须通过两项控制优势）",
        loc="left",
        fontweight="bold",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if values[row, column] >= AUC_FLOOR else "normal",
            )
    axis.text(
        0.0,
        -0.24,
        f"完整确认门：AUC ≥ {AUC_FLOOR:.3f}，超过原始密文 ≥ {RAW_MARGIN:+.3f}，"
        f"超过标签打乱 ≥ {LABEL_SHUFFLE_MARGIN:+.3f}。",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    axis.tick_params(length=0)


def _decision_color(status: str) -> str:
    return {"pass": "#166534", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1bn_svg"]
