from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


AUC_ROWS = (
    ("exact_16pair_auc", "16对：正确结构"),
    ("wrong_sbox_16pair_auc", "16对：错误 S盒"),
    ("invariant_16pair_auc", "16对：抹除位置"),
    ("exact_4pair_anchor_auc", "4对：正确结构锚点"),
)
MARGIN_ROWS = (
    ("exact_minus_wrong_sbox", "正确结构 - 错误 S盒", 0.010),
    ("exact_16pair_minus_exact_4pair", "16对 - 4对", 0.010),
    ("exact_minus_invariant", "保留位置 - 抹除位置", 0.010),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-V pair-count chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1v_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1v_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"3", "4"}:
        raise ValueError("K1-V plot requires seed3 and seed4 results")
    columns = ("3", "4")
    auc_values = np.asarray(
        [[float(seed_results[seed][field]) for seed in columns] for field, _ in AUC_ROWS],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [float(seed_results[seed][field]) for seed in columns]
            for field, _, _ in MARGIN_ROWS
        ],
        dtype=float,
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.8,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.6))
        figure.subplots_adjust(
            left=0.17,
            right=0.92,
            top=0.67,
            bottom=0.13,
            wspace=0.43,
        )
        figure.suptitle(
            "创新1 K1-V：每条样本从4对密文提升到16对，是否带来稳定增益",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "固定 uKNIT-BC 第5轮、cell11 差分、2048/class、相同网络和10个 epoch；唯一变量是每条样本的密文对数量。",
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.82,
            _decision_text(gate),
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.755,
            "左图比较绝对 AUC；右图分开显示正确语义、pair 数和位置保留的净差值，参考门槛写在行标签中。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _heatmap(
            axes[0],
            auc_values,
            labels=[label for _, label in AUC_ROWS],
            title="跨密钥验证 AUC（越高越好）",
            lower=min(0.45, float(auc_values.min()) - 0.03),
            upper=max(0.82, float(auc_values.max()) + 0.03),
            signed=False,
        )
        margin_limit = max(0.08, float(np.max(np.abs(margin_values))) + 0.03)
        _heatmap(
            axes[1],
            margin_values,
            labels=[
                f"{label}\n参考门槛 +{threshold:.3f}"
                for _, label, threshold in MARGIN_ROWS
            ],
            title="16对正确结构的净优势（正值表示更好）",
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
        "width_inches": 15.5,
        "height_inches": 8.6,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_pair_count": True,
        "values_annotated_to_four_decimals": True,
        "separated_auc_and_margin_panels": True,
    }


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    labels: list[str],
    title: str,
    lower: float,
    upper: float,
    signed: bool,
) -> None:
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks((0, 1), ("seed3", "seed4"))
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
                fontsize=11,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_family_ctspn_k1v_16pair_added_value_supported": (
            "裁决：两颗 seed 都保留正确 S盒优势，并显示出 16 对的新增价值。"
        ),
        "innovation1_uknit_family_ctspn_k1v_16pair_no_added_value": (
            "裁决：正确 S盒仍有作用，但 16 对没有在两颗 seed 上都产生新增价值。"
        ),
        "innovation1_uknit_family_ctspn_k1v_16pair_semantic_attribution_lost": (
            "裁决：增加到16对后，正确 S盒没有稳定优于错误 S盒。"
        ),
        "innovation1_uknit_family_ctspn_k1v_protocol_invalid": (
            "裁决：计划、缓存、输入形状或训练产物不完整，本次指标不可解释。"
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


__all__ = ["main", "parse_args", "render_k1v_svg"]
