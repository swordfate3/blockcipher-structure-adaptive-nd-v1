from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


COLUMNS = (
    ("uknit64", "3", "uKNIT r5\nseed3"),
    ("uknit64", "4", "uKNIT r5\nseed4"),
    ("dialga128", "0", "Dialga r4\nseed0"),
    ("dialga128", "1", "Dialga r4\nseed1"),
)
AUC_ROWS = (
    ("compact_exact_auc", "紧凑网络：正确 S盒"),
    ("compact_wrong_sbox_auc", "紧凑网络：错误 S盒"),
    ("anchor_auc", "历史同预算锚点"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT/Dialga K1-W result chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1w_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1w_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"uknit64", "dialga128"}:
        raise ValueError("K1-W plot requires uKNIT and Dialga results")
    values = [seed_results[cipher][seed] for cipher, seed, _label in COLUMNS]
    auc_values = np.asarray(
        [[float(result[field]) for result in values] for field, _label in AUC_ROWS],
        dtype=float,
    )
    margin_values = np.asarray(
        [
            [float(result["exact_minus_anchor"]) for result in values],
            [float(result["exact_minus_wrong_sbox"]) for result in values],
        ],
        dtype=float,
    )
    retention_thresholds = [
        float(result["retention_threshold"]) - float(result["anchor_auc"])
        for result in values
    ]
    semantic_thresholds: list[float | None] = [0.010, 0.010, None, None]

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
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 8.8))
        figure.subplots_adjust(
            left=0.15,
            right=0.94,
            top=0.64,
            bottom=0.13,
            wspace=0.35,
        )
        figure.suptitle(
            "创新1 K1-W：紧凑的不变直方图网络能否同时保留 uKNIT 与 Dialga 信号",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "固定4对密文、2048/class、10个 epoch；只把重复的16-cell投影压缩为运行时 cell 数无关的40维投影，参数量为137516。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.81,
            _decision_text(gate),
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.745,
            "左图比较正确结构、错误 S盒与历史锚点；右图直接标出每个独立门控是否通过，避免相近数值难以分辨。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_heatmap(axes[0], auc_values)
        _margin_heatmap(
            axes[1],
            margin_values,
            retention_thresholds=retention_thresholds,
            semantic_thresholds=semantic_thresholds,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.2,
        "height_inches": 8.8,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_cipher_rounds_and_compaction": True,
        "values_annotated_to_four_decimals": True,
        "per_cell_gate_status_visible": True,
    }


def _auc_heatmap(axis: plt.Axes, values: np.ndarray) -> None:
    lower = min(0.45, float(values.min()) - 0.03)
    upper = max(1.0, float(values.max()) + 0.03)
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks(range(len(COLUMNS)), [column[2] for column in COLUMNS])
    axis.set_yticks(range(len(AUC_ROWS)), [row[1] for row in AUC_ROWS])
    axis.set_title("跨密钥验证 AUC（越高越好）", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.4f}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _margin_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    retention_thresholds: list[float],
    semantic_thresholds: list[float | None],
) -> None:
    limit = max(0.08, float(np.max(np.abs(values))) + 0.02)
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(range(len(COLUMNS)), [column[2] for column in COLUMNS])
    axis.set_yticks(
        (0, 1),
        (
            "正确结构 - 历史锚点\n门槛因任务而异",
            "正确结构 - 错误 S盒\nuKNIT 门槛 +0.010",
        ),
    )
    axis.set_title("净差值与独立门控", loc="left", fontweight="bold", pad=14)
    for column in range(values.shape[1]):
        retention = float(values[0, column])
        retention_passed = retention >= retention_thresholds[column]
        axis.text(
            column,
            0,
            f"{retention:+.4f}\n{'通过' if retention_passed else '未通过'}",
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#111827",
        )
        semantic = float(values[1, column])
        semantic_threshold = semantic_thresholds[column]
        semantic_status = (
            "仅描述"
            if semantic_threshold is None
            else ("通过" if semantic >= semantic_threshold else "未通过")
        )
        axis.text(
            column,
            1,
            f"{semantic:+.4f}\n{semantic_status}",
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#111827",
        )
    axis.tick_params(length=0, axis="both", pad=9)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_family_ctspn_k1w_compact_invariant_supported": (
            "裁决：uKNIT 与 Dialga 的每颗 seed 都保留锚点，紧凑架构可以进入独立 pair 数比较。"
        ),
        "innovation1_uknit_family_ctspn_k1w_semantic_attribution_failed": (
            "裁决：Dialga 保留强信号，但 uKNIT 没有稳定保留正确 S盒优势；暂停扩样，先审计直方图分支贡献。"
        ),
        "innovation1_uknit_family_ctspn_k1w_uknit_retention_failed": (
            "裁决：Dialga 保留，但 uKNIT 两颗 seed 未保留历史锚点；暂停该紧凑优化。"
        ),
        "innovation1_uknit_family_ctspn_k1w_dialga_retention_failed": (
            "裁决：uKNIT 通过，但 Dialga 未保留强信号；不能形成家族架构结论。"
        ),
        "innovation1_uknit_family_ctspn_k1w_protocol_invalid": (
            "裁决：计划、缓存、折叠或训练产物不完整，本次指标不可解释。"
        ),
    }
    decision = str(gate.get("decision", ""))
    return labels.get(decision, f"裁决：{decision}")


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status,
        "#374151",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1w_svg"]
