from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


AUC_COLUMNS = (
    ("3", "k1w", "K1-W 紧凑\nseed3"),
    ("4", "k1w", "K1-W 紧凑\nseed4"),
    ("3", "k1t", "K1-T 折叠\nseed3"),
    ("4", "k1t", "K1-T 折叠\nseed4"),
)
AUC_ROWS = (
    ("exact", "正确 S盒"),
    ("zero", "关闭直方图分支"),
    ("wrong", "同权重错误 S盒"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-X audit chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1x_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1x_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-X plot requires seed3 and seed4 results")
    auc_values = np.asarray(
        [
            [
                _auc_value(seeds[seed], family, condition)
                for seed, family, _label in AUC_COLUMNS
            ]
            for condition, _label in AUC_ROWS
        ],
        dtype=float,
    )
    gate_values = np.asarray(
        [
            [float(seeds[seed]["folded_effective_update_ratio"]) for seed in ("3", "4")],
            [abs(float(seeds[seed]["k1w_full_minus_zero_auc"])) for seed in ("3", "4")],
            [abs(float(seeds[seed]["k1w_exact_minus_wrong_sbox_auc"])) for seed in ("3", "4")],
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
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 8.8))
        figure.subplots_adjust(
            left=0.15,
            right=0.95,
            top=0.64,
            bottom=0.13,
            wspace=0.38,
        )
        figure.suptitle(
            "创新1 K1-X：紧凑直方图网络的16倍更新假设是否足以解释失败",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "uKNIT r5，固定4对密文与原 seed3/4 检查点；只做推理和梯度读取，训练行数与优化器步数均为0。",
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
            "两颗 seed 都精确出现16倍折叠梯度，但 seed4 的结构分支仍有可测贡献，因此不能把失败只归因于更新过慢。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_heatmap(axes[0], auc_values)
        _gate_table(axes[1], gate_values)
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
        "all_values_annotated": True,
        "gate_status_visible_per_seed": True,
    }


def _auc_value(result: Mapping[str, Any], family: str, condition: str) -> float:
    prefixes = {"k1w": "k1w", "k1t": "k1t_folded"}
    suffixes = {
        "exact": "exact_auc",
        "zero": "zero_histogram_auc",
        "wrong": "wrong_sbox_same_checkpoint_auc",
    }
    return float(result[f"{prefixes[family]}_{suffixes[condition]}"])


def _auc_heatmap(axis: plt.Axes, values: np.ndarray) -> None:
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.48, vmax=0.61)
    axis.set_xticks(range(len(AUC_COLUMNS)), [column[2] for column in AUC_COLUMNS])
    axis.set_yticks(range(len(AUC_ROWS)), [row[1] for row in AUC_ROWS])
    axis.set_title("同一验证集上的 AUC", loc="left", fontweight="bold", pad=14)
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


def _gate_table(axis: plt.Axes, values: np.ndarray) -> None:
    status_values = np.asarray(
        [
            [1.0 if 15.999 <= value <= 16.001 else 0.0 for value in values[0]],
            [1.0 if value <= 0.010 else 0.0 for value in values[1]],
            [1.0 if value <= 0.010 else 0.0 for value in values[2]],
        ]
    )
    axis.imshow(status_values, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks((0, 1), ("uKNIT r5\nseed3", "uKNIT r5\nseed4"))
    axis.set_yticks(
        (0, 1, 2),
        (
            "折叠有效更新比\n门槛 15.999～16.001",
            "|完整 - 关闭分支| AUC\n门槛 ≤ 0.010",
            "|正确 - 错误 S盒| AUC\n门槛 ≤ 0.010",
        ),
    )
    axis.set_title("机制门控（每颗 seed 独立）", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            passed = bool(status_values[row, column])
            formatted = f"{value:.6f}" if row else f"{value:.3f}×"
            axis.text(
                column,
                row,
                f"{formatted}\n{'通过' if passed else '未通过'}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold",
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_family_ctspn_k1x_16x_optimization_geometry_supported": (
            "裁决：16倍更新关系与弱分支贡献在两颗 seed 都成立，可以进入单变量 K1-Y。"
        ),
        "innovation1_uknit_family_ctspn_k1x_optimization_geometry_not_sufficient": (
            "裁决：16倍关系成立，但不足以独立解释失败；不启动学习率放大，转向分支干扰审计。"
        ),
        "innovation1_uknit_family_ctspn_k1x_protocol_invalid": (
            "裁决：源绑定、重放或零训练约束失败，本次机制指标不可解释。"
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


__all__ = ["main", "parse_args", "render_k1x_svg"]
