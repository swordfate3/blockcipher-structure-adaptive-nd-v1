from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-Y result chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1y_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1y_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-Y plot requires seed3 and seed4 results")
    rows = (
        ("projection16x_exact_auc", "K1-Y：正确 S盒"),
        ("projection16x_wrong_sbox_auc", "K1-Y：错误 S盒"),
        ("k1w_exact_auc", "K1-W：原紧凑训练"),
        ("k1t_invariant_auc", "K1-T：历史锚点"),
    )
    auc_values = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in rows]
    )
    margin_rows = (
        ("exact_minus_k1w", "K1-Y 正确 - K1-W\n门槛 ≥ +0.020"),
        ("exact_minus_wrong_sbox", "K1-Y 正确 - 错误 S盒\n门槛 ≥ +0.010"),
        ("exact_minus_k1t", "K1-Y 正确 - K1-T\n保留门槛含0.550下限"),
    )
    margins = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in margin_rows]
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
            left=0.16,
            right=0.95,
            top=0.64,
            bottom=0.13,
            wspace=0.40,
        )
        figure.suptitle(
            "创新1 K1-Y：只加快紧凑投影权重能否恢复 uKNIT r5 信号",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "固定4对密文、2048/class、10个 epoch；仅5120个投影权重使用16倍学习率，其余132396个参数和全部数据协议不变。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.81,
            "裁决：两颗 seed 都明显提升且保留正确 S盒优势；seed4恢复锚点，seed3距0.550下限仅差0.0011，整体仍为 hold。",
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.745,
            "16倍更新方向得到强支持，但未在每颗 seed 独立通过；停止倍率调参，下一步改为固定几何的冗余投影参数化。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_table(axes[0], auc_values, rows)
        _margin_table(axes[1], margins, margin_rows, seeds)
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
        "per_seed_gate_status_visible": True,
    }


def _auc_table(axis: plt.Axes, values: np.ndarray, rows: tuple) -> None:
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.49, vmax=0.61)
    axis.set_xticks((0, 1), ("uKNIT r5\nseed3", "uKNIT r5\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("跨密钥验证 AUC", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.4f}",
                ha="center",
                va="center",
                fontsize=10.6,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


def _margin_table(
    axis: plt.Axes,
    values: np.ndarray,
    rows: tuple,
    seeds: Mapping[str, Any],
) -> None:
    limit = max(0.10, float(np.abs(values).max()) + 0.02)
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    axis.set_xticks((0, 1), ("uKNIT r5\nseed3", "uKNIT r5\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("净提升与独立门控", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column, seed in enumerate(("3", "4")):
            value = float(values[row, column])
            if row == 0:
                passed = value >= 0.020
            elif row == 1:
                passed = value >= 0.010
            else:
                passed = float(seeds[seed]["projection16x_exact_auc"]) >= float(
                    seeds[seed]["retention_threshold"]
                )
            axis.text(
                column,
                row,
                f"{value:+.4f}\n{'通过' if passed else '未通过'}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold",
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1y_svg"]
