from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1z import ALPHAS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-Z audit chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    grid = read_jsonl(args.grid)
    report = render_k1z_svg(gate, grid, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1z_svg(
    gate: Mapping[str, Any],
    grid_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-Z plot requires seed3 and seed4 results")
    grid = {
        (int(row["seed"]), float(row["alpha"])): float(row["auc"])
        for row in grid_rows
    }
    if set(grid) != {(seed, alpha) for seed in (3, 4) for alpha in ALPHAS}:
        raise ValueError("K1-Z plot alpha grid is incomplete")

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
            left=0.11,
            right=0.95,
            top=0.64,
            bottom=0.14,
            wspace=0.36,
        )
        figure.suptitle(
            "创新1 K1-Z：放大已有结构分支能否恢复 uKNIT 信号",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "训练缓存只用于选择倍率；跨密钥验证缓存只用于确认。权重、数据和结构均未改变，优化器步数为0。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.81,
            "裁决：倍率放大保留正确 S盒优势，但两颗 seed 都未恢复 K1-T 锚点；仅靠推理缩放不足，转向投影权重优化。",
            ha="left",
            fontsize=11.2,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.745,
            "这不能证明紧凑表示无能力：折叠 K1-T 已构造出同一表示中的更高 AUC，只能说明当前学到的投影方向需要改变。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _grid_plot(axes[0], grid, seeds)
        _confirmation_table(axes[1], seeds)
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
        "selection_and_confirmation_separated": True,
        "all_confirmation_values_annotated": True,
    }


def _grid_plot(
    axis: plt.Axes,
    grid: Mapping[tuple[int, float], float],
    seeds: Mapping[str, Any],
) -> None:
    positions = np.arange(len(ALPHAS))
    colors = {3: "#2563EB", 4: "#D97706"}
    for seed in (3, 4):
        values = [grid[(seed, alpha)] for alpha in ALPHAS]
        axis.plot(
            positions,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=colors[seed],
            label=f"seed{seed} 训练 AUC",
        )
        selected = float(seeds[str(seed)]["selected_alpha"])
        index = ALPHAS.index(selected)
        axis.scatter(
            [index],
            [values[index]],
            marker="*",
            s=190,
            color=colors[seed],
            edgecolor="#111827",
            linewidth=0.8,
            zorder=4,
        )
        axis.annotate(
            f"选 {selected:g}×\n{values[index]:.4f}",
            (index, values[index]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axis.set_xticks(positions, [f"{alpha:g}" for alpha in ALPHAS])
    axis.set_xlabel("结构残差倍率 alpha（训练集选择）")
    axis.set_ylabel("训练缓存 AUC")
    axis.set_title("倍率发现曲线", loc="left", fontweight="bold", pad=14)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(loc="lower right", frameon=False)
    axis.set_ylim(0.47, 0.66)


def _confirmation_table(axis: plt.Axes, seeds: Mapping[str, Any]) -> None:
    rows = (
        ("validation_selected_exact_auc", "选定倍率：正确 S盒"),
        ("validation_selected_wrong_sbox_auc", "选定倍率：错误 S盒"),
        ("validation_alpha1_auc", "原倍率 1×"),
        ("anchor_auc", "K1-T 历史锚点"),
    )
    values = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in rows]
    )
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.48, vmax=0.61)
    axis.set_xticks((0, 1), ("uKNIT r5\nseed3", "uKNIT r5\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("跨密钥验证确认", loc="left", fontweight="bold", pad=14)
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
    for column, seed in enumerate(("3", "4")):
        passed = bool(
            float(seeds[seed]["validation_selected_exact_auc"])
            >= float(seeds[seed]["retention_threshold"])
        )
        axis.text(
            column,
            -0.42,
            "保留锚点：通过" if passed else "保留锚点：未通过",
            ha="center",
            va="center",
            fontsize=9.7,
            fontweight="bold",
            color="#047857" if passed else "#B91C1C",
        )
    axis.tick_params(length=0, axis="both", pad=9)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1z_svg"]
