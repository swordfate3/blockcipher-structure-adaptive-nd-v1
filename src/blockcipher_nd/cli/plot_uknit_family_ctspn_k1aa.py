from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AA chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1aa_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1aa_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-AA plot requires seed3 and seed4 results")
    auc_rows = (
        ("virtual_slot_exact_auc", "K1-AA：正确 S盒"),
        ("virtual_slot_wrong_sbox_auc", "K1-AA：错误 S盒"),
        ("k1y_exact_auc", "K1-Y：16倍学习率锚点"),
        ("k1t_invariant_auc", "K1-T：历史冗余锚点"),
    )
    auc_values = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in auc_rows]
    )
    margin_rows = (
        ("exact_minus_wrong_sbox", "正确 - 错误 S盒\n门槛 ≥ +0.010"),
        ("exact_minus_k1y", "K1-AA - K1-Y\n允许最多回退0.005"),
        ("exact_minus_k1t", "K1-AA - K1-T\n允许最多回退0.010"),
    )
    margins = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in margin_rows]
    )
    passed = gate.get("status") == "pass"

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
            left=0.17,
            right=0.95,
            top=0.64,
            bottom=0.13,
            wspace=0.42,
        )
        figure.suptitle(
            "创新1 K1-AA：固定16个虚拟投影槽能否稳定恢复 uKNIT 5轮信号",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "固定4对密文、2048/class、10个 epoch；只改变投影参数化，全部参数统一使用 Adam 1e-4。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        if passed:
            decision = (
                "裁决：两颗 seed 均保留历史信号并显著领先错误 S盒；虚拟槽参数化通过本地机制门。"
            )
            next_text = (
                "下一步：保持 K1-AA 不变，单独比较每条样本4对与16对密文；仍不直接远程放大。"
            )
            decision_color = "#047857"
        else:
            decision = (
                "裁决：至少一颗 seed 未同时通过信号保留与正确 S盒归因门槛，K1-AA 暂缓。"
            )
            next_text = (
                "下一步按门控审查初始化或语义归因；不得同时增加pair数、数据量或学习率。"
            )
            decision_color = "#B45309"
        figure.text(
            0.05,
            0.81,
            decision,
            ha="left",
            fontsize=11.1,
            fontweight="bold",
            color=decision_color,
        )
        figure.text(
            0.05,
            0.745,
            next_text,
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_table(axes[0], auc_values, auc_rows)
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
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.49, vmax=0.63)
    axis.set_xticks((0, 1), ("uKNIT 5轮\nseed3", "uKNIT 5轮\nseed4"))
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
    axis.set_xticks((0, 1), ("uKNIT 5轮\nseed3", "uKNIT 5轮\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("结构归因与锚点差值", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column, seed in enumerate(("3", "4")):
            value = float(values[row, column])
            if row == 0:
                cell_passed = value >= 0.010
            else:
                cell_passed = float(seeds[seed]["virtual_slot_exact_auc"]) >= float(
                    seeds[seed]["retention_threshold"]
                )
            axis.text(
                column,
                row,
                f"{value:+.4f}\n{'通过' if cell_passed else '未通过'}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold",
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1aa_svg"]
