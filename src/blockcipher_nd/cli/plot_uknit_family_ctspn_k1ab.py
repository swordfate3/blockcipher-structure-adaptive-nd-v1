from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AB chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ab_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ab_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-AB plot requires seed3 and seed4 results")
    auc_rows = (
        ("exact_16pair_auc", "K1-AB：16对正确 S盒"),
        ("wrong_sbox_16pair_auc", "K1-AB：16对错误 S盒"),
        ("k1aa_exact_4pair_auc", "K1-AA：4对正确 S盒"),
        ("k1v_invariant_16pair_auc", "K1-V：16对不变锚点"),
    )
    auc_values = np.asarray(
        [[float(seeds[seed][field]) for seed in ("3", "4")] for field, _ in auc_rows]
    )
    margin_rows = (
        ("exact16_minus_exact4", "16对 - 4对\n门槛 ≥ +0.010"),
        ("exact16_minus_wrong_sbox16", "正确 - 错误 S盒\n门槛 ≥ +0.010"),
        ("exact16_minus_k1v_invariant16", "K1-AB - K1-V不变\n允许回退0.020"),
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
            "创新1 K1-AB：同一虚拟槽网络从4对提升到16对密文是否有效",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "uKNIT 5轮、2048/class、10个 epoch；模型与训练协议不变，只增加每条样本的独立密文对数量。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        if passed:
            decision = (
                "裁决：两颗 seed 均获得16对增益、保留正确 S盒归因，并复现历史16对不变分支能力。"
            )
            next_text = (
                "下一步：保留 K1-AA + 16对设置，单独检查 Dialga 4轮保留性；本结果仍不是正式规模训练。"
            )
            decision_color = "#047857"
        else:
            decision = (
                "裁决：至少一颗 seed 未通过增加密文对、正确 S盒归因或历史16对能力门槛。"
            )
            next_text = (
                "下一步按失败门控保留4对或审查聚合；不得同时扩大数据、轮数或修改网络。"
            )
            decision_color = "#B45309"
        figure.text(
            0.05,
            0.81,
            decision,
            ha="left",
            fontsize=11.0,
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
        _table(axes[0], auc_values, auc_rows, "跨密钥验证 AUC", 0.49, 0.75)
        _margin_table(axes[1], margins, margin_rows)
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


def _table(
    axis: plt.Axes,
    values: np.ndarray,
    rows: tuple,
    title: str,
    minimum: float,
    maximum: float,
) -> None:
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=minimum, vmax=maximum)
    axis.set_xticks((0, 1), ("uKNIT 5轮\nseed3", "uKNIT 5轮\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
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


def _margin_table(axis: plt.Axes, values: np.ndarray, rows: tuple) -> None:
    limit = max(0.12, float(np.abs(values).max()) + 0.02)
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    axis.set_xticks((0, 1), ("uKNIT 5轮\nseed3", "uKNIT 5轮\nseed4"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("增加密文对的净价值", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            passed = value >= (0.010 if row < 2 else -0.020)
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


__all__ = ["main", "parse_args", "render_k1ab_svg"]
