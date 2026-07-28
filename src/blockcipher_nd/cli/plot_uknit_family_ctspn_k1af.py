from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AF chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1af_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1af_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"0", "1"}:
        raise ValueError("K1-AF plot requires seed0 and seed1 results")
    if gate.get("status") != "hold" or gate.get("failed_protocol_checks"):
        raise ValueError("K1-AF plot requires a protocol-valid held result")

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.8,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure = plt.figure(figsize=(16.2, 9.2))
        grid = figure.add_gridspec(
            2,
            2,
            left=0.11,
            right=0.96,
            top=0.61,
            bottom=0.18,
            hspace=0.66,
            wspace=0.32,
            height_ratios=(0.9, 1.1),
        )
        auc_axis = figure.add_subplot(grid[0, 0])
        margin_axis = figure.add_subplot(grid[0, 1])
        position_axis = figure.add_subplot(grid[1, :])

        figure.suptitle(
            "创新1 K1-AF：Dialga 单 pair 解除了饱和，但正确 S盒仍未领先",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "冻结 K1-AC 检查点和验证数据，把每条 16-pair 样本拆成单 pair 观察；不训练、不选位置、不生成新数据。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.805,
            "裁决：两颗 seed 的单-pair AUC 都约为 0.80，但正确 S盒均低于错误 S盒，关闭 Dialga 单-pair 训练路线。",
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.735,
            "下一步：保留 uKNIT 五轮 16-pair 正向证据；Dialga 只作为 GF(2) 信号校准，转向另一共享组件密码表面。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )

        _auc_table(auc_axis, seeds)
        _margin_table(margin_axis, seeds)
        _position_panel(position_axis, seeds)

        aggregate = "；".join(
            f"seed{seed}={float(seeds[seed]['mean_query_exact_auc']):.6f}"
            for seed in ("0", "1")
        )
        figure.text(
            0.05,
            0.085,
            f"支持指标：16 次单-pair 概率取均值后的 AUC 为 {aggregate}。这是应用级多查询聚合，不是原始单-pair 指标。",
            ha="left",
            fontsize=9.8,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.040,
            "范围：Dialga-128 4轮，2048/class 既有跨密钥验证缓存，两颗 seed，零训练机制审计；不是正式训练或 SOTA 证据。",
            ha="left",
            fontsize=9.4,
            color="#6B7280",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.2,
        "height_inches": 9.2,
        "language": "zh-CN",
        "panels": 3,
        "close_values_rendered_as_tables": True,
        "per_position_margin_panel": True,
        "aggregation_scope_disclaimer": True,
    }


def _auc_table(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    rows = (
        ("pooled_exact_auc", "正确 S盒"),
        ("pooled_wrong_sbox_auc", "错误 S盒"),
    )
    axis.set_title("汇总单-pair AUC（32768 行/seed）", loc="left", fontweight="bold", pad=14)
    _table_axes(axis, len(rows))
    for row, (field, label) in enumerate(rows):
        axis.text(-0.06, row + 0.5, label, ha="right", va="center", fontsize=10.4)
        for column, seed in enumerate(("0", "1")):
            value = float(seeds[seed][field])
            axis.add_patch(
                Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor="#E0F2FE" if field == "pooled_exact_auc" else "#F3F4F6",
                    edgecolor="#CBD5E1",
                )
            )
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:.6f}",
                ha="center",
                va="center",
                fontsize=11.0,
                fontweight="bold",
            )


def _margin_table(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    axis.set_title("正确 - 错误 S盒（要求 ≥ +0.010）", loc="left", fontweight="bold", pad=14)
    _table_axes(axis, 2)
    for row, seed in enumerate(("0", "1")):
        value = float(seeds[seed]["pooled_exact_minus_wrong_auc"])
        passed = value >= 0.010
        axis.text(-0.06, row + 0.5, f"seed{seed}", ha="right", va="center", fontsize=10.4)
        axis.add_patch(
            Rectangle(
                (0, row),
                2,
                1,
                facecolor="#DCFCE7" if passed else "#FEE2E2",
                edgecolor="#CBD5E1",
            )
        )
        axis.text(
            1,
            row + 0.5,
            f"{value:+.6f}  {'通过' if passed else '未通过'}",
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
        )
    axis.set_xticks(())


def _position_panel(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    colors = {"0": "#0369A1", "1": "#C2410C"}
    offsets = {"0": -0.10, "1": 0.10}
    for seed in ("0", "1"):
        positions = seeds[seed]["per_position"]
        x = np.asarray([int(row["pair_position"]) for row in positions], dtype=float)
        y = np.asarray([float(row["exact_minus_wrong_auc"]) for row in positions])
        axis.scatter(
            x + offsets[seed],
            y,
            s=48,
            color=colors[seed],
            edgecolor="white",
            linewidth=0.7,
            label=f"seed{seed}",
            zorder=3,
        )
    axis.axhline(0.010, color="#15803D", linewidth=1.4, linestyle="--", label="通过门槛 +0.010")
    axis.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle=":")
    axis.set_title("16 个原始 pair 位置的正确-S盒优势（仅稳定性检查，不选有利位置）", loc="left", fontweight="bold", pad=12)
    axis.set_xlabel("原始 pair 位置")
    axis.set_ylabel("AUC 差值")
    axis.set_xticks(range(16))
    axis.set_xlim(-0.6, 15.6)
    axis.set_ylim(-0.018, 0.013)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", ncols=3, frameon=False)


def _table_axes(axis: plt.Axes, row_count: int) -> None:
    axis.set_xlim(0, 2)
    axis.set_ylim(0, row_count)
    axis.invert_yaxis()
    axis.set_xticks((0.5, 1.5), ("seed0", "seed1"))
    axis.set_yticks(())
    axis.tick_params(axis="x", length=0, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1af_svg"]
