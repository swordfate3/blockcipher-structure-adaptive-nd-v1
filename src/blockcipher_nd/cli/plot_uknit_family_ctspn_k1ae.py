from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AE chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ae_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ae_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"0", "1"}:
        raise ValueError("K1-AE plot requires seed0 and seed1 results")

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
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 9.2))
        figure.subplots_adjust(left=0.10, right=0.95, top=0.61, bottom=0.12, wspace=0.30)
        figure.suptitle(
            "创新1 K1-AE：Dialga 的近满分信号究竟来自哪条网络路径",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "冻结同一检查点和验证数据，只关闭两个结构残差的标量门；基础路径仍读取 GF(2) 线性扩散，但不执行 S盒。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.805,
            "裁决：只保留 GF(2) 基础路径仍接近满分，16对 Dialga 4轮任务被线性信号饱和，不适合审判 S盒语义。",
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.735,
            "下一步：先用现有缓存做单对重放审计，确认降低 pair 数是否解除饱和；不直接远程放大或更换网络。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_panel(axes[0], seeds)
        _drop_panel(axes[1], seeds)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.2,
        "height_inches": 9.2,
        "language": "zh-CN",
        "panels": 2,
        "all_values_annotated": True,
        "gf2_base_semantics_explained": True,
    }


def _auc_panel(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    rows = (
        ("full_auc", "完整模型"),
        ("histogram_off_auc", "关闭结构直方图残差"),
        ("edge_off_auc", "关闭精确组合边残差"),
        ("base_only_auc", "仅保留 GF(2) 基础路径"),
    )
    axis.set_title("同检查点跨密钥验证 AUC", loc="left", fontweight="bold", pad=14)
    _table_axes(axis, len(rows))
    for row, (field, label) in enumerate(rows):
        axis.text(-0.08, row + 0.5, label, ha="right", va="center", fontsize=10.2)
        for column, seed in enumerate(("0", "1")):
            value = float(seeds[seed][field])
            axis.add_patch(Rectangle((column, row), 1, 1, facecolor="#E0F2FE", edgecolor="#CBD5E1"))
            axis.text(column + 0.5, row + 0.5, f"{value:.6f}", ha="center", va="center", fontsize=10.8, fontweight="bold")


def _drop_panel(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    rows = (
        ("full_minus_histogram_off_auc", "完整 - 关闭直方图"),
        ("full_minus_edge_off_auc", "完整 - 关闭边残差"),
        ("full_minus_base_only_auc", "完整 - 仅基础路径"),
    )
    axis.set_title("结构残差的 AUC 必要性（门槛 ≥ +0.010）", loc="left", fontweight="bold", pad=14)
    _table_axes(axis, len(rows))
    for row, (field, label) in enumerate(rows):
        axis.text(-0.08, row + 0.5, label, ha="right", va="center", fontsize=10.2)
        for column, seed in enumerate(("0", "1")):
            value = float(seeds[seed][field])
            passed = value >= 0.010
            axis.add_patch(Rectangle((column, row), 1, 1, facecolor="#DCFCE7" if passed else "#FEE2E2", edgecolor="#CBD5E1"))
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:+.6f}\n{'必要' if passed else '不必要'}",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                linespacing=1.4,
            )


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


__all__ = ["main", "parse_args", "render_k1ae_svg"]
