from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AD chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ad_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ad_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"0", "1"}:
        raise ValueError("K1-AD plot requires seed0 and seed1 results")
    passed = gate.get("status") == "pass"

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
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 8.8))
        figure.subplots_adjust(left=0.08, right=0.95, top=0.62, bottom=0.13, wspace=0.25)
        figure.suptitle(
            "创新1 K1-AD：同一组权重是否真正依赖 Dialga 的正确 S盒",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "冻结每颗 seed 的正确分支最佳权重和验证数据，只把运行时 S盒从正确版本替换为错误版本；全程零训练。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        if passed:
            decision = "裁决：两颗 seed 的正确 S盒都带来至少 +0.010 AUC，支持同检查点功能性使用。"
            next_text = "下一步：保持数据和网络不变，只测试一种训练时反事实归因约束。"
            color = "#047857"
        else:
            decision = "裁决：同一检查点下正确 S盒没有稳定带来 +0.010 AUC，判别性结构归因未通过。"
            next_text = "下一步：K1-AE 零训练拆分基础路径与结构直方图残差，先查强信号走了哪条路径。"
            color = "#B45309"
        figure.text(0.05, 0.805, decision, ha="left", fontsize=11.0, fontweight="bold", color=color)
        figure.text(0.05, 0.74, next_text, ha="left", fontsize=10.2, color="#4B5563")

        _auc_panel(axes[0], seeds)
        _attribution_panel(axes[1], seeds)
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
        "same_checkpoint_explanation_visible": True,
    }


def _auc_panel(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    axis.set_title("同检查点跨密钥验证 AUC", loc="left", fontweight="bold", pad=14)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.invert_yaxis()
    labels = ("正确 S盒", "错误 S盒")
    fields = ("exact_auc", "wrong_sbox_auc")
    for row, (label, field) in enumerate(zip(labels, fields)):
        axis.text(-0.08, row + 0.5, label, ha="right", va="center", fontsize=10.8)
        for column, seed in enumerate(("0", "1")):
            value = float(seeds[seed][field])
            axis.add_patch(Rectangle((column, row), 1, 1, facecolor="#DCFCE7", edgecolor="#CBD5E1"))
            axis.text(column + 0.5, row + 0.5, f"{value:.6f}", ha="center", va="center", fontsize=11.2, fontweight="bold")
    axis.set_xticks((0.5, 1.5), ("seed0", "seed1"))
    axis.set_yticks(())
    axis.tick_params(axis="x", length=0, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _attribution_panel(axis: plt.Axes, seeds: Mapping[str, Mapping[str, Any]]) -> None:
    axis.set_title("逐 seed 结构归因门槛", loc="left", fontweight="bold", pad=14)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.invert_yaxis()
    rows = (
        ("exact_minus_wrong_sbox_auc", "正确 - 错误 S盒 AUC\n门槛 ≥ +0.010", 0.010),
        ("max_abs_probability_delta", "最大预测概率变化\n门槛 > 0.000001", 1e-6),
    )
    for row, (field, label, threshold) in enumerate(rows):
        axis.text(-0.08, row + 0.5, label, ha="right", va="center", fontsize=10.2, linespacing=1.3)
        for column, seed in enumerate(("0", "1")):
            value = float(seeds[seed][field])
            passed = value >= threshold if row == 0 else value > threshold
            face = "#DCFCE7" if passed else "#FEE2E2"
            axis.add_patch(Rectangle((column, row), 1, 1, facecolor=face, edgecolor="#CBD5E1"))
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:+.6f}\n{'通过' if passed else '未通过'}",
                ha="center",
                va="center",
                fontsize=10.6,
                fontweight="bold",
                linespacing=1.4,
            )
    axis.set_xticks((0.5, 1.5), ("seed0", "seed1"))
    axis.set_yticks(())
    axis.tick_params(axis="x", length=0, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ad_svg"]
