from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from blockcipher_nd.planning.cross_spn_e5_source_objective_gate import (
    gate_cross_spn_e5_source_objective_joint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly gate E5-R0 target seeds 2 and 3."
    )
    parser.add_argument("--seed2-gate", required=True, type=Path)
    parser.add_argument("--seed3-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--plot-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.seed2_gate, args.seed3_gate)
    ]
    report = gate_cross_spn_e5_source_objective_joint(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["status"] == "pass":
        if args.summary_csv is not None:
            write_joint_summary_csv(report, args.summary_csv)
        if args.plot_output is not None:
            render_joint_gate_svg(report, args.plot_output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def write_joint_summary_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_seed",
        "control_role",
        "candidate_auc",
        "control_auc",
        "auc_difference",
        "ci_lower",
        "ci_upper",
        "point_pass",
        "ci_pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in (2, 3):
            seed_report = report["per_seed"][str(seed)]
            candidate_auc = float(seed_report["aucs"]["candidate_transfer"])
            for role in ("off_transfer", "placebo_transfer", "scratch"):
                comparison = seed_report["bootstrap"]["comparisons"][role]
                writer.writerow(
                    {
                        "target_seed": seed,
                        "control_role": role,
                        "candidate_auc": candidate_auc,
                        "control_auc": float(seed_report["aucs"][role]),
                        "auc_difference": float(seed_report["margins"][role]),
                        "ci_lower": float(comparison["ci_lower"]),
                        "ci_upper": float(comparison["ci_upper"]),
                        "point_pass": bool(seed_report["point_pass"][role]),
                        "ci_pass": bool(seed_report["ci_pass"][role]),
                    }
                )


def render_joint_gate_svg(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roles = (
        ("off_transfer", "候选 - 关闭辅助损失的迁移基线"),
        ("placebo_transfer", "候选 - 同容量打乱安慰剂"),
        ("scratch", "候选 - GIFT 从零训练"),
    )
    colors = {2: "#2563EB", 3: "#059669"}
    rc_params = {
        "font.family": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"],
        "font.size": 9.8,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0F172A",
        "xtick.color": "#475569",
        "ytick.color": "#334155",
        "text.color": "#0F172A",
        "savefig.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
    }
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(3, 1, figsize=(12.6, 9.4), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.12, hspace=0.55)
        fig.suptitle(
            "创新1 E5-R0：源拓扑反事实辅助目标的迁移门控",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        fig.text(
            0.075,
            0.915,
            "PRESENT-80 r7 源 seed 0 → GIFT-64 r6｜训练 8,192/类｜"
            "目标只训练 1 轮｜误差线为配对 bootstrap 95% CI",
            ha="left",
            color="#526070",
            fontsize=9.8,
        )
        fig.text(
            0.075,
            0.875,
            "通过要求：每个差值 ≥ +0.004，且置信区间下界 > 0；两颗目标 seed 必须全部通过。",
            ha="left",
            color="#526070",
            fontsize=9.4,
        )
        all_comparisons = [
            report["per_seed"][str(seed)]["bootstrap"]["comparisons"][role]
            for role, _ in roles
            for seed in (2, 3)
        ]
        lower = min(-0.004, min(float(item["ci_lower"]) for item in all_comparisons))
        upper = max(0.004, max(float(item["ci_upper"]) for item in all_comparisons))
        span = max(upper - lower, 0.02)
        limits = (lower - 0.08 * span, upper + 0.20 * span)
        for axis, (role, title) in zip(axes, roles, strict=True):
            axis.set_xlim(*limits)
            axis.axvline(0.0, color="#64748B", linewidth=1.0, zorder=1)
            axis.axvline(
                0.004,
                color="#DC2626",
                linewidth=1.15,
                linestyle=(0, (4, 3)),
                zorder=1,
            )
            axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.75)
            axis.grid(False, axis="y")
            axis.set_title(title, loc="left", fontweight="bold", pad=8)
            axis.set_yticks((1, 0), ("目标 seed 2", "目标 seed 3"))
            for y, seed in zip((1, 0), (2, 3), strict=True):
                seed_report = report["per_seed"][str(seed)]
                point = float(seed_report["margins"][role])
                comparison = seed_report["bootstrap"]["comparisons"][role]
                ci_lower = float(comparison["ci_lower"])
                ci_upper = float(comparison["ci_upper"])
                axis.errorbar(
                    point,
                    y,
                    xerr=[[point - ci_lower], [ci_upper - point]],
                    fmt="o",
                    color=colors[seed],
                    ecolor=colors[seed],
                    markersize=7.0,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    elinewidth=2.0,
                    capsize=4.0,
                    zorder=3,
                )
                axis.text(
                    ci_upper + 0.02 * span,
                    y,
                    f"{point:+.4f}",
                    va="center",
                    ha="left",
                    fontsize=9.0,
                    color="#334155",
                )
        axes[-1].set_xlabel("AUC 差值（候选 - 对照）")
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[2], label="目标 seed 2", markersize=7),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[3], label="目标 seed 3", markersize=7),
                Line2D([0], [0], color="#DC2626", linestyle=(0, (4, 3)), label="点差阈值 +0.004"),
            ],
            loc="upper right",
            bbox_to_anchor=(0.96, 0.97),
            frameon=False,
        )
        fig.text(
            0.075,
            0.035,
            "裁决：两颗 seed 均未超过关闭辅助损失的迁移基线；拒绝 E5-R0，"
            "不运行 source seed 1 或 65,536/class 远程扩展。",
            ha="left",
            fontsize=9.6,
            color="#334155",
        )
        fig.savefig(path, format="svg")
        plt.close(fig)


__all__ = ["main", "parse_args", "render_joint_gate_svg", "write_joint_summary_csv"]
