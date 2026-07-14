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

from blockcipher_nd.planning.cross_spn_e4_e6_synthesis import (
    build_cross_spn_e4_e6_synthesis,
)


PANEL_TITLES = {
    "off_transfer": "候选源目标 - 未修改的 off-transfer anchor",
    "placebo_transfer": "候选源目标 - 同机制 placebo",
    "scratch": "候选迁移 - GIFT 从零训练",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize frozen E4, E5, and E6 cross-SPN evidence."
    )
    parser.add_argument("--e4-synthesis", required=True, type=Path)
    parser.add_argument("--e5-gate", required=True, type=Path)
    parser.add_argument("--e6-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, errors = _load_inputs(
        [args.e4_synthesis, args.e5_gate, args.e6_gate]
    )
    report = (
        build_cross_spn_e4_e6_synthesis(*inputs)
        if not errors
        else {
            "status": "fail",
            "decision": "invalid_e4_e6_synthesis",
            "errors": errors,
            "next_action": "repair_input_selection_without_interpreting_metrics",
        }
    )
    report["input_artifacts"] = {
        "e4_synthesis": str(args.e4_synthesis),
        "e5_gate": str(args.e5_gate),
        "e6_gate": str(args.e6_gate),
    }
    if report["status"] == "pass":
        report["artifacts"] = {
            name: str(output_dir / filename)
            for name, filename in {
                "results": "results.jsonl",
                "cells": "cells.csv",
                "summary": "summary.json",
                "gate": "gate.json",
                "curves": "curves.svg",
            }.items()
        }
        _write_results(output_dir / "results.jsonl", report["objective_cells"])
        _write_cells_csv(output_dir / "cells.csv", report["objective_cells"])
        _render_svg(output_dir / "curves.svg", report)
        (output_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "errors": report["errors"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


def _load_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"input {path} must contain a JSON object")
            continue
        payloads.append(payload)
    if len(payloads) != 3 and not errors:
        errors.append("synthesis requires exactly three input objects")
    return payloads, errors


def _write_results(path: Path, cells: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps(cell, ensure_ascii=False, sort_keys=True) + "\n")


def _write_cells_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_stage",
        "objective_label",
        "source_seed",
        "target_seed",
        "samples_per_class",
        "candidate_auc",
        "off_auc",
        "placebo_auc",
        "scratch_auc",
        "candidate_minus_off",
        "candidate_minus_off_ci_lower",
        "candidate_minus_off_ci_upper",
        "candidate_minus_placebo",
        "candidate_minus_placebo_ci_lower",
        "candidate_minus_placebo_ci_upper",
        "candidate_minus_scratch",
        "candidate_minus_scratch_ci_lower",
        "candidate_minus_scratch_ci_upper",
        "complete_gate_pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            off = cell["comparisons"]["off_transfer"]
            placebo = cell["comparisons"]["placebo_transfer"]
            scratch = cell["comparisons"]["scratch"]
            writer.writerow(
                {
                    "experiment_stage": cell["experiment_stage"],
                    "objective_label": cell["objective_label"],
                    "source_seed": cell["source_seed"],
                    "target_seed": cell["target_seed"],
                    "samples_per_class": cell["samples_per_class"],
                    "candidate_auc": cell["candidate_auc"],
                    "off_auc": cell["off_auc"],
                    "placebo_auc": cell["placebo_auc"],
                    "scratch_auc": cell["scratch_auc"],
                    "candidate_minus_off": off["delta_auc"],
                    "candidate_minus_off_ci_lower": off["ci_lower"],
                    "candidate_minus_off_ci_upper": off["ci_upper"],
                    "candidate_minus_placebo": placebo["delta_auc"],
                    "candidate_minus_placebo_ci_lower": placebo["ci_lower"],
                    "candidate_minus_placebo_ci_upper": placebo["ci_upper"],
                    "candidate_minus_scratch": scratch["delta_auc"],
                    "candidate_minus_scratch_ci_lower": scratch["ci_lower"],
                    "candidate_minus_scratch_ci_upper": scratch["ci_upper"],
                    "complete_gate_pass": cell["complete_gate_pass"],
                }
            )


def _render_svg(path: Path, report: dict[str, Any]) -> None:
    colors = {"e5": "#2563EB", "e6": "#059669"}
    cells = report["objective_cells"]
    labels = [
        f"{cell['experiment_stage'].upper()} · 目标 seed {cell['target_seed']}"
        for cell in cells
    ]
    y_positions = list(reversed(range(len(cells))))
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
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.0), sharey=True)
        fig.subplots_adjust(left=0.19, right=0.96, top=0.80, bottom=0.12, hspace=0.52)
        fig.suptitle(
            "创新1 E4-E6：跨 SPN 源目标与结构归因综合",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        fig.text(
            0.075,
            0.915,
            "E5/E6：GIFT-64 r6 · 8,192/类 · 目标训练 1 轮 · "
            "配对 bootstrap 95% CI；E4 中等规模结果仅在页脚单独汇总",
            ha="left",
            color="#526070",
            fontsize=9.5,
        )
        comparisons = [
            item
            for role in PANEL_TITLES
            for cell in cells
            for item in (cell["comparisons"][role],)
        ]
        lower = min(0.0, min(float(item["ci_lower"]) for item in comparisons))
        upper = max(0.004, max(float(item["ci_upper"]) for item in comparisons))
        span = max(upper - lower, 0.02)
        for axis, role in zip(axes, PANEL_TITLES, strict=True):
            axis.set_xlim(lower - 0.08 * span, upper + 0.18 * span)
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
            axis.set_title(PANEL_TITLES[role], loc="left", fontweight="bold", pad=8)
            axis.set_yticks(y_positions, labels)
            for y, cell in zip(y_positions, cells, strict=True):
                comparison = cell["comparisons"][role]
                point = float(comparison["delta_auc"])
                ci_lower = float(comparison["ci_lower"])
                ci_upper = float(comparison["ci_upper"])
                color = colors[cell["experiment_stage"]]
                axis.errorbar(
                    point,
                    y,
                    xerr=[[point - ci_lower], [ci_upper - point]],
                    fmt="o",
                    color=color,
                    ecolor=color,
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
                Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["e5"], label="E5 topology-identity BCE", markersize=7),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["e6"], label="E6 functional margin", markersize=7),
                Line2D([0], [0], color="#DC2626", linestyle=(0, (4, 3)), label="点差阈值 +0.004"),
            ],
            loc="upper right",
            bbox_to_anchor=(0.96, 0.97),
            frameon=False,
        )
        e4 = report["e4_representation"]
        fig.text(
            0.075,
            0.045,
            "E4 中等规模独立结论：source topology 4/4，target topology 4/4，"
            f"scratch efficiency {e4['scratch_margin']['pass_count']}/4。"
            "E5/E6 新源目标完整门控 0/4；保留 typed topology，停止源目标扩展。",
            ha="left",
            fontsize=9.5,
            color="#334155",
        )
        fig.savefig(path, format="svg")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
