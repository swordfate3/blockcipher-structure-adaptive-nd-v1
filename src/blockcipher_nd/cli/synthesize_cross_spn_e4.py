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

from blockcipher_nd.planning.cross_spn_e4_synthesis import (
    COMPARISON_SPECS,
    build_cross_spn_e4_synthesis,
)


PANEL_TITLES = {
    "scratch_margin": "相对同容量 scratch 的适配差值",
    "source_topology_margin": "正确 source topology 相对 shuffled source",
    "target_topology_margin": "正确 target topology 相对 shuffled target",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize the four verified E4-R4/R5 paired gate reports."
    )
    parser.add_argument("--gates", nargs=4, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "cells.csv", "summary.json", "curves.svg", "gate.json"):
        (output_dir / name).unlink(missing_ok=True)
    reports, load_errors = _load_gate_reports(args.gates)
    report = (
        build_cross_spn_e4_synthesis(reports)
        if not load_errors
        else {
            "status": "fail",
            "decision": "invalid_e4_cross_spn_synthesis",
            "errors": load_errors,
            "next_action": "repair_input_selection_without_interpreting_metrics",
        }
    )
    report["input_gates"] = [str(path) for path in args.gates]
    if report["status"] == "pass":
        artifacts = {
            "results": str(output_dir / "results.jsonl"),
            "cells": str(output_dir / "cells.csv"),
            "summary": str(output_dir / "summary.json"),
            "curves": str(output_dir / "curves.svg"),
            "gate": str(output_dir / "gate.json"),
        }
        report["artifacts"] = artifacts
        _write_results_jsonl(output_dir / "results.jsonl", report["cells"])
        _write_cells_csv(output_dir / "cells.csv", report["cells"])
        _render_forest_svg(output_dir / "curves.svg", report)
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


def _load_gate_reports(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read gate {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"gate {path} must contain a JSON object")
            continue
        reports.append(payload)
    return reports, errors


def _write_results_jsonl(path: Path, cells: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps(cell, ensure_ascii=False, sort_keys=True) + "\n")


def _write_cells_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_seed",
        "target_seed",
        "experiment_stage",
        "true_auc",
        "scratch_auc",
        "scratch_delta",
        "scratch_ci_lower",
        "scratch_ci_upper",
        "scratch_gate_pass",
        "source_topology_delta",
        "source_topology_ci_lower",
        "source_topology_ci_upper",
        "source_topology_gate_pass",
        "target_topology_delta",
        "target_topology_ci_lower",
        "target_topology_ci_upper",
        "target_topology_gate_pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            scratch = cell["comparisons"]["scratch_margin"]
            source = cell["comparisons"]["source_topology_margin"]
            target = cell["comparisons"]["target_topology_margin"]
            writer.writerow(
                {
                    "source_seed": cell["source_seed"],
                    "target_seed": cell["target_seed"],
                    "experiment_stage": cell["experiment_stage"],
                    "true_auc": cell["true_auc"],
                    "scratch_auc": cell["scratch_auc"],
                    "scratch_delta": scratch["delta_auc"],
                    "scratch_ci_lower": scratch["ci_lower"],
                    "scratch_ci_upper": scratch["ci_upper"],
                    "scratch_gate_pass": scratch["gate_pass"],
                    "source_topology_delta": source["delta_auc"],
                    "source_topology_ci_lower": source["ci_lower"],
                    "source_topology_ci_upper": source["ci_upper"],
                    "source_topology_gate_pass": source["gate_pass"],
                    "target_topology_delta": target["delta_auc"],
                    "target_topology_ci_lower": target["ci_lower"],
                    "target_topology_ci_upper": target["ci_upper"],
                    "target_topology_gate_pass": target["gate_pass"],
                }
            )


def _render_forest_svg(path: Path, report: dict[str, Any]) -> None:
    colors = {0: "#2563EB", 1: "#059669"}
    cells = report["cells"]
    y_positions = list(reversed(range(len(cells))))
    labels = [
        f"源 seed {cell['source_seed']} → 目标 seed {cell['target_seed']}"
        for cell in cells
    ]
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
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.8), sharey=True)
        fig.subplots_adjust(left=0.235, right=0.965, top=0.82, bottom=0.09, hspace=0.52)
        fig.suptitle(
            "创新1 E4：跨 SPN 控制差值与 95% 置信区间",
            x=0.08,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        fig.text(
            0.08,
            0.915,
            "GIFT-64 r6 · 65536/class · 目标训练恰好 1 轮 · "
            "四个 target cell；source 与 target seed 未完全交叉",
            ha="left",
            color="#526070",
            fontsize=9.8,
        )
        for axis, margin_key in zip(axes, COMPARISON_SPECS, strict=True):
            threshold = float(COMPARISON_SPECS[margin_key]["threshold"])
            comparisons = [cell["comparisons"][margin_key] for cell in cells]
            lower = min(0.0, min(float(item["ci_lower"]) for item in comparisons))
            upper = max(threshold, max(float(item["ci_upper"]) for item in comparisons))
            span = max(upper - lower, 0.01)
            axis.set_xlim(lower - 0.10 * span, upper + 0.24 * span)
            axis.axvline(0.0, color="#64748B", linewidth=1.0, zorder=1)
            axis.axvline(
                threshold,
                color="#DC2626",
                linewidth=1.15,
                linestyle=(0, (4, 3)),
                zorder=1,
            )
            axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.75)
            axis.grid(False, axis="y")
            axis.set_title(PANEL_TITLES[margin_key], loc="left", fontweight="bold", pad=8)
            axis.set_xlabel("AUC 差值（候选 - 对照）")
            axis.set_yticks(y_positions, labels)
            for y, cell, comparison in zip(
                y_positions,
                cells,
                comparisons,
                strict=True,
            ):
                point = float(comparison["delta_auc"])
                ci_lower = float(comparison["ci_lower"])
                ci_upper = float(comparison["ci_upper"])
                color = colors[int(cell["source_seed"])]
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
                    ci_upper + 0.025 * span,
                    y,
                    f"{point:+.4f}",
                    va="center",
                    ha="left",
                    fontsize=9.0,
                    color="#334155",
                )
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0], label="PRESENT source seed 0", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[1], label="PRESENT source seed 1", markersize=7),
            Line2D([0], [0], color="#DC2626", linestyle=(0, (4, 3)), label="预声明点阈值"),
        ]
        fig.legend(
            handles=legend,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.972),
            frameon=False,
            ncol=1,
        )
        fig.text(
            0.08,
            0.025,
            "裁决：source/target topology 归因 4/4 通过；相对 scratch 仅 2/4 通过。"
            "保留类型拓扑结果，停止 E4-R6 与机械扩样本。",
            ha="left",
            fontsize=9.6,
            color="#334155",
        )
        fig.savefig(path, format="svg")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
