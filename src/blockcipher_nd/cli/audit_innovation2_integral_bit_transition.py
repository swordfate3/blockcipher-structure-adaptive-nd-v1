from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    ACTIVE_BIT_WIDTHS,
    BitTransitionAuditConfig,
    run_bit_transition_audit,
)


COLORS = {5: "#2563EB", 6: "#059669", 7: "#DC2626"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r6 output properties at 5/6/7 active bits."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structures-per-width", type=int, default=64)
    parser.add_argument("--keys-per-structure", type=int, default=256)
    parser.add_argument("--structure-chunk-size", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BitTransitionAuditConfig(
        run_id=args.run_id,
        rounds=args.rounds,
        seed=args.seed,
        structures_per_width=args.structures_per_width,
        keys_per_structure=args.keys_per_structure,
        structure_chunk_size=args.structure_chunk_size,
        ridge_alpha=args.ridge_alpha,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "rounds": args.rounds,
            "seed": args.seed,
            "structures_per_width": args.structures_per_width,
            "keys_per_structure": args.keys_per_structure,
            "active_bit_widths": list(ACTIVE_BIT_WIDTHS),
            "training_performed": False,
        },
        mode="w",
    )
    result = run_bit_transition_audit(
        config,
        progress_callback=progress_callback,
    )
    results_path = args.output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "structure_rates.csv", result["structure_rows"])
    _write_csv(
        args.output_root / "marginal_predictions.csv",
        result["marginal_rows"],
    )
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_bit_transition_svg(
        result["rows"],
        result["structure_rows"],
        result["gate"],
        args.output_root / "curves.svg",
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "selected_active_bit_width": result["gate"][
                    "selected_active_bit_width"
                ],
                "output_root": str(args.output_root),
                "next_action": result["gate"]["next_action"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_bit_transition_svg(
    summary_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_width = {int(row["active_bit_width"]): row for row in summary_rows}
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.9))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.22,
            wspace=0.30,
        )
        figure.suptitle(
            "创新2 E8：PRESENT 6轮输出性质的细粒度活动 bit 审计",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            (
                f"每种宽度 {int(summary_rows[0]['structures'])} 个结构、"
                f"每结构 {int(summary_rows[0]['keys_per_structure'])} 把密钥；"
                "两半密钥独立估计可重复残差信号，无神经训练。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        distribution_axis, residual_axis = axes
        rng = np.random.default_rng(23)
        for x, width in enumerate(ACTIVE_BIT_WIDTHS):
            rates = np.asarray(
                [
                    float(row["balance_rate"])
                    for row in structure_rows
                    if int(row["active_bit_width"]) == width
                ]
            )
            jitter = rng.uniform(-0.16, 0.16, size=len(rates))
            distribution_axis.scatter(
                np.full(len(rates), x) + jitter,
                rates,
                s=24,
                alpha=0.66,
                color=COLORS[width],
                edgecolors="none",
            )
            distribution_axis.hlines(
                float(rates.mean()),
                x - 0.24,
                x + 0.24,
                color="#0F172A",
                linewidth=2.1,
            )
        distribution_axis.set_title(
            "结构级平衡率分布",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        distribution_axis.set_ylabel("跨 256 把密钥的平衡率  P(q=0)")
        distribution_axis.set_xticks(
            range(len(ACTIVE_BIT_WIDTHS)),
            labels=[f"{width} bit（{1 << width} 明文）" for width in ACTIVE_BIT_WIDTHS],
        )
        distribution_axis.set_ylim(-0.03, 1.03)
        distribution_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        distribution_axis.grid(False, axis="x")

        metric_names = ["结构可重复差异", "扣除输出位置", "扣除全部字段边际"]
        x_positions = np.arange(len(metric_names), dtype=np.float64)
        bar_width = 0.24
        for width_index, width in enumerate(ACTIVE_BIT_WIDTHS):
            values = [
                float(by_width[width]["cross_half_structure_std"]),
                float(
                    by_width[width]["cross_half_output_position_residual_std"]
                ),
                float(
                    by_width[width]["cross_half_combined_marginal_residual_std"]
                ),
            ]
            bars = residual_axis.bar(
                x_positions + (width_index - 1) * bar_width,
                values,
                bar_width,
                color=COLORS[width],
                label=f"{width} 活动 bit",
            )
            residual_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
        residual_axis.hlines(
            0.03,
            -0.38,
            0.38,
            color="#64748B",
            linestyle="--",
            linewidth=1.0,
            label="结构差异门槛 0.03",
        )
        residual_axis.hlines(
            0.02,
            1.62,
            2.38,
            color="#94A3B8",
            linestyle=":",
            linewidth=1.0,
            label="组合残差门槛 0.02",
        )
        residual_axis.set_title(
            "两半密钥可重复信号",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        residual_axis.set_ylabel("协方差估计标准差")
        residual_axis.set_xticks(x_positions, labels=metric_names)
        residual_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        residual_axis.grid(False, axis="x")
        maximum_bar = max(
            float(by_width[width][metric])
            for width in ACTIVE_BIT_WIDTHS
            for metric in (
                "cross_half_structure_std",
                "cross_half_output_position_residual_std",
                "cross_half_combined_marginal_residual_std",
            )
        )
        residual_axis.set_ylim(0.0, max(0.04, maximum_bar * 1.28))
        residual_axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=8.7,
        )

        decision_labels = {
            "innovation2_r6_active_bit_transition_benchmark_ready": (
                "通过，进入所选宽度的本地输出预测训练"
            ),
            "innovation2_r6_active_bit_transition_benchmark_not_ready": (
                "未通过，停止 r6 当前结构描述训练路线"
            ),
            "innovation2_r6_active_bit_transition_audit_invalid": (
                "审计无效，先修复数据或统计校验"
            ),
        }
        correlations = ", ".join(
            f"{width}bit={float(by_width[width]['cross_half_combined_marginal_residual_correlation']):.2f}"
            for width in ACTIVE_BIT_WIDTHS
        )
        figure.text(
            0.075,
            0.075,
            (
                f"组合边际残差两半相关：{correlations}；"
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str = "a",
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
