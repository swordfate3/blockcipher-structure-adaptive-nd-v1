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

from blockcipher_nd.tasks.innovation2.integral_transition_audit import (
    TransitionAuditConfig,
    run_transition_audit,
)


COLORS = {1: "#2563EB", 2: "#DC2626"}
LABELS = {1: "1 个活动 nibble（16 明文）", 2: "2 个活动 nibble（256 明文）"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the PRESENT r6 output-property transition by active width."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structures-per-width", type=int, default=64)
    parser.add_argument("--keys-per-structure", type=int, default=32)
    parser.add_argument("--structure-chunk-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TransitionAuditConfig(
        run_id=args.run_id,
        rounds=args.rounds,
        seed=args.seed,
        structures_per_width=args.structures_per_width,
        keys_per_structure=args.keys_per_structure,
        structure_chunk_size=args.structure_chunk_size,
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
            "training_performed": False,
        },
        mode="w",
    )
    result = run_transition_audit(config, progress_callback=progress_callback)
    results_path = args.output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "structure_rates.csv", result["structure_rows"])
    _write_csv(args.output_root / "position_priors.csv", result["position_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_transition_svg(
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
                "output_root": str(args.output_root),
                "next_action": result["gate"]["next_action"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_transition_svg(
    summary_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_width = {int(row["active_nibble_count"]): row for row in summary_rows}
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
        figure, axes = plt.subplots(1, 2, figsize=(13.6, 6.8))
        figure.subplots_adjust(left=0.075, right=0.975, top=0.72, bottom=0.22, wspace=0.30)
        figure.suptitle(
            "创新2 E7：PRESENT 6轮输出平衡性质的活动宽度过渡审计",
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
                "每个点是一种输入结构在 "
                f"{int(summary_rows[0]['keys_per_structure'])} 把独立密钥上的平衡率；"
                "本图不含神经训练结果。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        distribution_axis, residual_axis = axes
        rng = np.random.default_rng(17)
        for x, width in enumerate((1, 2)):
            rates = np.asarray(
                [
                    float(row["balance_rate"])
                    for row in structure_rows
                    if int(row["active_nibble_count"]) == width
                ]
            )
            jitter = rng.uniform(-0.16, 0.16, size=len(rates))
            distribution_axis.scatter(
                np.full(len(rates), x) + jitter,
                rates,
                s=25,
                alpha=0.66,
                color=COLORS[width],
                edgecolors="none",
            )
            distribution_axis.hlines(
                float(rates.mean()),
                x - 0.24,
                x + 0.24,
                color="#0F172A",
                linewidth=2.2,
            )
        distribution_axis.set_title("结构级平衡率分布", loc="left", fontweight="bold", pad=10)
        distribution_axis.set_ylabel("跨密钥平衡率  P(q=0)")
        distribution_axis.set_xticks((0, 1), labels=[LABELS[1], LABELS[2]])
        distribution_axis.set_ylim(-0.03, 1.03)
        distribution_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        distribution_axis.grid(False, axis="x")

        metric_names = [
            "观测结构差异",
            "扣除密钥采样噪声",
            "再扣除输出位置先验",
        ]
        x_positions = np.arange(len(metric_names), dtype=np.float64)
        width_bar = 0.34
        for offset, width in ((-width_bar / 2, 1), (width_bar / 2, 2)):
            values = [
                float(by_width[width]["balance_rate_std"]),
                float(by_width[width]["excess_balance_rate_std"]),
                float(by_width[width]["excess_output_position_residual_std"]),
            ]
            bars = residual_axis.bar(
                x_positions + offset,
                values,
                width_bar,
                color=COLORS[width],
                label=LABELS[width],
            )
            residual_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9.0)
        residual_axis.hlines(
            0.03,
            0.55,
            1.45,
            color="#64748B",
            linestyle="--",
            linewidth=1.0,
            label="结构差异门槛 0.03",
        )
        residual_axis.hlines(
            0.02,
            1.55,
            2.45,
            color="#94A3B8",
            linestyle=":",
            linewidth=1.0,
            label="位置残差门槛 0.02",
        )
        residual_axis.set_title("有限密钥噪声校正后的可学习差异", loc="left", fontweight="bold", pad=10)
        residual_axis.set_ylabel("标准差")
        residual_axis.set_xticks(x_positions, labels=metric_names)
        residual_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        residual_axis.grid(False, axis="x")
        residual_axis.legend(loc="upper right", frameon=False, fontsize=9.0)

        candidate = by_width[2]
        decision_labels = {
            "innovation2_r6_two_nibble_output_prediction_benchmark_ready": (
                "通过，可进入本地输出预测训练"
            ),
            "innovation2_r6_two_nibble_almost_always_balanced": (
                "两活动 nibble 几乎总平衡，转 r7 过渡审计"
            ),
            "innovation2_r6_two_nibble_output_prediction_benchmark_not_ready": (
                "未通过，转 5--7 活动 bit 细粒度审计"
            ),
            "innovation2_output_property_transition_audit_invalid": (
                "审计无效，先修复数据或校验"
            ),
        }
        figure.text(
            0.075,
            0.075,
            (
                f"两活动 nibble：q=1 比例 {float(candidate['q1_rate']):.3f}，"
                f"混合结构占比 {float(candidate['mixed_structure_fraction']):.1%}；"
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
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
