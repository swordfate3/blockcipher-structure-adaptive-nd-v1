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

from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    StableSubspaceAuditConfig,
    run_stable_subspace_audit,
)


COLORS = {5: "#2563EB", 6: "#059669", 7: "#DC2626"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit stable PRESENT output-balance mask subspaces."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structures-per-width", type=int, default=64)
    parser.add_argument("--keys-per-structure", type=int, default=256)
    parser.add_argument("--structure-chunk-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StableSubspaceAuditConfig(
        run_id=args.run_id,
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
            "seed": args.seed,
            "structures_per_width": args.structures_per_width,
            "keys_per_structure": args.keys_per_structure,
            "rounds": [4, 5, 6],
            "training_performed": False,
        },
        mode="w",
    )
    result = run_stable_subspace_audit(
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
    _write_csv(args.output_root / "subspaces.csv", result["subspace_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_subspace_svg(
        result["rows"],
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


def render_subspace_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_setting = {
        (int(row["rounds"]), int(row["active_bit_width"])): row for row in rows
    }
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
        figure, axes = plt.subplots(1, 2, figsize=(13.6, 6.7))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.22,
            wspace=0.30,
        )
        figure.suptitle(
            "创新2 E9：PRESENT 输出平衡 mask 子空间稳定性审计",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            "r4 单活动 nibble 是已知校准 fixture，r5 为参考，r6 为目标；所有 kernel 均使用两组互斥密钥。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        fraction_axis, dimension_axis = axes
        x_positions = np.arange(2, dtype=np.float64)
        bar_width = 0.24
        for width_index, width in enumerate((5, 6, 7)):
            fractions = [
                float(by_setting[(rounds, width)]["nontrivial_joint_kernel_fraction"])
                for rounds in (5, 6)
            ]
            bars = fraction_axis.bar(
                x_positions + (width_index - 1) * bar_width,
                fractions,
                bar_width,
                color=COLORS[width],
                label=f"{width} 活动 bit",
            )
            fraction_axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.8)
        fraction_axis.axhline(
            0.10,
            color="#64748B",
            linestyle="--",
            linewidth=1.0,
            label="非平凡结构比例门槛 0.10",
        )
        fraction_axis.set_title(
            "存在稳定非平凡 joint kernel 的结构比例",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        fraction_axis.set_ylabel("结构比例")
        fraction_axis.set_xticks(x_positions, labels=("PRESENT 5轮参考", "PRESENT 6轮目标"))
        fraction_axis.set_ylim(0.0, 1.08)
        fraction_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        fraction_axis.grid(False, axis="x")
        fraction_axis.legend(loc="upper right", frameon=False, fontsize=8.5)

        dimension_x = np.arange(3, dtype=np.float64)
        for rounds_index, rounds in enumerate((5, 6)):
            values = [
                float(by_setting[(rounds, width)]["mean_joint_kernel_dimension"])
                for width in (5, 6, 7)
            ]
            bars = dimension_axis.bar(
                dimension_x + (rounds_index - 0.5) * 0.34,
                values,
                0.34,
                color="#2563EB" if rounds == 5 else "#DC2626",
                label=f"{rounds}轮",
            )
            dimension_axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.8)
        dimension_axis.set_title(
            "平均稳定 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_ylabel("dim ker([M0; M1])")
        dimension_axis.set_xticks(dimension_x, labels=("5 bit", "6 bit", "7 bit"))
        dimension_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        dimension_axis.grid(False, axis="x")
        dimension_axis.legend(loc="upper right", frameon=False)

        decision_labels = {
            "innovation2_r6_stable_balance_subspace_ready": (
                "r6 稳定子空间存在，可进入结构条件预测 readiness"
            ),
            "innovation2_r6_stable_balance_subspace_not_found": (
                "r6 当前结构族无可用稳定子空间"
            ),
            "innovation2_stable_balance_subspace_protocol_invalid": (
                "协议无效，先修 parity word 或 GF(2) kernel"
            ),
        }
        figure.text(
            0.075,
            0.075,
            (
                "r4 已知 fixture 校准："
                f"{'通过' if gate.get('r4_known_fixture_calibration_pass') else '未通过'}；"
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "本图无神经训练结果。"
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
