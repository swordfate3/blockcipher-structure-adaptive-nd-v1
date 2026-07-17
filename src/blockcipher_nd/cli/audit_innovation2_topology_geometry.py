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

from blockcipher_nd.tasks.innovation2.integral_topology_geometry import (
    TopologyGeometryConfig,
    run_topology_geometry_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r7 P-layer topology kernel diversity."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TopologyGeometryConfig(
        run_id=args.run_id,
        seed=args.seed,
        keys=args.keys,
        key_chunk_size=args.key_chunk_size,
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
            "rounds": 7,
            "structures": 12,
            "keys_per_structure": args.keys,
            "key_chunk_size": args.key_chunk_size,
            "plaintexts_per_structure_per_key": 1 << 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_topology_geometry_audit(
        config,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_topology_svg(
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
                "distinct_joint_kernel_signatures": result["gate"][
                    "distinct_joint_kernel_signatures"
                ],
                "nontrivial_joint_kernel_structures": result["gate"][
                    "nontrivial_joint_kernel_structures"
                ],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_topology_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (int(row["base_start_nibble"]), int(row["p_power"])),
    )
    labels = [
        f"块{int(row['base_start_nibble'])}\nP^{int(row['p_power'])}"
        for row in ordered
    ]
    joint = [int(row["joint_kernel_dimension"]) for row in ordered]
    signatures = [str(row["joint_basis_signature"]) for row in ordered]
    signature_ids = {
        signature: index + 1 for index, signature in enumerate(sorted(set(signatures)))
    }
    signature_values = [signature_ids[signature] for signature in signatures]
    x = np.arange(len(rows), dtype=np.float64)
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
        figure, axes = plt.subplots(1, 2, figsize=(14.6, 7.0))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.25,
            wspace=0.30,
        )
        figure.suptitle(
            "创新2 E15：PRESENT P-layer 拓扑活动几何的输出 kernel",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "四个边界块分别取 P^0、P^1、P^2，共12个唯一16-bit活动几何；每个使用128把密钥。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        dimension_axis, signature_axis = axes
        colors = [
            ("#2563EB", "#059669", "#DC2626")[int(row["p_power"])]
            for row in ordered
        ]
        bars = dimension_axis.bar(x, joint, color=colors, width=0.68)
        dimension_axis.bar_label(bars, padding=3, fontsize=8.5)
        dimension_axis.set_title(
            "各拓扑几何的 joint kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_ylabel("kernel 维数")
        dimension_axis.set_xticks(x, labels=labels)
        dimension_axis.set_ylim(0, max(2.0, max(joint) * 1.20 + 0.5))
        dimension_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        dimension_axis.grid(False, axis="x")

        signature_axis.scatter(
            x,
            signature_values,
            s=72,
            c=joint,
            cmap="viridis",
            edgecolors="#FFFFFF",
            linewidths=0.8,
        )
        for position, signature_id, dimension in zip(
            x, signature_values, joint, strict=True
        ):
            signature_axis.annotate(
                f"d={dimension}",
                (position, signature_id),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
            )
        signature_axis.set_title(
            "joint kernel 签名类别",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        signature_axis.set_ylabel("签名类别编号")
        signature_axis.set_xticks(x, labels=labels)
        signature_axis.set_yticks(sorted(set(signature_values)))
        signature_axis.grid(True, color="#E5E7EB", linewidth=0.8)

        decision_labels = {
            "innovation2_topology_geometry_kernel_diversity_ready": (
                "P-layer拓扑几何形成足够多稳定 kernel"
            ),
            "innovation2_topology_geometry_kernel_diversity_insufficient": (
                "P-layer拓扑几何的 kernel 多样性不足"
            ),
            "innovation2_topology_geometry_protocol_invalid": (
                "协议或 Hwang anchor 校验无效"
            ),
        }
        figure.text(
            0.07,
            0.065,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是结构条件输出标签审计，不是神经训练，也不是积分/随机二分类。"
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
