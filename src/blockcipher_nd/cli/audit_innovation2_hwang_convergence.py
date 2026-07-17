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

from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import (
    HwangKernelConvergenceConfig,
    run_hwang_kernel_convergence_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit convergence of the PRESENT r7 Hwang output kernel."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=1)
    parser.add_argument(
        "--input-orientation",
        choices=("low_0_15", "high_48_63"),
        default="low_0_15",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HwangKernelConvergenceConfig(
        run_id=args.run_id,
        seed=args.seed,
        keys=args.keys,
        key_chunk_size=args.key_chunk_size,
        input_orientation=args.input_orientation,
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
            "keys": args.keys,
            "key_chunk_size": args.key_chunk_size,
            "input_orientation": args.input_orientation,
            "plaintexts_per_key": 1 << 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_hwang_kernel_convergence_audit(
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
    render_convergence_svg(
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
                "joint_kernel_dimension": result["gate"][
                    "joint_kernel_dimension"
                ],
                "joint_kernel_equals_paper_span": result["gate"][
                    "joint_kernel_equals_paper_span"
                ],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_convergence_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = [
        "发现密钥\n64把",
        "验证密钥\n64把",
        "联合密钥\n128把",
    ]
    by_split = {str(row["split"]): row for row in rows}
    ordered = [by_split[split] for split in ("discovery", "validation", "joint")]
    dimensions = [int(row["kernel_dimension"]) for row in ordered]
    paper_failures = [int(row["paper_mask_failures"]) for row in ordered]
    control_failures = [int(row["control_mask_failures"]) for row in ordered]
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
        figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.70,
            bottom=0.24,
            wspace=0.30,
        )
        input_orientation = str(gate.get("input_orientation", "low_0_15"))
        experiment_label = "E11" if input_orientation == "low_0_15" else "E11b"
        active_label = "低16位" if input_orientation == "low_0_15" else "高16位"
        figure.suptitle(
            f"创新2 {experiment_label}：PRESENT 7轮论文四维输出 kernel 收敛审判",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            f"{active_label}完整活动；128把新密钥与 E10 完全互斥；每把密钥使用完整 2^16 明文集合。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        x = np.arange(3, dtype=np.float64)
        dimension_axis, failure_axis = axes
        dimension_bars = dimension_axis.bar(
            x,
            dimensions,
            color=("#2563EB", "#059669", "#DC2626"),
            width=0.62,
        )
        dimension_axis.bar_label(dimension_bars, padding=3, fontsize=9.0)
        dimension_axis.axhline(
            4,
            color="#64748B",
            linestyle="--",
            linewidth=1.2,
            label="论文目标维数 = 4",
        )
        dimension_axis.set_title(
            "经验输出平衡 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_ylabel("kernel 维数")
        dimension_axis.set_xticks(x, labels=labels)
        dimension_axis.set_ylim(0, max(6.0, max(dimensions) * 1.18 + 0.5))
        dimension_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        dimension_axis.grid(False, axis="x")
        dimension_axis.legend(loc="upper right", frameon=False)

        width = 0.34
        paper_bars = failure_axis.bar(
            x - width / 2,
            paper_failures,
            width,
            color="#059669",
            label="四个论文 mask",
        )
        control_bars = failure_axis.bar(
            x + width / 2,
            control_failures,
            width,
            color="#DC2626",
            label="非论文同权重控制",
        )
        failure_axis.bar_label(paper_bars, padding=3, fontsize=9.0)
        failure_axis.bar_label(control_bars, padding=3, fontsize=9.0)
        failure_axis.set_title(
            "输出 mask 的非零 parity 次数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        failure_axis.set_ylabel("失败次数")
        failure_axis.set_xticks(x, labels=labels)
        failure_axis.set_ylim(
            0,
            max(1.5, max(control_failures + paper_failures) * 1.18 + 0.5),
        )
        failure_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        failure_axis.grid(False, axis="x")
        failure_axis.legend(loc="upper left", frameon=False)

        decision_labels = {
            "innovation2_present_r7_hwang_kernel_reproduced": (
                "论文四维输出 kernel 已在两组新密钥上复现"
            ),
            "innovation2_present_r7_hwang_kernel_underconstrained": (
                "论文 mask 稳定，但经验 kernel 尚未收敛到四维"
            ),
            "innovation2_present_r7_hwang_kernel_not_reproduced": (
                "论文 mask 未在新密钥上保持稳定"
            ),
            "innovation2_present_r7_hwang_convergence_protocol_invalid": (
                "协议校验无效，先修复数据或 kernel 计算"
            ),
        }
        figure.text(
            0.075,
            0.075,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是积分输出性质复现，不是积分/随机二分类，也不含神经训练。"
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
