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
    ActiveBlockKernelDiversityConfig,
    run_active_block_kernel_diversity_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r7 active-block output-kernel diversity."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ActiveBlockKernelDiversityConfig(
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
            "active_blocks": ["0..15", "16..31", "32..47", "48..63"],
            "keys_per_structure": args.keys,
            "key_chunk_size": args.key_chunk_size,
            "plaintexts_per_structure_per_key": 1 << 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_active_block_kernel_diversity_audit(
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
    render_diversity_svg(
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


def render_diversity_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = [
        "活动块\n0..15",
        "活动块\n16..31",
        "活动块\n32..47",
        "活动块\n48..63",
    ]
    ordered = sorted(rows, key=lambda row: int(row["active_start"]))
    discovery = [int(row["discovery_kernel_dimension"]) for row in ordered]
    validation = [int(row["validation_kernel_dimension"]) for row in ordered]
    joint = [int(row["joint_kernel_dimension"]) for row in ordered]
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
        figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.9))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.24,
            wspace=0.30,
        )
        figure.suptitle(
            "创新2 E12：PRESENT 7轮不同活动块的输出 kernel 多样性",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "四个结构仅改变16-bit活动块位置；每个结构使用同一组128把密钥和完整 2^16 明文集合。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        x = np.arange(4, dtype=np.float64)
        dimension_axis, failure_axis = axes
        width = 0.24
        discovery_bars = dimension_axis.bar(
            x - width,
            discovery,
            width,
            color="#2563EB",
            label="发现64把",
        )
        validation_bars = dimension_axis.bar(
            x,
            validation,
            width,
            color="#059669",
            label="验证64把",
        )
        joint_bars = dimension_axis.bar(
            x + width,
            joint,
            width,
            color="#DC2626",
            label="联合128把",
        )
        for bars in (discovery_bars, validation_bars, joint_bars):
            dimension_axis.bar_label(bars, padding=3, fontsize=8.5)
        dimension_axis.set_title(
            "各活动结构的经验 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_ylabel("kernel 维数")
        dimension_axis.set_xticks(x, labels=labels)
        dimension_axis.set_ylim(
            0,
            max(2.0, max(discovery + validation + joint) * 1.20 + 0.5),
        )
        dimension_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        dimension_axis.grid(False, axis="x")
        dimension_axis.legend(loc="upper right", frameon=False, fontsize=8.7)

        failure_width = 0.34
        paper_bars = failure_axis.bar(
            x - failure_width / 2,
            paper_failures,
            failure_width,
            color="#059669",
            label="Hwang 四个 mask",
        )
        control_bars = failure_axis.bar(
            x + failure_width / 2,
            control_failures,
            failure_width,
            color="#DC2626",
            label="非论文同权重控制",
        )
        failure_axis.bar_label(paper_bars, padding=3, fontsize=8.5)
        failure_axis.bar_label(control_bars, padding=3, fontsize=8.5)
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
            max(1.5, max(paper_failures + control_failures) * 1.18 + 0.5),
        )
        failure_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        failure_axis.grid(False, axis="x")
        failure_axis.legend(loc="upper right", frameon=False, fontsize=8.7)

        decision_labels = {
            "innovation2_present_r7_active_block_kernel_diversity_ready": (
                "不同活动块产生多个稳定 kernel，可构造结构条件标签表"
            ),
            "innovation2_present_r7_active_block_kernel_not_diverse": (
                "活动块位置没有形成足够的 kernel 多样性"
            ),
            "innovation2_present_r7_active_block_diversity_protocol_invalid": (
                "协议校验无效，先修复结构或 Hwang anchor"
            ),
        }
        figure.text(
            0.07,
            0.07,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是输出标签 readiness，不是神经训练，也不是积分/随机二分类。"
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
