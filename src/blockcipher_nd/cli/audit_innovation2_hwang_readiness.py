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
    HwangReadinessConfig,
    run_hwang_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r7 Hwang kernel bit-order readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--key-chunk-size", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HwangReadinessConfig(
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
            "keys": args.keys,
            "key_chunk_size": args.key_chunk_size,
            "plaintexts_per_structure": 1 << 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_hwang_readiness_audit(
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
    _write_csv(args.output_root / "mask_checks.csv", result["mask_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_hwang_readiness_svg(
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
                "passing_candidates": result["gate"]["passing_candidates"],
                "selected_candidate": result["gate"]["selected_candidate"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_hwang_readiness_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = [
        (
            "输入低16 bit\n输出同向"
            if row["candidate_id"] == "low_0_15__direct"
            else "输入低16 bit\n输出反向"
            if row["candidate_id"] == "low_0_15__reflected"
            else "输入高16 bit\n输出同向"
            if row["candidate_id"] == "high_48_63__direct"
            else "输入高16 bit\n输出反向"
        )
        for row in rows
    ]
    paper_failures = [int(row["paper_mask_failures_joint"]) for row in rows]
    control_failures = [int(row["control_mask_failures_joint"]) for row in rows]
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
        figure.suptitle(
            "创新2 E10：PRESENT 7轮论文积分输出 mask 的 bit-order 校准",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "每个候选使用完整 2^16 明文集合和 8 把密钥；左图应为 0，右图应大于 0。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        x = np.arange(len(rows), dtype=np.float64)
        paper_axis, control_axis = axes
        paper_bars = paper_axis.bar(x, paper_failures, color="#DC2626", width=0.62)
        paper_axis.bar_label(paper_bars, padding=3, fontsize=9.0)
        paper_axis.set_title(
            "四个论文 mask 的总失败次数（越低越好）",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        paper_axis.set_ylabel("非零 parity 次数")
        paper_axis.set_xticks(x, labels=labels)
        paper_axis.set_ylim(0, max(1.5, max(paper_failures, default=0) * 1.18 + 0.5))
        paper_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        paper_axis.grid(False, axis="x")

        control_bars = control_axis.bar(
            x,
            control_failures,
            color="#2563EB",
            width=0.62,
        )
        control_axis.bar_label(control_bars, padding=3, fontsize=9.0)
        control_axis.set_title(
            "非论文同权重控制 mask 的总失败次数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        control_axis.set_ylabel("非零 parity 次数")
        control_axis.set_xticks(x, labels=labels)
        control_axis.set_ylim(
            0,
            max(1.5, max(control_failures, default=0) * 1.18 + 0.5),
        )
        control_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        control_axis.grid(False, axis="x")

        decision_labels = {
            "innovation2_present_r7_hwang_bitorder_ready": (
                "唯一 bit-order 复现论文 mask，可扩大新密钥复核"
            ),
            "innovation2_present_r7_hwang_bitorder_ambiguous": (
                "多个方向通过，需增加密钥并核对状态布局"
            ),
            "innovation2_present_r7_hwang_bitorder_not_reproduced": (
                "当前四种映射均未复现，停止训练并审计协议"
            ),
            "innovation2_present_r7_hwang_protocol_invalid": (
                "实现校验未通过，先修复协议"
            ),
        }
        figure.text(
            0.075,
            0.075,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是输出平衡性质校准，不是积分/随机二分类，也不含神经训练。"
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
