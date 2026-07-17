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

from blockcipher_nd.tasks.innovation2.integral_output_label_readiness import (
    OutputLabelReadinessConfig,
    run_output_label_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Innovation 2 structure-mask label shortcuts."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = OutputLabelReadinessConfig(
        run_id=args.run_id,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_basis_rows = _read_csv(args.source_root / "kernel_basis.csv")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_run_id": source_gate.get("run_id"),
            "training_performed": False,
        },
        mode="w",
    )
    result = run_output_label_readiness_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_basis_rows=source_basis_rows,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "labels.csv", result["label_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_label_readiness_svg(
        result["rows"],
        result["label_rows"],
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
                "positive_rate": result["gate"]["positive_rate"],
                "flipping_masks": result["gate"]["flipping_masks"],
                "baseline_accuracies": result["gate"]["baseline_accuracies"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_label_readiness_svg(
    baseline_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    block_ids = ("block_0_15", "block_16_31", "block_32_47", "block_48_63")
    block_labels = ("活动块\n0..15", "活动块\n16..31", "活动块\n32..47", "活动块\n48..63")
    positives = [
        sum(
            int(row["balanced_label"])
            for row in label_rows
            if row["block_id"] == block_id
        )
        for block_id in block_ids
    ]
    plotted = [row for row in baseline_rows if not row["is_oracle"]]
    baseline_labels = [str(row["baseline_label"]) for row in plotted]
    accuracies = [float(row["accuracy"]) for row in plotted]
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
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.1))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.27,
            wspace=0.32,
        )
        figure.suptitle(
            "创新2 E13：结构-mask 输出平衡标签的边际捷径审计",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "标签来自 E12 的稳定 GF(2) kernel；本实验不重新加密，也不训练神经网络。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        x = np.arange(4, dtype=np.float64)
        label_axis, baseline_axis = axes
        bars = label_axis.bar(x, positives, color="#2563EB", width=0.62)
        label_axis.bar_label(bars, padding=3, fontsize=9.0)
        label_axis.set_title(
            "每个活动结构的平衡 mask 数量",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        label_axis.set_ylabel("正标签数量")
        label_axis.set_xticks(x, labels=block_labels)
        label_axis.set_ylim(0, max(2.0, max(positives) * 1.22 + 0.5))
        label_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        label_axis.grid(False, axis="x")

        y = np.arange(len(plotted), dtype=np.float64)
        colors = [
            "#DC2626" if row["baseline"] == "block_mask_additive" else "#059669"
            for row in plotted
        ]
        baseline_bars = baseline_axis.barh(y, accuracies, color=colors, height=0.62)
        baseline_axis.bar_label(
            baseline_bars,
            labels=[f"{value:.3f}" for value in accuracies],
            padding=4,
            fontsize=8.8,
        )
        baseline_axis.axvline(
            0.98,
            color="#64748B",
            linestyle="--",
            linewidth=1.1,
            label="捷径停止线 0.98",
        )
        baseline_axis.set_title(
            "无需神经网络的基线准确率",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        baseline_axis.set_xlabel("准确率")
        baseline_axis.set_yticks(y, labels=baseline_labels)
        baseline_axis.set_xlim(0.0, 1.08)
        baseline_axis.invert_yaxis()
        baseline_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        baseline_axis.grid(False, axis="y")
        baseline_axis.legend(loc="lower right", frameon=False, fontsize=8.7)

        decision_labels = {
            "innovation2_output_label_interaction_ready": (
                "边际基线未完全解释标签，可扩大结构族"
            ),
            "innovation2_output_label_shortcut_dominated": (
                "简单活动块+mask边际已解释标签，禁止直接训练"
            ),
            "innovation2_output_label_readiness_protocol_invalid": (
                "源证据或标签构造无效"
            ),
        }
        figure.text(
            0.07,
            0.065,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是输出预测任务 readiness，不是积分/随机二分类。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
