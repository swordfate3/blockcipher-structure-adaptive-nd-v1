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

from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    ContextLabelReadinessConfig,
    run_context_label_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Innovation 2 context-mask output-label shortcuts."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ContextLabelReadinessConfig(
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
    result = run_context_label_readiness_audit(
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
    render_context_label_svg(
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
                "distinct_context_label_signatures": result["gate"][
                    "distinct_context_label_signatures"
                ],
                "baseline_accuracies": result["gate"]["baseline_accuracies"],
                "baseline_aucs": result["gate"]["baseline_aucs"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_context_label_svg(
    baseline_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
    *,
    title: str = "创新2 E17：context-mask 输出平衡标签的捷径审计",
    subtitle: str = (
        "标签来自 E16 的16个稳定 joint kernel；18个候选 mask，共288行；"
        "不重新加密、不训练网络。"
    ),
    primary_stop: float = 0.95,
    primary_stop_label: str = "捷径 AUC 停止线 0.95",
    secondary_stop: float | None = 0.98,
    secondary_stop_label: str = "身份加性停止线 0.98",
) -> None:
    context_ids = np.arange(16, dtype=np.int64)
    positives = [
        sum(
            int(row["balanced_label"])
            for row in label_rows
            if int(row["context_id"]) == context_id
        )
        for context_id in context_ids
    ]
    plotted = [row for row in baseline_rows if not row["is_oracle"]]
    baseline_labels = [str(row["baseline_label"]) for row in plotted]
    accuracies = [float(row["accuracy"]) for row in plotted]
    aucs = [float(row["auc"]) for row in plotted]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.4))
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.70,
            bottom=0.29,
            wspace=0.38,
        )
        figure.suptitle(
            title,
            x=0.065,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.065,
            0.89,
            subtitle,
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        label_axis, baseline_axis = axes
        bars = label_axis.bar(context_ids, positives, color="#2563EB", width=0.68)
        label_axis.bar_label(bars, padding=3, fontsize=8.2)
        label_axis.set_title(
            "每个固定上下文的平衡候选 mask 数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        label_axis.set_xlabel("context 编号")
        label_axis.set_ylabel("正标签数量")
        label_axis.set_xticks(context_ids)
        label_axis.set_ylim(0, max(2.0, max(positives) * 1.22 + 0.5))
        label_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        label_axis.grid(False, axis="x")

        y = np.arange(len(plotted), dtype=np.float64)
        accuracy_bars = baseline_axis.barh(
            y - 0.17,
            accuracies,
            color="#2563EB",
            height=0.30,
            label="准确率",
        )
        auc_bars = baseline_axis.barh(
            y + 0.17,
            aucs,
            color="#DC2626",
            height=0.30,
            label="AUC",
        )
        for bars_for_metric, values in (
            (accuracy_bars, accuracies),
            (auc_bars, aucs),
        ):
            for bar, value in zip(bars_for_metric, values, strict=True):
                inside = value >= 0.90
                baseline_axis.text(
                    value - 0.008 if inside else value + 0.010,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    ha="right" if inside else "left",
                    va="center",
                    fontsize=7.7,
                    color="#FFFFFF" if inside else "#334155",
                    zorder=4,
                )
        baseline_axis.axvline(
            primary_stop,
            color="#D97706",
            linestyle="--",
            linewidth=1.1,
            label=primary_stop_label,
            zorder=0,
        )
        if secondary_stop is not None:
            baseline_axis.axvline(
                secondary_stop,
                color="#64748B",
                linestyle=":",
                linewidth=1.2,
                label=secondary_stop_label,
                zorder=0,
            )
        baseline_axis.set_title(
            "无需神经网络的基线准确率与 AUC",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        baseline_axis.set_xlabel("指标值")
        baseline_axis.set_yticks(y, labels=baseline_labels)
        baseline_axis.set_xlim(0.0, 1.08)
        baseline_axis.invert_yaxis()
        baseline_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        baseline_axis.grid(False, axis="y")
        baseline_axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.20),
            ncol=4,
            frameon=False,
            fontsize=8.3,
        )

        decision_labels = {
            "innovation2_context_label_interaction_ready": (
                "强基线未解释 context-mask 交互，可做 fresh-key 验证"
            ),
            "innovation2_context_label_shortcut_dominated": (
                "简单 context/mask 捷径已解释标签，禁止训练"
            ),
            "innovation2_context_label_readiness_protocol_invalid": (
                "E16源证据或标签构造无效"
            ),
            "innovation2_equal_prevalence_context_label_ready": (
                "等流行率 mask 未被强捷径解释，可做 fresh-key 验证"
            ),
            "innovation2_equal_prevalence_context_label_shortcut_dominated": (
                "等流行率标签仍被简单捷径解释，禁止训练"
            ),
            "innovation2_equal_prevalence_label_protocol_invalid": (
                "E16 span 或等流行率标签构造无效"
            ),
        }
        figure.text(
            0.065,
            0.065,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是结构条件输出预测 readiness，不是积分/随机二分类。"
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
