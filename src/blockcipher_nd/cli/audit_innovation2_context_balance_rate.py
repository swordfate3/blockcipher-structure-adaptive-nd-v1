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

from blockcipher_nd.tasks.innovation2.integral_context_balance_rate import (
    ContextBalanceRateConfig,
    run_context_balance_rate_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r7 context-mask cross-key balance rates."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ContextBalanceRateConfig(
        run_id=args.run_id,
        seed=args.seed,
        contexts=args.contexts,
        keys=args.keys,
        key_chunk_size=args.key_chunk_size,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_result_rows = _read_jsonl(args.source_root / "results.jsonl")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_run_id": source_gate.get("run_id"),
            "contexts": args.contexts,
            "keys": args.keys,
            "candidate_masks": 240,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_context_balance_rate_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_result_rows=source_result_rows,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "balance_rates.csv", result["cell_rows"])
    np.save(args.output_root / "xor_words.npy", result["xor_words"], allow_pickle=False)
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_balance_rate_svg(
        result["cell_rows"],
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
            "xor_words_cache": "xor_words.npy",
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "metrics": result["gate"]["metrics"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_balance_rate_svg(
    cell_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    discovery_rates = np.asarray(
        [float(row["discovery_balance_rate"]) for row in cell_rows],
        dtype=np.float64,
    )
    validation_rates = np.asarray(
        [float(row["validation_balance_rate"]) for row in cell_rows],
        dtype=np.float64,
    )
    metrics = gate["metrics"]
    metric_specs = (
        ("两半 rate 相关", "rate_half_correlation", 0.25, "min"),
        (
            "两半 interaction 残差相关",
            "interaction_residual_half_correlation",
            0.20,
            "min",
        ),
        (
            "context shuffle |相关|",
            "context_shuffle_residual_correlation",
            0.10,
            "max_abs",
        ),
        (
            "label shuffle |相关|",
            "label_shuffle_residual_correlation",
            0.10,
            "max_abs",
        ),
    )
    displayed_values = [
        abs(float(metrics[key])) if direction == "max_abs" else float(metrics[key])
        for _, key, _, direction in metric_specs
    ]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.1))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.70,
            bottom=0.25,
            wspace=0.34,
        )
        figure.suptitle(
            "创新2 E19：PRESENT 7轮跨密钥输出平衡概率审计",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "64个 context × 240个4-bit输出 mask；比较两个互斥64-key half；不训练网络。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        scatter_axis, metric_axis = axes
        scatter_axis.scatter(
            discovery_rates,
            validation_rates,
            s=7,
            alpha=0.22,
            color="#2563EB",
            edgecolors="none",
            rasterized=True,
        )
        scatter_axis.plot([0, 1], [0, 1], color="#64748B", linestyle="--", linewidth=1.0)
        scatter_axis.set_title(
            "15,360个 cell 的两半平衡率",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        scatter_axis.set_xlabel("发现64把密钥的平衡率")
        scatter_axis.set_ylabel("验证64把密钥的平衡率")
        scatter_axis.set_xlim(-0.02, 1.02)
        scatter_axis.set_ylim(-0.02, 1.02)
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.grid(True, color="#E5E7EB", linewidth=0.8)

        y = np.arange(len(metric_specs), dtype=np.float64)
        passes = [
            value >= threshold if direction == "min" else value < threshold
            for value, (_, _, threshold, direction) in zip(
                displayed_values, metric_specs, strict=True
            )
        ]
        bars = metric_axis.barh(
            y,
            displayed_values,
            color=["#059669" if passed else "#DC2626" for passed in passes],
            height=0.56,
        )
        metric_axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in displayed_values],
            padding=4,
            fontsize=8.6,
        )
        for position, (_, _, threshold, direction) in enumerate(metric_specs):
            metric_axis.plot(
                threshold,
                position,
                marker="|",
                markersize=16,
                markeredgewidth=2.0,
                color="#D97706",
            )
            metric_axis.text(
                threshold,
                position - 0.38,
                f"{'≥' if direction == 'min' else '<'}{threshold:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#92400E",
            )
        metric_axis.set_title(
            "可重复交互与打乱控制",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        metric_axis.set_xlabel("相关系数 / 绝对相关")
        metric_axis.set_yticks(y, labels=[label for label, *_ in metric_specs])
        metric_axis.set_xlim(0.0, max(0.45, max(displayed_values) * 1.18))
        metric_axis.invert_yaxis()
        metric_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        metric_axis.grid(False, axis="y")

        decision_labels = {
            "innovation2_balance_rate_interaction_ready": (
                "跨密钥 interaction 残差可重复，可设计连续预测"
            ),
            "innovation2_balance_rate_interaction_not_reproducible": (
                "interaction 残差弱、噪声化或被控制解释，停止该分支"
            ),
            "innovation2_balance_rate_protocol_invalid": (
                "E18重放、XOR缓存、mask网格或标量校验无效"
            ),
        }
        figure.text(
            0.075,
            0.055,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                f" MAD={metrics['mean_absolute_half_rate_difference']:.3f}，"
                f"残差std={metrics['validation_residual_standard_deviation']:.3f}，"
                f"excess var={metrics['interaction_excess_variance']:.5f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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
