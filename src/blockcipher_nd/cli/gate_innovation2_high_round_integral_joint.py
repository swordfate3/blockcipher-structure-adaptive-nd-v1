from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.high_round_integral_joint import (
    adjudicate_joint_high_round_integral,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly adjudicate two verified Innovation 2 PRESENT-80 r8 bridge "
            "artifact directories without training."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-artifacts", nargs=2, required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [_load_source(path) for path in args.source_artifacts]
    result = adjudicate_joint_high_round_integral(
        run_id=args.run_id,
        sources=sources,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_artifacts": [str(path) for path in args.source_artifacts],
            "training_performed": False,
        },
        mode="w",
    )
    results_path = args.output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    gate_path = args.output_root / "gate.json"
    gate_path.write_text(
        json.dumps(result["gate"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    csv_path = args.output_root / "seed_metrics.csv"
    _write_csv(csv_path, result["rows"])
    curves_path = args.output_root / "curves.svg"
    render_joint_bridge_svg(result["rows"], result["gate"], curves_path)
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
        mode="a",
    )
    report = {
        "status": result["gate"]["status"],
        "decision": result["gate"]["decision"],
        "run_id": args.run_id,
        "output_root": str(args.output_root),
        "results": str(results_path),
        "gate": str(gate_path),
        "metrics": str(csv_path),
        "curves": str(curves_path),
        "next_action": result["gate"]["next_action"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if result["gate"]["status"] == "fail" else 0


def _load_source(path: Path) -> dict[str, Any]:
    gate_path = path / "gate.local.json"
    results_path = path / "results.jsonl"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not isinstance(gate, dict):
        raise ValueError(f"expected JSON object: {gate_path}")
    return {"artifact_root": str(path), "gate": gate, "rows": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("joint bridge CSV requires rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str,
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_joint_bridge_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    rc = {
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#0F172A",
        "svg.fonttype": "none",
    }
    auc_series = (
        ("候选网络", "candidate_test_auc", "#2563EB"),
        ("线性基线", "linear_test_auc", "#D97706"),
        ("乱标签控制", "shuffled_test_auc", "#64748B"),
        ("架构先验", "architecture_prior_oriented_auc", "#7C3AED"),
        ("最强 parity", "strongest_oriented_fixed_parity_auc", "#059669"),
    )
    margin_series = (
        ("相对线性", "candidate_linear_auc_delta", "#D97706"),
        ("相对架构先验", "candidate_architecture_prior_auc_delta", "#7C3AED"),
        ("相对 parity", "candidate_strongest_fixed_parity_auc_delta", "#059669"),
    )
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
        fig.subplots_adjust(left=0.075, right=0.975, top=0.76, bottom=0.18, wspace=0.24)
        fig.suptitle(
            "创新2：PRESENT-80 8轮高轮积分神经双 seed 联合裁决",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.075,
            0.91,
            "每颗 seed：262,144 总训练行（约 131,072/类）｜2 个 multiset｜5 epochs｜独立测试 65,536 行",
            ha="left",
            fontsize=10,
            color="#475569",
        )

        x = list(range(len(rows)))
        width = 0.15
        auc_values: list[float] = []
        for series_index, (label, key, color) in enumerate(auc_series):
            offset = (series_index - 2) * width
            values = [float(row[key]) for row in rows]
            auc_values.extend(values)
            bars = axes[0].bar(
                [position + offset for position in x],
                values,
                width=width,
                label=label,
                color=color,
                alpha=0.92,
            )
            axes[0].bar_label(bars, fmt="%.3f", fontsize=7.2, padding=2, rotation=90)
        axes[0].axhline(0.5, color="#94A3B8", linewidth=1.0, linestyle=(0, (2, 2)))
        axes[0].axhline(0.53, color="#DC2626", linewidth=1.0, linestyle=(0, (4, 3)))
        axes[0].set_ylim(max(0.47, min(auc_values) - 0.012), min(1.0, max(auc_values) + 0.02))
        axes[0].set_xticks(x, [f"seed {int(row['seed'])}" for row in rows])
        axes[0].set_ylabel("独立测试 AUC")
        axes[0].set_title("候选网络与同协议控制", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.7, alpha=0.75)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        margin_values: list[float] = []
        margin_width = 0.22
        for series_index, (label, key, color) in enumerate(margin_series):
            offset = (series_index - 1) * margin_width
            values = [float(row[key]) for row in rows]
            margin_values.extend(values)
            bars = axes[1].bar(
                [position + offset for position in x],
                values,
                width=margin_width,
                label=label,
                color=color,
                alpha=0.92,
            )
            axes[1].bar_label(bars, fmt="%+.3f", fontsize=8, padding=3)
        axes[1].axhline(0.0, color="#94A3B8", linewidth=0.9)
        axes[1].axhline(0.01, color="#DC2626", linewidth=1.0, linestyle=(0, (4, 3)))
        lower = min(-0.01, min(margin_values) - 0.008)
        upper = max(0.025, max(margin_values) + 0.012)
        axes[1].set_ylim(lower, upper)
        axes[1].set_xticks(x, [f"seed {int(row['seed'])}" for row in rows])
        axes[1].set_ylabel("候选网络 AUC 优势")
        axes[1].set_title("归因 margin（红线为 +0.01）", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.7, alpha=0.75)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        verdict = {
            "pass": "通过：两颗 seed 均达到 r8 有用神经信号门槛",
            "hold": "暂缓：至少一颗 seed 未确认全部信号 margin",
            "fail": "无效：source、协议或控制证据不完整",
        }.get(str(gate.get("status")), str(gate.get("decision", "")))
        fig.text(0.075, 0.045, verdict, ha="left", fontsize=10.2, fontweight="bold")
        fig.savefig(output, format="svg")
        plt.close(fig)


__all__ = ["main", "parse_args", "render_joint_bridge_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
