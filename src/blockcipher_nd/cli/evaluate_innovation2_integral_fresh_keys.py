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

from blockcipher_nd.tasks.innovation2.integral_fresh_key_validation import (
    SELECTORS,
    FreshKeyValidationConfig,
    evaluate_fresh_key_enrichment,
)


COLORS = {
    "structure_mlp": "#DC2626",
    "linear_same_input": "#2563EB",
    "p_layer_reachability": "#D97706",
    "fixed_random": "#64748B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Innovation 2 E4 selectors on a frozen batch of fresh "
            "PRESENT-80 keys without retraining."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranking-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--fresh-keys", required=True, type=int)
    parser.add_argument("--key-seed", required=True, type=int)
    parser.add_argument("--random-selector-seed", required=True, type=int)
    parser.add_argument("--key-chunk-size", type=int, default=256)
    parser.add_argument(
        "--gate-mode",
        choices=("fresh-key-smoke", "fresh-key-enrichment"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ranking_rows = _read_csv(args.ranking_root / "ranking.csv")
    ranking_gate = _read_json(args.ranking_root / "gate.json")
    source_summary = _read_json(args.source_root / "dataset_summary.json")
    config = FreshKeyValidationConfig(
        run_id=args.run_id,
        top_k=args.top_k,
        fresh_keys=args.fresh_keys,
        key_seed=args.key_seed,
        random_selector_seed=args.random_selector_seed,
        gate_mode=args.gate_mode,
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
            "ranking_root": str(args.ranking_root),
            "source_root": str(args.source_root),
            "top_k": args.top_k,
            "fresh_keys": args.fresh_keys,
            "key_seed": args.key_seed,
            "random_selector_seed": args.random_selector_seed,
            "gate_mode": args.gate_mode,
            "training_performed": False,
        },
        mode="w",
    )
    result = evaluate_fresh_key_enrichment(
        config,
        ranking_rows=ranking_rows,
        ranking_gate=ranking_gate,
        source_summary=source_summary,
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
    _write_csv(args.output_root / "fresh_key_rates.csv", result["rate_rows"])
    _write_csv(args.output_root / "selector_overlaps.csv", result["overlap_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_fresh_key_svg(
        result["rows"],
        result["rate_rows"],
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
    report = {
        "run_id": args.run_id,
        "status": result["gate"]["status"],
        "decision": result["gate"]["decision"],
        "output_root": str(args.output_root),
        "results": str(results_path),
        "gate": str(args.output_root / "gate.json"),
        "curves": str(args.output_root / "curves.svg"),
        "next_action": result["gate"]["next_action"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if result["gate"]["status"] != "fail" else 1


def render_fresh_key_svg(
    selector_rows: list[dict[str, Any]],
    rate_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_selector = {str(row["selector"]): row for row in selector_rows}
    labels = [str(by_selector[selector]["selector_label"]) for selector in SELECTORS]
    means = [float(by_selector[selector]["mean_balance_rate"]) for selector in SELECTORS]
    minimums = [
        float(by_selector[selector]["minimum_balance_rate"])
        for selector in SELECTORS
    ]
    zero_counts = [
        int(by_selector[selector]["zero_observed_failure_structures"])
        for selector in SELECTORS
    ]
    colors = [COLORS[selector] for selector in SELECTORS]
    fresh_keys = int(selector_rows[0]["fresh_keys"])
    top_k = int(selector_rows[0]["top_k"])

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
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
        figure.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.72,
            bottom=0.22,
            wspace=0.28,
        )
        figure.suptitle(
            "创新2 E5：PRESENT 5轮候选在全新密钥上的平衡率验证",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.895,
            (
                f"四种方法各固定选择 {top_k} 个未见几何组合，并使用相同的 "
                f"{fresh_keys} 把独立密钥重新评估；本图是统计验证，不是所有密钥证明。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        mean_axis, zero_axis = axes
        bars = mean_axis.bar(labels, means, color=colors, width=0.62)
        for selector_index, selector in enumerate(SELECTORS):
            rates = [
                float(row["balance_rate"])
                for row in rate_rows
                if row["selector"] == selector
            ]
            offsets = np.linspace(-0.18, 0.18, len(rates))
            mean_axis.scatter(
                selector_index + offsets,
                rates,
                color="#0F172A",
                alpha=0.58,
                s=14,
                zorder=3,
            )
        mean_axis.set_title(
            "各选择器 top-k 的 fresh-key 平衡率",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        mean_axis.set_ylabel("平均平衡率（点表示单个结构）")
        mean_axis.set_ylim(max(0.0, min(minimums) - 0.08), 1.035)
        mean_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        mean_axis.grid(False, axis="x")
        label_y = mean_axis.get_ylim()[0] + 0.012
        for bar, mean, minimum in zip(bars, means, minimums, strict=True):
            mean_axis.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"均值 {mean:.3f}\n最低 {minimum:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="white",
                fontweight="bold",
            )

        zero_bars = zero_axis.bar(labels, zero_counts, color=colors, width=0.62)
        zero_axis.set_title(
            f"{fresh_keys} 把密钥中零次观察到失衡的结构数",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        zero_axis.set_ylabel("结构数量（只表示统计零失败）")
        zero_axis.set_ylim(0, max(top_k, max(zero_counts) + 1))
        zero_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        zero_axis.grid(False, axis="x")
        for bar, count in zip(zero_bars, zero_counts, strict=True):
            zero_axis.text(
                bar.get_x() + bar.get_width() / 2,
                count + 0.18,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=0, pad=8)

        metrics = gate["metrics"]
        figure.text(
            0.07,
            0.075,
            (
                f"MLP 相对线性 {metrics['candidate_linear_mean_advantage']:+.3f}；"
                f"相对 P层启发式 {metrics['candidate_reachability_mean_advantage']:+.3f}；"
                f"相对随机 {metrics['candidate_random_mean_advantage']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
