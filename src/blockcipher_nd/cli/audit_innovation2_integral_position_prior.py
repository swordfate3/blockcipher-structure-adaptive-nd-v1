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

from blockcipher_nd.tasks.innovation2.integral_position_prior_audit import (
    SELECTORS,
    PositionPriorAuditConfig,
    evaluate_position_prior_audit,
)


COLORS = {
    "structure_mlp": "#DC2626",
    "train_output_position_prior": "#7C3AED",
    "position_matched_linear": "#2563EB",
    "position_matched_random": "#64748B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the Innovation 2 E5 enrichment is explained by a "
            "training-only output-position prior."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranking-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--fresh-keys", required=True, type=int)
    parser.add_argument("--key-seed", required=True, type=int)
    parser.add_argument("--matched-random-seed", required=True, type=int)
    parser.add_argument("--experiment-seed", required=True, type=int)
    parser.add_argument("--structure-chunk-size", type=int, default=16)
    parser.add_argument(
        "--gate-mode",
        choices=("position-prior-smoke", "position-prior-audit"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ranking_rows = _read_csv(args.ranking_root / "ranking.csv")
    ranking_gate = _read_json(args.ranking_root / "gate.json")
    source_summary = _read_json(args.source_root / "dataset_summary.json")
    config = PositionPriorAuditConfig(
        run_id=args.run_id,
        top_k=args.top_k,
        fresh_keys=args.fresh_keys,
        key_seed=args.key_seed,
        matched_random_seed=args.matched_random_seed,
        experiment_seed=args.experiment_seed,
        gate_mode=args.gate_mode,
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
            "ranking_root": str(args.ranking_root),
            "source_root": str(args.source_root),
            "top_k": args.top_k,
            "fresh_keys": args.fresh_keys,
            "key_seed": args.key_seed,
            "matched_random_seed": args.matched_random_seed,
            "experiment_seed": args.experiment_seed,
            "gate_mode": args.gate_mode,
            "training_performed": False,
        },
        mode="w",
    )
    result = evaluate_position_prior_audit(
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
    _write_csv(args.output_root / "train_position_priors.csv", result["position_rows"])
    _write_csv(args.output_root / "selector_overlaps.csv", result["overlap_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_position_prior_svg(
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


def render_position_prior_svg(
    selector_rows: list[dict[str, Any]],
    rate_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_selector = {str(row["selector"]): row for row in selector_rows}
    labels = [str(by_selector[selector]["selector_label"]) for selector in SELECTORS]
    means = [float(by_selector[selector]["mean_balance_rate"]) for selector in SELECTORS]
    colors = [COLORS[selector] for selector in SELECTORS]
    histograms = np.zeros((len(SELECTORS), 16), dtype=np.int64)
    for selector_index, selector in enumerate(SELECTORS):
        for row in rate_rows:
            if row["selector"] == selector:
                histograms[selector_index, int(row["output_nibble"])] += 1
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
        figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.6))
        figure.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.72,
            bottom=0.22,
            wspace=0.31,
        )
        figure.suptitle(
            "创新2 E6：输出位置先验能否解释神经候选富集",
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
                f"只用冻结训练集构造位置先验；四种方法各取 {top_k} 个结构，"
                f"并在相同 {fresh_keys} 把新密钥上评估。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        rate_axis, heatmap_axis = axes
        bars = rate_axis.bar(labels, means, color=colors, width=0.62)
        for selector_index, selector in enumerate(SELECTORS):
            rates = [
                float(row["balance_rate"])
                for row in rate_rows
                if row["selector"] == selector
            ]
            offsets = np.linspace(-0.18, 0.18, len(rates))
            rate_axis.scatter(
                selector_index + offsets,
                rates,
                color="#0F172A",
                alpha=0.58,
                s=14,
                zorder=3,
            )
        rate_axis.set_title(
            "top-k fresh-key 平衡率",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        rate_axis.set_ylabel("平均平衡率（点表示单个结构）")
        rate_axis.set_ylim(
            max(
                0.0,
                min(float(row["balance_rate"]) for row in rate_rows) - 0.08,
            ),
            1.035,
        )
        rate_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        rate_axis.grid(False, axis="x")
        label_y = rate_axis.get_ylim()[0] + 0.012
        for bar, mean in zip(bars, means, strict=True):
            rate_axis.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="white",
            )

        image = heatmap_axis.imshow(
            histograms,
            cmap="YlOrRd",
            aspect="auto",
            vmin=0,
            vmax=max(1, int(histograms.max())),
        )
        heatmap_axis.set_title(
            "各选择器命中的输出 nibble 数量",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        heatmap_axis.set_xlabel("输出 nibble（0-based）")
        heatmap_axis.set_xticks(range(16))
        heatmap_axis.set_yticks(range(len(SELECTORS)), labels=labels)
        for row_index in range(histograms.shape[0]):
            for column_index in range(histograms.shape[1]):
                count = int(histograms[row_index, column_index])
                if count:
                    heatmap_axis.text(
                        column_index,
                        row_index,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=("white" if count > histograms.max() / 2 else "#0F172A"),
                    )
        figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=0, pad=7)

        metrics = gate["metrics"]
        figure.text(
            0.07,
            0.075,
            (
                f"MLP 相对训练位置先验 {metrics['candidate_position_prior_advantage']:+.3f}；"
                f"相对位置匹配线性 {metrics['candidate_matched_linear_advantage']:+.3f}；"
                f"相对位置匹配随机 {metrics['candidate_matched_random_advantage']:+.3f}。"
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
