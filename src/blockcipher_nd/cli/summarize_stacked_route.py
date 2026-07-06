from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


PASS_DECISION = "stable_stacked_selection_improves_best_single"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize route-level stacking selection stability across seed summaries."
    )
    parser.add_argument("--summaries", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-positive-seed-fraction", type=float, default=1.0)
    return parser.parse_args(argv)


def summarize_stacked_route(
    summaries: list[dict[str, Any]],
    *,
    min_positive_seed_fraction: float = 1.0,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one summary is required")
    if min_positive_seed_fraction < 0.0 or min_positive_seed_fraction > 1.0:
        raise ValueError("min_positive_seed_fraction must be in [0, 1]")

    rows = [_row_from_summary(index, summary) for index, summary in enumerate(summaries)]
    passed_seed_count = sum(bool(row["passes_seed_gate"]) for row in rows)
    positive_seed_fraction = passed_seed_count / len(rows)
    delta_means = [float(row["delta_mean_vs_best_single_auc"]) for row in rows]
    all_seed_summaries_positive = positive_seed_fraction >= min_positive_seed_fraction
    decision = (
        "stable_cross_seed_stacking_improves_best_single"
        if all_seed_summaries_positive
        else "stable_but_mixed_cross_seed_stacking_diagnostic"
    )

    return {
        "status": "pass",
        "decision": decision,
        "seed_count": len(rows),
        "passed_seed_count": passed_seed_count,
        "positive_seed_fraction": positive_seed_fraction,
        "min_positive_seed_fraction": float(min_positive_seed_fraction),
        "all_seed_summaries_positive": all_seed_summaries_positive,
        "delta_mean_vs_best_single_auc": {
            "min": min(delta_means),
            "max": max(delta_means),
            "mean": mean(delta_means),
        },
        "rows": rows,
        "claim_scope": (
            "route-level diagnostic across precomputed per-seed stacking selection summaries; "
            "does not train models, change validation data, or provide formal SPN/PRESENT evidence"
        ),
    }


def _row_from_summary(index: int, summary: dict[str, Any]) -> dict[str, Any]:
    delta = summary["delta_vs_best_single_auc"]
    delta_mean = float(delta["mean"])
    decision = str(summary.get("decision", ""))
    positive_fraction = float(summary.get("positive_fraction", 0.0))
    passes_seed_gate = decision == PASS_DECISION and delta_mean > 0.0 and positive_fraction >= 1.0
    return {
        "summary_index": index,
        "decision": decision,
        "report_count": int(summary.get("report_count", 0)),
        "positive_count": int(summary.get("positive_count", 0)),
        "positive_fraction": positive_fraction,
        "same_selection": bool(summary.get("same_selection", False)),
        "delta_mean_vs_best_single_auc": delta_mean,
        "delta_min_vs_best_single_auc": float(delta["min"]),
        "delta_max_vs_best_single_auc": float(delta["max"]),
        "passes_seed_gate": passes_seed_gate,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    summary = summarize_stacked_route(
        summaries,
        min_positive_seed_fraction=args.min_positive_seed_fraction,
    )
    summary["summary_paths"] = [str(path) for path in args.summaries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
