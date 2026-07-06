from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train-holdout stacking selection stability across JSON reports."
    )
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-positive-fraction", type=float, default=1.0)
    parser.add_argument("--require-same-selection", action="store_true")
    return parser.parse_args(argv)


def summarize_stacked_selection(
    reports: list[dict[str, Any]],
    *,
    min_positive_fraction: float = 1.0,
    require_same_selection: bool = False,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    if min_positive_fraction < 0.0 or min_positive_fraction > 1.0:
        raise ValueError("min_positive_fraction must be in [0, 1]")

    rows = [_row_from_report(report) for report in reports]
    deltas = [float(row["delta_vs_best_single_auc"]) for row in rows]
    positive_count = sum(delta > 0.0 for delta in deltas)
    positive_fraction = positive_count / len(deltas)
    selection_counts = Counter(
        (
            str(row["feature_space"]),
            float(row["l2"]),
            bool(row["standardize"]),
        )
        for row in rows
    )
    dominant_selection, dominant_count = selection_counts.most_common(1)[0]
    same_selection = dominant_count == len(rows)
    passes_positive_gate = positive_fraction >= min_positive_fraction
    passes_selection_gate = same_selection or not require_same_selection
    decision = (
        "stable_stacked_selection_improves_best_single"
        if passes_positive_gate and passes_selection_gate
        else "mixed_or_unstable_stacked_selection_diagnostic"
    )
    return {
        "status": "pass",
        "decision": decision,
        "report_count": len(rows),
        "positive_count": positive_count,
        "positive_fraction": positive_fraction,
        "min_positive_fraction": float(min_positive_fraction),
        "delta_vs_best_single_auc": {
            "min": min(deltas),
            "max": max(deltas),
            "mean": mean(deltas),
        },
        "same_selection": same_selection,
        "require_same_selection": bool(require_same_selection),
        "dominant_selection": {
            "feature_space": dominant_selection[0],
            "l2": dominant_selection[1],
            "standardize": dominant_selection[2],
            "count": dominant_count,
        },
        "selection_counts": [
            {
                "feature_space": feature_space,
                "l2": l2,
                "standardize": standardize,
                "count": count,
            }
            for (feature_space, l2, standardize), count in sorted(selection_counts.items())
        ],
        "rows": rows,
        "claim_scope": (
            "stacking selection stability diagnostic across precomputed reports; "
            "does not train models, change validation data, or provide formal SPN/PRESENT evidence"
        ),
    }


def _row_from_report(report: dict[str, Any]) -> dict[str, Any]:
    selection = report.get("selection", {})
    selected = selection.get("selected", {})
    fit = report.get("fit", {})
    feature_space = str(selected.get("feature_space", report.get("feature_space", "")))
    l2 = float(selected.get("l2", fit.get("l2", 0.0)))
    standardize = bool(selected.get("standardize", fit.get("standardize", False)))
    return {
        "selection_seed": int(selection.get("selection_seed", -1)),
        "feature_space": feature_space,
        "l2": l2,
        "standardize": standardize,
        "validation_auc": float(report["validation_metrics"]["auc"]),
        "best_single_auc": float(report["validation_best_single"]["metrics"]["auc"]),
        "delta_vs_best_single_auc": float(report["delta_stacked_vs_validation_best_single_auc"]),
        "delta_vs_fixed_ensemble_auc": float(
            report["delta_stacked_vs_validation_best_fixed_ensemble_auc"]
        ),
        "decision": str(report.get("decision", "")),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    summary = summarize_stacked_selection(
        reports,
        min_positive_fraction=args.min_positive_fraction,
        require_same_selection=args.require_same_selection,
    )
    summary["report_paths"] = [str(path) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
