from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


NORMAL_DECISION = "compressed_feature_expert_local_screen_positive_needs_controls"
SHUFFLE_DECISION = "compressed_feature_expert_shuffle_train_labels_control"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize compact compressed-span summary retention against flat span reports."
    )
    parser.add_argument("--flat-span-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--summary-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--summary-shuffle-control-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-summary-auc", type=float, default=0.99)
    parser.add_argument("--max-shuffle-control-auc", type=float, default=0.55)
    parser.add_argument("--max-auc-drop-vs-flat", type=float, default=0.001)
    return parser.parse_args(argv)


def summarize_compressed_span_route(
    *,
    flat_span_reports: list[dict[str, Any]],
    summary_reports: list[dict[str, Any]],
    summary_shuffle_control_reports: list[dict[str, Any]],
    min_summary_auc: float = 0.99,
    max_shuffle_control_auc: float = 0.55,
    max_auc_drop_vs_flat: float = 0.001,
) -> dict[str, Any]:
    if not flat_span_reports:
        raise ValueError("at least one flat span report is required")
    if len(flat_span_reports) != len(summary_reports):
        raise ValueError("flat span and summary report counts must match")
    if len(flat_span_reports) != len(summary_shuffle_control_reports):
        raise ValueError("flat span and shuffle-control report counts must match")
    if min_summary_auc < 0.0 or min_summary_auc > 1.0:
        raise ValueError("min_summary_auc must be in [0, 1]")
    if max_shuffle_control_auc < 0.0 or max_shuffle_control_auc > 1.0:
        raise ValueError("max_shuffle_control_auc must be in [0, 1]")
    if max_auc_drop_vs_flat < 0.0:
        raise ValueError("max_auc_drop_vs_flat must be non-negative")

    rows = [
        _row(index, flat_report, summary_report, shuffle_report, min_summary_auc, max_shuffle_control_auc, max_auc_drop_vs_flat)
        for index, (flat_report, summary_report, shuffle_report) in enumerate(
            zip(flat_span_reports, summary_reports, summary_shuffle_control_reports, strict=True)
        )
    ]
    flat_feature_counts = {int(report.get("feature_count", 0)) for report in flat_span_reports}
    summary_feature_counts = {int(report.get("feature_count", 0)) for report in summary_reports}
    flat_feature_count = _only_count(flat_feature_counts, "flat span")
    summary_feature_count = _only_count(summary_feature_counts, "summary")
    all_summary_positive = all(row["summary_passes_gate"] for row in rows)
    all_shuffle_controls_random_like = all(row["shuffle_control_passes_gate"] for row in rows)
    all_summary_retains_flat_signal = all(row["retention_passes_gate"] for row in rows)
    if all_summary_positive and all_shuffle_controls_random_like and all_summary_retains_flat_signal:
        decision = "compressed_span_summary_retains_flat_signal_controls_pass"
    elif all_summary_positive and all_shuffle_controls_random_like:
        decision = "compressed_span_summary_positive_but_drop_audit"
    else:
        decision = "compressed_span_summary_hold_or_audit"
    return {
        "status": "pass",
        "decision": decision,
        "seed_count": len(rows),
        "feature_counts": {
            "flat_span": flat_feature_count,
            "summary": summary_feature_count,
        },
        "feature_reduction_ratio": summary_feature_count / flat_feature_count,
        "thresholds": {
            "min_summary_auc": float(min_summary_auc),
            "max_shuffle_control_auc": float(max_shuffle_control_auc),
            "max_auc_drop_vs_flat": float(max_auc_drop_vs_flat),
        },
        "flat_auc": _stats([row["flat_auc"] for row in rows]),
        "summary_auc": _stats([row["summary_auc"] for row in rows]),
        "shuffle_control_auc": _stats([row["shuffle_control_auc"] for row in rows]),
        "auc_drop_vs_flat": _stats([row["auc_drop_vs_flat"] for row in rows]),
        "all_summary_positive": all_summary_positive,
        "all_shuffle_controls_random_like": all_shuffle_controls_random_like,
        "all_summary_retains_flat_signal": all_summary_retains_flat_signal,
        "rows": rows,
        "claim_scope": (
            "compressed span summary retention diagnostic only; compares existing local reports, "
            "does not train, score, alter labels, alter negatives, launch remote runs, or provide "
            "formal SPN/PRESENT evidence"
        ),
    }


def _row(
    index: int,
    flat_report: dict[str, Any],
    summary_report: dict[str, Any],
    shuffle_report: dict[str, Any],
    min_summary_auc: float,
    max_shuffle_control_auc: float,
    max_auc_drop_vs_flat: float,
) -> dict[str, Any]:
    flat_auc = float(flat_report["validation_metrics"]["auc"])
    summary_auc = float(summary_report["validation_metrics"]["auc"])
    shuffle_auc = float(shuffle_report["validation_metrics"]["auc"])
    auc_drop = flat_auc - summary_auc
    return {
        "summary_index": int(index),
        "flat_decision": str(flat_report.get("decision", "")),
        "summary_decision": str(summary_report.get("decision", "")),
        "shuffle_decision": str(shuffle_report.get("decision", "")),
        "flat_auc": flat_auc,
        "summary_auc": summary_auc,
        "shuffle_control_auc": shuffle_auc,
        "auc_drop_vs_flat": auc_drop,
        "flat_feature_count": int(flat_report.get("feature_count", 0)),
        "summary_feature_count": int(summary_report.get("feature_count", 0)),
        "summary_passes_gate": (
            str(summary_report.get("decision", "")) == NORMAL_DECISION
            and summary_auc >= min_summary_auc
        ),
        "shuffle_control_passes_gate": (
            str(shuffle_report.get("decision", "")) == SHUFFLE_DECISION
            and shuffle_auc <= max_shuffle_control_auc
        ),
        "retention_passes_gate": auc_drop <= max_auc_drop_vs_flat,
    }


def _only_count(values: set[int], label: str) -> int:
    if len(values) != 1:
        raise ValueError(f"{label} feature counts must be stable, got {sorted(values)}")
    value = next(iter(values))
    if value <= 0:
        raise ValueError(f"{label} feature count must be positive")
    return value


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    flat_span_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.flat_span_reports]
    summary_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary_reports]
    summary_shuffle_control_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summary_shuffle_control_reports
    ]
    report = summarize_compressed_span_route(
        flat_span_reports=flat_span_reports,
        summary_reports=summary_reports,
        summary_shuffle_control_reports=summary_shuffle_control_reports,
        min_summary_auc=args.min_summary_auc,
        max_shuffle_control_auc=args.max_shuffle_control_auc,
        max_auc_drop_vs_flat=args.max_auc_drop_vs_flat,
    )
    report["flat_span_report_paths"] = [str(path) for path in args.flat_span_reports]
    report["summary_report_paths"] = [str(path) for path in args.summary_reports]
    report["summary_shuffle_control_report_paths"] = [
        str(path) for path in args.summary_shuffle_control_reports
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
