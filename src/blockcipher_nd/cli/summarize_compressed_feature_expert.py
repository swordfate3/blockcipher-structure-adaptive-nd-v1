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
        description=(
            "Summarize compressed SPN feature expert reports, shuffle-label controls, "
            "and frozen-score ensemble diagnostics across seeds."
        )
    )
    parser.add_argument("--normal-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--shuffle-control-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--ensemble-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-normal-auc", type=float, default=0.99)
    parser.add_argument("--max-shuffle-control-auc", type=float, default=0.55)
    parser.add_argument("--min-ensemble-delta", type=float, default=0.0)
    return parser.parse_args(argv)


def summarize_compressed_feature_expert(
    *,
    normal_reports: list[dict[str, Any]],
    shuffle_control_reports: list[dict[str, Any]],
    ensemble_reports: list[dict[str, Any]],
    min_normal_auc: float = 0.99,
    max_shuffle_control_auc: float = 0.55,
    min_ensemble_delta: float = 0.0,
) -> dict[str, Any]:
    if not normal_reports:
        raise ValueError("at least one normal report is required")
    if len(normal_reports) != len(shuffle_control_reports):
        raise ValueError("normal and shuffle-control report counts must match")
    if len(normal_reports) != len(ensemble_reports):
        raise ValueError("normal and ensemble report counts must match")

    normal_rows = [_normal_row(index, report, min_normal_auc) for index, report in enumerate(normal_reports)]
    shuffle_rows = [
        _shuffle_row(index, report, max_shuffle_control_auc)
        for index, report in enumerate(shuffle_control_reports)
    ]
    ensemble_rows = [
        _ensemble_row(index, report, min_ensemble_delta)
        for index, report in enumerate(ensemble_reports)
    ]
    normal_passed = sum(row["passes_gate"] for row in normal_rows)
    shuffle_passed = sum(row["passes_gate"] for row in shuffle_rows)
    ensemble_gain_passed = sum(row["passes_gate"] for row in ensemble_rows)
    all_normal_positive = normal_passed == len(normal_rows)
    all_shuffle_controls_random_like = shuffle_passed == len(shuffle_rows)
    all_ensemble_gains_positive = ensemble_gain_passed == len(ensemble_rows)
    if all_normal_positive and all_shuffle_controls_random_like and all_ensemble_gains_positive:
        decision = "compressed_feature_local_positive_controls_pass_ensemble_gain"
    elif all_normal_positive and all_shuffle_controls_random_like:
        decision = "compressed_feature_local_positive_controls_pass_not_ensemble_gain"
    else:
        decision = "compressed_feature_route_hold_or_audit"

    validation_aucs = [float(row["validation_auc"]) for row in normal_rows]
    shuffle_aucs = [float(row["validation_auc"]) for row in shuffle_rows]
    ensemble_deltas = [float(row["delta_best_ensemble_vs_single_auc"]) for row in ensemble_rows]
    return {
        "status": "pass",
        "decision": decision,
        "seed_count": len(normal_rows),
        "normal_passed_seed_count": normal_passed,
        "shuffle_control_passed_seed_count": shuffle_passed,
        "ensemble_gain_passed_seed_count": ensemble_gain_passed,
        "all_normal_positive": all_normal_positive,
        "all_shuffle_controls_random_like": all_shuffle_controls_random_like,
        "all_ensemble_gains_positive": all_ensemble_gains_positive,
        "thresholds": {
            "min_normal_auc": float(min_normal_auc),
            "max_shuffle_control_auc": float(max_shuffle_control_auc),
            "min_ensemble_delta": float(min_ensemble_delta),
        },
        "validation_auc": _stats(validation_aucs),
        "shuffle_control_auc": _stats(shuffle_aucs),
        "ensemble_delta_vs_best_single_auc": _stats(ensemble_deltas),
        "normal_rows": normal_rows,
        "shuffle_control_rows": shuffle_rows,
        "ensemble_rows": ensemble_rows,
        "claim_scope": (
            "route-level compressed SPN feature expert diagnostic only; this gate "
            "does not train models, alter validation data, prove formal SPN/PRESENT "
            "evidence, or convert a strong single expert into a multi-network claim"
        ),
    }


def _normal_row(index: int, report: dict[str, Any], min_normal_auc: float) -> dict[str, Any]:
    decision = str(report.get("decision", ""))
    auc = float(report["validation_metrics"]["auc"])
    return {
        "summary_index": index,
        "decision": decision,
        "validation_auc": auc,
        "validation_accuracy": float(report["validation_metrics"].get("accuracy", 0.0)),
        "feature_count": int(report.get("feature_count", 0)),
        "validation_rows": int(report.get("validation_rows", 0)),
        "passes_gate": decision == NORMAL_DECISION and auc >= min_normal_auc,
    }


def _shuffle_row(index: int, report: dict[str, Any], max_shuffle_control_auc: float) -> dict[str, Any]:
    decision = str(report.get("decision", ""))
    auc = float(report["validation_metrics"]["auc"])
    return {
        "summary_index": index,
        "decision": decision,
        "validation_auc": auc,
        "validation_accuracy": float(report["validation_metrics"].get("accuracy", 0.0)),
        "passes_gate": decision == SHUFFLE_DECISION and auc <= max_shuffle_control_auc,
    }


def _ensemble_row(index: int, report: dict[str, Any], min_ensemble_delta: float) -> dict[str, Any]:
    delta = float(report["delta_best_ensemble_vs_single_auc"])
    return {
        "summary_index": index,
        "best_single_model_key": str(report.get("best_single", {}).get("model_key", "")),
        "best_single_auc": float(report["best_single"]["metrics"]["auc"]),
        "best_ensemble_auc": float(report["best_ensemble"]["metrics"]["auc"]),
        "delta_best_ensemble_vs_single_auc": delta,
        "passes_gate": delta > min_ensemble_delta,
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    normal_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.normal_reports]
    shuffle_control_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.shuffle_control_reports
    ]
    ensemble_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.ensemble_reports]
    summary = summarize_compressed_feature_expert(
        normal_reports=normal_reports,
        shuffle_control_reports=shuffle_control_reports,
        ensemble_reports=ensemble_reports,
        min_normal_auc=args.min_normal_auc,
        max_shuffle_control_auc=args.max_shuffle_control_auc,
        min_ensemble_delta=args.min_ensemble_delta,
    )
    summary["normal_report_paths"] = [str(path) for path in args.normal_reports]
    summary["shuffle_control_report_paths"] = [str(path) for path in args.shuffle_control_reports]
    summary["ensemble_report_paths"] = [str(path) for path in args.ensemble_reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
