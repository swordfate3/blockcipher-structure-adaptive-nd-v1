from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
)


DEFAULT_IMPROVEMENT_MARGIN = 0.001
DEFAULT_MAX_ERROR_JACCARD = 0.85


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aligned PRESENT trail-position frozen scores against the global-stat control."
    )
    parser.add_argument("--global-artifact", required=True, type=Path)
    parser.add_argument("--candidate-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--improvement-margin", type=float, default=DEFAULT_IMPROVEMENT_MARGIN)
    parser.add_argument("--max-error-jaccard", type=float, default=DEFAULT_MAX_ERROR_JACCARD)
    return parser.parse_args(argv)


def analyze_trail_position_scores(
    *,
    global_artifact_dir: Path,
    candidate_artifact_dir: Path,
    improvement_margin: float = DEFAULT_IMPROVEMENT_MARGIN,
    max_error_jaccard: float = DEFAULT_MAX_ERROR_JACCARD,
) -> dict[str, Any]:
    global_artifact = load_score_artifact(global_artifact_dir)
    candidate_artifact = load_score_artifact(candidate_artifact_dir)
    summary = evaluate_frozen_score_ensemble([global_artifact, candidate_artifact])
    global_report = summary["models"][0]
    candidate_report = summary["models"][1]
    pairwise = summary["diversity"]["pairwise"][0]
    overlap = _error_overlap_report(global_artifact, candidate_artifact)

    margins = {
        "auc": _metric(candidate_report, "auc") - _metric(global_report, "auc"),
        "accuracy": _metric(candidate_report, "accuracy") - _metric(global_report, "accuracy"),
        "calibrated_accuracy": _metric(candidate_report, "calibrated_accuracy")
        - _metric(global_report, "calibrated_accuracy"),
    }
    candidate_clears_global = margins["auc"] >= improvement_margin
    acceptable_overlap = pairwise["error_jaccard_at_0_5"] <= max_error_jaccard
    candidate_adds_unique_corrections = (
        overlap["candidate_correct_global_wrong_rate_at_0_5"]
        > overlap["global_correct_candidate_wrong_rate_at_0_5"]
    )

    if candidate_clears_global and acceptable_overlap and candidate_adds_unique_corrections:
        decision = "support_trail_position_score_residual"
        action = "run_residual_and_error_overlap_gate_before_any_ensemble_claim"
        interpretation = (
            "Trail-position frozen scores beat the same-input global-stat control and correct more "
            "of the control's errors than they introduce at the 0.5 threshold."
        )
    elif candidate_clears_global:
        decision = "hold_trail_position_high_overlap_or_threshold_risk"
        action = "inspect_error_overlap_before_scaling_or_ensemble_promotion"
        interpretation = (
            "Trail-position frozen scores beat the global-stat control by AUC, but overlap or "
            "threshold-side errors need inspection before treating the signal as complementary."
        )
    else:
        decision = "hold_trail_position_score_residual"
        action = "do_not_promote_score_artifacts_beyond_diagnostic_use"
        interpretation = "Trail-position frozen scores do not clear the same-input global-stat control by the gate margin."

    return {
        "status": "pass",
        "global_artifact": str(global_artifact_dir),
        "candidate_artifact": str(candidate_artifact_dir),
        "rows": int(len(global_artifact.labels)),
        "global_control": global_report,
        "candidate": candidate_report,
        "margins_vs_global_control": margins,
        "improvement_margin": improvement_margin,
        "pairwise": pairwise,
        "overlap_at_0_5": overlap,
        "max_allowed_error_jaccard_at_0_5": max_error_jaccard,
        "best_ensemble": summary["best_ensemble"],
        "delta_best_ensemble_vs_single_auc": summary["delta_best_ensemble_vs_single_auc"],
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "PRESENT r8 trail-position frozen-score residual diagnostic only; "
            "not a diverse ensemble claim, not raw single-sample SOTA, and not formal SPN/PRESENT evidence"
        ),
    }


def _metric(report: dict[str, Any], key: str) -> float:
    return float(report["metrics"][key])


def _error_overlap_report(
    global_artifact: EnsembleScoreArtifact,
    candidate_artifact: EnsembleScoreArtifact,
) -> dict[str, Any]:
    labels = global_artifact.labels.astype(np.float32, copy=False)
    global_pred = global_artifact.probabilities >= 0.5
    candidate_pred = candidate_artifact.probabilities >= 0.5
    global_correct = global_pred == labels
    candidate_correct = candidate_pred == labels
    global_wrong = ~global_correct
    candidate_wrong = ~candidate_correct
    either_wrong = global_wrong | candidate_wrong
    both_wrong = global_wrong & candidate_wrong
    n = max(1, len(labels))
    either_wrong_count = int(either_wrong.sum())
    return {
        "both_correct_rate_at_0_5": float((global_correct & candidate_correct).sum() / n),
        "both_wrong_rate_at_0_5": float(both_wrong.sum() / n),
        "global_correct_candidate_wrong_rate_at_0_5": float((global_correct & candidate_wrong).sum() / n),
        "candidate_correct_global_wrong_rate_at_0_5": float((candidate_correct & global_wrong).sum() / n),
        "global_wrong_count_at_0_5": int(global_wrong.sum()),
        "candidate_wrong_count_at_0_5": int(candidate_wrong.sum()),
        "both_wrong_count_at_0_5": int(both_wrong.sum()),
        "either_wrong_count_at_0_5": either_wrong_count,
        "error_jaccard_at_0_5": float(both_wrong.sum() / either_wrong_count) if either_wrong_count else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_trail_position_scores(
        global_artifact_dir=args.global_artifact,
        candidate_artifact_dir=args.candidate_artifact,
        improvement_margin=args.improvement_margin,
        max_error_jaccard=args.max_error_jaccard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
