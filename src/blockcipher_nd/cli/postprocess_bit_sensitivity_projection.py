from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
)


DEFAULT_MIN_PROJECTION_AUC = 0.52
DEFAULT_MIN_MARGIN_VS_GLOBAL = 0.001
DEFAULT_MAX_ERROR_JACCARD_WITH_ANCHOR = 0.75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate a bit-sensitivity projection score artifact against global and trail-position controls."
    )
    parser.add_argument("--global-artifact", required=True, type=Path)
    parser.add_argument("--anchor-artifact", required=True, type=Path)
    parser.add_argument("--projection-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-projection-auc", type=float, default=DEFAULT_MIN_PROJECTION_AUC)
    parser.add_argument("--min-margin-vs-global", type=float, default=DEFAULT_MIN_MARGIN_VS_GLOBAL)
    parser.add_argument(
        "--max-error-jaccard-with-anchor",
        type=float,
        default=DEFAULT_MAX_ERROR_JACCARD_WITH_ANCHOR,
    )
    return parser.parse_args(argv)


def postprocess_bit_sensitivity_projection(
    *,
    global_artifact_dir: Path,
    anchor_artifact_dir: Path,
    projection_artifact_dir: Path,
    min_projection_auc: float = DEFAULT_MIN_PROJECTION_AUC,
    min_margin_vs_global: float = DEFAULT_MIN_MARGIN_VS_GLOBAL,
    max_error_jaccard_with_anchor: float = DEFAULT_MAX_ERROR_JACCARD_WITH_ANCHOR,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        global_artifact = load_score_artifact(global_artifact_dir)
        anchor_artifact = load_score_artifact(anchor_artifact_dir)
        projection_artifact = load_score_artifact(projection_artifact_dir)
        ensemble = evaluate_frozen_score_ensemble(
            [global_artifact, anchor_artifact, projection_artifact]
        )
    except Exception as exc:  # noqa: BLE001 - postprocess reports artifact/protocol failures.
        return _failure_report(
            global_artifact_dir=global_artifact_dir,
            anchor_artifact_dir=anchor_artifact_dir,
            projection_artifact_dir=projection_artifact_dir,
            errors=[f"{type(exc).__name__}:{exc}"],
        )

    _check_required_metadata(global_artifact, anchor_artifact, projection_artifact, errors)
    models = ensemble.get("models", [])
    global_report = _report_for_artifact(models, global_artifact)
    anchor_report = _report_for_artifact(models, anchor_artifact)
    projection_report = _report_for_artifact(models, projection_artifact)
    if global_report is None:
        errors.append("missing_global_control_report")
        global_report = {}
    if anchor_report is None:
        errors.append("missing_anchor_report")
        anchor_report = {}
    if projection_report is None:
        errors.append("missing_projection_report")
        projection_report = {}

    global_auc = _metric_auc(global_report)
    anchor_auc = _metric_auc(anchor_report)
    projection_auc = _metric_auc(projection_report)
    if global_auc is None:
        errors.append("missing_global_control_auc")
    if anchor_auc is None:
        errors.append("missing_anchor_auc")
    if projection_auc is None:
        errors.append("missing_projection_auc")

    overlap_with_anchor = _pairwise_for_artifacts(
        ensemble,
        left=anchor_artifact,
        right=projection_artifact,
    )
    if overlap_with_anchor is None:
        errors.append("missing_anchor_projection_pairwise_overlap")
        overlap_with_anchor = {}

    margin_vs_global = (
        float(projection_auc - global_auc)
        if projection_auc is not None and global_auc is not None
        else None
    )
    margin_vs_anchor = (
        float(projection_auc - anchor_auc)
        if projection_auc is not None and anchor_auc is not None
        else None
    )
    if errors:
        return _failure_report(
            global_artifact_dir=global_artifact_dir,
            anchor_artifact_dir=anchor_artifact_dir,
            projection_artifact_dir=projection_artifact_dir,
            errors=errors,
            ensemble=ensemble,
        )

    hold_reasons = _hold_reasons(
        projection_auc=float(projection_auc),
        margin_vs_global=float(margin_vs_global),
        overlap_with_anchor=overlap_with_anchor,
        min_projection_auc=min_projection_auc,
        min_margin_vs_global=min_margin_vs_global,
        max_error_jaccard_with_anchor=max_error_jaccard_with_anchor,
    )
    if hold_reasons:
        decision = "hold_projection_duplicate_or_weak"
        action = "do_not_promote_projection_to_diverse_pool_or_remote_scale"
        interpretation = (
            "The projection artifact is aligned, but it is too weak or too similar "
            "to the trail-position anchor for a clean non-neighbor expert claim."
        )
        next_action = {
            "branch": "hold_bit_sensitivity_projection",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    else:
        decision = "projection_expert_ready_for_local_screen"
        action = "run_local_diversity_or_ensemble_screen_with_this_projection_artifact"
        interpretation = (
            "The projection clears the same-input global control and has acceptable "
            "error overlap with the trail-position anchor."
        )
        next_action = {
            "branch": "bit_sensitivity_projection_local_screen_ready",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "recommended_followup": (
                "evaluate frozen-score diversity against global control and trail-position "
                "before any remote training slot"
            ),
        }

    return {
        "status": "pass",
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "global_artifact": str(global_artifact_dir),
        "anchor_artifact": str(anchor_artifact_dir),
        "projection_artifact": str(projection_artifact_dir),
        "global_control": global_report,
        "anchor": anchor_report,
        "projection": projection_report,
        "margins_vs_global_control": {"auc": margin_vs_global},
        "margins_vs_anchor": {"auc": margin_vs_anchor},
        "overlap_with_anchor": overlap_with_anchor,
        "hold_reasons": hold_reasons,
        "thresholds": {
            "min_projection_auc": min_projection_auc,
            "min_margin_vs_global": min_margin_vs_global,
            "max_error_jaccard_with_anchor": max_error_jaccard_with_anchor,
        },
        "ensemble_summary": ensemble,
        "next_action": next_action,
        "claim_scope": (
            "bit-sensitivity projection frozen-score gate only; local diagnostic for "
            "diverse expert readiness, not a trained neural-model result, not remote "
            "evidence, and not formal SPN/PRESENT evidence"
        ),
    }


def _failure_report(
    *,
    global_artifact_dir: Path,
    anchor_artifact_dir: Path,
    projection_artifact_dir: Path,
    errors: list[str],
    ensemble: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "fail_protocol_alignment",
        "action": "repair_score_artifact_alignment_before_interpreting_projection",
        "global_artifact": str(global_artifact_dir),
        "anchor_artifact": str(anchor_artifact_dir),
        "projection_artifact": str(projection_artifact_dir),
        "errors": errors,
        "ensemble_summary": ensemble,
        "next_action": {
            "branch": "repair_bit_sensitivity_projection_artifacts",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": "protocol_or_artifact_alignment_failed",
        },
        "claim_scope": (
            "bit-sensitivity projection frozen-score gate only; no metric or expert-pool "
            "claim is allowed while artifact alignment fails"
        ),
    }


def _check_required_metadata(
    global_artifact: EnsembleScoreArtifact,
    anchor_artifact: EnsembleScoreArtifact,
    projection_artifact: EnsembleScoreArtifact,
    errors: list[str],
) -> None:
    if global_artifact.metadata.get("negative_mode") != "encrypted_random_plaintexts":
        errors.append("global_control_negative_mode_not_strict")
    if anchor_artifact.metadata.get("negative_mode") != "encrypted_random_plaintexts":
        errors.append("anchor_negative_mode_not_strict")
    if projection_artifact.metadata.get("negative_mode") != "encrypted_random_plaintexts":
        errors.append("projection_negative_mode_not_strict")
    if projection_artifact.metadata.get("expert_family") != "bit_sensitivity_projection":
        errors.append("projection_expert_family_not_bit_sensitivity_projection")


def _report_for_artifact(
    models: Any,
    artifact: EnsembleScoreArtifact,
) -> dict[str, Any] | None:
    if not isinstance(models, list):
        return None
    model_key = str(artifact.metadata.get("model_key", ""))
    expert_family = str(artifact.metadata.get("expert_family", ""))
    for row in models:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if (
            str(row.get("model_key", "")) == model_key
            and str(metadata.get("expert_family", "")) == expert_family
        ):
            return row
    return None


def _pairwise_for_artifacts(
    ensemble: dict[str, Any],
    *,
    left: EnsembleScoreArtifact,
    right: EnsembleScoreArtifact,
) -> dict[str, Any] | None:
    pairwise = (ensemble.get("diversity") or {}).get("pairwise", [])
    left_key = str(left.metadata.get("model_key", ""))
    right_key = str(right.metadata.get("model_key", ""))
    if not isinstance(pairwise, list):
        return None
    for row in pairwise:
        if not isinstance(row, dict):
            continue
        observed = {str(row.get("left", "")), str(row.get("right", ""))}
        if observed == {left_key, right_key}:
            return row
    return None


def _hold_reasons(
    *,
    projection_auc: float,
    margin_vs_global: float,
    overlap_with_anchor: dict[str, Any],
    min_projection_auc: float,
    min_margin_vs_global: float,
    max_error_jaccard_with_anchor: float,
) -> list[str]:
    reasons: list[str] = []
    if projection_auc < min_projection_auc:
        reasons.append("projection_auc_below_gate")
    if margin_vs_global < min_margin_vs_global:
        reasons.append("does_not_clear_global_control")
    error_jaccard = _float_or_none(overlap_with_anchor.get("error_jaccard_at_0_5"))
    if error_jaccard is None:
        reasons.append("missing_anchor_error_overlap")
    elif error_jaccard > max_error_jaccard_with_anchor:
        reasons.append("high_error_overlap_with_anchor")
    return reasons


def _metric_auc(report: dict[str, Any]) -> float | None:
    metrics = report.get("metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict):
        return None
    return _float_or_none(metrics.get("auc"))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_bit_sensitivity_projection(
        global_artifact_dir=args.global_artifact,
        anchor_artifact_dir=args.anchor_artifact,
        projection_artifact_dir=args.projection_artifact,
        min_projection_auc=args.min_projection_auc,
        min_margin_vs_global=args.min_margin_vs_global,
        max_error_jaccard_with_anchor=args.max_error_jaccard_with_anchor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
