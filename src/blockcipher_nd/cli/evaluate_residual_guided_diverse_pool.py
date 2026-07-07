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


DEFAULT_POOL_PLAN = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the PRESENT r8 residual-guided Pool 3 fixed-fusion diagnostic "
            "after the residual-focus gate has promoted a focus candidate."
        )
    )
    parser.add_argument("--pool-plan", type=Path, default=DEFAULT_POOL_PLAN)
    parser.add_argument("--trail-position-artifact", type=Path)
    parser.add_argument("--raw117-artifact", type=Path)
    parser.add_argument("--residual-focus-artifact", type=Path)
    parser.add_argument("--uniform-control-artifact", type=Path)
    parser.add_argument("--labelshuffle-control-artifact", type=Path)
    parser.add_argument("--min-auc-delta", type=float, default=0.0)
    parser.add_argument("--min-control-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def evaluate_residual_guided_diverse_pool(
    *,
    pool_plan: Path,
    trail_position_artifact: Path | None,
    raw117_artifact: Path | None,
    residual_focus_artifact: Path | None,
    uniform_control_artifact: Path | None,
    labelshuffle_control_artifact: Path | None,
    min_auc_delta: float = 0.0,
    min_control_delta: float = 0.0,
) -> dict[str, Any]:
    plan = _load_json(pool_plan)
    plan_ready = bool(plan.get("should_run_pool")) and plan.get("status") == "pass"
    if not plan_ready:
        return {
            "status": _pending_or_hold_status(plan),
            "decision": "wait_for_residual_guided_pool_plan",
            "should_run_pool": False,
            "pool_plan": str(pool_plan),
            "pool_plan_decision": str(plan.get("decision", "")),
            "next_action": _next_action(plan, default="finish_residual_focus_262k_gate"),
            "claim_scope": (
                "local Pool 3 fixed-fusion guard only; residual-guided ensemble "
                "evaluation waits until the residual-focus 262144/class gate passes"
            ),
        }

    artifact_paths = {
        "trail_position_artifact": trail_position_artifact,
        "raw117_artifact": raw117_artifact,
        "residual_focus_artifact": residual_focus_artifact,
        "uniform_control_artifact": uniform_control_artifact,
        "labelshuffle_control_artifact": labelshuffle_control_artifact,
    }
    missing = [name for name, path in artifact_paths.items() if path is None or not path.exists()]
    if missing:
        return {
            "status": "pending",
            "decision": "wait_for_pool3_score_artifacts",
            "should_run_pool": True,
            "pool_plan": str(pool_plan),
            "selected_residual_candidate": str(plan.get("selected_residual_candidate", "")),
            "missing_score_artifacts": missing,
            "next_action": {
                "branch": "retrieve_or_generate_pool3_score_artifacts",
                "should_launch_remote": False,
            },
            "claim_scope": (
                "local Pool 3 fixed-fusion guard only; score artifacts must exist and align "
                "before any residual-guided ensemble diagnostic can be interpreted"
            ),
        }

    trail = load_score_artifact(trail_position_artifact)
    raw117 = load_score_artifact(raw117_artifact)
    residual_focus = load_score_artifact(residual_focus_artifact)
    uniform = load_score_artifact(uniform_control_artifact)
    labelshuffle = load_score_artifact(labelshuffle_control_artifact)

    comparisons = [
        _comparison("trail_position_plus_raw117", [trail, raw117]),
        _comparison("trail_position_plus_raw117_plus_residual_focus", [trail, raw117, residual_focus]),
        _comparison("trail_position_plus_raw117_plus_uniform_control", [trail, raw117, uniform]),
        _comparison("trail_position_plus_raw117_plus_labelshuffle_control", [trail, raw117, labelshuffle]),
    ]
    by_name = {row["name"]: row for row in comparisons}
    base_auc = by_name["trail_position_plus_raw117"]["best_ensemble_auc"]
    candidate_auc = by_name["trail_position_plus_raw117_plus_residual_focus"]["best_ensemble_auc"]
    uniform_auc = by_name["trail_position_plus_raw117_plus_uniform_control"]["best_ensemble_auc"]
    labelshuffle_auc = by_name["trail_position_plus_raw117_plus_labelshuffle_control"]["best_ensemble_auc"]
    candidate_delta_vs_base = float(candidate_auc - base_auc)
    candidate_delta_vs_uniform = float(candidate_auc - uniform_auc)
    candidate_delta_vs_labelshuffle = float(candidate_auc - labelshuffle_auc)

    supported = (
        candidate_delta_vs_base > min_auc_delta
        and candidate_delta_vs_uniform >= min_control_delta
        and candidate_delta_vs_labelshuffle > min_control_delta
    )
    return {
        "status": "pass" if supported else "hold",
        "decision": (
            "support_residual_guided_pool3_fixed_fusion"
            if supported
            else "residual_guided_pool3_fixed_fusion_diagnostic_only"
        ),
        "should_run_pool": True,
        "pool_plan": str(pool_plan),
        "selected_residual_candidate": str(plan.get("selected_residual_candidate", "")),
        "artifact_dirs": {name: str(path) for name, path in artifact_paths.items()},
        "comparisons": comparisons,
        "candidate_delta_vs_base_auc": candidate_delta_vs_base,
        "candidate_delta_vs_uniform_control_auc": candidate_delta_vs_uniform,
        "candidate_delta_vs_labelshuffle_control_auc": candidate_delta_vs_labelshuffle,
        "min_auc_delta": min_auc_delta,
        "min_control_delta": min_control_delta,
        "next_action": {
            "branch": (
                "document_residual_guided_pool3_fixed_fusion"
                if supported
                else "repair_residual_guided_pool3_before_scaleup"
            ),
            "should_launch_remote": False,
        },
        "claim_scope": (
            "application-level medium diagnostic evidence for residual-guided fixed-score "
            "expert combination only; not raw single-sample SOTA, not formal SPN/PRESENT "
            "evidence, and not a PRESENT r8 breakthrough claim"
        ),
    }


def _comparison(name: str, artifacts: list[EnsembleScoreArtifact]) -> dict[str, Any]:
    summary = evaluate_frozen_score_ensemble(artifacts)
    return {
        "name": name,
        "artifact_model_keys": [str(artifact.metadata.get("model_key", "")) for artifact in artifacts],
        "best_single_model_key": str(summary["best_single"]["model_key"]),
        "best_single_auc": float(summary["best_single"]["metrics"]["auc"]),
        "best_ensemble_mode": str(summary["best_ensemble"]["mode"]),
        "best_ensemble_auc": float(summary["best_ensemble"]["metrics"]["auc"]),
        "delta_best_ensemble_vs_single_auc": float(summary["delta_best_ensemble_vs_single_auc"]),
        "summary": summary,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _pending_or_hold_status(plan: dict[str, Any]) -> str:
    status = str(plan.get("status", "pending"))
    if status in {"hold", "fail"}:
        return "hold"
    return "pending"


def _next_action(plan: dict[str, Any], *, default: str) -> dict[str, Any]:
    next_action = plan.get("next_action")
    if isinstance(next_action, dict):
        return {
            "branch": str(next_action.get("branch", default)),
            "should_launch_remote": bool(next_action.get("should_launch_remote", False)),
        }
    return {"branch": default, "should_launch_remote": False}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_residual_guided_diverse_pool(
        pool_plan=args.pool_plan,
        trail_position_artifact=args.trail_position_artifact,
        raw117_artifact=args.raw117_artifact,
        residual_focus_artifact=args.residual_focus_artifact,
        uniform_control_artifact=args.uniform_control_artifact,
        labelshuffle_control_artifact=args.labelshuffle_control_artifact,
        min_auc_delta=args.min_auc_delta,
        min_control_delta=args.min_control_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
