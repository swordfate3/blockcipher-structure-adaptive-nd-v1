from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.analyze_trail_position_scores import analyze_trail_position_scores
from blockcipher_nd.cli.verify_score_artifacts import verify_score_artifacts


REQUIRED_MODELS = [
    "present_pairset_global_stats:trail_position_global_control:near_neighbor_control",
    "present_trail_position_stats_pairset:trail_position:weak_positive",
]
REQUIRED_SCORE_FILES = (
    "models.json",
    "labels.npy",
    "probabilities.npy",
    "logits.npy",
    "sample_ids.npy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess PRESENT r8 trail-position score artifacts for retrieved runs."
    )
    parser.add_argument("--run-root", action="append", required=True, type=Path)
    parser.add_argument("--expected-score-rows", required=True, type=int)
    parser.add_argument("--expected-matrix-rows", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def postprocess_trail_position_results(
    run_roots: list[Path],
    *,
    expected_score_rows: int,
    expected_matrix_rows: int = 2,
) -> dict[str, Any]:
    runs = [
        _postprocess_single_run(
            run_root,
            expected_score_rows=expected_score_rows,
            expected_matrix_rows=expected_matrix_rows,
        )
        for run_root in run_roots
    ]
    failed = [run for run in runs if run["status"] == "fail"]
    pending = [run for run in runs if run["status"] == "pending"]
    ready = [run for run in runs if run["status"] == "pass"]

    if failed:
        status = "fail"
        decision = "hold_trail_position_postprocess_failed"
        action = "inspect_failed_verification_before_interpreting_262k_scores"
    elif pending:
        status = "pending"
        decision = "wait_for_trail_position_score_artifacts"
        action = "let_tmux_watchers_finish_retrieval_before_score_claims"
    else:
        status = "pass"
        decisions = {str(run.get("analysis", {}).get("decision", "")) for run in ready}
        if decisions == {"support_trail_position_score_residual"}:
            decision = "support_trail_position_score_residual_all_runs"
            action = "update_experiment_record_and_compare_against_medium_gate_scope"
        else:
            decision = "hold_trail_position_score_residual_mixed_runs"
            action = "inspect_per_seed_overlap_before_scale_or_ensemble_promotion"

    margins = [
        float(run["analysis"]["margins_vs_global_control"]["auc"])
        for run in ready
        if isinstance(run.get("analysis"), dict)
    ]
    return {
        "status": status,
        "decision": decision,
        "action": action,
        "expected_score_rows": int(expected_score_rows),
        "expected_matrix_rows": int(expected_matrix_rows),
        "ready_run_count": len(ready),
        "pending_run_count": len(pending),
        "failed_run_count": len(failed),
        "min_candidate_auc_margin_vs_global": min(margins) if margins else None,
        "runs": runs,
        "claim_scope": (
            "PRESENT r8 trail-position score postprocess only; 262144/class is medium diagnostic, "
            "not formal SPN/PRESENT evidence, not a breakthrough claim, and not a diverse ensemble claim"
        ),
    }


def _postprocess_single_run(
    run_root: Path,
    *,
    expected_score_rows: int,
    expected_matrix_rows: int,
) -> dict[str, Any]:
    score_root = run_root / "score_artifacts"
    global_artifact = score_root / "global_stats_control"
    candidate_artifact = score_root / "trail_position"
    train_matrix = run_root / "results" / "train_matrix.jsonl"
    train_rows = _jsonl_line_count(train_matrix)
    missing = _missing_score_files(global_artifact, candidate_artifact)
    base = {
        "run_id": run_root.name,
        "run_root": str(run_root),
        "train_matrix": str(train_matrix),
        "train_rows": train_rows,
        "missing_score_files": missing,
        "score_artifacts_root": str(score_root),
    }
    if train_rows < expected_matrix_rows or missing:
        return {
            **base,
            "status": "pending",
            "reason": "train_matrix_or_score_artifacts_not_ready",
        }

    verification = verify_score_artifacts(
        [global_artifact, candidate_artifact],
        expected_rows=expected_score_rows,
        required_models=REQUIRED_MODELS,
    )
    verification_path = score_root / "verification_summary_local.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if verification["status"] != "pass":
        return {
            **base,
            "status": "fail",
            "verification": verification,
            "verification_summary": str(verification_path),
            "reason": "score_artifact_verification_failed",
        }

    analysis = analyze_trail_position_scores(
        global_artifact_dir=global_artifact,
        candidate_artifact_dir=candidate_artifact,
    )
    analysis_path = score_root / "trail_position_score_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **base,
        "status": "pass",
        "verification": verification,
        "analysis": analysis,
        "verification_summary": str(verification_path),
        "score_analysis": str(analysis_path),
    }


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _missing_score_files(global_artifact: Path, candidate_artifact: Path) -> list[str]:
    missing: list[str] = []
    for artifact_dir in (global_artifact, candidate_artifact):
        for filename in REQUIRED_SCORE_FILES:
            path = artifact_dir / filename
            if not path.exists():
                missing.append(str(path))
    return missing


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_trail_position_results(
        args.run_root,
        expected_score_rows=args.expected_score_rows,
        expected_matrix_rows=args.expected_matrix_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
