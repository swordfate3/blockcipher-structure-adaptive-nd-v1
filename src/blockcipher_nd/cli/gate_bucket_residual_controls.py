from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_bucket_residual_controls_gate.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate the local PRESENT r8 V16 bucket-conditioned residual controls "
            "before migrating the route to the 262144/class trail-position artifacts."
        )
    )
    parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    parser.add_argument("--two-score-ensemble", type=Path, action="append", required=True)
    parser.add_argument("--three-score-ensemble", type=Path, action="append", required=True)
    parser.add_argument("--shuffle-label-report", type=Path, action="append", required=True)
    parser.add_argument("--train-bucket-shuffle-report", type=Path, action="append", required=True)
    parser.add_argument("--train-bucket-shuffle-ensemble", type=Path, action="append", required=True)
    parser.add_argument("--validation-bucket-shuffle-report", type=Path, action="append", required=True)
    parser.add_argument("--validation-bucket-shuffle-ensemble", type=Path, action="append", required=True)
    parser.add_argument("--no-bucket-report", type=Path, action="append", required=True)
    parser.add_argument("--min-candidate-vs-nobucket-delta", type=float, default=0.0)
    parser.add_argument("--min-three-vs-two-delta", type=float, default=0.0)
    parser.add_argument("--max-shuffle-label-auc", type=float, default=0.60)
    parser.add_argument("--max-shuffle-three-vs-two-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def gate_bucket_residual_controls(
    *,
    candidate_reports: list[Path],
    two_score_ensembles: list[Path],
    three_score_ensembles: list[Path],
    shuffle_label_reports: list[Path],
    train_bucket_shuffle_reports: list[Path],
    train_bucket_shuffle_ensembles: list[Path],
    validation_bucket_shuffle_reports: list[Path],
    validation_bucket_shuffle_ensembles: list[Path],
    no_bucket_reports: list[Path],
    min_candidate_vs_nobucket_delta: float = 0.0,
    min_three_vs_two_delta: float = 0.0,
    max_shuffle_label_auc: float = 0.60,
    max_shuffle_three_vs_two_delta: float = 0.0,
) -> dict[str, Any]:
    path_groups = {
        "candidate_reports": candidate_reports,
        "two_score_ensembles": two_score_ensembles,
        "three_score_ensembles": three_score_ensembles,
        "shuffle_label_reports": shuffle_label_reports,
        "train_bucket_shuffle_reports": train_bucket_shuffle_reports,
        "train_bucket_shuffle_ensembles": train_bucket_shuffle_ensembles,
        "validation_bucket_shuffle_reports": validation_bucket_shuffle_reports,
        "validation_bucket_shuffle_ensembles": validation_bucket_shuffle_ensembles,
        "no_bucket_reports": no_bucket_reports,
    }
    lengths = {name: len(paths) for name, paths in path_groups.items()}
    seed_count = lengths["candidate_reports"]
    errors: list[str] = []
    for name, count in lengths.items():
        if count != seed_count:
            errors.append(f"{name}: expected {seed_count} paths, got {count}")
    if errors:
        return _report("fail", errors=errors, runs=[], seed_count=seed_count)

    runs: list[dict[str, Any]] = []
    for index in range(seed_count):
        run_errors: list[str] = []
        candidate = _load_json(candidate_reports[index])
        no_bucket = _load_json(no_bucket_reports[index])
        shuffle_label = _load_json(shuffle_label_reports[index])
        train_bucket_shuffle = _load_json(train_bucket_shuffle_reports[index])
        validation_bucket_shuffle = _load_json(validation_bucket_shuffle_reports[index])
        two_score = _load_json(two_score_ensembles[index])
        three_score = _load_json(three_score_ensembles[index])
        train_bucket_shuffle_ensemble = _load_json(train_bucket_shuffle_ensembles[index])
        validation_bucket_shuffle_ensemble = _load_json(validation_bucket_shuffle_ensembles[index])

        candidate_auc = _validation_auc(candidate)
        no_bucket_auc = _validation_auc(no_bucket)
        shuffle_label_auc = _validation_auc(shuffle_label)
        train_bucket_shuffle_auc = _validation_auc(train_bucket_shuffle)
        validation_bucket_shuffle_auc = _validation_auc(validation_bucket_shuffle)
        two_score_auc = _best_ensemble_auc(two_score)
        three_score_auc = _best_ensemble_auc(three_score)
        train_bucket_shuffle_three_score_auc = _best_ensemble_auc(train_bucket_shuffle_ensemble)
        validation_bucket_shuffle_three_score_auc = _best_ensemble_auc(validation_bucket_shuffle_ensemble)

        bucket_vs_nobucket_delta = candidate_auc - no_bucket_auc
        three_vs_two_delta = three_score_auc - two_score_auc
        trainbucket_shuffle_three_vs_two_delta = train_bucket_shuffle_three_score_auc - two_score_auc
        valbucket_shuffle_three_vs_two_delta = validation_bucket_shuffle_three_score_auc - two_score_auc

        seed_name = f"seed{index}"
        if bucket_vs_nobucket_delta <= min_candidate_vs_nobucket_delta:
            run_errors.append(f"{seed_name}: candidate_not_above_no_bucket")
        if three_vs_two_delta <= min_three_vs_two_delta:
            run_errors.append(f"{seed_name}: three_score_not_above_two_score")
        if shuffle_label_auc > max_shuffle_label_auc:
            run_errors.append(f"{seed_name}: shuffle_label_auc_too_high")
        if trainbucket_shuffle_three_vs_two_delta > max_shuffle_three_vs_two_delta:
            run_errors.append(f"{seed_name}: train_bucket_shuffle_three_score_not_below_two_score")
        if valbucket_shuffle_three_vs_two_delta > max_shuffle_three_vs_two_delta:
            run_errors.append(f"{seed_name}: validation_bucket_shuffle_three_score_not_below_two_score")

        runs.append(
            {
                "seed": index,
                "status": "fail" if run_errors else "pass",
                "errors": run_errors,
                "candidate_auc": candidate_auc,
                "no_bucket_auc": no_bucket_auc,
                "bucket_vs_nobucket_auc_delta": bucket_vs_nobucket_delta,
                "two_score_auc": two_score_auc,
                "three_score_auc": three_score_auc,
                "three_vs_two_auc_delta": three_vs_two_delta,
                "shuffle_label_validation_auc": shuffle_label_auc,
                "train_bucket_shuffle_validation_auc": train_bucket_shuffle_auc,
                "validation_bucket_shuffle_validation_auc": validation_bucket_shuffle_auc,
                "trainbucket_shuffle_three_score_auc": train_bucket_shuffle_three_score_auc,
                "trainbucket_shuffle_three_vs_two_delta": trainbucket_shuffle_three_vs_two_delta,
                "valbucket_shuffle_three_score_auc": validation_bucket_shuffle_three_score_auc,
                "valbucket_shuffle_three_vs_two_delta": valbucket_shuffle_three_vs_two_delta,
            }
        )
        errors.extend(run_errors)

    return _report("fail" if errors else "pass", errors=errors, runs=runs, seed_count=seed_count)


def _report(status: str, *, errors: list[str], runs: list[dict[str, Any]], seed_count: int) -> dict[str, Any]:
    pass_report = status == "pass"
    return {
        "status": status,
        "decision": (
            "bucket_conditioned_residual_controls_pass_local_diagnostic"
            if pass_report
            else "hold_bucket_conditioned_residual_controls_failed"
        ),
        "action": (
            "keep_as_262k_migration_candidate_wait_for_trail_position_artifacts"
            if pass_report
            else "do_not_migrate_bucket_residual_until_controls_are_repaired"
        ),
        "seed_count": seed_count,
        "errors": errors,
        "min_bucket_vs_nobucket_auc_delta": _min_run_value(runs, "bucket_vs_nobucket_auc_delta"),
        "min_three_vs_two_auc_delta": _min_run_value(runs, "three_vs_two_auc_delta"),
        "max_shuffle_label_validation_auc": _max_run_value(runs, "shuffle_label_validation_auc"),
        "max_trainbucket_shuffle_three_vs_two_delta": _max_run_value(
            runs, "trainbucket_shuffle_three_vs_two_delta"
        ),
        "max_valbucket_shuffle_three_vs_two_delta": _max_run_value(runs, "valbucket_shuffle_three_vs_two_delta"),
        "runs": runs,
        "next_action": {
            "branch": "wait_for_262k_trail_position_artifacts_then_run_v16_planner",
            "should_launch_remote": False,
            "requires_implementation": False,
        },
        "claim_scope": (
            "local 2048/class frozen-score control gate only; not remote evidence, not formal SPN/PRESENT "
            "evidence, not a breakthrough claim, and not a raw single-sample SOTA claim"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _validation_auc(report: dict[str, Any]) -> float:
    return float(report["validation_metrics"]["auc"])


def _best_ensemble_auc(report: dict[str, Any]) -> float:
    return float(report["best_ensemble"]["metrics"]["auc"])


def _min_run_value(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if key in run]
    return min(values) if values else None


def _max_run_value(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if key in run]
    return max(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_bucket_residual_controls(
        candidate_reports=args.candidate_report,
        two_score_ensembles=args.two_score_ensemble,
        three_score_ensembles=args.three_score_ensemble,
        shuffle_label_reports=args.shuffle_label_report,
        train_bucket_shuffle_reports=args.train_bucket_shuffle_report,
        train_bucket_shuffle_ensembles=args.train_bucket_shuffle_ensemble,
        validation_bucket_shuffle_reports=args.validation_bucket_shuffle_report,
        validation_bucket_shuffle_ensembles=args.validation_bucket_shuffle_ensemble,
        no_bucket_reports=args.no_bucket_report,
        min_candidate_vs_nobucket_delta=args.min_candidate_vs_nobucket_delta,
        min_three_vs_two_delta=args.min_three_vs_two_delta,
        max_shuffle_label_auc=args.max_shuffle_label_auc,
        max_shuffle_three_vs_two_delta=args.max_shuffle_three_vs_two_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
