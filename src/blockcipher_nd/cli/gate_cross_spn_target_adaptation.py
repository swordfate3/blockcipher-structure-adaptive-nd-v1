from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_target_adaptation_gate import (
    gate_cross_spn_target_adaptation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate one E4-R4 or E4-R5 cross-SPN target-adaptation seed."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--typed-scratch-scores", required=True, type=Path)
    parser.add_argument("--true-to-true-scores", required=True, type=Path)
    parser.add_argument("--shuffled-to-true-scores", required=True, type=Path)
    parser.add_argument("--true-to-shuffled-scores", required=True, type=Path)
    parser.add_argument("--expected-seed", required=True, choices=(2, 3, 4, 5), type=int)
    parser.add_argument(
        "--experiment-stage", choices=("e4_r4", "e4_r5"), default="e4_r4"
    )
    parser.add_argument("--samples-per-class", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--paired-scores-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_target_adaptation(
        plan_path=args.plan,
        results_path=args.results,
        progress_path=args.progress,
        score_artifact_paths={
            "typed_scratch": args.typed_scratch_scores,
            "true_to_true": args.true_to_true_scores,
            "shuffled_to_true": args.shuffled_to_true_scores,
            "true_to_shuffled": args.true_to_shuffled_scores,
        },
        expected_seed=args.expected_seed,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        experiment_stage=args.experiment_stage,
        readiness_only=args.readiness_only,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        paired_scores_output=args.paired_scores_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
