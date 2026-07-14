from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_e5_source_objective_gate import (
    gate_cross_spn_e5_source_objective,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate one E5-R0 topology-counterfactual target seed."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--scratch-scores", required=True, type=Path)
    parser.add_argument("--off-transfer-scores", required=True, type=Path)
    parser.add_argument("--candidate-transfer-scores", required=True, type=Path)
    parser.add_argument("--placebo-transfer-scores", required=True, type=Path)
    parser.add_argument("--expected-target-seed", required=True, choices=(2, 3), type=int)
    parser.add_argument("--expected-source-seed", default=0, type=int)
    parser.add_argument("--bootstrap-replicates", default=10000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260715, type=int)
    parser.add_argument("--paired-scores-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_e5_source_objective(
        plan_path=args.plan,
        results_path=args.results,
        score_artifact_paths={
            "scratch": args.scratch_scores,
            "off_transfer": args.off_transfer_scores,
            "candidate_transfer": args.candidate_transfer_scores,
            "placebo_transfer": args.placebo_transfer_scores,
        },
        expected_target_seed=args.expected_target_seed,
        expected_source_seed=args.expected_source_seed,
        paired_scores_output=args.paired_scores_output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
