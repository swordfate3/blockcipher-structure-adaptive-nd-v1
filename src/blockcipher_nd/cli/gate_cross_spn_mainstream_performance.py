from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_mainstream_performance_gate import (
    gate_cross_spn_mainstream_performance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--typed-scratch-scores", required=True, type=Path)
    parser.add_argument("--typed-source0-scores", required=True, type=Path)
    parser.add_argument("--typed-source1-scores", required=True, type=Path)
    parser.add_argument("--lstm-scores", required=True, type=Path)
    parser.add_argument("--resnet-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_mainstream_performance(
        plan_path=args.plan,
        results_path=args.results,
        score_artifact_paths={
            "typed_scratch": args.typed_scratch_scores,
            "typed_source0": args.typed_source0_scores,
            "typed_source1": args.typed_source1_scores,
            "lstm": args.lstm_scores,
            "resnet": args.resnet_scores,
        },
        expected_seed=args.expected_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
