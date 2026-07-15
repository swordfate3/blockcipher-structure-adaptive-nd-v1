from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.feistel_balanced_gate import (
    gate_feistel_balanced_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and gate the balanced-Feistel round-relation matrix."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--samples-per-class", required=True, type=int)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--final-repeats", required=True, type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--calibration", action="store_true")
    mode.add_argument("--layout-repair", action="store_true")
    mode.add_argument("--scale-probe", action="store_true")
    mode.add_argument("--scale-confirmation", action="store_true")
    mode.add_argument("--target-round-probe", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_feistel_balanced_results(
        plan_path=args.plan,
        results_path=args.results,
        expected_samples_per_class=args.samples_per_class,
        expected_seeds=tuple(args.seeds),
        expected_epochs=args.epochs,
        expected_final_repeats=args.final_repeats,
        readiness=args.readiness,
        calibration=args.calibration,
        layout_repair=args.layout_repair,
        scale_probe=args.scale_probe,
        scale_confirmation=args.scale_confirmation,
        target_round_probe=args.target_round_probe,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
