from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_typed_transfer_gate import (
    gate_cross_spn_typed_transfer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate a frozen PRESENT-to-GIFT typed checkpoint transfer stage."
    )
    parser.add_argument("--results", action="append", required=True, type=Path)
    parser.add_argument("--progress", action="append", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--expected-seed", choices=(0, 1), type=int, default=0)
    parser.add_argument(
        "--experiment-stage",
        choices=("e4_r2", "e4_r3"),
        default="e4_r2",
    )
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_typed_transfer(
        args.results,
        progress_paths=args.progress,
        expected_seeds=(args.expected_seed,),
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        readiness_only=args.readiness_only,
        experiment_stage=args.experiment_stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
