from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_typed_cell_gate import (
    gate_cross_spn_typed_cell,
)


def expected_seeds_arg(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(seed) for seed in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected seeds must be comma-separated integers"
        ) from exc
    if not seeds or any(not part for part in value.split(",")):
        raise argparse.ArgumentTypeError(
            "expected seeds must be comma-separated integers"
        )
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the E4 cross-SPN typed-cell source/scratch experiment."
    )
    parser.add_argument(
        "--present-results", action="append", required=True, type=Path
    )
    parser.add_argument(
        "--present-progress", action="append", required=True, type=Path
    )
    parser.add_argument("--gift-results", action="append", required=True, type=Path)
    parser.add_argument("--gift-progress", action="append", required=True, type=Path)
    parser.add_argument("--expected-seeds", type=expected_seeds_arg, default="0")
    parser.add_argument("--samples-per-class", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_typed_cell(
        args.present_results,
        present_progress_paths=args.present_progress,
        gift_results_paths=args.gift_results,
        gift_progress_paths=args.gift_progress,
        expected_seeds=args.expected_seeds,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        readiness_only=args.readiness_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["expected_seeds_arg", "main", "parse_args"]
