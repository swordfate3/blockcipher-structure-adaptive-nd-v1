from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.invp_state_matrix_conv2d_gate import (
    gate_invp_state_matrix_conv2d,
)


def expected_seeds_arg(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected seeds must be comma-separated integers"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the strict PRESENT r7 InvP state-matrix Conv2D experiment."
    )
    parser.add_argument("--results", action="append", required=True, type=Path)
    parser.add_argument("--progress", action="append", required=True, type=Path)
    parser.add_argument("--expected-seeds", type=expected_seeds_arg, default="0")
    parser.add_argument("--samples-per-class", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_invp_state_matrix_conv2d(
        args.results,
        progress_paths=args.progress,
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
