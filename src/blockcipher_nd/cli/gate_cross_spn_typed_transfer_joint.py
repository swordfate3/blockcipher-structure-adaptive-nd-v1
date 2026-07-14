from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_typed_transfer_gate import (
    gate_cross_spn_typed_transfer_joint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the frozen two-seed E4-R2 PRESENT-to-GIFT transfer."
    )
    parser.add_argument("--seed0-results", required=True, type=Path)
    parser.add_argument("--seed0-progress", required=True, type=Path)
    parser.add_argument("--seed1-results", required=True, type=Path)
    parser.add_argument("--seed1-progress", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_cross_spn_typed_transfer_joint(
        [args.seed0_results, args.seed1_results],
        progress_paths=[args.seed0_progress, args.seed1_progress],
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
