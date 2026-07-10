from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.autond_typed_invp_gate import gate_autond_typed_invp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the local AutoND public-code typed InvP attribution matrix."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=4)
    parser.add_argument("--required-margin", type=float, default=0.01)
    parser.add_argument("--train-rows", type=int, default=16_384)
    parser.add_argument("--validation-rows", type=int, default=4_096)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--final-rows", type=int, default=4_096)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_autond_typed_invp(
        args.results,
        expected_rows=args.expected_rows,
        required_margin=args.required_margin,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        final_repeats=args.final_repeats,
        final_rows=args.final_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]

