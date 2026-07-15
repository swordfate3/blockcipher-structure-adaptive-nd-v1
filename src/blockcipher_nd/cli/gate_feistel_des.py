from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.feistel_des_gate import gate_feistel_des_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and gate the DES Feistel Innovation 1 matrix."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--samples-per-class", required=True, type=int)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--final-repeats", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_feistel_des_results(
        plan_path=args.plan,
        results_path=args.results,
        expected_samples_per_class=args.samples_per_class,
        expected_seeds=tuple(args.seeds),
        expected_epochs=args.epochs,
        expected_final_repeats=args.final_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
