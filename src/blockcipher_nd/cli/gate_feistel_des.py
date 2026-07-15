from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.feistel_des_gate import (
    gate_feistel_des_official_attribution,
    gate_feistel_des_official_calibration,
    gate_feistel_des_results,
)


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--official-calibration", action="store_true")
    mode.add_argument("--official-attribution", action="store_true")
    mode.add_argument("--official-raw-attribution", action="store_true")
    parser.add_argument("--expected-rounds", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.official_calibration:
        gate = gate_feistel_des_official_calibration
    elif args.official_attribution or args.official_raw_attribution:
        gate = gate_feistel_des_official_attribution
    else:
        gate = gate_feistel_des_results
    gate_kwargs = dict(
        plan_path=args.plan,
        results_path=args.results,
        expected_samples_per_class=args.samples_per_class,
        expected_seeds=tuple(args.seeds),
        expected_epochs=args.epochs,
        expected_final_repeats=args.final_repeats,
    )
    if args.official_attribution or args.official_raw_attribution:
        gate_kwargs["expected_rounds"] = args.expected_rounds
        gate_kwargs["raw_mapping_only"] = args.official_raw_attribution
    report = gate(**gate_kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
