from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_target_adaptation_gate import (
    gate_cross_spn_target_adaptation_joint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the two-seed E4-R4 target-adaptation confirmation."
    )
    parser.add_argument("--seed2-gate", required=True, type=Path)
    parser.add_argument("--seed3-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [
        json.loads(args.seed2_gate.read_text(encoding="utf-8")),
        json.loads(args.seed3_gate.read_text(encoding="utf-8")),
    ]
    report = gate_cross_spn_target_adaptation_joint(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
