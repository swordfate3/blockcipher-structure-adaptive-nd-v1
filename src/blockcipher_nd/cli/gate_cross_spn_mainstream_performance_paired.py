from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_mainstream_performance_gate import (
    paired_mainstream_performance_interval_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed6-gate", required=True, type=Path)
    parser.add_argument("--seed6-scores", required=True, type=Path)
    parser.add_argument("--seed7-gate", required=True, type=Path)
    parser.add_argument("--seed7-scores", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = paired_mainstream_performance_interval_gate(
        seed_gate_paths={6: args.seed6_gate, 7: args.seed7_gate},
        primary_score_paths={6: args.seed6_scores, 7: args.seed7_scores},
        replicates=args.replicates,
        chunk_size=args.chunk_size,
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
