from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.cross_spn_e6_readiness import (
    gate_e6_source_readiness,
    gate_e6_source_diagnostic,
    gate_e6_target_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate E6 source or target readiness.")
    parser.add_argument(
        "--stage",
        choices=("source", "source-diagnostic", "target"),
        required=True,
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--anchor-results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "source":
        report, manifest = gate_e6_source_readiness(args.plan, args.results)
        if report["status"] == "pass":
            if args.manifest_output is None:
                raise ValueError("source readiness requires --manifest-output")
            args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report["manifest_output"] = str(args.manifest_output)
    elif args.stage == "source-diagnostic":
        if args.anchor_results is None:
            raise ValueError("source diagnostic requires --anchor-results")
        report, manifest = gate_e6_source_diagnostic(
            args.plan,
            args.results,
            args.anchor_results,
        )
        if report["status"] == "pass":
            if args.manifest_output is None:
                raise ValueError("source diagnostic requires --manifest-output")
            args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report["manifest_output"] = str(args.manifest_output)
    else:
        if args.progress is None:
            raise ValueError("target readiness requires --progress")
        report = gate_e6_target_readiness(args.plan, args.results, args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


__all__ = ["main", "parse_args"]
