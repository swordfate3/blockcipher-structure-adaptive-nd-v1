from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.planning.innovation2_paper_reference_precondition import (
    paper_reference_seed1_precondition_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether verified Innovation 2 paper-reference seed0 evidence "
            "allows generation of the frozen seed1 remote package."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = paper_reference_seed1_precondition_report(
        plan_path=args.plan,
        source_root=args.source_artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
