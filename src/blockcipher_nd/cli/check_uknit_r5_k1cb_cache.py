from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_r5_published_comparison_k1cb import (
    RUN_ID,
    audit_source_caches,
    read_tasks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that K1-CB can reuse all four K1-CA caches without generating data."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-CB run_id must remain frozen as {RUN_ID}")
    report = audit_source_caches(read_tasks(args.plan), args.source_cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
