from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.evaluation.result_index import (
    DEFAULT_RESULT_ROOTS,
    write_result_index,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a recency-sorted index for local and retrieved experiment results."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_RESULT_ROOTS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = write_result_index(
        args.outputs_root,
        markdown_output=args.markdown_output,
        json_output=args.json_output,
        roots=tuple(args.roots),
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0
