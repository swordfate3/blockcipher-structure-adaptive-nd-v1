from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_r6_pair_amplification_k1bv import (
    build_readiness,
    read_tasks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run zero-training K1-BV readiness.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_readiness(read_tasks(args.plan))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
