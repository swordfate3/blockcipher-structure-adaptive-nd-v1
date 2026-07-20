from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import (
    execute_phase,
    git_head,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an E104 PRESENT r9 ATM split (3,3,3) phase."
    )
    parser.add_argument("--mode", choices=("readiness", "probe", "search"), required=True)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--e103-anchor", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor = json.loads(args.e103_anchor.read_text(encoding="utf-8"))
    result = execute_phase(
        args.output_root,
        atm_root=args.atm_root,
        e103_gate=anchor,
        e103_gate_sha256=str(anchor.get("artifact_sha256", "")),
        actual_atm_commit=git_head(args.atm_root),
        mode=args.mode,
    )
    gate = result["gate"]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if gate["status"] == "pass":
        return 0
    return 1 if gate["status"] == "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
