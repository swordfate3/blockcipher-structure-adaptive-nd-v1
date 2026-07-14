from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a relocatable progress JSONL copy for local result re-adjudication."
    )
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def localize_progress_output(
    progress_path: Path,
    results_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_done = [row for row in rows if row.get("event") == "run_done"]
    if len(run_done) != 1:
        raise ValueError(f"progress must contain exactly one run_done event: {run_done!r}")
    original_output = run_done[0].get("output")
    if not isinstance(original_output, str) or not original_output:
        raise ValueError("progress run_done output must be a non-empty string")
    run_done[0].setdefault("remote_output", original_output)
    run_done[0]["output"] = str(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "progress": str(progress_path),
        "results": str(results_path),
        "output": str(output_path),
        "rows": len(rows),
        "remote_output": run_done[0]["remote_output"],
        "localized_output": run_done[0]["output"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = localize_progress_output(args.progress, args.results, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


__all__ = ["localize_progress_output", "main", "parse_args"]
