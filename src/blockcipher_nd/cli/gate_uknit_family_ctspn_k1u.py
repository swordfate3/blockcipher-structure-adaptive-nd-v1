from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    read_jsonl,
    write_json,
)
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import read_tasks
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate_k1u,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the remote uKNIT K1-U 65536/class medium diagnostic."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-U run_id must remain frozen as {RUN_ID}")
    source_commit = _read_sha(args.source_commit_file)
    expected_source_commit = _read_sha(args.expected_source_commit_file)
    tasks = read_tasks(args.plan)
    rows = read_jsonl(args.results)
    progress = read_jsonl(args.progress)
    gate = adjudicate_k1u(
        tasks=tasks,
        result_rows=rows,
        progress_events=progress,
        source_checks={
            "source_revision_matches_launch_pin": (
                bool(source_commit) and source_commit == expected_source_commit
            )
        },
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "source_commit": source_commit,
        "expected_source_commit": expected_source_commit,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_history_csv(args.results, args.output_root / "history.csv")
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _read_sha(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        return ""
    return value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
