from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.dialga_r4_dfc1 import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate,
    read_tasks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate Dialga DFC1 scale results.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"DFC1 run_id must remain frozen as {RUN_ID}")
    source_commit = _read_sha(args.source_commit_file)
    expected_commit = _read_sha(args.expected_source_commit_file)
    rows = _read_jsonl(args.results)
    checkpoints = sorted(args.checkpoint_root.glob("*.pt"))
    gate = adjudicate(
        tasks=read_tasks(args.plan),
        result_rows=rows,
        progress_events=_read_jsonl(args.progress),
        source_checks={
            "source_revision_matches_launch_pin": bool(source_commit)
            and source_commit == expected_commit,
            "six_nonempty_checkpoints_present": len(checkpoints)
            == EXPECTED_RESULT_ROWS
            and all(path.stat().st_size > 0 for path in checkpoints),
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
        "expected_source_commit": expected_commit,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(
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
    _write_history_csv(rows, args.output_root / "history.csv")
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _read_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_index", "run_label", "cipher", "model", "rounds", "seed",
        "samples_per_class", "pairs_per_sample", "epoch", "train_loss",
        "train_eval_loss", "train_accuracy", "train_auc", "val_loss",
        "val_accuracy", "val_auc", "learning_rate",
    ]
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        label = (
            f"run{index}: {row.get('cipher', '')} r{row.get('rounds', '')} "
            f"{row.get('model', '')} seed{row.get('seed', '')}"
        )
        for item in row.get("history", []):
            if isinstance(item, dict):
                records.append(
                    {
                        "run_index": index,
                        "run_label": label,
                        "cipher": row.get("cipher", ""),
                        "model": row.get("model", ""),
                        "rounds": row.get("rounds", ""),
                        "seed": row.get("seed", ""),
                        "samples_per_class": row.get("samples_per_class", ""),
                        "pairs_per_sample": row.get("pairs_per_sample", ""),
                        **item,
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())
