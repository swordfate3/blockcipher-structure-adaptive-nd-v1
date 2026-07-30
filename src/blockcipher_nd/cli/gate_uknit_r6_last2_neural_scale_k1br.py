from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_r6_last2_neural_scale_k1br import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate_k1br,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate uKNIT r6 K1-BR scale diagnostic."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_commit = _read_sha(args.source_commit_file)
    expected_commit = _read_sha(args.expected_source_commit_file)
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    rows = _read_jsonl(args.results)
    gate = adjudicate_k1br(
        tasks=tasks,
        result_rows=rows,
        progress_events=_read_jsonl(args.progress),
        source_checks={
            "source_revision_matches_launch_pin": bool(source_commit)
            and source_commit == expected_commit
        },
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
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
            key: gate[key]
            for key in (
                "run_id",
                "status",
                "decision",
                "tier",
                "aucs",
                "best_candidate_auc",
                "attribution_margin",
                "next_action",
                "claim_scope",
            )
        },
    )
    _write_history(rows, args.output_root / "history.csv")
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _read_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return (
        value
        if len(value) == 40 and all(c in "0123456789abcdef" for c in value)
        else ""
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
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


def _write_history(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "run_index",
        "model",
        "epoch",
        "train_loss",
        "train_auc",
        "val_loss",
        "val_auc",
        "learning_rate",
    ]
    records = []
    for index, row in enumerate(rows, start=1):
        for item in row.get("history", []):
            if isinstance(item, dict):
                records.append(
                    {"run_index": index, "model": row.get("model", ""), **item}
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())
