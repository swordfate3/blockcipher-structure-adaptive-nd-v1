from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    evaluate_matched_contrast,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E32b train-only matched-contrast readjudication."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": args.run_id, "source_root": str(args.source_root)},
    )
    labels = np.load(args.source_root / "labels.npy")
    source_metadata = json.loads(
        (args.source_root / "metadata.json").read_text(encoding="utf-8")
    )
    source_gate = json.loads(
        (args.source_root / "gate.json").read_text(encoding="utf-8")
    )
    result = evaluate_matched_contrast(
        run_id=args.run_id,
        labels=labels,
        source_metadata=source_metadata,
        source_gate=source_gate,
    )
    np.save(args.output_root / "selected_mask.npy", result["selected_mask"])
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "summary.json", result["summary"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_selected(args.output_root / "selected_cells.csv", result["selected_rows"])
    _write_progress(
        progress_path,
        "run_done",
        {
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(json.dumps({"gate": result["gate"], "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if result["gate"]["status"] == "fail" else 0


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_selected(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
