from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.speck_hwang_position_labels import (
    SpeckPositionLabelConfig,
    run_position_label_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered E28 SPECK position-mask label-width audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    gate_path = (
        args.source_root / "gate.local.json"
        if (args.source_root / "gate.local.json").is_file()
        else args.source_root / "gate.json"
    )
    source_gate = _read_json(gate_path)
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_rows = _read_jsonl(args.source_root / "results.jsonl")
    config = SpeckPositionLabelConfig(
        run_id=args.run_id,
        ridge_alpha=args.ridge_alpha,
    )
    result = run_position_label_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_rows=source_rows,
    )
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_csv(args.output_root / "label_rows.csv", result["label_rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": config.run_id,
            "source_root": str(args.source_root),
            "rows": result["rows"],
            "label_rows": result["label_rows"],
            "gate": result["gate"],
            "metadata": result["metadata"],
        },
    )
    print(
        json.dumps(
            {"gate": result["gate"], "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if result["gate"]["status"] == "fail" else 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
