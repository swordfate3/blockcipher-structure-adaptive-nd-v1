from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.evaluate_innovation2_integral_ranking import (
    render_joint_ranking_svg,
)
from blockcipher_nd.tasks.innovation2.integral_property_ranking import (
    adjudicate_joint_integral_ranking,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly adjudicate the frozen seed0 and seed1 Innovation 2 E2 "
            "ranking gates without training."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-gates", nargs=2, required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_gates = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.source_gates
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_gates": [str(path) for path in args.source_gates],
            "training_performed": False,
        },
        mode="w",
    )
    result = adjudicate_joint_integral_ranking(
        run_id=args.run_id,
        source_gates=source_gates,
    )
    results_path = args.output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    metrics_path = args.output_root / "seed_metrics.csv"
    _write_csv_rows(metrics_path, result["rows"])
    gate_path = args.output_root / "gate.json"
    gate_path.write_text(
        json.dumps(result["gate"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    curves_path = args.output_root / "curves.svg"
    render_joint_ranking_svg(result["rows"], result["gate"], curves_path)
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
        mode="a",
    )
    report = {
        "status": result["gate"]["status"],
        "decision": result["gate"]["decision"],
        "run_id": args.run_id,
        "output_root": str(args.output_root),
        "results": str(results_path),
        "metrics": str(metrics_path),
        "gate": str(gate_path),
        "curves": str(curves_path),
        "next_action": result["gate"]["next_action"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if result["gate"]["status"] != "fail" else 1


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str,
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
