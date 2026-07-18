from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.neural_architecture_boundary_synthesis import (
    NeuralArchitectureBoundaryConfig,
    adjudicate_boundary_synthesis,
    architecture_rows,
    load_gate_sources,
    serializable_config,
    source_hashes,
    validate_gate_sources,
)


SOURCE_ARGS = {
    "formal_method": "formal_root",
    "skinny_residual": "skinny_root",
    "shared_operator": "shared_root",
    "rectangle_labels": "rectangle_labels_root",
    "rectangle_untyped": "rectangle_untyped_root",
    "rectangle_row_mechanism": "rectangle_row_mechanism_root",
    "rectangle_row_operator": "rectangle_row_operator_root",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize E93 Innovation 2 architecture evidence boundaries."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--skinny-root", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--rectangle-labels-root", required=True, type=Path)
    parser.add_argument("--rectangle-untyped-root", required=True, type=Path)
    parser.add_argument("--rectangle-row-mechanism-root", required=True, type=Path)
    parser.add_argument("--rectangle-row-operator-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = NeuralArchitectureBoundaryConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False})
    roots = {role: getattr(args, name) for role, name in SOURCE_ARGS.items()}
    sources = load_gate_sources(roots)
    checks = validate_gate_sources(sources)
    rows = architecture_rows(sources)
    gate = adjudicate_boundary_synthesis(config, checks, rows)
    _write_progress(progress, "sources_replayed", checks)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_neural_architecture_boundary_synthesis",
        "config": serializable_config(config),
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "gate": gate,
        "source_hashes": source_hashes(sources),
        "architecture_rows": rows,
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_neural_architecture_boundary_synthesis",
            "decision": gate["decision"],
            "training_performed": False,
            **row,
        }
        for row in rows
    ]
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_csv(args.output_root / "architecture_ranking.csv", rows)
    _write_json(args.output_root / "source_hashes.json", source_hashes(sources))
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)}, sort_keys=True
        )
    )
    return 1 if gate["status"] == "fail" else 0


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
