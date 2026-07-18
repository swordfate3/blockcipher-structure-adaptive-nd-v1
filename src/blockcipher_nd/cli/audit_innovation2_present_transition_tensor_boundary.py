from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_transition_tensor_boundary_audit import (
    TensorBoundaryAuditConfig,
    evaluate_tensor_boundary_audit,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E54 retained-boundary feasibility before tensor contraction."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exact-summary", required=True, type=Path)
    parser.add_argument("--exact-fixtures", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TensorBoundaryAuditConfig(run_id=args.run_id, mode=args.mode)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {"run_id": config.run_id, "mode": config.mode, "training": False},
    )
    exact_summary = json.loads(args.exact_summary.read_text(encoding="utf-8"))
    fixtures = [
        json.loads(line)
        for line in args.exact_fixtures.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_progress(
        progress,
        "source_ready",
        {
            "exact_run_id": exact_summary["run_id"],
            "fixtures": len(fixtures),
        },
    )
    evaluation = evaluate_tensor_boundary_audit(
        config, exact_summary=exact_summary, fixtures=fixtures
    )
    gate = evaluation["gate"]
    _write_progress(
        progress,
        "boundary_complete",
        {
            "retained_variables": gate["metrics"]["required_retained_variables"],
            "dense_entries": gate["metrics"]["required_dense_entries"],
            "internal_factor_graph_constructed": False,
        },
    )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_transition_tensor_boundary_audit",
        "config": serializable_config(config),
        "source_summary": str(args.exact_summary),
        "source_fixtures": str(args.exact_fixtures),
        "cipher": "PRESENT-80",
        "target_rounds": 5,
        "training_performed": False,
        "remote_scale": False,
        "five_round_subset_executed": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "boundary_manifest": evaluation["boundary_manifest"],
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_json(args.output_root / "factor_manifest.json", evaluation["boundary_manifest"])
    _write_jsonl(
        args.output_root / "elimination_orders.jsonl", evaluation["elimination_rows"]
    )
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
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


if __name__ == "__main__":
    raise SystemExit(main())
