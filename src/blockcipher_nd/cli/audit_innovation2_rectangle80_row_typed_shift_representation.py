from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    load_rectangle_profile_sources,
    to_rectangle_model_order,
    validate_rectangle_model_order,
    validate_rectangle_profile_sources,
)
from blockcipher_nd.tasks.innovation2.rectangle80_row_typed_shift_representation_audit import (
    Rectangle80RowTypedAuditConfig,
    adjudicate_row_typed_audit,
    evaluate_row_typed_ridges,
    load_e90_source,
    result_rows,
    serializable_config,
    validate_e90_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E91 RECTANGLE row-typed ShiftRow representation."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--e90-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Rectangle80RowTypedAuditConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False})
    physical_sources = load_rectangle_profile_sources(args.profile_root)
    profile_checks = validate_rectangle_profile_sources(physical_sources)
    sources = to_rectangle_model_order(physical_sources)
    model_order_checks = validate_rectangle_model_order(
        physical_sources, sources
    )
    e90_source = load_e90_source(args.e90_root)
    e90_checks = validate_e90_source(e90_source)
    if (
        not all(profile_checks.values())
        or not all(model_order_checks.values())
        or not all(e90_checks.values())
    ):
        raise ValueError("E88/E90 source or cell-major replay validation failed")
    _write_progress(
        progress,
        "sources_validated",
        {
            "profile": profile_checks,
            "model_order": model_order_checks,
            "e90": e90_checks,
        },
    )
    reports = evaluate_row_typed_ridges(config, sources)
    gate = adjudicate_row_typed_audit(
        config,
        profile_checks,
        model_order_checks,
        e90_checks,
        reports,
    )
    rows = result_rows(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_row_typed_shift_representation_audit",
        "config": serializable_config(config),
        "profile_source_run_id": physical_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": physical_sources["source_hashes"],
        "e90_source_run_id": e90_source["gate"].get("run_id"),
        "e90_source_hashes": e90_source["hashes"],
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "profile_checks": profile_checks,
        "model_order_checks": model_order_checks,
        "e90_checks": e90_checks,
        "gate": gate,
        "result_rows": rows,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
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


if __name__ == "__main__":
    raise SystemExit(main())
