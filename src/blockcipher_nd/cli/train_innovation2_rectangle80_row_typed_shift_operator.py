from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    load_rectangle_profile_sources,
    r3_only_sources,
    to_rectangle_model_order,
    validate_rectangle_model_order,
    validate_rectangle_profile_sources,
)
from blockcipher_nd.tasks.innovation2.rectangle80_row_typed_shift_operator_readiness import (
    Rectangle80RowTypedOperatorConfig,
    adjudicate_row_typed_readiness,
    load_e91_source,
    measure_row_typed_contract,
    serializable_config,
    train_row_typed_matrix,
    validate_e91_source,
)
from blockcipher_nd.tasks.innovation2.rectangle80_row_typed_shift_representation_audit import (
    load_e90_source,
    validate_e90_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E92 RECTANGLE Row-Typed Shift Operator readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--e90-root", required=True, type=Path)
    parser.add_argument("--e91-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Rectangle80RowTypedOperatorConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"training": True, "epochs": config.epochs, "seed": config.seed},
    )
    physical_sources = load_rectangle_profile_sources(args.profile_root)
    profile_checks = validate_rectangle_profile_sources(physical_sources)
    model_sources = to_rectangle_model_order(physical_sources)
    model_order_checks = validate_rectangle_model_order(
        physical_sources, model_sources
    )
    e90_source = load_e90_source(args.e90_root)
    e90_checks = validate_e90_source(e90_source)
    e91_source = load_e91_source(args.e91_root)
    e91_checks = validate_e91_source(e91_source)
    if not all(
        all(checks.values())
        for checks in (
            profile_checks,
            model_order_checks,
            e90_checks,
            e91_checks,
        )
    ):
        raise ValueError("E88/E90/E91 source validation failed")
    sources = r3_only_sources(model_sources)
    profile_checks["r3_shape_is_192x64x13"] = sources[
        "prefix_features"
    ].shape == (192, 64, 13)
    _write_progress(
        progress,
        "sources_validated",
        {
            "profile": profile_checks,
            "model_order": model_order_checks,
            "e90": e90_checks,
            "e91": e91_checks,
        },
    )
    contract = measure_row_typed_contract(config, sources)
    training = train_row_typed_matrix(config, sources, args.output_root)
    gate = adjudicate_row_typed_readiness(
        config,
        profile_checks,
        model_order_checks,
        e90_checks,
        e91_checks,
        contract,
        training,
        e91_source,
    )
    for row in training["trained_rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_row_typed_shift_operator_readiness",
        "config": serializable_config(config),
        "profile_source_run_id": physical_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": physical_sources["source_hashes"],
        "e90_source_run_id": e90_source["gate"].get("run_id"),
        "e90_source_hashes": e90_source["hashes"],
        "e91_source_run_id": e91_source["gate"].get("run_id"),
        "e91_source_hashes": e91_source["hashes"],
        "checkpoint_transfer": False,
        "parameter_neutral_row_typing": True,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "profile_checks": profile_checks,
        "model_order_checks": model_order_checks,
        "e90_checks": e90_checks,
        "e91_checks": e91_checks,
        "contract": contract,
        "trained_rows": training["trained_rows"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", training["trained_rows"])
    _write_csv(args.output_root / "history.csv", training["history_rows"])
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
