from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.rectangle80_r3_only_profile_operator_readiness import (
    Rectangle80R3ProfileReadinessConfig,
    adjudicate_rectangle_r3_readiness,
    fair_topology_ridges,
    load_rectangle_profile_sources,
    measure_rectangle_r3_contract,
    r3_only_sources,
    serializable_config,
    to_rectangle_model_order,
    train_rectangle_r3_matrix,
    validate_rectangle_model_order,
    validate_rectangle_profile_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E89 RECTANGLE-80 r3-only profile-operator readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Rectangle80R3ProfileReadinessConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"epochs": config.epochs, "seed": config.seed, "training": True},
    )
    physical_sources = load_rectangle_profile_sources(args.profile_root)
    source_checks = validate_rectangle_profile_sources(physical_sources)
    if not all(source_checks.values()):
        raise ValueError("E88 RECTANGLE profile source validation failed")
    model_sources = to_rectangle_model_order(physical_sources)
    model_order_checks = validate_rectangle_model_order(
        physical_sources, model_sources
    )
    if not all(model_order_checks.values()):
        raise ValueError("RECTANGLE cell-major topology adapter validation failed")
    ridges = fair_topology_ridges(config, model_sources)
    sources = r3_only_sources(model_sources)
    source_checks["r3_shape_is_192x64x13"] = sources[
        "prefix_features"
    ].shape == (192, 64, 13)
    source_checks["r3_features_match_full_columns_26_to_38"] = bool(
        (
            sources["prefix_features"]
            == model_sources["prefix_features"][:, :, 26:39]
        ).all()
    )
    _write_progress(progress, "sources_validated", source_checks)
    _write_progress(progress, "cell_major_adapter_validated", model_order_checks)
    _write_progress(progress, "fair_topology_ridges_complete", ridges)
    contract = measure_rectangle_r3_contract(config, sources)
    training = train_rectangle_r3_matrix(config, sources, args.output_root)
    gate = adjudicate_rectangle_r3_readiness(
        config,
        source_checks,
        model_order_checks,
        contract,
        ridges,
        training,
    )
    for row in training["trained_rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_r3_only_profile_readiness",
        "config": serializable_config(config),
        "profile_source_run_id": physical_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": physical_sources["source_hashes"],
        "cell_major_order": sources["physical_by_model"].tolist(),
        "checkpoint_transfer": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "model_order_checks": model_order_checks,
        "deterministic_ridges": ridges,
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
