from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_attribution import (
    Gift64R3AttributionConfig,
    adjudicate_gift_r3_attribution,
    load_e77_source,
    serializable_config,
    validate_e77_source,
)
from blockcipher_nd.tasks.innovation2.gift64_r3_only_profile_operator_readiness import (
    load_gift_profile_sources,
    measure_gift_r3_contract,
    r3_only_sources,
    train_gift_r3_matrix,
    validate_gift_profile_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E78 GIFT-64 r3-only 30-epoch seed0 attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--e77-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Gift64R3AttributionConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"epochs": config.epochs, "seed": config.seed, "training": True},
    )

    full_sources = load_gift_profile_sources(args.profile_root)
    profile_checks = validate_gift_profile_sources(full_sources)
    e77_source = load_e77_source(args.e77_root)
    e77_checks = validate_e77_source(e77_source)
    if not all(profile_checks.values()) or not all(e77_checks.values()):
        raise ValueError("E75/E77 source validation failed")
    sources = r3_only_sources(full_sources)
    profile_checks["r3_shape_is_192x64x13"] = sources["prefix_features"].shape == (
        192,
        64,
        13,
    )
    _write_progress(
        progress,
        "sources_validated",
        {"profile": profile_checks, "e77": e77_checks},
    )
    contract = measure_gift_r3_contract(config, sources)
    training = train_gift_r3_matrix(config, sources, args.output_root)
    gate = adjudicate_gift_r3_attribution(
        config,
        profile_checks,
        e77_checks,
        contract,
        training,
        e77_source,
    )
    for row in training["trained_rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": config.task_name,
        "config": serializable_config(config),
        "profile_source_run_id": full_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": full_sources["source_hashes"],
        "e77_source_run_id": e77_source["gate"].get("run_id"),
        "e77_source_hashes": e77_source["hashes"],
        "checkpoint_transfer": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "profile_checks": profile_checks,
        "e77_checks": e77_checks,
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
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
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
