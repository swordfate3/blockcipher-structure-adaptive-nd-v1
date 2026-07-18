from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    load_profile_operator_sources,
    measure_profile_operator_contract,
    train_profile_operator_matrix,
    validate_profile_operator_sources,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_replication import (
    ProfileOperatorReplicationConfig,
    adjudicate_profile_operator_replication,
    load_seed0_source,
    serializable_config,
    validate_seed0_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E68 seed1 profile operator replication."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProfileOperatorReplicationConfig(run_id=args.run_id)
    training_config = config.as_training_config()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"epochs": config.epochs, "seed": config.seed})
    seed0_source = load_seed0_source(args.seed0_root)
    seed0_checks = validate_seed0_source(seed0_source)
    profile_sources = load_profile_operator_sources(args.profile_root, args.atlas_root)
    profile_checks = validate_profile_operator_sources(profile_sources, strict=True)
    if not all(seed0_checks.values()) or not all(
        value for value in profile_checks.values() if isinstance(value, bool)
    ):
        raise ValueError("E65/E67 source validation failed")
    _write_progress(
        progress,
        "sources_validated",
        {"seed0": seed0_checks, "profile": profile_checks},
    )
    contract = measure_profile_operator_contract(training_config, profile_sources)
    training = train_profile_operator_matrix(
        training_config, profile_sources, args.output_root
    )
    gate = adjudicate_profile_operator_replication(
        config,
        seed0_checks,
        profile_checks,
        contract,
        seed0_source,
        training,
    )
    for row in training["trained_rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_balance_profile_operator_replication",
        "config": serializable_config(config),
        "seed0_source_run_id": seed0_source["gate"].get("run_id"),
        "seed0_source_hashes": seed0_source["hashes"],
        "profile_source_run_id": profile_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": profile_sources["source_hashes"],
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "seed0_checks": seed0_checks,
        "profile_checks": profile_checks,
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
