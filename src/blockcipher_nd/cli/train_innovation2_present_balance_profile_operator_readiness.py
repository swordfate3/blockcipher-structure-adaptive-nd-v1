from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    ProfileOperatorReadinessConfig,
    adjudicate_profile_operator_readiness,
    load_profile_operator_sources,
    measure_profile_operator_contract,
    serializable_config,
    train_profile_operator_matrix,
    validate_profile_operator_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E66 prefix-guided balance profile operator readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "readiness"), default="readiness")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProfileOperatorReadinessConfig(
        run_id=args.run_id,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        seed=args.seed,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"mode": config.mode, "training": True})
    sources = load_profile_operator_sources(args.profile_root, args.atlas_root)
    source_checks = validate_profile_operator_sources(
        sources, strict=config.mode == "readiness"
    )
    _write_progress(progress, "sources_validated", source_checks)
    if not all(value for value in source_checks.values() if isinstance(value, bool)):
        raise ValueError("E43/E65 profile source validation failed")
    contract = measure_profile_operator_contract(config, sources)
    _write_progress(progress, "model_contract_complete", contract)
    training = train_profile_operator_matrix(config, sources, args.output_root)
    gate = adjudicate_profile_operator_readiness(
        config, source_checks, contract, training
    )
    for row in training["trained_rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_balance_profile_operator_readiness",
        "config": serializable_config(config),
        "source_run_id": sources["profile_gate"].get("run_id"),
        "source_hashes": sources["source_hashes"],
        "prefix_semantics": "E45/E65 r1-r3 ANF support/degree features per output node",
        "forbidden_inputs": [
            "r4 full-cube certificate",
            "negative witness",
            "key/offset parity",
            "label-derived feature",
        ],
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "contract": contract,
        "trained_rows": training["trained_rows"],
        "gate": gate,
    }
    np.save(args.output_root / "prefix_features.npy", sources["prefix_features"])
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
