from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_gift_shared_profile_operator_readiness import (
    SharedProfileReadinessConfig,
    adjudicate_shared_profile_readiness,
    load_e85_sources,
    measure_shared_profile_contract,
    prepare_e85_sources,
    serializable_config,
    train_shared_profile_matrix,
    validate_e85_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E85 PRESENT/GIFT shared profile-operator readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--present-profile-root", required=True, type=Path)
    parser.add_argument("--present-atlas-root", required=True, type=Path)
    parser.add_argument("--gift-profile-root", required=True, type=Path)
    parser.add_argument("--present-readiness-root", required=True, type=Path)
    parser.add_argument("--gift-readiness-root", required=True, type=Path)
    parser.add_argument("--e80-root", required=True, type=Path)
    parser.add_argument("--e84-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SharedProfileReadinessConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"epochs": config.epochs, "seed": config.seed, "training": True},
    )
    sources = load_e85_sources(
        args.present_profile_root,
        args.present_atlas_root,
        args.gift_profile_root,
        args.present_readiness_root,
        args.gift_readiness_root,
        args.e80_root,
        args.e84_root,
    )
    source_checks = validate_e85_sources(sources)
    if not all(source_checks.values()):
        raise ValueError("E65/E75/E73/E76/E80/E84 source validation failed")
    data = prepare_e85_sources(sources)
    source_checks.update(
        {
            "present_r3_shape_is_96x64x13": data["present"][
                "prefix_features"
            ].shape
            == (96, 64, 13),
            "gift_r3_shape_is_192x64x13": data["gift"]["prefix_features"].shape
            == (192, 64, 13),
            "present_r3_matches_full_columns_26_to_38": bool(
                (
                    data["present"]["prefix_features"]
                    == sources["present_full"]["prefix_features"][:, :, 26:39]
                ).all()
            ),
            "gift_r3_matches_full_columns_26_to_38": bool(
                (
                    data["gift"]["prefix_features"]
                    == sources["gift_full"]["prefix_features"][:, :, 26:39]
                ).all()
            ),
        }
    )
    _write_progress(progress, "sources_validated", source_checks)
    contract = measure_shared_profile_contract(config, data)
    _write_progress(progress, "model_contract_complete", contract)
    matrix = train_shared_profile_matrix(config, data, args.output_root)
    gate = adjudicate_shared_profile_readiness(
        config, source_checks, contract, matrix
    )
    for row in matrix["rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
        row["training_performed"] = True
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_gift_shared_profile_operator_readiness",
        "architecture": "Topology-Parameterized Shared Profile Operator",
        "config": serializable_config(config),
        "present_profile_source_run_id": sources["present_full"][
            "profile_gate"
        ].get("run_id"),
        "gift_profile_source_run_id": sources["gift_full"]["profile_gate"].get(
            "run_id"
        ),
        "profile_source_hashes": {
            "present": sources["present_full"]["source_hashes"],
            "gift": sources["gift_full"]["source_hashes"],
        },
        "route_source_hashes": sources["route_hashes"],
        "shared_checkpoint": True,
        "cipher_id_input": False,
        "checkpoint_transfer": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "contract": contract,
        "schedule_audits": matrix["schedule_audits"],
        "rows": matrix["rows"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", matrix["rows"])
    _write_csv(args.output_root / "history.csv", matrix["history"])
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
