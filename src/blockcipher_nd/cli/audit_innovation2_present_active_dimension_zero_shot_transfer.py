from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_active_dimension_zero_shot_transfer import (
    ACTIVE_DIMENSIONS,
    ActiveDimensionTransferConfig,
    adjudicate_active_dimension_transfer,
    build_dimension_labels,
    build_transfer_rows,
    evaluate_zero_shot_models,
    load_transfer_sources,
    make_transfer_structures,
    result_rows_for_transfer,
    scalar_validate_negatives,
    serializable_config,
    validate_transfer_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E70 PRESENT active-dimension zero-shot transfer."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--e43-root", required=True, type=Path)
    parser.add_argument("--e65-root", required=True, type=Path)
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--seed1-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ActiveDimensionTransferConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False})
    sources = load_transfer_sources(
        args.e43_root, args.e65_root, args.seed0_root, args.seed1_root
    )
    source_checks = validate_transfer_sources(sources)
    if not all(value for value in source_checks.values() if isinstance(value, bool)):
        raise ValueError("E43/E65/E67/E68 transfer source validation failed")
    _write_progress(progress, "sources_validated", source_checks)
    dimension_data = {}
    serialized_structures = {}
    for dimension in ACTIVE_DIMENSIONS:
        structures = make_transfer_structures(dimension)
        data = build_dimension_labels(config, structures)
        transfer = build_transfer_rows(
            data["labels"], dimension, config.checkerboard_attempts
        )
        scalar = scalar_validate_negatives(config, structures, data)
        dimension_data[dimension] = {
            "structures": structures,
            "data": data,
            "transfer": transfer,
            "scalar_validation": scalar,
        }
        serialized_structures[str(dimension)] = [
            {
                "index": structure.index,
                "structure_id": structure.structure_id,
                "dimension": structure.dimension,
                "active_bits": list(structure.active_bits),
            }
            for structure in structures
        ]
        np.save(args.output_root / f"d{dimension}_labels.npy", data["labels"])
        np.save(
            args.output_root / f"d{dimension}_witness_key_indices.npy",
            data["witness_key_indices"],
        )
        np.save(
            args.output_root / f"d{dimension}_witness_offsets.npy",
            data["witness_offsets"],
        )
        np.save(
            args.output_root / f"d{dimension}_prefix_features.npy",
            data["prefix_features"],
        )
        _write_progress(
            progress,
            "dimension_labels_complete",
            {
                "dimension": dimension,
                "positive": int(np.sum(data["labels"] == 1)),
                "negative": int(np.sum(data["labels"] == 0)),
                "unknown": int(np.sum(data["labels"] < 0)),
                "matched_rows": transfer["metrics"]["rows"],
                "scalar_validation": scalar,
                "provider_complete": data["provider_complete"],
                "provider_cap_events": data["provider_cap_events"],
                "completed_structures": data["completed_structures"],
            },
        )
    transfer_reports = evaluate_zero_shot_models(sources, dimension_data)
    gate = adjudicate_active_dimension_transfer(
        config, source_checks, dimension_data, transfer_reports
    )
    results = result_rows_for_transfer(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_active_dimension_zero_shot_transfer",
        "config": serializable_config(config),
        "source_hashes": sources["hashes"],
        "structures": serialized_structures,
        "training_performed": False,
        "checkpoint_inference_only": True,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", results)
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


if __name__ == "__main__":
    raise SystemExit(main())
