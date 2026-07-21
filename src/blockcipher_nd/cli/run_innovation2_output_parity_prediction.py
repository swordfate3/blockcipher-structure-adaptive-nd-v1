from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    MASKS,
    RUN_ID,
    OutputParityPredictionConfig,
    adjudicate_output_prediction_readiness,
    generate_output_prediction_data,
    serializable_config,
    train_output_prediction_matrix,
    validate_output_prediction_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OP1 fixed-key ciphertext parity prediction readiness."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = OutputParityPredictionConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {
            "training": True,
            "sample_classification": False,
            "rounds": config.rounds,
            "seed": config.seed,
        },
    )
    data = generate_output_prediction_data(config)
    protocol_checks = validate_output_prediction_contract(config, data)
    _write_progress(progress, "data_ready", protocol_checks)
    training = train_output_prediction_matrix(config, data)
    gate = adjudicate_output_prediction_readiness(config, protocol_checks, training)
    for row in training["rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_parity_prediction",
        "experiment": "op1",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "one fixed unknown key; disjoint plaintext splits",
        "sample_classification": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    dataset_summary = {
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
        "test_rows": config.test_rows,
        "input_shape": [64],
        "full_output_shape": [64],
        "direct_parity_shape": [16],
        "parity_masks": [f"{mask:016x}" for mask in MASKS],
        "train_parity_prevalence": data["train"].parity_targets.mean(axis=0).tolist(),
        "validation_parity_prevalence": data["validation"]
        .parity_targets.mean(axis=0)
        .tolist(),
        "test_parity_prevalence": data["test"].parity_targets.mean(axis=0).tolist(),
        "protocol_checks": protocol_checks,
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "dataset_summary": dataset_summary,
        "protocol_checks": protocol_checks,
        "trained_rows": training["rows"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", training["rows"])
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "dataset_summary.json", dataset_summary)
    _write_masks(args.output_root / "masks.csv")
    _write_arrays(args.output_root, data)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _write_arrays(output_root: Path, data: dict[str, Any]) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        split = data[split_name]
        split_root = data_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        np.save(split_root / "plaintexts.npy", split.plaintexts)
        np.save(split_root / "features.npy", split.features)
        np.save(split_root / "full_targets.npy", split.full_targets)
        np.save(split_root / "parity_targets.npy", split.parity_targets)


def _write_masks(path: Path) -> None:
    rows = [
        {
            "mask_index": index,
            "output_nibble": index,
            "mask_hex": f"{mask:016x}",
            "output_bits_lsb_first": ",".join(str(4 * index + bit) for bit in range(4)),
        }
        for index, mask in enumerate(MASKS)
    ]
    _write_csv(path, rows)


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
