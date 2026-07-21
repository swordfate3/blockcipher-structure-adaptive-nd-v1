from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    PAPER_RUN_ID,
    RUN_ID,
    KimuraOutputPredictionConfig,
    adjudicate_kimura_output_prediction,
    parameter_counts,
    prepare_disk_output_prediction_data,
    serializable_config,
    train_kimura_output_matrix,
    validate_kimura_output_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OP9 PRESENT r3 Kimura-LSTM output prediction."
    )
    parser.add_argument(
        "--mode", choices=("smoke", "paper_calibration"), default="smoke"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "paper_calibration":
        config = KimuraOutputPredictionConfig.paper_calibration(
            run_id=args.run_id or PAPER_RUN_ID,
            seed=args.seed,
            device=args.device or "cuda",
        )
    else:
        config = KimuraOutputPredictionConfig(
            run_id=args.run_id or RUN_ID,
            seed=args.seed,
            device=args.device or "cpu",
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "training": True,
            "sample_classification": False,
            "target": "full_64_bit_true_ciphertext_output",
        },
    )
    data = prepare_disk_output_prediction_data(
        config, args.output_root, progress=progress
    )
    protocol_checks = validate_kimura_output_contract(config, data)
    progress("data_ready", protocol_checks)
    training = train_kimura_output_matrix(
        config, data, args.output_root, progress=progress
    )
    gate = adjudicate_kimura_output_prediction(
        config, protocol_checks, training
    )
    for row in training["rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "op9_present_r3_kimura_lstm",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_config(config),
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "one fixed unknown key; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "64 MSB-first true ciphertext bits",
        "sample_classification": False,
        "paper_family": "Kimura et al. output prediction",
        "paper_exact_reproduction": False,
        "parameter_counts": parameter_counts(config),
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "protocol_checks": protocol_checks,
        "trained_rows": training["rows"],
        "checkpoint_manifest": training["checkpoints"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", training["rows"])
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "checkpoint_manifest.json", training["checkpoints"]
    )
    progress(
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
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
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
