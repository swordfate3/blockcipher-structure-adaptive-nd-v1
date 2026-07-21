from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    REMOTE_RUN_ID,
    RUN_ID,
    SelectedOutputBitHeadConfig,
    adjudicate_selected_output_head,
    parameter_counts,
    prepare_selected_output_data,
    serializable_config,
    train_selected_output_matrix,
    validate_selected_output_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OP11 selected output-bit head confirmation."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "independent_key_confirmation"),
        default="smoke",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "independent_key_confirmation":
        config = SelectedOutputBitHeadConfig.independent_key_confirmation(
            run_id=args.run_id or REMOTE_RUN_ID,
            device=args.device or "cuda",
        )
    else:
        config = SelectedOutputBitHeadConfig(
            run_id=args.run_id or RUN_ID,
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
            "target": "eight_preregistered_true_ciphertext_output_bits",
        },
    )
    data = prepare_selected_output_data(config, args.output_root, progress=progress)
    protocol_checks = validate_selected_output_contract(config, data)
    progress("data_ready", protocol_checks)
    training = train_selected_output_matrix(
        config, data, args.output_root, progress=progress
    )
    gate = adjudicate_selected_output_head(config, protocol_checks, training)
    for row in training["rows"]:
        row["run_id"] = config.run_id
        row["rounds"] = config.rounds
        row["seed"] = config.seed
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "op11_present_r3_selected8_independent_key",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_config(config),
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "second independent fixed unknown key; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "eight preregistered MSB-first true ciphertext bits",
        "sample_classification": False,
        "selected_msb_indices": list(config.selected_msb_indices),
        "parameter_counts": parameter_counts(config),
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "protocol_checks": protocol_checks,
        "model_summaries": training["summaries"],
        "bit_rows": training["rows"],
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
        "run_done", {"status": gate["status"], "decision": gate["decision"]}
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
