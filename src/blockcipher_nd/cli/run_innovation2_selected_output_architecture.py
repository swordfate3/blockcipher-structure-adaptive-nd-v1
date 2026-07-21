from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.plot_innovation2_selected_output_architecture import (
    render_selected_output_architecture,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    REMOTE_RUN_ID,
    RUN_ID,
    SelectedOutputArchitectureConfig,
    adjudicate_architecture_gate,
    parameter_counts,
    prepare_architecture_data,
    serializable_config,
    train_architecture_matrix,
    validate_architecture_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OPA1 five-model selected-output screen."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "phase_a_screen"),
        default="smoke",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "phase_a_screen":
        config = SelectedOutputArchitectureConfig.phase_a_screen(
            run_id=args.run_id or REMOTE_RUN_ID,
            device=args.device or "cuda",
        )
    else:
        config = SelectedOutputArchitectureConfig(
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
    data = prepare_architecture_data(config, args.output_root, progress=progress)
    protocol_checks = validate_architecture_contract(config, data)
    progress("data_ready", protocol_checks)
    training = train_architecture_matrix(
        config,
        data,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_architecture_gate(config, protocol_checks, training)
    for row in training["rows"]:
        row["run_id"] = config.run_id
        row["rounds"] = config.rounds
        row["seed"] = config.seed
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "opa1_present_r3_selected8_architecture_screen",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_config(config),
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "third independent fixed unknown key; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "eight preregistered MSB-first true ciphertext bits",
        "phase": "A discovery screen; architecture claims require seed3 matched-shuffle confirmation",
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
        args.output_root / "checkpoint_manifest.json",
        training["checkpoints"],
    )
    if config.mode == "smoke":
        render_selected_output_architecture(
            summary,
            args.output_root / "curves.svg",
        )
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            sort_keys=True,
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
