from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.selected_output_architecture_confirmation import (
    ArchitectureConfirmationConfig,
    adjudicate_confirmation,
    candidate_from_phase_a_gate,
    confirmation_parameter_counts,
    prepare_confirmation_data,
    serializable_confirmation_config,
    train_confirmation_matrix,
    validate_confirmation_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gate-selected Innovation 2 OPA2 architecture confirmation."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "phase_b_confirmation"),
        default="smoke",
    )
    parser.add_argument("--phase-a-gate", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phase_a_gate = json.loads(args.phase_a_gate.read_text(encoding="utf-8"))
    candidate = candidate_from_phase_a_gate(phase_a_gate)
    if args.mode == "phase_b_confirmation":
        config = ArchitectureConfirmationConfig.phase_b_confirmation(
            candidate,
            run_id=args.run_id,
            device=args.device or "cuda",
        )
    else:
        default = ArchitectureConfirmationConfig.smoke(candidate)
        config = ArchitectureConfirmationConfig(
            **{
                **default.__dict__,
                "run_id": args.run_id or default.run_id,
                "device": args.device or "cpu",
            }
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
            "candidate_architecture": candidate,
            "phase_a_gate": str(args.phase_a_gate),
            "sample_classification": False,
        },
    )
    data = prepare_confirmation_data(config, args.output_root, progress=progress)
    protocol_checks = validate_confirmation_contract(config, data, phase_a_gate)
    progress("data_ready", protocol_checks)
    training = train_confirmation_matrix(
        config,
        data,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_confirmation(config, protocol_checks, training)
    for row in training["rows"]:
        row.update(
            {
                "run_id": config.run_id,
                "rounds": config.rounds,
                "seed": config.seed,
                "status": gate["status"],
                "decision": gate["decision"],
            }
        )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "opa2_present_r3_selected8_architecture_confirmation",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_confirmation_config(config),
        "candidate_architecture": candidate,
        "phase_a_gate_run_id": phase_a_gate["run_id"],
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "fourth independent fixed unknown key; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "eight preregistered MSB-first true ciphertext bits",
        "sample_classification": False,
        "selected_msb_indices": list(config.selected_msb_indices),
        "parameter_counts": confirmation_parameter_counts(config),
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "phase_a_gate": phase_a_gate,
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
        from blockcipher_nd.cli.plot_innovation2_selected_output_architecture_confirmation import (
            render_architecture_confirmation,
        )

        render_architecture_confirmation(summary, args.output_root / "curves.svg")
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
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
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
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
