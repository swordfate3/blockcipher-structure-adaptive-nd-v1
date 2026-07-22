from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.selected8_output_parity import (
    RUN_ID_PREFIX,
    SELECTED8_PARITY_MASK,
    SHIFTED_CONTROL_MASK,
    Selected8ParityConfig,
    adjudicate_selected8_parity,
    parameter_counts,
    prepare_selected8_parity_data,
    serializable_config,
    train_selected8_parity_matrix,
    validate_selected8_parity_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OPE1 selected-eight output parity prediction."
    )
    parser.add_argument("--mode", choices=("smoke", "diagnostic"), default="smoke")
    parser.add_argument("--rounds", choices=(3, 4), default=3, type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--r3-gate", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "diagnostic":
        config = Selected8ParityConfig.diagnostic(
            rounds=args.rounds,
            run_id=args.run_id,
            device=args.device or "cpu",
        )
    else:
        config = Selected8ParityConfig(
            run_id=args.run_id
            or f"{RUN_ID_PREFIX}_r{args.rounds}_smoke_seed1_20260722",
            rounds=args.rounds,
            device=args.device or "cpu",
        )
    if config.mode == "diagnostic" and config.rounds == 4 and args.r3_gate is None:
        raise ValueError("OPE1 r4 diagnostic requires --r3-gate")
    r3_gate = (
        json.loads(args.r3_gate.read_text(encoding="utf-8"))
        if args.r3_gate is not None
        else None
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "rounds": config.rounds,
            "training": True,
            "sample_classification": False,
            "target": "xor_of_eight_preregistered_true_ciphertext_output_bits",
        },
    )
    data = prepare_selected8_parity_data(
        config, args.output_root, progress=progress
    )
    protocol_checks = validate_selected8_parity_contract(config, data)
    progress("data_ready", protocol_checks)
    training = train_selected8_parity_matrix(
        config,
        data,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_selected8_parity(
        config,
        protocol_checks,
        training,
        r3_gate=r3_gate,
    )
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
        "experiment": "ope1_present_r3_r4_selected8_full_parity",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_config(config),
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "same second fixed unknown key as OP11 and OP12; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "XOR of all eight preregistered true ciphertext output bits",
        "sample_classification": False,
        "selected8_parity_mask": list(SELECTED8_PARITY_MASK),
        "shifted_geometry_control_mask": list(SHIFTED_CONTROL_MASK),
        "parameter_counts": parameter_counts(config),
        "source_r3_gate": str(args.r3_gate) if args.r3_gate is not None else None,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "protocol_checks": protocol_checks,
        "model_summaries": training["summaries"],
        "result_rows": training["rows"],
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
    if args.r3_gate is not None:
        (args.output_root / "r3_gate.json").write_bytes(args.r3_gate.read_bytes())
    from blockcipher_nd.cli.plot_innovation2_selected8_parity import (
        render_selected8_parity,
    )

    render_selected8_parity(summary, args.output_root / "curves.svg")
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


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
