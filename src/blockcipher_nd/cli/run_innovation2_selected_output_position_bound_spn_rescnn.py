from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.selected_output_position_bound_spn_rescnn import (
    OPD1_GATE_SHA256,
    OPF1_GATE_SHA256,
    OPC1_GATE_SHA256,
    OPN1_GATE_SHA256,
    PositionBoundSpnResCnnConfig,
    adjudicate_position_bound,
    authorize_from_source_gates,
    authorize_round_extension_from_opd1_gate,
    authorize_scale_extension_from_opf1_gate,
    position_bound_parameter_counts,
    prepare_position_bound_data,
    serializable_position_bound_config,
    train_position_bound_matrix,
    validate_position_bound_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OPD1 position-bound SPN-ResCNN prediction."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "position_bound_head",
            "round_extension_smoke",
            "round_extension",
            "scale_extension_smoke",
            "scale_extension",
        ),
        default="smoke",
    )
    parser.add_argument("--opc1-gate", type=Path)
    parser.add_argument("--opn1-gate", type=Path)
    parser.add_argument("--opd1-gate", type=Path)
    parser.add_argument("--opf1-gate", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.opc1_gate is None or args.opn1_gate is None:
        raise ValueError("OPD1 requires --opc1-gate and --opn1-gate")
    source_gate_paths = (args.opc1_gate, args.opn1_gate)
    source_gate_bytes = tuple(path.read_bytes() for path in source_gate_paths)
    source_gates = tuple(json.loads(payload) for payload in source_gate_bytes)
    authorize_from_source_gates(*source_gates)
    source_gate_checks = {
        "opc1_gate_sha256_matches_frozen_source": hashlib.sha256(
            source_gate_bytes[0]
        ).hexdigest()
        == OPC1_GATE_SHA256,
        "opn1_gate_sha256_matches_frozen_source": hashlib.sha256(
            source_gate_bytes[1]
        ).hexdigest()
        == OPN1_GATE_SHA256,
    }
    is_round_extension = args.mode.startswith("round_extension")
    is_scale_extension = args.mode.startswith("scale_extension")
    is_r4 = is_round_extension or is_scale_extension
    opd1_gate_bytes = None
    opd1_gate = None
    if is_r4:
        if args.opd1_gate is None:
            raise ValueError("PRESENT r4 extensions require --opd1-gate")
        opd1_gate_bytes = args.opd1_gate.read_bytes()
        opd1_gate = json.loads(opd1_gate_bytes)
        authorize_round_extension_from_opd1_gate(opd1_gate)
        source_gate_checks["opd1_gate_sha256_matches_frozen_source"] = (
            hashlib.sha256(opd1_gate_bytes).hexdigest() == OPD1_GATE_SHA256
        )
    opf1_gate_bytes = None
    opf1_gate = None
    if is_scale_extension:
        if args.opf1_gate is None:
            raise ValueError("OPF2 scale extension requires --opf1-gate")
        opf1_gate_bytes = args.opf1_gate.read_bytes()
        opf1_gate = json.loads(opf1_gate_bytes)
        authorize_scale_extension_from_opf1_gate(opf1_gate)
        source_gate_checks["opf1_gate_sha256_matches_frozen_source"] = (
            hashlib.sha256(opf1_gate_bytes).hexdigest() == OPF1_GATE_SHA256
        )
    if args.mode == "position_bound_head":
        config = PositionBoundSpnResCnnConfig.formal(
            run_id=args.run_id,
            device=args.device or "cuda",
        )
    elif args.mode == "round_extension":
        config = PositionBoundSpnResCnnConfig.round_extension(
            run_id=args.run_id,
            device=args.device or "cuda",
        )
    elif args.mode == "round_extension_smoke":
        config = PositionBoundSpnResCnnConfig.round_extension(
            run_id=args.run_id,
            device=args.device or "cpu",
            smoke=True,
        )
    elif args.mode == "scale_extension":
        config = PositionBoundSpnResCnnConfig.scale_extension(
            run_id=args.run_id,
            device=args.device or "cuda",
        )
    elif args.mode == "scale_extension_smoke":
        config = PositionBoundSpnResCnnConfig.scale_extension(
            run_id=args.run_id,
            device=args.device or "cpu",
            smoke=True,
        )
    else:
        default = PositionBoundSpnResCnnConfig()
        config = PositionBoundSpnResCnnConfig(
            **{
                **default.__dict__,
                "run_id": args.run_id or default.run_id,
                "device": args.device or "cpu",
            }
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
            "training": True,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
        },
    )
    data = prepare_position_bound_data(config, args.output_root, progress=progress)
    protocol_checks = validate_position_bound_contract(config, data)
    protocol_checks.update(source_gate_checks)
    progress("data_ready", protocol_checks)
    training = train_position_bound_matrix(
        config,
        data,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_position_bound(
        config,
        protocol_checks,
        training,
        reference_gate=opf1_gate if is_scale_extension else opd1_gate,
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
        "experiment": (
            "opf2_present_r4_selected8_position_bound_spn_rescnn_2p20_scale"
            if is_scale_extension
            else (
                "opf1_present_r4_selected8_position_bound_spn_rescnn_round_extension"
                if is_round_extension
                else "opd1_present_r3_selected8_position_bound_spn_rescnn"
            )
        ),
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_position_bound_config(config),
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "eighth independent fixed unknown key; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "eight preregistered MSB-first true ciphertext bits",
        "sample_classification": False,
        "selected_msb_indices": list(config.selected_msb_indices),
        "parameter_counts": position_bound_parameter_counts(config),
        "opc1_gate_decision": source_gates[0]["decision"],
        "opn1_gate_decision": source_gates[1]["decision"],
        "opd1_gate_decision": opd1_gate["decision"] if opd1_gate else None,
        "opf1_gate_decision": opf1_gate["decision"] if opf1_gate else None,
        "split_layout": data.get("split_layout"),
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
    _write_json(args.output_root / "checkpoint_manifest.json", training["checkpoints"])
    _write_source_gate(args.output_root / "opc1_gate.json", source_gate_bytes[0])
    _write_source_gate(args.output_root / "opn1_gate.json", source_gate_bytes[1])
    if opd1_gate_bytes is not None:
        _write_source_gate(args.output_root / "opd1_gate.json", opd1_gate_bytes)
    if opf1_gate_bytes is not None:
        _write_source_gate(args.output_root / "opf1_gate.json", opf1_gate_bytes)
    if config.mode in {
        "smoke",
        "round_extension_smoke",
        "scale_extension_smoke",
    }:
        from blockcipher_nd.cli.plot_innovation2_selected_output_position_bound_spn_rescnn import (
            render_position_bound_spn_rescnn,
        )

        render_position_bound_spn_rescnn(summary, args.output_root / "curves.svg")
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(
        json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True)
    )
    return 1 if gate["status"] == "fail" else 0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_source_gate(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


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
