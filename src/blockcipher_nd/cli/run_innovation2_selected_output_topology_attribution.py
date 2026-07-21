from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.selected_output_topology_attribution import (
    TopologyAttributionConfig,
    adjudicate_topology,
    authorize_from_opa2_gate,
    prepare_topology_data,
    serializable_topology_config,
    topology_parameter_counts,
    train_topology_matrix,
    validate_topology_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gate-authorized Innovation 2 OPA3 PRESENT topology attribution."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "topology_attribution"),
        default="smoke",
    )
    parser.add_argument("--opa2-gate", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    opa2_gate = json.loads(args.opa2_gate.read_text(encoding="utf-8"))
    authorize_from_opa2_gate(opa2_gate)
    if args.mode == "topology_attribution":
        config = TopologyAttributionConfig.formal(
            run_id=args.run_id,
            device=args.device or "cuda",
        )
    else:
        default = TopologyAttributionConfig()
        config = TopologyAttributionConfig(
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
            "opa2_gate": str(args.opa2_gate),
            "sample_classification": False,
        },
    )
    data = prepare_topology_data(config, args.output_root, progress=progress)
    protocol_checks = validate_topology_contract(config, data, opa2_gate)
    progress("data_ready", protocol_checks)
    training = train_topology_matrix(
        config,
        data,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_topology(config, protocol_checks, training, opa2_gate)
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
        "experiment": "opa3_present_r3_selected8_topology_attribution",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "config": serializable_topology_config(config),
        "opa2_gate_run_id": opa2_gate["run_id"],
        "secret_key_hex": f"{int(data['secret_key']):020x}",
        "key_protocol": "same fourth fixed unknown key as OPA2; disjoint plaintext train/test",
        "input": "64 MSB-first plaintext bits",
        "target": "eight preregistered MSB-first true ciphertext bits",
        "sample_classification": False,
        "selected_msb_indices": list(config.selected_msb_indices),
        "parameter_counts": topology_parameter_counts(config),
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "opa2_gate": opa2_gate,
        "protocol_checks": protocol_checks,
        "model_summaries": training["summaries"],
        "bit_rows": training["rows"],
        "checkpoint_manifest": training["checkpoints"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", training["rows"])
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "opa2_gate.json", opa2_gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "checkpoint_manifest.json",
        training["checkpoints"],
    )
    if config.mode == "smoke":
        from blockcipher_nd.cli.plot_innovation2_selected_output_topology_attribution import (
            render_topology_attribution,
        )

        render_topology_attribution(summary, args.output_root / "curves.svg")
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
