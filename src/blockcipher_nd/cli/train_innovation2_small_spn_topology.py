from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    TopologyTrainingConfig,
    adjudicate_topology_training,
    load_topology_training_data,
    train_topology_row,
    training_matrix,
    validate_training_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E33 small-SPN GraphGPS/SCGT attribution matrix."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--contrast-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    smoke = args.mode == "smoke"
    config = TopologyTrainingConfig(
        run_id=args.run_id,
        mode=args.mode,
        hidden_dim=args.hidden_dim or (32 if smoke else 64),
        blocks=args.blocks or (2 if smoke else 3),
        heads=4,
        epochs=args.epochs or (8 if smoke else 40),
        batch_size=128,
        device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    data = load_topology_training_data(args.label_root, args.contrast_root)
    readiness = validate_training_contract(data)
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "matrix_rows": len(training_matrix(config)),
            "device": config.device,
        },
    )
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for row_spec in training_matrix(config):
        _write_progress(progress_path, "training_row_start", {"row_id": row_spec.row_id})
        trained = train_topology_row(config, row_spec, data)
        rows.append(trained["result"])
        history.extend(trained["history"])
        torch.save(trained["state_dict"], checkpoint_root / f"{row_spec.row_id}.pt")
        _write_progress(
            progress_path,
            "training_row_done",
            {
                "row_id": row_spec.row_id,
                "best_epoch": trained["result"]["best_epoch"],
                "dual_unseen_auc": trained["result"]["dual_unseen_auc"],
            },
        )
    gate = adjudicate_topology_training(config, readiness, rows)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_topology_prediction",
        "mode": config.mode,
        "label_source_run": data["label_gate"].get("run_id"),
        "contrast_source_run": data["contrast_gate"].get("run_id"),
        "fit_cells": len(data["fit_cells"]),
        "validation_cells": len(data["validation_cells"]),
        "heldout_used_for_training_or_selection": False,
        "cipher_id_embedding": False,
        "config": {
            "hidden_dim": config.hidden_dim,
            "blocks": config.blocks,
            "heads": config.heads,
            "dropout": config.dropout,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "device": config.device,
        },
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_history(args.output_root / "history.csv", history)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(
        args.output_root / "summary.json",
        {"run_id": config.run_id, "rows": rows, "gate": gate, "metadata": metadata},
    )
    _write_progress(
        progress_path,
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


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
