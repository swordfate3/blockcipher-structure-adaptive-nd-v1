from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.tasks.innovation2.small_spn_rcca_training import (
    RelationTrainingConfig,
    adjudicate_relation_training,
    load_relation_training_data,
    measure_relation_model_contract,
    phase_a_matrix,
    readiness_matrix,
    train_relation_row,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E63 DeepSets/RCCA.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--e37-root", required=True, type=Path)
    parser.add_argument("--e62-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "smoke":
        config = RelationTrainingConfig(run_id=args.run_id, device=args.device)
        matrix = readiness_matrix()
    else:
        config = RelationTrainingConfig(
            run_id=args.run_id,
            mode="full",
            hidden_dim=64,
            layers=2,
            heads=4,
            epochs=40,
            batch_size=128,
            dropout=0.10,
            device=args.device,
        )
        matrix = phase_a_matrix()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _write_progress(progress, "source_load_start", {"mode": config.mode})
    data = load_relation_training_data(args.e37_root, args.e62_root)
    contract = measure_relation_model_contract(config, data)
    _write_progress(progress, "contract_done", contract)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for row_spec in matrix:
        _write_progress(progress, "row_start", {"row_id": row_spec.row_id})
        trained = train_relation_row(config, row_spec, data)
        rows.append(trained["result"])
        history.extend(trained["history"])
        torch.save(
            trained["state_dict"], checkpoint_root / f"{row_spec.row_id}.pt"
        )
        _write_progress(
            progress,
            "row_done",
            {
                "row_id": row_spec.row_id,
                "best_epoch": trained["result"]["best_epoch"],
                "dual_unseen_auc": trained["result"]["dual_unseen_auc"],
            },
        )
    gate = adjudicate_relation_training(
        config, data=data, contract=contract, rows=rows
    )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_rcca",
        "mode": config.mode,
        "e37_root": str(args.e37_root),
        "e62_root": str(args.e62_root),
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "heads": config.heads,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "device": config.device,
        "fit_relations": len(data["fit_relations"]),
        "validation_relations": len(data["validation_relations"]),
        "checkpoint_metric": "train-topology validation-relation AUC",
        "heldout_topologies_used_for_checkpoint": False,
        "exact_key_vectors_used_as_model_features": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "contract": contract,
        "rows": rows,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_history(args.output_root / "history.csv", history)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "rows": rows}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": event,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


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
    fieldnames = (
        "row_id",
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_auc",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
