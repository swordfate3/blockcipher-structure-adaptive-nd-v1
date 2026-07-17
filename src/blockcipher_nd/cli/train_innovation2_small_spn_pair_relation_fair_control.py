from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.tasks.innovation2.small_spn_expanded_neural_screen import (
    load_expanded_neural_data,
    validate_expanded_neural_contract,
)
from blockcipher_nd.tasks.innovation2.small_spn_pair_relation_reasoner import (
    PairRelationTrainingConfig,
    adjudicate_pair_relation_attribution,
    measure_pair_relation_contract,
    pair_relation_fair_control_matrix,
    train_pair_relation_row,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E39 pair-relation fair-corrupted P attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--phase-a-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PairRelationTrainingConfig(run_id=args.run_id, device=args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    data = load_expanded_neural_data(args.source_root)
    readiness = validate_expanded_neural_contract(data)
    contract = measure_pair_relation_contract(
        data, hidden_dim=config.hidden_dim, path_rank=config.path_rank
    )
    source_gate = _read_json(args.phase_a_root / "gate.json")
    source_rows = _read_jsonl(args.phase_a_root / "results.jsonl")
    true_rows = [
        row
        for row in source_rows
        if row.get("topology_mode") == "true" and row.get("label_mode") == "true"
    ]
    matrix = pair_relation_fair_control_matrix()
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "source_run_id": source_gate.get("run_id"),
            "matrix_rows": len(matrix),
            "device": config.device,
            **contract,
        },
    )
    control_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for row_spec in matrix:
        _write_progress(progress_path, "training_row_start", {"row_id": row_spec.row_id})
        trained = train_pair_relation_row(config, row_spec, data)
        trained["result"]["task"] = (
            "innovation2_small_spn_pair_relation_fair_control"
        )
        control_rows.append(trained["result"])
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
    gate = adjudicate_pair_relation_attribution(
        config, readiness, contract, source_gate, true_rows, control_rows
    )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_pair_relation_fair_control",
        "source_run": data["source_gate"].get("run_id"),
        "phase_a_run": source_gate.get("run_id"),
        "heldout_used_for_training_selection_or_checkpoint": False,
        "control": "fixed destination-cell rotation composed with each variant P-layer",
        "model_contract": contract,
        "config": {
            "hidden_dim": config.hidden_dim,
            "path_rank": config.path_rank,
            "shared_triangle_blocks": 1,
            "step_schedule": [2, 3, 4, 5],
            "dropout": config.dropout,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "device": config.device,
        },
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(args.output_root / "results.jsonl", control_rows)
    _write_history(args.output_root / "history.csv", history)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": config.run_id,
            "true_rows": true_rows,
            "control_rows": control_rows,
            "gate": gate,
            "metadata": metadata,
        },
    )
    _write_progress(
        progress_path,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)}, sort_keys=True
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
