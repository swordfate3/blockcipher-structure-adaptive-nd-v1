from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_pair_state_neural_attribution import (
    PresentPairStateTrainingConfig,
    adjudicate_e44,
    load_e43_source,
    measure_model_contract,
    serializable_config,
    train_e44_matrix,
    validate_e43_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E44 PRESENT r4 pair-state topology attribution matrix."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--path-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PresentPairStateTrainingConfig(
        run_id=args.run_id,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        path_rank=args.path_rank,
        seed=args.seed,
        device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "mode": config.mode, "training": True},
    )
    data = load_e43_source(args.source_root)
    source_checks = validate_e43_source(data, strict=config.mode == "full")
    _write_progress(progress_path, "source_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43 source validation failed")
    contract = measure_model_contract(config, data)
    _write_progress(progress_path, "model_contract_complete", contract)
    matrix = train_e44_matrix(config, data)
    for row in matrix["rows"]:
        _write_progress(
            progress_path,
            "matrix_row_complete",
            {
                "row_id": row["row_id"],
                "validation_auc": row["validation_auc"],
            },
        )
    gate = adjudicate_e44(config, source_checks, contract, matrix)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_pair_state_neural_attribution",
        "config": serializable_config(config),
        "source_run_id": data["gate"].get("run_id"),
        "source_hashes": data["source_hashes"],
        "selected_processor": matrix["selected_processor"],
        "training_performed": True,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "model_contract": contract,
        "rows": matrix["rows"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", matrix["rows"])
    _write_csv(args.output_root / "history.csv", matrix["history"])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
