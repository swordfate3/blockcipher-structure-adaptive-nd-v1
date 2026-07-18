from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_cgpr_attribution import (
    CgprAttributionConfig,
    adjudicate_e51,
    build_prefix_ridge_bundle,
    load_e51_sources,
    measure_cgpr_contract,
    serializable_config,
    train_e51_matrix,
    validate_e51_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E51 PRESENT CGPR seed0 formal attribution matrix."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--e44-root", required=True, type=Path)
    parser.add_argument("--e45-root", required=True, type=Path)
    parser.add_argument("--e49-root", required=True, type=Path)
    parser.add_argument("--e50-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--path-rank", type=int, default=2)
    parser.add_argument("--residual-bound", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = CgprAttributionConfig(
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        path_rank=args.path_rank,
        residual_bound=args.residual_bound,
        seed=args.seed,
        device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "training": True, "remote_scale": False},
    )
    sources = load_e51_sources(
        args.atlas_root,
        args.e44_root,
        args.e45_root,
        args.e49_root,
        args.e50_root,
    )
    source_checks = validate_e51_sources(sources)
    _write_progress(progress_path, "sources_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43/E44/E45/E49/E50 source validation failed")
    ridge = build_prefix_ridge_bundle(sources["atlas"])
    contract = measure_cgpr_contract(config, sources["atlas"], ridge)
    _write_progress(progress_path, "cgpr_contract_complete", contract)
    matrix = train_e51_matrix(config, sources["atlas"], ridge)
    for row in matrix["rows"]:
        _write_progress(
            progress_path,
            "matrix_row_complete",
            {
                "row_id": row["row_id"],
                "validation_auc": row["validation_auc"],
                "best_epoch": row["best_epoch"],
                "training_performed": row["training_performed"],
            },
        )
    gate = adjudicate_e51(config, source_checks, contract, matrix)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_cgpr_attribution",
        "config": serializable_config(config),
        "source_hashes": {
            **sources["atlas"]["source_hashes"],
            **sources["neural_hashes"],
            **sources["e50_hashes"],
            **sources["e51_hashes"],
        },
        "prefix_source": "PRESENT S-box ANF and true P-layer rounds 1-3",
        "final_round_oracle_used": False,
        "certificate_or_witness_input_used": False,
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
