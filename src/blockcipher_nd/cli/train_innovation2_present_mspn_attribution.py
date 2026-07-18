from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_mspn_attribution import (
    MspnAttributionConfig,
    adjudicate_e47,
    load_attribution_sources,
    serializable_config,
    train_attribution_matrix,
    validate_attribution_sources,
)
from blockcipher_nd.tasks.innovation2.present_mspn_readiness import (
    measure_mspn_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E47 PRESENT MSPN seed0 attribution matrix."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--e44-root", required=True, type=Path)
    parser.add_argument("--e45-root", required=True, type=Path)
    parser.add_argument("--e46-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--degree-channels", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = MspnAttributionConfig(
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        degree_channels=args.degree_channels,
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
    sources = load_attribution_sources(
        args.atlas_root, args.e44_root, args.e45_root, args.e46_root
    )
    source_checks = validate_attribution_sources(sources)
    _write_progress(progress_path, "sources_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43/E44/E45/E46 source validation failed")
    contract = measure_mspn_contract(config, sources["atlas"])
    _write_progress(progress_path, "model_contract_complete", contract)
    matrix = train_attribution_matrix(config, sources["atlas"])
    for row in matrix["rows"]:
        _write_progress(
            progress_path,
            "matrix_row_complete",
            {
                "row_id": row["row_id"],
                "validation_auc": row["validation_auc"],
                "training_performed": row["training_performed"],
            },
        )
    gate = adjudicate_e47(config, source_checks, contract, matrix)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_mspn_attribution",
        "config": serializable_config(config),
        "source_hashes": {
            **sources["atlas"]["source_hashes"],
            **sources["source_hashes"],
            **sources["e46_hashes"],
        },
        "forbidden_precomputed_features_used": False,
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
