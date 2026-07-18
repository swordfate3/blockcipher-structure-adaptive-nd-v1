from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.skinny64_r4_only_sparse_profile_operator_readiness import (
    r4_only_sources,
)
from blockcipher_nd.tasks.innovation2.skinny64_true_ridge_sparse_residual import (
    Skinny64TrueRidgeResidualConfig,
    adjudicate_true_ridge_residual,
    build_true_ridge_bundle,
    load_e84_sources,
    measure_true_ridge_residual_contract,
    serializable_config,
    train_true_ridge_residual_matrix,
    validate_e84_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E84 SKINNY true-ridge-guided sparse residual readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--e83-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Skinny64TrueRidgeResidualConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"epochs": config.epochs, "seed": config.seed, "training": True},
    )
    full_sources = load_e84_sources(args.profile_root, args.e83_root)
    source_checks = validate_e84_sources(full_sources)
    if not all(source_checks.values()):
        raise ValueError("E82/E83 source validation failed")
    sources = r4_only_sources(full_sources)
    source_checks["r4_shape_is_96x64x13"] = sources["prefix_features"].shape == (
        96,
        64,
        13,
    )
    _write_progress(progress, "sources_validated", source_checks)
    ridge = build_true_ridge_bundle(sources)
    _write_progress(
        progress,
        "true_ridge_reproduced",
        {
            "train_auc": ridge["train_auc"],
            "validation_auc": ridge["validation_auc"],
        },
    )
    contract = measure_true_ridge_residual_contract(config, sources, ridge)
    _write_progress(progress, "model_contract_complete", contract)
    matrix = train_true_ridge_residual_matrix(
        config, sources, ridge, args.output_root
    )
    gate = adjudicate_true_ridge_residual(
        config, source_checks, contract, matrix
    )
    for row in matrix["rows"]:
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_skinny64_true_ridge_sparse_residual_readiness",
        "architecture": "True-Ridge-Guided Sparse Residual",
        "config": serializable_config(config),
        "profile_source_run_id": full_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": full_sources["source_hashes"],
        "e83_source_run_id": full_sources["e83_gate"].get("run_id"),
        "e83_source_hashes": full_sources["e83_hashes"],
        "checkpoint_transfer": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "ridge": {
            "train_auc": ridge["train_auc"],
            "validation_auc": ridge["validation_auc"],
            "train_standardization_only": ridge["train_standardization_only"],
            "ridge_lambda": ridge["ridge_lambda"],
        },
        "contract": contract,
        "rows": matrix["rows"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", matrix["rows"])
    _write_csv(args.output_root / "history.csv", matrix["history"])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
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
