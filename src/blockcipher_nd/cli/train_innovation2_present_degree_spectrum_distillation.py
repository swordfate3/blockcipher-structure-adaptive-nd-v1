from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_degree_spectrum_distillation import (
    DegreeSpectrumDistillationConfig,
    adjudicate_e49,
    build_teacher_bundle,
    load_e49_sources,
    measure_distillation_contract,
    serializable_config,
    teacher_metric_rows,
    train_distillation_matrix,
    validate_e49_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E49 PRESENT intermediate degree-spectrum readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--e45-root", required=True, type=Path)
    parser.add_argument("--e47-root", required=True, type=Path)
    parser.add_argument("--e48-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--degree-channels", type=int, default=9)
    parser.add_argument("--auxiliary-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DegreeSpectrumDistillationConfig(
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        degree_channels=args.degree_channels,
        auxiliary_scale=args.auxiliary_scale,
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
    sources = load_e49_sources(
        args.atlas_root, args.e45_root, args.e47_root, args.e48_root
    )
    source_checks = validate_e49_sources(sources)
    _write_progress(progress_path, "sources_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43/E45/E47/E48 source validation failed")
    teachers = build_teacher_bundle(sources["atlas"])
    contract = measure_distillation_contract(config, sources["atlas"], teachers)
    _write_progress(progress_path, "distillation_contract_complete", contract)
    matrix = train_distillation_matrix(config, sources["atlas"], teachers)
    for row in matrix["rows"]:
        _write_progress(
            progress_path,
            "matrix_row_complete",
            {
                "row_id": row["row_id"],
                "validation_auc": row["validation_auc"],
                "validation_teacher_normalized_mse": row[
                    "validation_teacher_normalized_mse"
                ],
                "training_performed": row["training_performed"],
            },
        )
    gate = adjudicate_e49(config, source_checks, contract, teachers, matrix)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_degree_spectrum_distillation",
        "config": serializable_config(config),
        "source_hashes": {
            **sources["atlas"]["source_hashes"],
            **sources["source_hashes"],
            **sources["e48_hashes"],
        },
        "teacher_target_source": "PRESENT S-box ANF and selected P-layer rounds 1-3",
        "teacher_features_used_as_balance_input": False,
        "final_round_oracle_used": False,
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
    _write_csv(args.output_root / "teacher_metrics.csv", teacher_metric_rows(matrix))
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
