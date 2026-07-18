from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_unit_balance_profile_readiness import (
    UnitBalanceProfileConfig,
    adjudicate_unit_profile,
    build_profile_feature_table,
    build_unit_profile_benchmark,
    evaluate_profile_features,
    load_e43_unit_source,
    result_rows_for_profile,
    serializable_config,
    validate_unit_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E65 PRESENT unit-output balance profile readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--attempts", type=int, default=64)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = UnitBalanceProfileConfig(
        run_id=args.run_id,
        mode=args.mode,
        attempts=args.attempts,
        ridge_lambda=args.ridge_lambda,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(progress, "run_start", {"training": False, "mode": config.mode})
    source = load_e43_unit_source(args.atlas_root)
    source_checks = validate_unit_source(source, strict=config.mode == "audit")
    _write_progress(progress, "source_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43 unit source validation failed")
    benchmark = build_unit_profile_benchmark(source, attempts=config.attempts)
    table = build_profile_feature_table(source, benchmark)
    reports = evaluate_profile_features(config, table)
    gate = adjudicate_unit_profile(config, source_checks, benchmark, table, reports)
    results = result_rows_for_profile(config, gate, reports)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_unit_balance_profile_readiness",
        "config": serializable_config(config),
        "source_run_id": source["gate"].get("run_id"),
        "source_hashes": source["source_hashes"],
        "profile_semantics": (
            "64 unit-output universal-balance labels per active structure; "
            "unknown/unselected entries remain masked as -1"
        ),
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "reports": reports,
        "gate": gate,
    }
    np.save(args.output_root / "profile_targets.npy", benchmark["profile_targets"])
    np.save(args.output_root / "profile_observed.npy", benchmark["profile_observed"])
    _write_csv(args.output_root / "matched_unit_contrast.csv", benchmark["rows"])
    _write_csv(args.output_root / "features.csv", _flatten_feature_rows(table))
    _write_jsonl(args.output_root / "results.jsonl", results)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _flatten_feature_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, base in enumerate(table["rows"]):
        output = dict(base)
        for family, matrix in table["matrices"].items():
            for column in range(matrix.shape[1]):
                output[f"{family}_{column:02d}"] = float(matrix[row_index, column])
        rows.append(output)
    return rows


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
