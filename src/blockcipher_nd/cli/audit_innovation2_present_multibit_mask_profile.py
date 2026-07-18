from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_multibit_mask_profile_readiness import (
    MultibitMaskProfileConfig,
    adjudicate_multibit_profile,
    build_multibit_benchmark,
    build_multibit_feature_table,
    decompose_multibit_labels,
    evaluate_multibit_features,
    load_e43_multibit_source,
    result_rows_for_multibit,
    serializable_config,
    validate_multibit_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E69 PRESENT multi-bit mask profile readiness."
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
    config = MultibitMaskProfileConfig(
        run_id=args.run_id,
        mode=args.mode,
        attempts=args.attempts,
        ridge_lambda=args.ridge_lambda,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "mode": config.mode})
    source = load_e43_multibit_source(args.atlas_root)
    source_checks = validate_multibit_source(source, strict=config.mode == "audit")
    if not all(source_checks.values()):
        raise ValueError("E43 multi-bit source validation failed")
    benchmark = build_multibit_benchmark(source, attempts=config.attempts)
    decomposition = decompose_multibit_labels(source, benchmark)
    table = build_multibit_feature_table(source, benchmark)
    feature_reports = evaluate_multibit_features(config, table)
    gate = adjudicate_multibit_profile(
        config,
        source_checks,
        benchmark,
        decomposition,
        table,
        feature_reports,
    )
    results = result_rows_for_multibit(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_multibit_mask_profile_readiness",
        "config": serializable_config(config),
        "source_run_id": source["gate"].get("run_id"),
        "source_hashes": source["source_hashes"],
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "gate": gate,
    }
    _write_csv(args.output_root / "matched_multibit_contrast.csv", benchmark["rows"])
    _write_csv(args.output_root / "decomposition.csv", decomposition["rows"])
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
