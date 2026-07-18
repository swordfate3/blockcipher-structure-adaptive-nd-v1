from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    CertificateAttributionConfig,
    adjudicate_e45,
    build_feature_table,
    evaluate_feature_families,
    load_sources,
    serializable_config,
    validate_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E45 PRESENT certificate-complexity feature attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--neural-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = CertificateAttributionConfig(
        run_id=args.run_id, mode=args.mode, ridge_lambda=args.ridge_lambda
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "mode": config.mode, "training": False},
    )
    sources = load_sources(args.atlas_root, args.neural_root)
    source_checks = validate_sources(sources, strict=config.mode == "audit")
    _write_progress(progress_path, "sources_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43/E44 source validation failed")
    table = build_feature_table(sources)
    _write_progress(
        progress_path,
        "feature_table_complete",
        {
            "rows": len(table["rows"]),
            "feature_families": len(table["matrices"]),
        },
    )
    evaluation = evaluate_feature_families(config, table)
    gate = adjudicate_e45(config, source_checks, table, evaluation)
    feature_rows = _flatten_feature_rows(table)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_certificate_complexity_attribution",
        "config": serializable_config(config),
        "atlas_source_run_id": sources["atlas"]["gate"].get("run_id"),
        "neural_source_run_id": sources["neural_gate"].get("run_id"),
        "atlas_source_hashes": sources["atlas"]["source_hashes"],
        "neural_source_hashes": sources["neural_hashes"],
        "feature_names": {
            family: list(names) for family, names in table["feature_names"].items()
        },
        "training_performed": False,
        "oracle_is_not_comparable_baseline": True,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "reports": evaluation["reports"],
        "gate": gate,
    }
    _write_csv(args.output_root / "features.csv", feature_rows)
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
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


def _flatten_feature_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, base in enumerate(table["rows"]):
        output = dict(base)
        for family, matrix in table["matrices"].items():
            for column, name in enumerate(table["feature_names"][family]):
                output[f"{family}__{name}"] = float(matrix[row_index, column])
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
