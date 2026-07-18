from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_support_identity_collision import (
    SupportIdentityAuditConfig,
    adjudicate_e48,
    build_support_identity_table,
    evaluate_support_identity,
    export_feature_rows,
    load_e48_sources,
    serializable_config,
    validate_e48_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E48 PRESENT support identity collisions and sketches."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--e45-root", required=True, type=Path)
    parser.add_argument("--e47-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SupportIdentityAuditConfig(run_id=args.run_id, mode=args.mode)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "training": False, "remote_scale": False},
    )
    sources = load_e48_sources(args.atlas_root, args.e45_root, args.e47_root)
    source_checks = validate_e48_sources(sources, strict=config.mode == "audit")
    _write_progress(progress_path, "sources_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError("E43/E45/E47 source validation failed")
    table = build_support_identity_table(config, sources)
    _write_progress(
        progress_path,
        "support_identity_table_complete",
        {"rows": len(table["rows"]), "sketches": len(table["sketches"])},
    )
    evaluation = evaluate_support_identity(config, table)
    gate = adjudicate_e48(config, source_checks, table, evaluation)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_support_identity_collision",
        "config": serializable_config(config),
        "source_hashes": {
            **sources["atlas"]["source_hashes"],
            **sources["source_hashes"],
        },
        "rademacher_sha256": table["rademacher_sha256"],
        "binary_projection_sha256": table["binary_projection_sha256"],
        "final_round_oracle_used": False,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "reports": evaluation["reports"],
        "collisions": evaluation["collisions"],
        "gate": gate,
    }
    _write_csv(args.output_root / "features.csv", export_feature_rows(table))
    _write_csv(args.output_root / "collision.csv", evaluation["collision_rows"])
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
