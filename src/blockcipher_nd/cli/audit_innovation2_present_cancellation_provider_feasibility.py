from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_cancellation_provider_feasibility import (
    RUN_ID,
    CancellationProviderConfig,
    adjudicate_provider_feasibility,
    build_provider_portfolio,
    load_sources,
    select_frozen_panel,
    serializable_config,
    source_hashes,
    validate_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E97 PRESENT cancellation-aware strict-label provider feasibility."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = CancellationProviderConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "remote": False})

    sources = load_sources(args.outputs_root)
    source_checks = validate_sources(sources)
    _write_progress(progress, "sources_replayed", source_checks)
    labels_path = (
        args.outputs_root
        / "local_audits/i2_present_r5_strict_label_provider_coverage_20260718/labels.csv"
    )
    panel = select_frozen_panel(labels_path)
    _write_progress(progress, "query_panel_frozen", {"queries": len(panel)})
    providers = build_provider_portfolio(sources)
    gate = adjudicate_provider_feasibility(config, source_checks, panel, providers)

    provider_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_cancellation_provider_feasibility",
            "result_kind": "provider",
            "status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
            **row,
        }
        for row in providers
    ]
    panel_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_cancellation_provider_feasibility",
            "result_kind": "query",
            "status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
            **row,
        }
        for row in panel
    ]
    result_rows = provider_rows + panel_rows
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_cancellation_provider_feasibility",
        "experiment": "e97",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "rounds": 5,
        "query_count": len(panel),
        "provider_count": len(providers),
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "gate": gate,
        "result_rows": result_rows,
    }

    _write_json(args.output_root / "source_hashes.json", source_hashes(sources))
    _write_csv(args.output_root / "query_panel.csv", panel)
    _write_csv(args.output_root / "provider_portfolio.csv", providers)
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
