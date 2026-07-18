from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.real_spn_pair_state_transfer_readiness import (
    RealSpnTransferAuditConfig,
    adjudicate_real_spn_transfer_readiness,
    build_label_readiness,
    load_label_sources,
    make_present64_fixture,
    measure_model_grid,
    measure_present64_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E42 real-SPN pair-state transfer readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outputs-root", default=Path("outputs"), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RealSpnTransferAuditConfig(run_id=args.run_id, device=args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "device": config.device, "training": False},
    )

    sources = load_label_sources(args.outputs_root)
    source_records = [source["record"] for source in sources.values()]
    _write_progress(
        progress_path,
        "label_sources_loaded",
        {"source_count": len(source_records)},
    )
    label_rows = build_label_readiness(sources)
    _write_progress(
        progress_path,
        "label_readiness_complete",
        {
            "label_families": len(label_rows),
            "ready_families": sum(row["train_ready"] for row in label_rows),
        },
    )

    fixture = make_present64_fixture()
    model_contract = measure_present64_contract(fixture)
    _write_progress(progress_path, "model_contract_complete", model_contract)
    model_rows = measure_model_grid(fixture, device=config.device)
    _write_progress(
        progress_path,
        "model_grid_complete",
        {
            "model_cases": len(model_rows),
            "successful_model_cases": sum(row["success"] for row in model_rows),
        },
    )
    gate = adjudicate_real_spn_transfer_readiness(
        config, source_records, label_rows, model_contract, model_rows
    )

    label_csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"checks"}
        }
        | {
            "failed_checks": ";".join(
                key for key, passed in row["checks"].items() if not passed
            )
        }
        for row in label_rows
    ]
    result = {
        "run_id": config.run_id,
        "task": "innovation2_real_spn_pair_state_transfer_readiness",
        "status": gate["status"],
        "decision": gate["decision"],
        "ready_label_family_count": gate.get("metrics", {}).get(
            "ready_label_family_count", 0
        ),
        "model_ready": gate.get("metrics", {}).get("model_ready", False),
        "successful_model_cases": gate.get("metrics", {}).get(
            "successful_model_cases", 0
        ),
        "model_cases": len(model_rows),
        "training_performed": False,
        "remote_scale": False,
    }
    summary = {
        "run_id": config.run_id,
        "source_records": source_records,
        "label_rows": label_rows,
        "model_contract": model_contract,
        "model_rows": model_rows,
        "gate": gate,
        "result": result,
    }
    _write_json(args.output_root / "label_sources.json", {"sources": source_records})
    _write_csv(args.output_root / "label_readiness.csv", label_csv_rows)
    _write_csv(args.output_root / "model_memory.csv", model_rows)
    _write_jsonl(args.output_root / "results.jsonl", [result])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
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
