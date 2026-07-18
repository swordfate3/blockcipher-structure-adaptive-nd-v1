from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_query_cone_sparse_anf_growth import (
    FROZEN_FIXTURE_IDS,
    SparseAnfGrowthConfig,
    calibrate_against_e53a,
    evaluate_sparse_growth_gate,
    freeze_query_manifest,
    run_sparse_query,
    serializable_config,
    validate_or_instantiate_label,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E55 PRESENT r3 query-cone exact sparse-ANF growth."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exact-summary", required=True, type=Path)
    parser.add_argument("--exact-fixtures", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--maximum-terms", type=int, default=5_000_000)
    parser.add_argument("--maximum-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-memory-bytes", type=int, default=4 * (1 << 30))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SparseAnfGrowthConfig(
        run_id=args.run_id,
        mode=args.mode,
        rounds=args.rounds,
        maximum_terms=args.maximum_terms,
        maximum_seconds=args.maximum_seconds,
        maximum_memory_bytes=args.maximum_memory_bytes,
    )
    source_summary = json.loads(args.exact_summary.read_text(encoding="utf-8"))
    fixtures = _read_jsonl(args.exact_fixtures)
    query_manifest = freeze_query_manifest(fixtures)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "training": False,
            "remote_scale": False,
            "planned_queries": len(query_manifest),
            "caps": {
                "terms": config.maximum_terms,
                "seconds": config.maximum_seconds,
                "memory_bytes": config.maximum_memory_bytes,
            },
        },
    )
    calibration = calibrate_against_e53a(fixtures, config=config)
    _write_progress(
        progress,
        "calibration_done",
        {
            "checks": calibration["checks"],
            "completed_rows": calibration["completed_rows"],
            "expected_rows": calibration["expected_rows"],
        },
    )

    query_rows: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    calibration_passed = all(calibration["checks"].values())
    stop_reason: str | None = None
    for query in query_manifest:
        if not calibration_passed:
            stop_reason = "calibration_failed"
        if stop_reason is not None:
            query_rows.append(
                {
                    **query,
                    "status": "skipped",
                    "cap_reason": stop_reason,
                    "label": "unknown",
                    "superpoly_monomials": None,
                    "superpoly_sha256": None,
                }
            )
            continue
        _write_progress(
            progress,
            "query_start",
            {"query_id": query["query_id"], "source_fixture_id": query["source_fixture_id"]},
        )
        row = run_sparse_query(query, config=config)
        if row["status"] == "completed":
            label_evidence = validate_or_instantiate_label(row, config=config)
            row.update(label_evidence)
            if row["label"] == "positive":
                certificates.append(_certificate_row(row))
            else:
                witnesses.append(_witness_row(row))
        else:
            stop_reason = str(row["cap_reason"])
        query_rows.append(row)
        _write_progress(
            progress,
            "query_done",
            {
                "query_id": row["query_id"],
                "status": row["status"],
                "label": row["label"],
                "cap_reason": row.get("cap_reason"),
                "superpoly_monomials": row.get("superpoly_monomials"),
                "elapsed_seconds": row.get("elapsed_seconds"),
            },
        )

    evaluation = evaluate_sparse_growth_gate(
        config,
        source_summary=source_summary,
        calibration=calibration,
        query_rows=query_rows,
    )
    gate = evaluation["gate"]
    clean_rows = [{key: value for key, value in row.items() if key != "_superpoly"} for row in query_rows]
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_query_cone_sparse_anf_growth",
            "query_id": row["query_id"],
            "rounds": config.rounds,
            "query_type": row["query_type"],
            "status": row["status"],
            "label": row["label"],
            "superpoly_monomials": row.get("superpoly_monomials"),
            "maximum_observed_terms": row.get("maximum_observed_terms"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "cap_reason": row.get("cap_reason"),
            "gate_status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
        }
        for row in clean_rows
    ]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_query_cone_sparse_anf_growth",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "executed_rounds": config.rounds,
        "target_rounds": 5,
        "symbolic_variables": {"plaintext": 64, "master_key": 80, "total": 144},
        "source_summary": str(args.exact_summary),
        "source_fixtures": str(args.exact_fixtures),
        "training_performed": False,
        "remote_scale": False,
        "five_round_subset_executed": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "query_manifest": query_manifest,
        "calibration": calibration,
        "query_rows": clean_rows,
        "certificates": certificates,
        "witnesses": witnesses,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": result_rows,
    }
    _write_json(args.output_root / "query_manifest.json", {"queries": query_manifest})
    _write_jsonl(args.output_root / "calibration.jsonl", calibration["rows"])
    _write_jsonl(args.output_root / "certificates.jsonl", certificates)
    _write_jsonl(args.output_root / "witnesses.jsonl", witnesses)
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _certificate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "certificate": "exact_full_superpoly_is_zero",
        "active_bits": row["active_bits"],
        "output_mask_hex": row["output_mask_hex"],
        "superpoly_monomials": row["superpoly_monomials"],
        "superpoly_sha256": row["superpoly_sha256"],
        "scalar_rechecks": row["scalar_rechecks"],
    }


def _witness_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "certificate": "exact_nonzero_superpoly_with_scalar_witness",
        "active_bits": row["active_bits"],
        "output_mask_hex": row["output_mask_hex"],
        "superpoly_monomials": row["superpoly_monomials"],
        "superpoly_sha256": row["superpoly_sha256"],
        "witness_key_hex": row["witness_key_hex"],
        "witness_offset_hex": row["witness_offset_hex"],
        "witness_parity": row["witness_parity"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


if __name__ == "__main__":
    raise SystemExit(main())
