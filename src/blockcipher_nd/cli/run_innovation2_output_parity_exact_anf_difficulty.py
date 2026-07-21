from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.output_parity_exact_anf_difficulty import (
    RUN_ID,
    OutputParityAnfDifficultyConfig,
    adjudicate_exact_anf_difficulty,
    run_exact_anf_audit,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OP8 exact ANF output-parity difficulty audit."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--key-metadata", required=True, type=Path)
    parser.add_argument("--r1-gate", required=True, type=Path)
    parser.add_argument("--r2-gate", required=True, type=Path)
    parser.add_argument("--r3-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = OutputParityAnfDifficultyConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    key_metadata = _read_json(args.key_metadata)
    secret_key = int(str(key_metadata["secret_key_hex"]), 16)
    source_gates = {
        1: _read_json(args.r1_gate),
        2: _read_json(args.r2_gate),
        3: _read_json(args.r3_gate),
    }
    _write_progress(
        progress,
        "run_start",
        {
            "training": False,
            "sample_classification": False,
            "planned_functions": 48,
        },
    )
    rows = run_exact_anf_audit(config, secret_key=secret_key)
    for row in rows:
        _write_progress(
            progress,
            "function_done",
            {
                "rounds": row["rounds"],
                "mask_index": row["mask_index"],
                "status": row["status"],
                "exact_monomial_count": row.get("exact_monomial_count"),
            },
        )
    gate = adjudicate_exact_anf_difficulty(config, rows, source_gates)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_parity_prediction",
        "experiment": "op8_present_r1_r3_exact_anf_difficulty",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "secret_key_hex": f"{secret_key:020x}",
        "secret_key_source": str(args.key_metadata),
        "source_gates": {
            "r1": str(args.r1_gate),
            "r2": str(args.r2_gate),
            "r3": str(args.r3_gate),
        },
        "training": False,
        "sample_classification": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "gate": gate,
        "round_summaries": gate["round_summaries"],
        "result_rows": rows,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_csv(args.output_root / "round_summary.csv", gate["round_summaries"])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True)
    )
    return 1 if gate["status"] == "fail" else 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
