from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_atm_resumable_search_runner_fixture import (
    RUN_ID,
    execute_resumable_runner_fixture,
    result_rows,
    sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E102 route-owned ATM resumable-search fixture gate."
    )
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--e101-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    e101_gate = json.loads(args.e101_gate.read_text(encoding="utf-8"))
    summary = execute_resumable_runner_fixture(
        args.output_root,
        actual_atm_commit=_git_head(args.atm_root),
        actual_search_sha256=sha256(args.atm_root / "Modelling/Search.py"),
        e101_gate=e101_gate,
        e101_gate_sha256=sha256(args.e101_gate),
    )
    gate = summary["gate"]
    metadata = {
        "run_id": RUN_ID,
        "task": "innovation2_present_atm_resumable_search_runner_fixture",
        "experiment": "e102",
        "device": "local_cpu",
        "source_commit": summary["config"]["source_commit"],
        "search_source_sha256": summary["config"]["search_source_sha256"],
        "cipher_round_function": "PRESENT",
        "key_model": "not instantiated; deterministic three-bit Avec fixture only",
        "training_performed": False,
        "present_high_round_search_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary["metadata"] = metadata
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_jsonl(args.output_root / "results.jsonl", result_rows(summary))
    _write_csv(args.output_root / "fixture_calls.csv", summary["call_rows"])
    _write_csv(args.output_root / "artifact_contract.csv", summary["artifact_rows"])
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {"training": False, "high_round_search": False, "remote": False},
    )
    _write_progress(
        progress,
        "fixture_protocol_evaluated",
        gate["metrics"],
    )
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
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
