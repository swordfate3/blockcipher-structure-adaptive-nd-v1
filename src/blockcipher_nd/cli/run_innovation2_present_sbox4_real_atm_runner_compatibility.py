from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_sbox4_real_atm_runner_compatibility import (
    RUN_ID,
    execute_real_atm_runner_compatibility,
    result_rows,
    sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E103 real ATM PRESENT S-box compatibility gate."
    )
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--e102-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hard-cap-seconds", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hard_cap_seconds < 1:
        raise ValueError("hard cap must be positive")
    e102_gate = json.loads(args.e102_gate.read_text(encoding="utf-8"))
    summary = execute_real_atm_runner_compatibility(
        args.output_root,
        atm_root=args.atm_root,
        actual_atm_commit=_git_head(args.atm_root),
        e102_gate=e102_gate,
        e102_gate_sha256=sha256(args.e102_gate),
        hard_cap_seconds=args.hard_cap_seconds,
    )
    gate = summary["gate"]
    metadata = {
        "run_id": RUN_ID,
        "task": "innovation2_present_sbox4_real_atm_runner_compatibility",
        "experiment": "e103",
        "device": "local_cpu",
        "state_bits": 4,
        "rounds": 3,
        "split": [1, 1, 1],
        "limit": 64,
        "workers": 2,
        "key_model": "independent 4-bit keys in ATM split construction",
        "linear_layer": "identity wiring; no PRESENT P-layer",
        "training_performed": False,
        "present_r9_r10_search_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary["metadata"] = metadata
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "source_hashes.json", summary["source_hashes"])
    _write_json(args.output_root / "bitset_build.json", summary["build"])
    _write_jsonl(args.output_root / "results.jsonl", result_rows(summary))
    environment_rows = [
        {"section": section, "check": name, "passed": passed}
        for section in ("source_checks", "environment_checks")
        for name, passed in gate[section].items()
    ]
    _write_csv(args.output_root / "environment_contract.csv", environment_rows)
    relation_rows = [
        {"metric": name, "value": value}
        for name, value in summary["relation_audit"].items()
    ]
    _write_csv(args.output_root / "relation_space.csv", relation_rows)
    oracle_rows = [
        {"route": route, **row}
        for route, rows in (
            ("official_anchor", summary["anchor_candidate_calls"]),
            ("resumed_runner", summary["runner_candidate_calls"]),
        )
        for row in rows
    ]
    _write_csv(args.output_root / "oracle_calls.csv", oracle_rows)
    official_root = args.output_root / "official_anchor"
    official_root.mkdir(exist_ok=True)
    _write_json(
        official_root / "result.json",
        {"relations": summary["anchor_relations"]},
    )
    _write_json(
        official_root / "metrics.json",
        {
            "candidate_calls": gate["metrics"]["official_candidate_calls"],
            "internal": summary["anchor_internal"],
            "seconds": gate["metrics"]["official_anchor_seconds"],
        },
    )
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {"training": False, "present_high_round_search": False, "remote": False},
    )
    _write_progress(
        progress,
        "real_atm_compatibility_evaluated",
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
