from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_r10_atm_source_generation_resume_readiness import (
    RUN_ID,
    SourceGenerationResumeConfig,
    adjudicate_source_generation_readiness,
    audit_source_generation_contract,
    result_rows,
    serializable_config,
    sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E101 PRESENT r9/r10 ATM source-generation resume readiness."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--e100-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SourceGenerationResumeConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "search": False, "remote": False})
    audit = audit_source_generation_contract(
        args.atm_root,
        actual_commit=_git_head(args.atm_root),
    )
    _write_progress(progress, "source_generation_contract_audited", audit["metrics"])
    e100_gate = json.loads(args.e100_gate.read_text(encoding="utf-8"))
    gate = adjudicate_source_generation_readiness(
        config,
        audit=audit,
        e100_gate=e100_gate,
        e100_gate_hash=sha256(args.e100_gate),
    )
    rows = result_rows(config, audit, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_r10_atm_source_generation_resume_readiness",
        "experiment": "e101",
        "config": serializable_config(config),
        "source_commit": audit["actual_commit"],
        "cipher_round_function": "PRESENT",
        "key_model": "independent 64-bit round keys",
        "rounds_audited": [9, 10],
        "training_performed": False,
        "search_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": audit["checks"],
        "notebook_contract": _serializable_notebook(audit["notebook"]),
        "search_contract": audit["search"],
        "split_coverage": audit["split_rows"],
        "historical_costs": audit["cost_rows"],
        "resume_contract": audit["resume_rows"],
        "environment_contract": audit["environment_rows"],
        "gate": gate,
    }
    _write_json(args.output_root / "source_hashes.json", audit["hashes"])
    _write_csv(args.output_root / "split_coverage.csv", audit["split_rows"])
    _write_csv(args.output_root / "historical_costs.csv", audit["cost_rows"])
    _write_csv(args.output_root / "resume_contract.csv", audit["resume_rows"])
    _write_csv(args.output_root / "environment_contract.csv", audit["environment_rows"])
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(progress, "run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _serializable_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    return {
        **notebook,
        "rounds": {str(rounds): contract for rounds, contract in notebook["rounds"].items()},
    }


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
