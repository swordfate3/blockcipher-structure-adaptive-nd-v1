from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    ProviderAuditConfig,
    audit_atm_results,
    evaluate_provider_contract,
    inspect_claasp_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E31 deterministic integral-label provider contract audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--claasp-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProviderAuditConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(progress_path, "run_start", {"run_id": config.run_id})
    actual_claasp_commit = _git_head(args.claasp_root)
    actual_atm_commit = _git_head(args.atm_root)
    claasp = inspect_claasp_contract(args.claasp_root)
    atm = audit_atm_results(args.atm_root / "Ciphers/PRESENT/Results")
    result = evaluate_provider_contract(
        config,
        claasp=claasp,
        atm=atm,
        actual_claasp_commit=actual_claasp_commit,
        actual_atm_commit=actual_atm_commit,
    )
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "summary.json", result["summary"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_progress(
        progress_path,
        "run_done",
        {
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(json.dumps({"gate": result["gate"], "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if result["gate"]["status"] == "fail" else 0


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
