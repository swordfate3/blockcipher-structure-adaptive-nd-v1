from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_atm_basis_merge_source_audit import (
    RUN_ID,
    AtmBasisMergeAuditConfig,
    adjudicate_basis_merge_audit,
    audit_relation_bases,
    audit_source_contract,
    build_split_coverage,
    result_rows,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E98-A PRESENT r9 ATM basis merge and split coverage."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--paper-text", required=True, type=Path)
    parser.add_argument("--e98-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AtmBasisMergeAuditConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "remote": False})

    source = audit_source_contract(
        args.atm_root,
        args.paper_text,
        actual_commit=_git_head(args.atm_root),
    )
    _write_progress(progress, "source_contract_replayed", source["checks"])
    results_root = args.atm_root / "Ciphers/PRESENT/Results"
    relations = audit_relation_bases(results_root)
    _write_progress(progress, "gf2_dependencies_recovered", relations["metrics"])
    splits = build_split_coverage(source, results_root)
    e98_gate = json.loads(args.e98_gate.read_text(encoding="utf-8"))
    gate = adjudicate_basis_merge_audit(
        config,
        source_audit=source,
        relation_audit=relations,
        split_rows=splits,
        e98_gate=e98_gate,
        e98_gate_hash=_sha256(args.e98_gate),
    )
    rows = result_rows(config, relations, splits, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_basis_merge_source_audit",
        "experiment": "e98-a",
        "config": serializable_config(config),
        "source_commit": source["actual_commit"],
        "cipher_round_function": "PRESENT",
        "key_model": "independent 64-bit round keys",
        "rounds": 9,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_audit": _serializable_source_audit(source),
        "file_ranks": relations["file_rows"],
        "dependencies": relations["dependency_rows"],
        "dependency_members": relations["member_rows"],
        "split_coverage": splits,
        "gate": gate,
    }

    _write_json(args.output_root / "source_hashes.json", source["source_hashes"])
    _write_csv(args.output_root / "file_ranks.csv", relations["file_rows"])
    _write_csv(args.output_root / "dependencies.csv", relations["dependency_rows"])
    _write_csv(args.output_root / "dependency_members.csv", relations["member_rows"])
    _write_csv(args.output_root / "split_coverage.csv", splits)
    _write_jsonl(args.output_root / "results.jsonl", rows)
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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable_source_audit(source: dict[str, Any]) -> dict[str, Any]:
    return {
        **source,
        "saved_dimensions": {
            "-".join(str(part) for part in split): dimension
            for split, dimension in source["saved_dimensions"].items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
