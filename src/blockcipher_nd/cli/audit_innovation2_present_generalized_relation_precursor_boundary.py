from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_generalized_relation_precursor_boundary import (
    PrecursorBoundaryConfig,
    audit_relation_costs,
    evaluate_precursor_boundary,
    load_relations,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E57 PRESENT r9 generalized-relation precursor boundary."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--maximum-scalar-plaintexts", type=int, default=1 << 24)
    parser.add_argument("--expected-relations", type=int, default=470)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PrecursorBoundaryConfig(
        run_id=args.run_id,
        mode=args.mode,
        rounds=args.rounds,
        maximum_scalar_plaintexts=args.maximum_scalar_plaintexts,
        expected_relations=args.expected_relations,
    )
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
            "scalar_encryption_executed": False,
        },
    )
    actual_commit = _git_head(args.atm_root)
    relations = load_relations(args.atm_root / "Ciphers/PRESENT/Results")
    cost_audit = audit_relation_costs(relations)
    evaluation = evaluate_precursor_boundary(
        config,
        actual_commit=actual_commit,
        relations=relations,
        cost_audit=cost_audit,
    )
    gate = evaluation["gate"]
    _write_progress(
        progress,
        "precursor_boundary_ready",
        {
            "relations": len(relations),
            "minimum_plaintexts": evaluation["metrics"][
                "minimum_precursor_plaintexts_per_relation_key"
            ],
            "maximum_plaintexts": evaluation["metrics"][
                "maximum_precursor_plaintexts_per_relation_key"
            ],
            "scalar_cap": config.maximum_scalar_plaintexts,
        },
    )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_generalized_relation_precursor_boundary",
        "config": serializable_config(config),
        "cipher_round_function": "PRESENT",
        "rounds": config.rounds,
        "actual_atm_commit": actual_commit,
        "external_source_root": str(args.atm_root),
        "external_files_copied_into_project": False,
        "scalar_encryption_executed": False,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "relation_costs": cost_audit["rows"],
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "relation_costs.jsonl", cost_audit["rows"])
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
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


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


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
