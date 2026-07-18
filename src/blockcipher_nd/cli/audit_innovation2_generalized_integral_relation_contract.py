from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import (
    audit_atm_results,
)
from blockcipher_nd.tasks.innovation2.generalized_integral_relation_contract import (
    GeneralizedRelationContractConfig,
    audit_relation_overlap,
    evaluate_generalized_relation_contract,
    inspect_present_key_model,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E56 ATM generalized-integral relation label contract."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--minimum-positives", type=int, default=256)
    parser.add_argument("--minimum-negatives", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = GeneralizedRelationContractConfig(
        run_id=args.run_id,
        mode=args.mode,
        rounds=args.rounds,
        minimum_positives=args.minimum_positives,
        minimum_negatives=args.minimum_negatives,
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
        },
    )
    actual_commit = _git_head(args.atm_root)
    results_root = args.atm_root / "Ciphers/PRESENT/Results"
    atm_audit = audit_atm_results(results_root)
    key_model = inspect_present_key_model(args.atm_root)
    overlap = audit_relation_overlap(results_root)
    _write_progress(
        progress,
        "source_and_relations_ready",
        {
            "actual_commit": actual_commit,
            "deduplicated_relations": overlap["metrics"]["deduplicated_relations"],
            "common_relations": overlap["metrics"]["relations_common_to_all_files"],
            "negative_relations": overlap["metrics"][
                "proven_key_dependent_negative_relations"
            ],
            "key_model": key_model["key_model"],
        },
    )
    evaluation = evaluate_generalized_relation_contract(
        config,
        actual_commit=actual_commit,
        atm_audit=atm_audit,
        key_model=key_model,
        overlap=overlap,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_generalized_integral_relation_contract",
        "config": serializable_config(config),
        "cipher_round_function": "PRESENT",
        "executed_rounds": config.rounds,
        "actual_atm_commit": actual_commit,
        "external_source_root": str(args.atm_root),
        "external_files_copied_into_project": False,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    source_contract = {
        "actual_commit": actual_commit,
        "expected_commit": config.expected_commit,
        "key_model": key_model,
        "atm_checks": atm_audit["checks"],
        "atm_semantic_checks": atm_audit["semantic_checks"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_contract": source_contract,
        "atm": atm_audit,
        "relation_overlap": overlap,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_json(args.output_root / "source_contract.json", source_contract)
    _write_json(args.output_root / "relation_overlap.json", overlap)
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
