from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    RUN_ID,
    PuRankingReadinessConfig,
    adjudicate_pu_readiness,
    audit_sources,
    build_ranking_audit,
    load_relation_groups,
    result_rows,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E98 PRESENT r9 positive-unlabeled ranking readiness."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PuRankingReadinessConfig(run_id=args.run_id)
    results_root = args.atm_root / "Ciphers/PRESENT/Results"
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "remote": False})

    actual_commit = _git_head(args.atm_root)
    source_audit = audit_sources(results_root, actual_commit=actual_commit)
    _write_progress(progress, "sources_replayed", source_audit["checks"])
    groups = load_relation_groups(results_root)
    ranking_audit = build_ranking_audit(
        groups,
        minimum_unlabeled_per_positive=config.minimum_unlabeled_per_positive,
    )
    _write_progress(progress, "ranking_pools_built", ranking_audit["metrics"])
    gate = adjudicate_pu_readiness(
        config,
        source_audit=source_audit,
        ranking_audit=ranking_audit,
    )
    rows = result_rows(config, ranking_audit, gate)
    pool_rows = [
        {
            "heldout_file": pool["heldout_file"],
            "positive_id": pool["positive_id"],
            "relation_size": pool["relation_size"],
            "unlabeled_count": pool["unlabeled_count"],
            "minimum_unlabeled_met": pool["minimum_unlabeled_met"],
        }
        for pool in ranking_audit["pools"]
    ]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_generalized_relation_pu_ranking_readiness",
        "experiment": "e98",
        "config": serializable_config(config),
        "cipher_round_function": "PRESENT",
        "key_model": "independent 64-bit round keys",
        "rounds": 9,
        "training_performed": False,
        "remote_scale": False,
        "source_commit": actual_commit,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_audit": source_audit,
        "folds": ranking_audit["folds"],
        "ranking_baselines": ranking_audit["baseline_rows"],
        "gate": gate,
    }

    _write_json(args.output_root / "source_hashes.json", source_audit["hashes"])
    _write_csv(args.output_root / "folds.csv", ranking_audit["folds"])
    _write_csv(args.output_root / "candidate_pools.csv", pool_rows)
    _write_csv(args.output_root / "ranking_baselines.csv", ranking_audit["baseline_rows"])
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


if __name__ == "__main__":
    raise SystemExit(main())
