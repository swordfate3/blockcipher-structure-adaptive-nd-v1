from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_identity_topology_residual_attribution import (
    RUN_ID,
    IdentityTopologyResidualConfig,
    adjudicate_identity_topology_residual,
    build_folds,
    result_rows,
    serializable_config,
    sha256,
    train_attribution_matrix,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    load_relation_groups,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E100 PRESENT r9 identity-preserving topology residual attribution."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--e99-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = IdentityTopologyResidualConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"device": args.device, "training": True, "remote": False})
    groups = load_relation_groups(args.results_root)
    folds = build_folds(groups)
    _write_progress(progress, "fold_pools_built", folds["metrics"])
    training = train_attribution_matrix(
        config,
        folds,
        device=args.device,
        progress_callback=lambda event, payload: _write_progress(progress, event, payload),
    )
    e99_gate = json.loads(args.e99_gate.read_text(encoding="utf-8"))
    gate = adjudicate_identity_topology_residual(
        config,
        fold_audit=folds,
        training=training,
        e99_gate=e99_gate,
        e99_gate_hash=sha256(args.e99_gate),
    )
    rows = result_rows(config, training, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_identity_topology_residual_attribution",
        "experiment": "e100",
        "config": serializable_config(config),
        "cipher_round_function": "PRESENT",
        "key_model": "independent 64-bit round keys",
        "rounds": 9,
        "device": args.device,
        "training_performed": True,
        "remote_scale": False,
        "candidate_semantics": "unlabeled, not negative",
        "checkpoint_selection": "final epoch; no test-selected checkpoint",
        "paired_control": "true-P versus cycle-conjugate wrong-P",
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "fold_audit": folds["fold_rows"],
        "aggregate_metrics": training["aggregate_rows"],
        "fold_metrics": training["fold_metrics"],
        "gate": gate,
    }
    _write_csv(args.output_root / "fold_audit.csv", folds["fold_rows"])
    _write_csv(args.output_root / "fold_metrics.csv", training["fold_metrics"])
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(progress, "run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


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
