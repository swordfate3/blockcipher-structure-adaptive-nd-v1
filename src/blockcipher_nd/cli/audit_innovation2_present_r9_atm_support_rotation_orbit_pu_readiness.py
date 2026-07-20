from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_atm_support_rotation_orbit_pu_readiness import (
    RUN_ID,
    SupportRotationOrbitPuConfig,
    adjudicate_support_rotation_orbit_readiness,
    build_support_rotation_orbit_audit,
    result_rows,
    serializable_config,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    load_relation_groups,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E98-C PRESENT r9 support-and-rotation-orbit PU readiness."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--e98b-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SupportRotationOrbitPuConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "remote": False})
    groups = load_relation_groups(args.results_root)
    audit = build_support_rotation_orbit_audit(groups, config)
    _write_progress(progress, "support_rotation_components_built", audit["metrics"])
    e98b_gate = json.loads(args.e98b_gate.read_text(encoding="utf-8"))
    gate = adjudicate_support_rotation_orbit_readiness(
        config,
        audit=audit,
        e98b_gate=e98b_gate,
        e98b_gate_hash=_sha256(args.e98b_gate),
    )
    rows = result_rows(config, audit, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_atm_support_rotation_orbit_pu_readiness",
        "experiment": "e98-c",
        "config": serializable_config(config),
        "cipher_round_function": "PRESENT",
        "key_model": "independent 64-bit round keys",
        "rounds": 9,
        "training_performed": False,
        "remote_scale": False,
        "candidate_semantics": "unlabeled, not negative",
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "groups": audit["group_rows"],
        "components": audit["component_rows"],
        "fold_audit": audit["fold_rows"],
        "ranking_baselines": audit["baseline_rows"],
        "gate": gate,
    }
    _write_csv(args.output_root / "groups.csv", audit["group_rows"])
    _write_csv(args.output_root / "components.csv", audit["component_rows"])
    _write_csv(args.output_root / "fold_audit.csv", audit["fold_rows"])
    _write_csv(args.output_root / "ranking_baselines.csv", audit["baseline_rows"])
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
