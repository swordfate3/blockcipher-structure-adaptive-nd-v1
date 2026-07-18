from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.gift64_topology_interaction_readjudication import (
    Gift64TopologyInteractionConfig,
    adjudicate_topology_interaction,
    checkpoint_topology_counterfactuals,
    load_e76_source,
    load_gift_profile_sources,
    result_rows,
    r3_only_sources,
    serializable_config,
    topology_expanded_ridges,
    validate_e76_source,
    validate_gift_profile_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E77 GIFT topology-interaction readjudication."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--e76-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Gift64TopologyInteractionConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False})

    full_sources = load_gift_profile_sources(args.profile_root)
    profile_checks = validate_gift_profile_sources(full_sources)
    e76_source = load_e76_source(args.e76_root)
    e76_checks = validate_e76_source(e76_source)
    _write_progress(
        progress,
        "sources_validated",
        {"profile": profile_checks, "e76": e76_checks},
    )
    if not all(profile_checks.values()) or not all(e76_checks.values()):
        raise ValueError("E75/E76 source validation failed")

    ridges = topology_expanded_ridges(config, full_sources)
    sources = r3_only_sources(full_sources)
    counterfactuals = checkpoint_topology_counterfactuals(
        config, sources, e76_source
    )
    gate = adjudicate_topology_interaction(
        config,
        profile_checks,
        e76_checks,
        ridges,
        counterfactuals,
        e76_source,
    )
    rows = result_rows(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_gift64_topology_interaction_readjudication",
        "config": serializable_config(config),
        "profile_source_run_id": full_sources["profile_gate"].get("run_id"),
        "profile_source_hashes": full_sources["source_hashes"],
        "e76_source_run_id": e76_source["gate"].get("run_id"),
        "e76_source_hashes": e76_source["hashes"],
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "profile_checks": profile_checks,
        "e76_checks": e76_checks,
        "gate": gate,
        "result_rows": rows,
    }
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
