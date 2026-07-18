from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_round_slice_direction_attribution import (
    RoundSliceDirectionConfig,
    adjudicate_round_slice_direction,
    evaluate_round_slice_direction,
    load_round_slice_sources,
    result_rows_for_round_slice,
    serializable_config,
    validate_round_slice_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E72 round-slice direction audit.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--seed1-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RoundSliceDirectionConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False})
    sources = load_round_slice_sources(
        args.profile_root,
        args.atlas_root,
        args.seed0_root,
        args.seed1_root,
    )
    source_checks = validate_round_slice_sources(sources)
    if not all(
        value for value in source_checks.values() if isinstance(value, bool)
    ):
        raise ValueError("E65/E67/E68 source validation failed")
    _write_progress(progress, "sources_validated", source_checks)
    evaluation = evaluate_round_slice_direction(config, sources)
    gate = adjudicate_round_slice_direction(config, source_checks, evaluation)
    results = result_rows_for_round_slice(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_round_slice_direction_attribution",
        "config": serializable_config(config),
        "source_hashes": {
            "profile": sources["profile"]["source_hashes"],
            "checkpoints": sources["hashes"],
        },
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", results)
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
