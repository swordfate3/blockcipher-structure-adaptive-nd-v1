from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_next_round_identifiability import (
    _validate_artifact_manifest,
    _write_artifact_manifest,
    _write_json,
    _write_jsonl,
    _write_progress,
)
from blockcipher_nd.tasks.innovation2.present_next_round_selected8_identifiability import (
    PresentSelected8IdentifiabilityConfig,
    audit_present_selected8_identifiability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether complete PRESENT current states and eight selected "
            "next-state bits reveal four round-key nibbles."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--master-keys", type=int, default=16)
    parser.add_argument("--heldout-states-per-round", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default = PresentSelected8IdentifiabilityConfig()
    config = PresentSelected8IdentifiabilityConfig(
        run_id=args.run_id or default.run_id,
        seed=args.seed,
        master_keys=args.master_keys,
        heldout_states_per_round=args.heldout_states_per_round,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, object]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "master_keys": config.master_keys,
            "rounds": config.rounds,
            "calibration_pairs": config.calibration_pairs,
            "heldout_states_per_round": config.heldout_states_per_round,
            "neural_training": False,
        },
    )
    audit = audit_present_selected8_identifiability(config, progress=progress)
    _write_jsonl(args.output_root / "results.jsonl", audit["rows"])
    _write_json(args.output_root / "metadata.json", audit["metadata"])
    _write_json(args.output_root / "summary.json", audit["summary"])
    _write_json(args.output_root / "gate.json", audit["gate"])
    progress(
        "run_done",
        {
            "status": audit["gate"]["status"],
            "decision": audit["gate"]["decision"],
        },
    )
    _write_artifact_manifest(args.output_root)
    _write_json(
        args.output_root / "validation.json",
        _validate_artifact_manifest(args.output_root),
    )
    print(
        json.dumps(
            {"gate": audit["gate"], "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 0 if audit["gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
