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
from blockcipher_nd.tasks.innovation2.present_key_blind_target_stability import (
    PresentKeyBlindStabilityConfig,
    audit_present_key_blind_target_stability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit key-blind cross-key target stability for PRESENT r3 selected outputs."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument("--plaintexts", type=int, default=1024)
    parser.add_argument("--reference-keys", type=int, default=256)
    parser.add_argument("--evaluation-keys", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default = PresentKeyBlindStabilityConfig()
    config = PresentKeyBlindStabilityConfig(
        run_id=args.run_id or default.run_id,
        seed=args.seed,
        plaintexts=args.plaintexts,
        reference_keys=args.reference_keys,
        evaluation_keys=args.evaluation_keys,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, object]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "plaintexts": config.plaintexts,
            "reference_keys": config.reference_keys,
            "evaluation_keys": config.evaluation_keys,
            "neural_training": False,
        },
    )
    audit = audit_present_key_blind_target_stability(config, progress=progress)
    _write_jsonl(args.output_root / "results.jsonl", audit["rows"])
    _write_json(args.output_root / "metadata.json", audit["metadata"])
    _write_json(args.output_root / "summary.json", audit["summary"])
    _write_json(args.output_root / "gate.json", audit["gate"])
    progress(
        "run_done",
        {"status": audit["gate"]["status"], "decision": audit["gate"]["decision"]},
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
    return 0 if audit["gate"]["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
