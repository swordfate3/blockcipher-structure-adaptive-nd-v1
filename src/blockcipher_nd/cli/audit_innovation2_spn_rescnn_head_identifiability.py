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
from blockcipher_nd.tasks.innovation2.spn_rescnn_head_identifiability import (
    SpnResCnnHeadIdentifiabilityConfig,
    audit_spn_rescnn_head_identifiability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit final-routing identifiability in the OPC1 global head."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default = SpnResCnnHeadIdentifiabilityConfig()
    config = SpnResCnnHeadIdentifiabilityConfig(
        run_id=args.run_id or default.run_id
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "neural_training": False},
    )
    audit = audit_spn_rescnn_head_identifiability(config)
    _write_jsonl(args.output_root / "results.jsonl", audit["rows"])
    _write_json(args.output_root / "metadata.json", audit["metadata"])
    _write_json(args.output_root / "summary.json", audit["summary"])
    _write_json(args.output_root / "gate.json", audit["gate"])
    _write_progress(
        progress_path,
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
