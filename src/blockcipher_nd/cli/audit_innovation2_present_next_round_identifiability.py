from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_next_round_identifiability import (
    PresentNextRoundIdentifiabilityConfig,
    audit_present_next_round_identifiability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether full PRESENT current/next internal states reveal each "
            "round key deterministically."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=20_260_722)
    parser.add_argument("--master-keys", type=int, default=16)
    parser.add_argument("--heldout-states-per-round", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default = PresentNextRoundIdentifiabilityConfig()
    config = PresentNextRoundIdentifiabilityConfig(
        run_id=args.run_id or default.run_id,
        seed=args.seed,
        master_keys=args.master_keys,
        heldout_states_per_round=args.heldout_states_per_round,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "master_keys": config.master_keys,
            "rounds": config.rounds,
            "heldout_states_per_round": config.heldout_states_per_round,
            "neural_training": False,
        },
    )
    audit = audit_present_next_round_identifiability(
        config,
        progress=progress,
    )
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


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_artifact_manifest(output_root: Path) -> None:
    names = (
        "results.jsonl",
        "metadata.json",
        "summary.json",
        "gate.json",
        "progress.jsonl",
    )
    rows = []
    for name in names:
        path = output_root / name
        rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_json(output_root / "artifact_manifest.json", rows)


def _validate_artifact_manifest(output_root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    rows_valid = []
    for row in manifest:
        path = output_root / row["path"]
        rows_valid.append(
            path.is_file()
            and int(row["bytes"]) == path.stat().st_size
            and str(row["sha256"])
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    checks = {
        "five_core_artifacts_manifested": len(manifest) == 5,
        "all_manifest_paths_are_local_files": all(
            "/" not in str(row["path"]) and "\\" not in str(row["path"])
            for row in manifest
        ),
        "all_manifest_sizes_and_sha256_match": bool(rows_valid)
        and all(rows_valid),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "manifest_rows": len(manifest),
    }


if __name__ == "__main__":
    raise SystemExit(main())
