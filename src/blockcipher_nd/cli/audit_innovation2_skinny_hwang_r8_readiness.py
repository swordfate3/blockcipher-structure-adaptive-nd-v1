from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.audit_innovation2_skinny_hwang_readiness import (
    render_skinny_hwang_svg,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_r8_readiness import (
    SkinnyHwangR8ReadinessConfig,
    run_skinny_hwang_r8_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the SKINNY-64/64 r8 two-active-cell kernel fixture."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-keys", type=int, default=512)
    parser.add_argument("--validation-keys", type=int, default=256)
    parser.add_argument("--key-chunk-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SkinnyHwangR8ReadinessConfig(
        run_id=args.run_id,
        seed=args.seed,
        discovery_keys=args.discovery_keys,
        validation_keys=args.validation_keys,
        key_chunk_size=args.key_chunk_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "cipher": "SKINNY-64/64",
            "rounds": 8,
            "target_active_cells": [14, 15],
            "control_active_cells": [0, 1],
            "discovery_keys": args.discovery_keys,
            "validation_keys": args.validation_keys,
            "plaintexts_per_key_per_role": 256,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_skinny_hwang_r8_readiness_audit(
        config,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    np.save(args.output_root / "keys.npy", result["keys"], allow_pickle=False)
    np.save(
        args.output_root / "parity_rows.npy",
        result["parity_rows"],
        allow_pickle=False,
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    render_skinny_hwang_svg(
        result["rows"],
        result["gate"],
        args.output_root / "curves.svg",
        experiment_code="E21",
        rounds=8,
        target_label="cells14+15",
        control_label="cells0+1",
        expected_rank=63,
        expected_nullity=1,
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "target": result["rows"][0],
                "control": result["rows"][1],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str = "a",
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
