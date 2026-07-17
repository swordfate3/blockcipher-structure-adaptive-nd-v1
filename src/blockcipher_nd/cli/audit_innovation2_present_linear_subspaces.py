from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_linear_subspace_diversity import (
    LinearSubspaceAuditConfig,
    evaluate_linear_subspaces,
    make_audit_keys,
    make_linear_subspaces,
    run_cached_subspace_parities,
    scalar_vectorized_fixture_matches,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E30 PRESENT-r7 linear-subspace kernel diversity readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--random-subspaces", type=int, default=32)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LinearSubspaceAuditConfig(
        run_id=args.run_id,
        mode=args.mode,
        random_subspaces=args.random_subspaces,
        keys=args.keys,
        key_chunk_size=args.key_chunk_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    structures = make_linear_subspaces(
        random_subspaces=config.random_subspaces,
        seed=config.subspace_seed,
    )
    keys = make_audit_keys(config)
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "structures": len(structures),
            "keys": len(keys),
            "plaintexts_per_structure": 1 << config.dimension,
            "training_performed": False,
        },
    )
    parity_rows: list[np.ndarray] = []
    completed: dict[str, bool] = {}
    first_rows: dict[str, int] = {}
    resumed_rows: dict[str, int] = {}
    for structure in structures:
        cache = args.output_root / "cache" / structure.structure_id
        first = run_cached_subspace_parities(
            config,
            structure,
            keys=keys,
            cache_root=cache,
            progress_callback=callback,
        )
        resumed = run_cached_subspace_parities(
            config,
            structure,
            keys=keys,
            cache_root=cache,
            progress_callback=callback,
        )
        parity_rows.append(first["parity_rows"])
        completed[structure.structure_id] = bool(first["completed"].all())
        first_rows[structure.structure_id] = int(first["rows_generated"])
        resumed_rows[structure.structure_id] = int(resumed["rows_generated"])

    parity_matrix = np.stack(parity_rows)
    result = evaluate_linear_subspaces(
        config,
        structures=structures,
        keys=keys,
        parity_rows=parity_matrix,
        completed=completed,
        resume_rows_generated=resumed_rows,
        scalar_vector_match=scalar_vectorized_fixture_matches(),
    )
    np.save(
        args.output_root / "bases.npy",
        np.asarray([structure.basis for structure in structures], dtype=np.uint64),
    )
    np.save(args.output_root / "parity_rows.npy", parity_matrix)
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": config.run_id,
            "first_rows_generated": first_rows,
            "resume_rows_generated": resumed_rows,
            "completed": completed,
            "gate": result["gate"],
            "metadata": result["metadata"],
        },
    )
    _write_keys(args.output_root / "keys.csv", keys)
    _write_progress(
        progress_path,
        "run_done",
        {
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(
        json.dumps(
            {"gate": result["gate"], "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 1 if result["gate"]["status"] == "fail" else 0


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


def _write_keys(path: Path, keys: tuple[int, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key_index", "split", "key_hex"])
        writer.writeheader()
        half = len(keys) // 2
        for index, key in enumerate(keys):
            writer.writerow(
                {
                    "key_index": index,
                    "split": "discovery" if index < half else "validation",
                    "key_hex": f"0x{key:020X}",
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
