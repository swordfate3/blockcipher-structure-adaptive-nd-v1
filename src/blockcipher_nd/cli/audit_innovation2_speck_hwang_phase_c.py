from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    SpeckHwangPhaseCConfig,
    collect_phase_c_parity_rows,
    evaluate_phase_c,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Innovation 2 SPECK32/64 Hwang Phase C with 32 discovery keys, "
            "32 fresh validation keys, and a fixed-position control."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-keys", type=int, default=32)
    parser.add_argument("--validation-keys", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1 << 24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    config = SpeckHwangPhaseCConfig(
        run_id=args.run_id,
        seed=args.seed,
        discovery_keys=args.discovery_keys,
        validation_keys=args.validation_keys,
        chunk_size=args.chunk_size,
        backend="torch_int32",
        device=args.device,
    )
    import torch

    row_started: dict[tuple[str, int, int], float] = {}
    timing_evidence = _load_timing_evidence(progress_path)

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        identity = (
            str(payload.get("role", "unknown")),
            int(payload.get("rounds", -1)),
            int(payload.get("key_index", -1)),
        )
        if event == "speck_parity_row_start":
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            row_started[identity] = time.perf_counter()
        elif event == "speck_parity_row_done" and identity in row_started:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
                peak_memory = int(torch.cuda.max_memory_allocated(args.device))
            else:
                peak_memory = 0
            evidence = {
                "elapsed_seconds": time.perf_counter() - row_started[identity],
                "peak_memory_bytes": peak_memory,
            }
            timing_evidence[identity] = evidence
            payload = {**payload, **evidence}
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "seed": config.seed,
            "discovery_keys": config.discovery_keys,
            "validation_keys": config.validation_keys,
            "anchor_fixed_bits": [5, 6],
            "anchor_rounds": [6, 7],
            "control_fixed_bits": [0, 1],
            "control_rounds": [7],
            "assignments_per_key_round_role": 1 << 30,
            "expected_exact_rows": 192,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "training_performed": False,
        },
    )
    collected = collect_phase_c_parity_rows(
        config,
        cache_root=args.output_root / "cache",
        progress_callback=progress_callback,
    )
    result = evaluate_phase_c(
        config,
        keys=collected["keys"],
        anchor_parity_rows=collected["anchor_parity_rows"],
        control_parity_rows=collected["control_parity_rows"],
        completed=collected["completed"],
        resume_rows_generated=collected["resume_rows_generated"],
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        timing_rows=len(timing_evidence),
    )
    runtime = _summarize_runtime(timing_evidence)
    result["gate"]["runtime"] = runtime
    for row in result["rows"]:
        runtime_key = f"{row['role']}_r{row['rounds']}"
        row.update(runtime.get(runtime_key, {}))

    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_csv(
        args.output_root / "kernel_basis.csv",
        result["basis_rows"],
        fieldnames=[
            "run_id",
            "role",
            "rounds",
            "split",
            "basis_index",
            "mask_hex",
            "mask_weight",
            "basis_valid",
        ],
    )
    _write_key_csv(
        args.output_root / "keys.csv",
        collected["keys"],
        discovery_keys=config.discovery_keys,
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    summary = {
        "run_id": config.run_id,
        "metadata": result["metadata"],
        "cache_metadata": collected["cache_metadata"],
        "first_rows_generated": collected["first_rows_generated"],
        "resume_rows_generated": collected["resume_rows_generated"],
        "completed": collected["completed"],
        "timing_rows": len(timing_evidence),
        "runtime": runtime,
        "rows": result["rows"],
        "gate": result["gate"],
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": config.run_id,
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


def _load_timing_evidence(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    evidence: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not path.exists():
        return evidence
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "speck_parity_row_done":
            continue
        if row.get("elapsed_seconds") is None or row.get("peak_memory_bytes") is None:
            continue
        identity = (str(row["role"]), int(row["rounds"]), int(row["key_index"]))
        evidence[identity] = {
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "peak_memory_bytes": int(row["peak_memory_bytes"]),
        }
    return evidence


def _summarize_runtime(
    evidence: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (role, rounds, _), row in evidence.items():
        grouped.setdefault(f"{role}_r{rounds}", []).append(row)
    return {
        group: {
            "timed_rows": len(rows),
            "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "mean_elapsed_seconds": (
                sum(float(row["elapsed_seconds"]) for row in rows) / len(rows)
            ),
            "max_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rows),
            "max_peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
        }
        for group, rows in grouped.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError("CSV rows must be non-empty when fieldnames are omitted")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_key_csv(path: Path, keys: tuple[int, ...], *, discovery_keys: int) -> None:
    rows = [
        {
            "key_index": index,
            "split": "discovery" if index < discovery_keys else "validation",
            "key_hex": f"0x{key:016X}",
        }
        for index, key in enumerate(keys)
    ]
    _write_csv(path, rows)


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
