from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.speck_hwang_topology_pairs import (
    SpeckHwangTopologyPairConfig,
    collect_topology_pair_rows,
    evaluate_topology_pairs,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_positions import (
    load_phase_c_position_baselines,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E27-N SPECK ROR7-to-addition topology pairs and controls."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase-c-root", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=1 << 24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = SpeckHwangTopologyPairConfig(
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    baseline = load_phase_c_position_baselines(config, phase_c_root=args.phase_c_root)
    if args.readiness_only:
        payload = {
            "run_id": config.run_id,
            "status": "pass",
            "phase_c_anchor_and_control_verified": True,
            "phase_c_keys": len(baseline["keys"]),
            "families": 2,
            "pairs_per_family": 16,
            "screen_keys": config.screen_keys,
            "max_per_family": config.max_per_family,
            "training_performed": False,
        }
        _write_json(args.output_root / "readiness.json", payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    import torch

    progress_path = args.output_root / "progress.jsonl"
    started: dict[tuple[str, int, str, int], float] = {}
    timing = _load_timing(progress_path)

    def callback(event: str, payload: dict[str, Any]) -> None:
        identity = (
            str(payload.get("family", "unknown")),
            int(payload.get("lane", -1)),
            str(payload.get("phase", "unknown")),
            int(payload.get("key_index", -1)),
        )
        if event == "speck_parity_row_start":
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started[identity] = time.perf_counter()
        elif event == "speck_parity_row_done" and identity in started:
            torch.cuda.synchronize(args.device)
            evidence = {
                "elapsed_seconds": time.perf_counter() - started[identity],
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            }
            timing[identity] = evidence
            payload = {**payload, **evidence}
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "families": ["ror7_add_aligned", "offset_minus_one"],
            "pairs_per_family": 16,
            "screen_keys": 8,
            "max_per_family": 4,
            "assignments_per_exact_row": 1 << 30,
            "training_performed": False,
        },
    )
    collected = collect_topology_pair_rows(
        config,
        phase_c_root=args.phase_c_root,
        cache_root=args.output_root / "cache",
        progress_callback=callback,
    )
    result = evaluate_topology_pairs(
        config,
        keys=collected["keys"],
        screen_parity_rows=collected["screen_parity_rows"],
        candidates=collected["candidates"],
        selected=collected["selected"],
        selected_specs=collected["selected_specs"],
        validation_parity_rows=collected["validation_parity_rows"],
        baseline_valid=True,
        caches_completed=collected["completed"],
        resume_rows_generated=collected["resume_rows_generated"],
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        timing_rows=len(timing),
    )
    result["gate"]["runtime"] = {
        "timed_rows": len(timing),
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in timing.values()),
        "max_peak_memory_bytes": max(
            (int(row["peak_memory_bytes"]) for row in timing.values()), default=0
        ),
    }
    baseline_output = args.output_root / "baseline_phase_c"
    for role in ("anchor", "control"):
        role_output = baseline_output / role
        role_output.mkdir(parents=True, exist_ok=True)
        for name, key in (
            ("parity_rows.npy", "parity_path"),
            ("metadata.json", "metadata_path"),
            ("completed.npy", "completed_path"),
        ):
            shutil.copy2(collected["baseline"][role][key], role_output / name)
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    _write_key_csv(args.output_root / "keys.csv", collected["keys"])
    np.save(args.output_root / "screen_parity_rows.npy", collected["screen_parity_rows"])
    np.save(
        args.output_root / "validation_parity_rows.npy",
        collected["validation_parity_rows"],
    )
    _write_json(
        args.output_root / "selected_candidates.json",
        {key: list(value) for key, value in collected["selected"].items()},
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": config.run_id,
            "cache_metadata": collected["cache_metadata"],
            "first_rows_generated": collected["first_rows_generated"],
            "resume_rows_generated": collected["resume_rows_generated"],
            "completed": collected["completed"],
            "timing_rows": len(timing),
            "rows": result["rows"],
            "gate": result["gate"],
            "metadata": result["metadata"],
        },
    )
    _write_progress(
        progress_path,
        "run_done",
        {"status": result["gate"]["status"], "decision": result["gate"]["decision"]},
    )
    print(json.dumps({"gate": result["gate"], "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if result["gate"]["status"] == "fail" else 0


def _load_timing(path: Path) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    evidence: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    if not path.exists():
        return evidence
    for row in _read_jsonl(path):
        if row.get("event") == "speck_parity_row_done" and row.get("elapsed_seconds") is not None:
            identity = (
                str(row["family"]),
                int(row["lane"]),
                str(row["phase"]),
                int(row["key_index"]),
            )
            evidence[identity] = {
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "peak_memory_bytes": int(row["peak_memory_bytes"]),
            }
    return evidence


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_key_csv(path: Path, keys: tuple[int, ...]) -> None:
    rows = [
        {"key_index": index, "split": "discovery" if index < 32 else "validation", "key_hex": f"0x{key:016X}"}
        for index, key in enumerate(keys)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key_index", "split", "key_hex"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
