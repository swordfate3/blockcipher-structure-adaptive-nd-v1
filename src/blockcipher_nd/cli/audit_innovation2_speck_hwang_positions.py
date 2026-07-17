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

from blockcipher_nd.tasks.innovation2.speck_hwang_positions import (
    SpeckHwangPositionConfig,
    collect_position_parity_rows,
    evaluate_position_family,
    load_phase_c_position_baselines,
    verify_position_mapping_fixture,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the E27 SPECK32/64 adjacent fixed-position kernel family."
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
    config = SpeckHwangPositionConfig(
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        backend="torch_int32",
        device=args.device,
    )
    mapping_valid = verify_position_mapping_fixture()
    baseline = load_phase_c_position_baselines(config, phase_c_root=args.phase_c_root)
    if args.readiness_only:
        payload = {
            "run_id": config.run_id,
            "status": "pass" if mapping_valid else "fail",
            "mapping_fixture_valid": mapping_valid,
            "phase_c_anchor_and_control_verified": True,
            "phase_c_keys": len(baseline["keys"]),
            "position_pairs": 30,
            "screen_keys": config.screen_keys,
            "max_validation_candidates": config.max_validation_candidates,
            "training_performed": False,
        }
        _write_json(args.output_root / "readiness.json", payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if mapping_valid else 1

    import torch

    progress_path = args.output_root / "progress.jsonl"
    row_started: dict[tuple[int, str, int], float] = {}
    timing_evidence = _load_timing_evidence(progress_path)

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        identity = (
            int(payload.get("position_start", -1)),
            str(payload.get("phase", "unknown")),
            int(payload.get("key_index", -1)),
        )
        if event == "speck_parity_row_start":
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            row_started[identity] = time.perf_counter()
        elif event == "speck_parity_row_done" and identity in row_started:
            torch.cuda.synchronize(args.device)
            evidence = {
                "elapsed_seconds": time.perf_counter() - row_started[identity],
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            }
            timing_evidence[identity] = evidence
            payload = {**payload, **evidence}
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "position_pairs": 30,
            "screen_keys": config.screen_keys,
            "max_validation_candidates": config.max_validation_candidates,
            "assignments_per_exact_row": 1 << 30,
            "chunk_size": config.chunk_size,
            "backend": config.backend,
            "device": config.device,
            "training_performed": False,
        },
    )
    collected = collect_position_parity_rows(
        config,
        phase_c_root=args.phase_c_root,
        cache_root=args.output_root / "cache",
        progress_callback=progress_callback,
    )
    result = evaluate_position_family(
        config,
        keys=collected["keys"],
        anchor_words=collected["baseline"]["anchor"]["parity_rows"][1],
        control_words=collected["baseline"]["control"]["parity_rows"][0],
        screen_parity_rows=collected["screen_parity_rows"],
        screen_candidates=collected["screen_candidates"],
        selected_candidates=collected["selected_candidates"],
        validation_parity_rows=collected["validation_parity_rows"],
        baseline_valid=True,
        caches_completed=collected["completed"],
        resume_rows_generated=collected["resume_rows_generated"],
        mapping_fixture_valid=mapping_valid,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        timing_rows=len(timing_evidence),
    )
    result["gate"]["runtime"] = _summarize_runtime(timing_evidence)

    baseline_output = args.output_root / "baseline_phase_c"
    for role in ("anchor", "control"):
        role_output = baseline_output / role
        role_output.mkdir(parents=True, exist_ok=True)
        for name, source_key in (
            ("parity_rows.npy", "parity_path"),
            ("metadata.json", "metadata_path"),
            ("completed.npy", "completed_path"),
        ):
            shutil.copy2(collected["baseline"][role][source_key], role_output / name)

    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_csv(
        args.output_root / "kernel_basis.csv",
        result["basis_rows"],
        [
            "run_id", "role", "rounds", "split", "basis_index", "mask_hex",
            "mask_weight", "basis_valid", "position_start",
        ],
    )
    _write_key_csv(args.output_root / "keys.csv", collected["keys"])
    np.save(args.output_root / "screen_parity_rows.npy", result["screen_parity_rows"])
    np.save(
        args.output_root / "validation_parity_rows.npy",
        result["validation_parity_rows"],
    )
    np.save(args.output_root / "selected_candidates.npy", result["selected_candidates"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": config.run_id,
            "metadata": result["metadata"],
            "cache_metadata": collected["cache_metadata"],
            "first_rows_generated": collected["first_rows_generated"],
            "resume_rows_generated": collected["resume_rows_generated"],
            "completed": collected["completed"],
            "mapping_fixture_valid": mapping_valid,
            "timing_rows": len(timing_evidence),
            "rows": result["rows"],
            "gate": result["gate"],
        },
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": config.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(json.dumps({"gate": result["gate"], "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if result["gate"]["status"] == "fail" else 0


def _load_timing_evidence(path: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    evidence: dict[tuple[int, str, int], dict[str, Any]] = {}
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
        identity = (
            int(row["position_start"]),
            str(row["phase"]),
            int(row["key_index"]),
        )
        evidence[identity] = {
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "peak_memory_bytes": int(row["peak_memory_bytes"]),
        }
    return evidence


def _summarize_runtime(
    evidence: dict[tuple[int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    rows = list(evidence.values())
    return {
        "timed_rows": len(rows),
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
        "max_peak_memory_bytes": max(
            (int(row["peak_memory_bytes"]) for row in rows), default=0
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_key_csv(path: Path, keys: tuple[int, ...]) -> None:
    rows = [
        {
            "key_index": index,
            "split": "discovery" if index < 32 else "validation",
            "key_hex": f"0x{key:016X}",
        }
        for index, key in enumerate(keys)
    ]
    _write_csv(path, rows, ["key_index", "split", "key_hex"])


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
