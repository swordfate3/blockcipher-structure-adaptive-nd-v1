from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    SPECK32_ACTIVE_BITS,
    SPECK32_FIXED_MASK,
    SpeckParityCacheConfig,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)


MAX_SECONDS_PER_ROW = 1800.0
MAX_PEAK_MEMORY_BYTES = 40 * 1024**3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Innovation 2 SPECK32 Hwang exact-structure single-key timing gate."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--key", type=lambda value: int(value, 0), default=0x1918111009080100)
    parser.add_argument("--chunk-size", type=int, default=1 << 24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    config = SpeckParityCacheConfig(
        run_id=args.run_id,
        rounds=(6, 7),
        keys=(args.key,),
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=0,
        chunk_size=args.chunk_size,
        backend="torch_int32",
        device=args.device,
    )
    import torch

    row_started: dict[int, float] = {}
    row_timings, peak_memory = _load_timing_evidence(progress_path)

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        rounds = int(payload.get("rounds", -1))
        if event == "speck_parity_row_start":
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            row_started[rounds] = time.perf_counter()
        elif event == "speck_parity_row_done" and rounds in row_started:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
                peak_memory[rounds] = int(torch.cuda.max_memory_allocated(args.device))
            else:
                peak_memory[rounds] = 0
            row_timings[rounds] = time.perf_counter() - row_started[rounds]
            payload = {
                **payload,
                "elapsed_seconds": row_timings[rounds],
                "peak_memory_bytes": peak_memory[rounds],
            }
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "rounds": [6, 7],
            "key": f"0x{args.key:016X}",
            "active_bits": 30,
            "assignments_per_key": 1 << 30,
            "chunk_size": args.chunk_size,
            "backend": "torch_int32",
            "device": args.device,
            "training_performed": False,
        },
    )
    first = run_cached_speck_parity_rows(
        config,
        cache_root=args.output_root / "cache",
        progress_callback=progress_callback,
    )
    second = run_cached_speck_parity_rows(
        config,
        cache_root=args.output_root / "cache",
        progress_callback=progress_callback,
    )
    rows = []
    for round_index, rounds in enumerate(config.rounds):
        parity_word = int(first["parity_rows"][round_index, 0])
        masks = hwang_speck_basis_masks(rounds)
        mask_checks = [((parity_word & mask).bit_count() % 2) == 0 for mask in masks]
        rows.append(
            {
                "run_id": args.run_id,
                "task": "innovation2_speck32_hwang_phase_b_single_key_timing",
                "rounds": rounds,
                "key": f"0x{args.key:016X}",
                "parity_word_hex": f"0x{parity_word:08X}",
                "paper_basis_masks": [f"0x{mask:08X}" for mask in masks],
                "paper_basis_valid_for_key": all(mask_checks),
                "paper_basis_direction_checks": mask_checks,
                "elapsed_seconds": row_timings.get(rounds),
                "peak_memory_bytes": peak_memory.get(rounds),
            }
        )
    gate = adjudicate_phase_b(
        run_id=args.run_id,
        rows=rows,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        device_name=(
            torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else "CPU"
        ),
        completed=bool(first["completed"].all()),
        resume_rows_generated=int(second["rows_generated"]),
        official_vector_matches=(
            Speck32_64(rounds=22, key=0x1918111009080100).encrypt(0x6574694C)
            == 0xA86842F2
        ),
        active_bits_exact=(
            SPECK32_ACTIVE_BITS == tuple(bit for bit in range(32) if bit not in {5, 6})
            and SPECK32_FIXED_MASK == 0x60
            and config.assignments == (1 << 30)
        ),
        parity_shape=tuple(first["parity_rows"].shape),
    )
    summary = {
        "run_id": args.run_id,
        "config": first["metadata"],
        "cache_status": first["cache_status"],
        "first_rows_generated": int(first["rows_generated"]),
        "resume_rows_generated": int(second["rows_generated"]),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": gate["device_name"],
        },
        "rows": rows,
        "gate": gate,
    }
    (args.output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress_path,
        "run_done",
        {"run_id": args.run_id, "status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 0 if gate["status"] != "fail" else 1


def adjudicate_phase_b(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    cuda_available: bool,
    device_count: int,
    device_name: str,
    completed: bool,
    resume_rows_generated: int,
    official_vector_matches: bool,
    active_bits_exact: bool,
    parity_shape: tuple[int, ...],
) -> dict[str, Any]:
    readiness_checks = {
        "official_speck32_vector_matches": official_vector_matches,
        "exact_wang_thirty_active_bit_structure": active_bits_exact,
        "cuda_available": cuda_available,
        "cuda_device_count_positive": device_count >= 1,
        "two_round_rows_present": len(rows) == 2 and {row.get("rounds") for row in rows} == {6, 7},
        "parity_shape_is_two_by_one": parity_shape == (2, 1),
        "cache_completed": completed,
        "resume_generates_zero_rows": resume_rows_generated == 0,
        "all_hwang_masks_valid_for_single_key": bool(rows)
        and all(bool(row.get("paper_basis_valid_for_key")) for row in rows),
        "timing_and_memory_evidence_present": bool(rows)
        and all(
            row.get("elapsed_seconds") is not None
            and row.get("peak_memory_bytes") is not None
            for row in rows
        ),
    }
    elapsed = [float(row["elapsed_seconds"]) for row in rows if row.get("elapsed_seconds") is not None]
    peaks = [int(row["peak_memory_bytes"]) for row in rows if row.get("peak_memory_bytes") is not None]
    performance_checks = {
        "max_seconds_per_row_at_most_1800": len(elapsed) == 2 and max(elapsed) <= MAX_SECONDS_PER_ROW,
        "max_peak_memory_at_most_40gib": len(peaks) == 2 and max(peaks) <= MAX_PEAK_MEMORY_BYTES,
    }
    if not all(readiness_checks.values()):
        status = "fail"
        decision = "innovation2_speck_hwang_phase_b_protocol_invalid"
        next_action = "repair CUDA, exact structure, cache, timing, or mask ownership"
    elif all(performance_checks.values()):
        status = "pass"
        decision = "innovation2_speck_hwang_phase_b_single_key_timing_ready"
        next_action = "preregister the minimum 32+32 fresh-key exact-kernel matrix"
    else:
        status = "hold"
        decision = "innovation2_speck_hwang_phase_b_direct_enumeration_not_scalable"
        next_action = "stop direct enumeration and rank Midori or algebraic cube-sum backends"
    return {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "performance_checks": performance_checks,
        "metrics": {
            "elapsed_seconds_by_round": {str(row["rounds"]): row.get("elapsed_seconds") for row in rows},
            "peak_memory_bytes_by_round": {str(row["rounds"]): row.get("peak_memory_bytes") for row in rows},
        },
        "device_name": device_name,
        "claim_scope": "remote one-key exact-2^30 timing and paper-mask readiness; not a kernel reproduction, multi-key validation, neural training, or paper-scale evidence",
        "next_action": {"action": next_action, "training": False, "remote_scale": False},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_timing_evidence(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    timings: dict[int, float] = {}
    peak_memory: dict[int, int] = {}
    if not path.exists():
        return timings, peak_memory
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "speck_parity_row_done":
            continue
        if row.get("elapsed_seconds") is None or row.get("peak_memory_bytes") is None:
            continue
        rounds = int(row["rounds"])
        timings[rounds] = float(row["elapsed_seconds"])
        peak_memory[rounds] = int(row["peak_memory_bytes"])
    return timings, peak_memory


if __name__ == "__main__":
    raise SystemExit(main())
