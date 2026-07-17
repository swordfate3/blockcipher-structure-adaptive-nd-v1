from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import (
    CONTROL_ACTIVE_BITS,
    SpeckHwangPhaseCConfig,
    evaluate_phase_c,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import SPECK32_ACTIVE_BITS


REQUIRED_FILES = (
    "SHA256SUMS",
    "gate.json",
    "metadata.json",
    "summary.json",
    "results.jsonl",
    "progress.jsonl",
    "keys.csv",
    "kernel_basis.csv",
    "source_expected_commit.txt",
    "cache/anchor/metadata.json",
    "cache/anchor/parity_rows.npy",
    "cache/anchor/completed.npy",
    "cache/control/metadata.json",
    "cache/control/parity_rows.npy",
    "cache/control/completed.npy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate and re-adjudicate a retrieved Innovation 2 "
            "SPECK32/64 Phase C archive."
        )
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation, local_gate = validate_phase_c_archive(
        args.artifact_root,
        expected_source_commit=args.expected_source_commit,
    )
    _write_json(args.artifact_root / "validation.local.json", validation)
    _write_json(args.artifact_root / "gate.local.json", local_gate)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "errors": validation["errors"],
                "gate": local_gate["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if validation["status"] == "pass" else 1


def validate_phase_c_archive(
    artifact_root: Path,
    *,
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (artifact_root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    marker = artifact_root / "retrieved_from_verified_result_branch.marker"
    if not marker.is_file():
        errors.append("missing verified result retrieval marker")
    if errors:
        return _failed_payloads(
            artifact_root,
            expected_source_commit=expected_source_commit,
            errors=errors,
        )

    errors.extend(_verify_manifest(artifact_root))
    source_commit = (artifact_root / "source_expected_commit.txt").read_text(
        encoding="utf-8"
    ).strip()
    if source_commit != expected_source_commit:
        errors.append(
            f"source commit mismatch: expected {expected_source_commit}, got {source_commit}"
        )

    metadata = _read_json(artifact_root / "metadata.json")
    archived_gate = _read_json(artifact_root / "gate.json")
    summary = _read_json(artifact_root / "summary.json")
    config = SpeckHwangPhaseCConfig(
        run_id=str(metadata["run_id"]),
        seed=int(metadata["seed"]),
        discovery_keys=int(metadata["discovery_keys"]),
        validation_keys=int(metadata["validation_keys"]),
        chunk_size=int(metadata["chunk_size"]),
        backend=str(metadata["backend"]),
        device=str(metadata["device"]),
    )
    errors.extend(_validate_run_metadata(metadata, config))
    keys, key_errors = _read_keys(artifact_root / "keys.csv", config)
    errors.extend(key_errors)
    anchor_cache_metadata = _read_json(
        artifact_root / "cache/anchor/metadata.json"
    )
    control_cache_metadata = _read_json(
        artifact_root / "cache/control/metadata.json"
    )
    errors.extend(
        _validate_cache_metadata(
            anchor_cache_metadata,
            config=config,
            keys=keys,
            role="anchor",
            rounds=[6, 7],
            active_bits=SPECK32_ACTIVE_BITS,
        )
    )
    errors.extend(
        _validate_cache_metadata(
            control_cache_metadata,
            config=config,
            keys=keys,
            role="control",
            rounds=[7],
            active_bits=CONTROL_ACTIVE_BITS,
        )
    )
    anchor_rows = np.load(artifact_root / "cache/anchor/parity_rows.npy")
    control_rows = np.load(artifact_root / "cache/control/parity_rows.npy")
    anchor_completed = np.load(artifact_root / "cache/anchor/completed.npy")
    control_completed = np.load(artifact_root / "cache/control/completed.npy")
    if anchor_rows.dtype != np.uint32 or anchor_rows.shape != (2, config.total_keys):
        errors.append("anchor parity cache has wrong shape or dtype")
    if control_rows.dtype != np.uint32 or control_rows.shape != (1, config.total_keys):
        errors.append("control parity cache has wrong shape or dtype")
    if (
        anchor_completed.dtype != np.bool_
        or anchor_completed.shape != (2, config.total_keys)
    ):
        errors.append("anchor completion cache has wrong shape or dtype")
    if (
        control_completed.dtype != np.bool_
        or control_completed.shape != (1, config.total_keys)
    ):
        errors.append("control completion cache has wrong shape or dtype")
    timing_rows = _timing_row_count(artifact_root / "progress.jsonl")
    resume_rows = summary.get("resume_rows_generated", {})
    recomputed = evaluate_phase_c(
        config,
        keys=keys,
        anchor_parity_rows=anchor_rows,
        control_parity_rows=control_rows,
        completed={
            "anchor": bool(anchor_completed.all()),
            "control": bool(control_completed.all()),
        },
        resume_rows_generated={
            "anchor": int(resume_rows.get("anchor", -1)),
            "control": int(resume_rows.get("control", -1)),
        },
        cuda_available=bool(
            archived_gate.get("readiness_checks", {}).get("cuda_available")
        ),
        device_count=(
            1
            if archived_gate.get("readiness_checks", {}).get(
                "cuda_device_count_positive"
            )
            else 0
        ),
        timing_rows=timing_rows,
    )
    recomputed_gate = recomputed["gate"]
    archived_rows = [
        json.loads(line)
        for line in (artifact_root / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    errors.extend(_compare_rows(recomputed["rows"], archived_rows))
    for field in ("status", "decision", "signal_checks"):
        if recomputed_gate.get(field) != archived_gate.get(field):
            errors.append(f"recomputed gate field differs from archive: {field}")
    if not all(recomputed_gate.get("readiness_checks", {}).values()):
        errors.append("recomputed readiness checks do not all pass")
    if not all(archived_gate.get("readiness_checks", {}).values()):
        errors.append("archived readiness checks do not all pass")

    validation = {
        "status": "pass" if not errors else "fail",
        "run_id": config.run_id,
        "artifact_root": str(artifact_root),
        "expected_source_commit": expected_source_commit,
        "source_commit": source_commit,
        "manifest_verified": not any(error.startswith("SHA256") for error in errors),
        "verified_result_marker_present": marker.is_file(),
        "timing_rows": timing_rows,
        "expected_timing_rows": 192,
        "remote_decision": archived_gate.get("decision"),
        "recomputed_decision": recomputed_gate.get("decision"),
        "errors": errors,
    }
    local_gate = dict(recomputed_gate)
    local_gate["local_validation"] = {
        "status": validation["status"],
        "source_commit_matches": source_commit == expected_source_commit,
        "manifest_verified": validation["manifest_verified"],
        "remote_gate_matches_recomputation": not any(
            error.startswith("recomputed gate field") for error in errors
        ),
        "errors": errors,
    }
    if errors:
        local_gate["status"] = "fail"
        local_gate["decision"] = "innovation2_speck_hwang_phase_c_protocol_invalid"
        local_gate["next_action"] = {
            "action": "repair retrieved archive, source alignment, cache, or local recomputation",
            "training": False,
            "remote_scale": False,
        }
    return validation, local_gate


def _verify_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"SHA256 manifest line is malformed: {line}")
            continue
        expected, relative = parts
        path = root / relative.strip()
        if not path.is_file():
            errors.append(f"SHA256 manifest file is missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    return errors


def _read_keys(
    path: Path, config: SpeckHwangPhaseCConfig
) -> tuple[tuple[int, ...], list[str]]:
    errors: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["key_index"]))
    keys = tuple(int(row["key_hex"], 0) for row in rows)
    if len(keys) != config.total_keys:
        errors.append(f"keys.csv row count is {len(keys)}, expected {config.total_keys}")
    for index, row in enumerate(rows):
        expected_split = "discovery" if index < config.discovery_keys else "validation"
        if int(row["key_index"]) != index or row["split"] != expected_split:
            errors.append(f"keys.csv split/index mismatch at row {index}")
            break
    return keys, errors


def _timing_row_count(path: Path) -> int:
    identities: set[tuple[str, int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "speck_parity_row_done":
            continue
        if row.get("elapsed_seconds") is None or row.get("peak_memory_bytes") is None:
            continue
        identities.add((str(row["role"]), int(row["rounds"]), int(row["key_index"])))
    return len(identities)


def _validate_cache_metadata(
    payload: dict[str, Any],
    *,
    config: SpeckHwangPhaseCConfig,
    keys: tuple[int, ...],
    role: str,
    rounds: list[int],
    active_bits: tuple[int, ...],
) -> list[str]:
    expected = {
        "run_id": f"{config.run_id}:{role}",
        "cipher": "SPECK32/64",
        "rounds": rounds,
        "keys": [f"0x{key:016X}" for key in keys],
        "active_bits": list(active_bits),
        "active_bit_mask": f"0x{sum(1 << bit for bit in active_bits):08X}",
        "fixed_plaintext": "0x00000000",
        "fixed_mask": f"0x{(((1 << 32) - 1) ^ sum(1 << bit for bit in active_bits)):08X}",
        "assignments_per_key": 1 << 30,
        "chunk_size": config.chunk_size,
        "output_bit_order": "LSB-first",
        "backend": config.backend,
        "device": config.device,
    }
    return [
        f"{role} cache metadata differs: {field}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _validate_run_metadata(
    payload: dict[str, Any], config: SpeckHwangPhaseCConfig
) -> list[str]:
    expected = {
        "task": "innovation2_speck32_hwang_phase_c_exact_kernel",
        "cipher": "SPECK32/64",
        "key_generation_seed": config.seed + 25031,
        "total_keys": config.total_keys,
        "anchor_fixed_bits": [5, 6],
        "anchor_rounds": [6, 7],
        "control_fixed_bits": [0, 1],
        "control_rounds": [7],
        "fixed_context": "00",
        "assignments_per_key_round_role": 1 << 30,
        "expected_exact_rows": 192,
        "output_bit_order": "LSB-first",
        "training_performed": False,
    }
    return [
        f"run metadata differs: {field}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _compare_rows(
    recomputed: list[dict[str, Any]], archived: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    archived_by_key = {
        (str(row.get("role")), int(row.get("rounds", -1))): row for row in archived
    }
    if len(archived_by_key) != 3:
        errors.append("archived results do not contain exactly three role/round rows")
    for row in recomputed:
        key = (str(row["role"]), int(row["rounds"]))
        other = archived_by_key.get(key)
        if other is None:
            errors.append(f"archived result row is missing: {key}")
            continue
        for field, value in row.items():
            if other.get(field) != value:
                errors.append(f"archived result differs for {key}: {field}")
                break
    return errors


def _failed_payloads(
    artifact_root: Path,
    *,
    expected_source_commit: str,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = {
        "status": "fail",
        "artifact_root": str(artifact_root),
        "expected_source_commit": expected_source_commit,
        "errors": errors,
    }
    gate = {
        "run_id": artifact_root.name,
        "status": "fail",
        "decision": "innovation2_speck_hwang_phase_c_protocol_invalid",
        "local_validation": {"status": "fail", "errors": errors},
        "claim_scope": "invalid or incomplete retrieved Phase C archive",
        "next_action": {
            "action": "retrieve or repair the complete verified result archive",
            "training": False,
            "remote_scale": False,
        },
    }
    return validation, gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
