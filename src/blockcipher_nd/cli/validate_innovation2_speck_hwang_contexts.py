from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.speck_hwang_contexts import (
    CONTEXTS,
    PHASE_C_METADATA_SHA256,
    PHASE_C_PARITY_SHA256,
    PHASE_C_RUN_ID,
    PHASE_C_SOURCE_COMMIT,
    SpeckHwangContextConfig,
    evaluate_context_audit,
    fixed_plaintext_for_context,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import SPECK32_ACTIVE_BITS
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys


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
    "context_parity_rows.npy",
    "direct_context11_rows.npy",
    "baseline_phase_c/parity_rows.npy",
    "baseline_phase_c/metadata.json",
    "baseline_phase_c/completed.npy",
    "cache/context01/metadata.json",
    "cache/context01/parity_rows.npy",
    "cache/context01/completed.npy",
    "cache/context10/metadata.json",
    "cache/context10/parity_rows.npy",
    "cache/context10/completed.npy",
    "cache/context11_direct/metadata.json",
    "cache/context11_direct/parity_rows.npy",
    "cache/context11_direct/completed.npy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a retrieved SPECK fixed-context archive."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation, gate = validate_context_archive(
        args.artifact_root,
        expected_source_commit=args.expected_source_commit,
    )
    _write_json(args.artifact_root / "validation.local.json", validation)
    _write_json(args.artifact_root / "gate.local.json", gate)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "errors": validation["errors"],
                "gate": gate["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if validation["status"] == "pass" else 1


def validate_context_archive(
    root: Path,
    *,
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = [
        f"missing required file: {relative}"
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    ]
    marker = root / "retrieved_from_verified_result_branch.marker"
    if not marker.is_file():
        errors.append("missing verified result retrieval marker")
    if errors:
        return _failed_payloads(root, expected_source_commit, errors)
    errors.extend(_verify_manifest(root))

    source_commit = (root / "source_expected_commit.txt").read_text(
        encoding="utf-8"
    ).strip()
    if source_commit != expected_source_commit:
        errors.append(
            f"source commit mismatch: expected {expected_source_commit}, got {source_commit}"
        )
    metadata = _read_json(root / "metadata.json")
    archived_gate = _read_json(root / "gate.json")
    summary = _read_json(root / "summary.json")
    config = SpeckHwangContextConfig(
        run_id=str(metadata["run_id"]),
        seed=0,
        discovery_keys=int(metadata["discovery_keys"]),
        validation_keys=int(metadata["validation_keys"]),
        chunk_size=int(metadata["chunk_size"]),
        backend=str(metadata["backend"]),
        device=str(metadata["device"]),
    )
    errors.extend(_validate_run_metadata(metadata, config))
    keys, key_errors = _read_keys(root / "keys.csv", config)
    errors.extend(key_errors)

    baseline_parity_path = root / "baseline_phase_c/parity_rows.npy"
    baseline_metadata_path = root / "baseline_phase_c/metadata.json"
    baseline_completed_path = root / "baseline_phase_c/completed.npy"
    baseline_valid = True
    if _sha256(baseline_parity_path) != PHASE_C_PARITY_SHA256:
        errors.append("Phase C baseline parity SHA256 mismatch")
        baseline_valid = False
    if _sha256(baseline_metadata_path) != PHASE_C_METADATA_SHA256:
        errors.append("Phase C baseline metadata SHA256 mismatch")
        baseline_valid = False
    baseline_parity = np.load(baseline_parity_path)
    baseline_completed = np.load(baseline_completed_path)
    if baseline_parity.shape != (2, 64) or baseline_parity.dtype != np.uint32:
        errors.append("Phase C baseline parity has wrong shape or dtype")
        baseline_valid = False
    if (
        baseline_completed.shape != (2, 64)
        or baseline_completed.dtype != np.bool_
        or not bool(baseline_completed.all())
    ):
        errors.append("Phase C baseline completion evidence is invalid")
        baseline_valid = False

    context_rows = np.load(root / "context_parity_rows.npy")
    direct_rows = np.load(root / "direct_context11_rows.npy")
    if context_rows.shape != (4, 2, 64) or context_rows.dtype != np.uint32:
        errors.append("context parity rows have wrong shape or dtype")
    if direct_rows.shape != (2, 1) or direct_rows.dtype != np.uint32:
        errors.append("direct context11 rows have wrong shape or dtype")
    if not np.array_equal(context_rows[0], baseline_parity):
        errors.append("context00 rows differ from verified Phase C baseline")
    if not np.array_equal(context_rows[3], context_rows[0] ^ context_rows[1] ^ context_rows[2]):
        errors.append("derived context11 rows violate the permutation partition identity")
    direct_checks = [
        int(context_rows[3, round_index, 0]) == int(direct_rows[round_index, 0])
        for round_index in range(2)
    ]
    if not all(direct_checks):
        errors.append("direct context11 exact-row crosscheck failed")

    caches_completed: dict[str, bool] = {}
    for context in ("01", "10", "11_direct"):
        cache_dir = root / f"cache/context{context}"
        cache_metadata = _read_json(cache_dir / "metadata.json")
        errors.extend(
            _validate_cache_metadata(
                cache_metadata,
                config=config,
                keys=keys,
                context=context,
            )
        )
        parity = np.load(cache_dir / "parity_rows.npy")
        completed = np.load(cache_dir / "completed.npy")
        expected_shape = (2, 1 if context == "11_direct" else 64)
        if parity.shape != expected_shape or parity.dtype != np.uint32:
            errors.append(f"context{context} parity cache has wrong shape or dtype")
        if completed.shape != expected_shape or completed.dtype != np.bool_:
            errors.append(f"context{context} completion cache has wrong shape or dtype")
        caches_completed[context] = bool(completed.all())
    if not np.array_equal(
        np.load(root / "cache/context01/parity_rows.npy"), context_rows[1]
    ):
        errors.append("context01 aggregate rows differ from cache")
    if not np.array_equal(
        np.load(root / "cache/context10/parity_rows.npy"), context_rows[2]
    ):
        errors.append("context10 aggregate rows differ from cache")
    if not np.array_equal(
        np.load(root / "cache/context11_direct/parity_rows.npy"), direct_rows
    ):
        errors.append("direct context11 rows differ from cache")

    timing_rows = _timing_row_count(root / "progress.jsonl")
    resume_rows = summary.get("resume_rows_generated", {})
    recomputed = evaluate_context_audit(
        config,
        keys=keys,
        context_parity_rows=context_rows,
        baseline_valid=baseline_valid,
        caches_completed=caches_completed,
        resume_rows_generated={
            context: int(resume_rows.get(context, -1))
            for context in ("01", "10", "11_direct")
        },
        direct_context11_checks=direct_checks,
        partition_fixture_valid=bool(summary.get("partition_fixture_valid")),
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
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors.extend(_compare_rows(recomputed["rows"], archived_rows))
    for field in ("status", "decision", "signal_checks", "metrics"):
        if recomputed_gate.get(field) != archived_gate.get(field):
            errors.append(f"recomputed gate field differs from archive: {field}")
    if not all(recomputed_gate.get("readiness_checks", {}).values()):
        errors.append("recomputed readiness checks do not all pass")
    if not all(archived_gate.get("readiness_checks", {}).values()):
        errors.append("archived readiness checks do not all pass")

    validation = {
        "status": "pass" if not errors else "fail",
        "run_id": config.run_id,
        "artifact_root": str(root),
        "expected_source_commit": expected_source_commit,
        "source_commit": source_commit,
        "manifest_verified": not any("SHA256" in error for error in errors),
        "phase_c_baseline_verified": baseline_valid,
        "verified_result_marker_present": marker.is_file(),
        "timing_rows": timing_rows,
        "expected_timing_rows": 258,
        "remote_decision": archived_gate.get("decision"),
        "recomputed_decision": recomputed_gate.get("decision"),
        "errors": errors,
    }
    local_gate = dict(recomputed_gate)
    local_gate["local_validation"] = {
        "status": validation["status"],
        "source_commit_matches": source_commit == expected_source_commit,
        "manifest_verified": validation["manifest_verified"],
        "phase_c_baseline_verified": baseline_valid,
        "remote_gate_matches_recomputation": not any(
            error.startswith("recomputed gate field") for error in errors
        ),
        "errors": errors,
    }
    if errors:
        local_gate["status"] = "fail"
        local_gate["decision"] = "innovation2_speck_hwang_context_protocol_invalid"
        local_gate["next_action"] = {
            "action": "repair archive, baseline provenance, context derivation, cache, or local recomputation",
            "training": False,
            "remote_scale": False,
        }
    return validation, local_gate


def _validate_run_metadata(
    payload: dict[str, Any], config: SpeckHwangContextConfig
) -> list[str]:
    expected = {
        "task": "innovation2_speck32_hwang_fixed_context_audit",
        "cipher": "SPECK32/64",
        "rounds": [6, 7],
        "fixed_bits": [5, 6],
        "contexts": list(CONTEXTS),
        "context00_source_run": PHASE_C_RUN_ID,
        "context00_source_commit": PHASE_C_SOURCE_COMMIT,
        "context00_parity_sha256": PHASE_C_PARITY_SHA256,
        "context00_metadata_sha256": PHASE_C_METADATA_SHA256,
        "context11_derivation": "parity00 xor parity01 xor parity10",
        "total_keys": 64,
        "key_generation_seed": 25031,
        "assignments_per_exact_row": 1 << 30,
        "new_exact_rows": 258,
        "training_performed": False,
    }
    return [
        f"run metadata differs: {field}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _validate_cache_metadata(
    payload: dict[str, Any],
    *,
    config: SpeckHwangContextConfig,
    keys: tuple[int, ...],
    context: str,
) -> list[str]:
    is_direct = context == "11_direct"
    context_value = "11" if is_direct else context
    expected_keys = keys[:1] if is_direct else keys
    expected = {
        "run_id": f"{config.run_id}:context{context}",
        "cipher": "SPECK32/64",
        "rounds": [6, 7],
        "keys": [f"0x{key:016X}" for key in expected_keys],
        "active_bits": list(SPECK32_ACTIVE_BITS),
        "active_bit_mask": "0xFFFFFF9F",
        "fixed_plaintext": f"0x{fixed_plaintext_for_context(context_value):08X}",
        "fixed_mask": "0x00000060",
        "assignments_per_key": 1 << 30,
        "chunk_size": config.chunk_size,
        "output_bit_order": "LSB-first",
        "backend": config.backend,
        "device": config.device,
    }
    return [
        f"context{context} cache metadata differs: {field}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _read_keys(
    path: Path, config: SpeckHwangContextConfig
) -> tuple[tuple[int, ...], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["key_index"]))
    keys = tuple(int(row["key_hex"], 0) for row in rows)
    errors: list[str] = []
    if keys != make_phase_c_keys(config.phase_c_config()):
        errors.append("keys.csv does not match the Phase C paired key set")
    for index, row in enumerate(rows):
        split = "discovery" if index < 32 else "validation"
        if int(row["key_index"]) != index or row["split"] != split:
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
        identities.add(
            (str(row["context"]), int(row["rounds"]), int(row["key_index"]))
        )
    return len(identities)


def _compare_rows(
    recomputed: list[dict[str, Any]], archived: list[dict[str, Any]]
) -> list[str]:
    archived_by_key = {
        (str(row.get("context")), int(row.get("rounds", -1))): row
        for row in archived
    }
    errors: list[str] = []
    if len(archived_by_key) != 8:
        errors.append("archived results do not contain eight context/round rows")
    for row in recomputed:
        key = (str(row["context"]), int(row["rounds"]))
        other = archived_by_key.get(key)
        if other is None:
            errors.append(f"archived result row is missing: {key}")
            continue
        for field, value in row.items():
            if other.get(field) != value:
                errors.append(f"archived result differs for {key}: {field}")
                break
    return errors


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
        elif _sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    return errors


def _failed_payloads(
    root: Path, expected_source_commit: str, errors: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = {
        "status": "fail",
        "artifact_root": str(root),
        "expected_source_commit": expected_source_commit,
        "errors": errors,
    }
    gate = {
        "run_id": root.name,
        "status": "fail",
        "decision": "innovation2_speck_hwang_context_protocol_invalid",
        "local_validation": {"status": "fail", "errors": errors},
        "claim_scope": "invalid or incomplete retrieved E26 context archive",
        "next_action": {
            "action": "retrieve or repair the complete verified E26 archive",
            "training": False,
            "remote_scale": False,
        },
    }
    return validation, gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
