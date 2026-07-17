from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.speck_hwang_positions import (
    ANCHOR_START,
    CONTROL_START,
    PHASE_C_ANCHOR_METADATA_SHA256,
    PHASE_C_ANCHOR_PARITY_SHA256,
    PHASE_C_CONTROL_METADATA_SHA256,
    PHASE_C_CONTROL_PARITY_SHA256,
    POSITION_STARTS,
    SpeckHwangPositionConfig,
    evaluate_position_family,
    mask_is_balanced,
    select_validation_candidates,
    verify_position_mapping_fixture,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import hwang_speck_basis_masks
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys


REQUIRED_FILES = (
    "SHA256SUMS",
    "source_expected_commit.txt",
    "results.jsonl",
    "summary.json",
    "gate.json",
    "metadata.json",
    "progress.jsonl",
    "kernel_basis.csv",
    "keys.csv",
    "screen_parity_rows.npy",
    "validation_parity_rows.npy",
    "selected_candidates.npy",
    "baseline_phase_c/anchor/parity_rows.npy",
    "baseline_phase_c/anchor/metadata.json",
    "baseline_phase_c/anchor/completed.npy",
    "baseline_phase_c/control/parity_rows.npy",
    "baseline_phase_c/control/metadata.json",
    "baseline_phase_c/control/completed.npy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a retrieved E27 SPECK position archive."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation, gate = validate_position_archive(
        args.artifact_root,
        expected_source_commit=args.expected_source_commit,
    )
    _write_json(args.artifact_root / "validation.local.json", validation)
    _write_json(args.artifact_root / "gate.local.json", gate)
    print(json.dumps(validation, sort_keys=True))
    return 0 if validation["status"] == "pass" else 1


def validate_position_archive(
    root: Path, *, expected_source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"required file missing: {relative}")
    if errors:
        return _failed(root, expected_source_commit, errors)

    errors.extend(_verify_manifest(root))
    source_commit = (root / "source_expected_commit.txt").read_text(encoding="utf-8").strip()
    if source_commit != expected_source_commit:
        errors.append("source commit does not match expected commit")
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    archived_gate = _read_json(root / "gate.json")
    config = SpeckHwangPositionConfig(
        run_id=str(metadata.get("run_id", root.name)),
        chunk_size=int(metadata.get("chunk_size", 0)),
        backend=str(metadata.get("backend", "")),
        device=str(metadata.get("device", "")),
    )
    keys = _read_keys(root / "keys.csv")
    expected_keys = make_phase_c_keys(config.phase_c_config())
    if keys != expected_keys:
        errors.append("keys.csv does not match deterministic Phase C keys")

    anchor_path = root / "baseline_phase_c/anchor/parity_rows.npy"
    control_path = root / "baseline_phase_c/control/parity_rows.npy"
    baseline_valid = True
    frozen_hashes = {
        anchor_path: PHASE_C_ANCHOR_PARITY_SHA256,
        root / "baseline_phase_c/anchor/metadata.json": PHASE_C_ANCHOR_METADATA_SHA256,
        control_path: PHASE_C_CONTROL_PARITY_SHA256,
        root / "baseline_phase_c/control/metadata.json": PHASE_C_CONTROL_METADATA_SHA256,
    }
    for path, expected in frozen_hashes.items():
        if _sha256(path) != expected:
            errors.append(f"frozen Phase C SHA256 mismatch: {path.relative_to(root)}")
            baseline_valid = False
    anchor_rows = np.load(anchor_path)
    control_rows = np.load(control_path)
    screen = np.load(root / "screen_parity_rows.npy")
    validation_rows = np.load(root / "validation_parity_rows.npy")
    selected = tuple(int(value) for value in np.load(root / "selected_candidates.npy"))
    paper_mask = hwang_speck_basis_masks(7)[0]
    screen_candidates = tuple(
        start
        for index, start in enumerate(POSITION_STARTS)
        if start not in {ANCHOR_START, CONTROL_START}
        and bool(mask_is_balanced(screen[index], paper_mask).all())
    )
    if selected != select_validation_candidates(screen_candidates):
        errors.append("selected candidates violate the frozen deterministic rule")

    cache_errors, completed, resumed = _verify_cache_tree(
        root,
        config=config,
        screen=screen,
        validation=validation_rows,
        selected=selected,
        summary=summary,
    )
    errors.extend(cache_errors)
    timing_rows = _count_timing_rows(root / "progress.jsonl")
    recomputed = evaluate_position_family(
        config,
        keys=keys,
        anchor_words=anchor_rows[1],
        control_words=control_rows[0],
        screen_parity_rows=screen,
        screen_candidates=screen_candidates,
        selected_candidates=selected,
        validation_parity_rows=validation_rows,
        baseline_valid=baseline_valid,
        caches_completed=completed,
        resume_rows_generated=resumed,
        mapping_fixture_valid=verify_position_mapping_fixture(),
        cuda_available=True,
        device_count=1,
        timing_rows=timing_rows,
    )
    archived_rows = _read_jsonl(root / "results.jsonl")
    if archived_rows != recomputed["rows"]:
        errors.append("results.jsonl differs from local GF(2) recomputation")
    gate_fields = ("status", "decision", "readiness_checks", "signal_checks", "metrics", "claim_scope")
    for field in gate_fields:
        if archived_gate.get(field) != recomputed["gate"].get(field):
            errors.append(f"remote gate differs from local recomputation: {field}")
    status = "pass" if not errors else "fail"
    local_gate = recomputed["gate"]
    local_gate["local_validation"] = {"status": status, "errors": errors}
    if errors:
        local_gate["status"] = "fail"
        local_gate["decision"] = "innovation2_speck_hwang_position_family_protocol_invalid"
    validation_payload = {
        "status": status,
        "artifact_root": str(root),
        "expected_source_commit": expected_source_commit,
        "source_commit": source_commit,
        "manifest_verified": not _verify_manifest(root),
        "phase_c_baselines_verified": baseline_valid,
        "timing_rows": timing_rows,
        "expected_timing_rows": recomputed["metadata"]["expected_new_exact_rows"],
        "remote_gate_matches_recomputation": not any(
            error.startswith("remote gate differs") for error in errors
        ),
        "errors": errors,
    }
    return validation_payload, local_gate


def _verify_cache_tree(
    root: Path,
    *,
    config: SpeckHwangPositionConfig,
    screen: np.ndarray,
    validation: np.ndarray,
    selected: tuple[int, ...],
    summary: dict[str, Any],
) -> tuple[list[str], dict[str, bool], dict[str, int]]:
    errors: list[str] = []
    completed: dict[str, bool] = {}
    resumed = {str(key): int(value) for key, value in summary.get("resume_rows_generated", {}).items()}
    selected_lookup = {start: index for index, start in enumerate(selected)}
    for position_index, start in enumerate(POSITION_STARTS):
        if start in {ANCHOR_START, CONTROL_START}:
            continue
        phases = ("screen", "validation") if start in selected_lookup else ("screen",)
        for phase in phases:
            cache_root = root / "cache" / f"position{start:02d}" / phase
            role = f"position{start:02d}_{phase}"
            required = [cache_root / name for name in ("metadata.json", "parity_rows.npy", "completed.npy")]
            if any(not path.is_file() for path in required):
                errors.append(f"cache files missing for {role}")
                completed[role] = False
                continue
            cached = np.load(cache_root / "parity_rows.npy")
            done = np.load(cache_root / "completed.npy")
            expected = (
                screen[position_index]
                if phase == "screen"
                else validation[selected_lookup[start]]
            )
            if cached.shape != (1, expected.size) or not np.array_equal(cached[0], expected):
                errors.append(f"cached parity rows differ from aggregate array for {role}")
            complete = done.shape == cached.shape and done.dtype == np.bool_ and bool(done.all())
            completed[role] = complete
            if not complete:
                errors.append(f"cache completion invalid for {role}")
            metadata = _read_json(cache_root / "metadata.json")
            expected_keys = make_phase_c_keys(config.phase_c_config())
            phase_keys = expected_keys[: config.screen_keys] if phase == "screen" else expected_keys[config.screen_keys :]
            if metadata.get("keys") != [f"0x{key:016X}" for key in phase_keys]:
                errors.append(f"cache key sequence mismatch for {role}")
            if metadata.get("active_bits") != [bit for bit in range(32) if bit not in {start, start + 1}]:
                errors.append(f"cache active bits mismatch for {role}")
    if set(resumed) != set(completed):
        errors.append("summary resume cache roles do not match archive cache roles")
    return errors, completed, resumed


def _count_timing_rows(path: Path) -> int:
    identities: set[tuple[int, str, int]] = set()
    for row in _read_jsonl(path):
        if row.get("event") == "speck_parity_row_done" and row.get("elapsed_seconds") is not None:
            identities.add((int(row["position_start"]), str(row["phase"]), int(row["key_index"])))
    return len(identities)


def _read_keys(path: Path) -> tuple[int, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["key_index"]))
    return tuple(int(row["key_hex"], 16) for row in rows)


def _verify_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"malformed SHA256 line: {line}")
            continue
        expected, relative = parts
        path = root / relative.strip()
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"SHA256 mismatch or missing file: {relative}")
    return errors


def _failed(
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
        "decision": "innovation2_speck_hwang_position_family_protocol_invalid",
        "local_validation": {"status": "fail", "errors": errors},
    }
    return validation, gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
