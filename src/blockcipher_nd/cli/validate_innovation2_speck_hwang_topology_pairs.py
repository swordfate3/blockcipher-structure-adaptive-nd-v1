from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.speck_hwang_parity import hwang_speck_basis_masks
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys
from blockcipher_nd.tasks.innovation2.speck_hwang_positions import (
    PHASE_C_ANCHOR_METADATA_SHA256,
    PHASE_C_ANCHOR_PARITY_SHA256,
    PHASE_C_CONTROL_METADATA_SHA256,
    PHASE_C_CONTROL_PARITY_SHA256,
    mask_is_balanced,
)
from blockcipher_nd.tasks.innovation2.speck_hwang_topology_pairs import (
    FAMILIES,
    LANES,
    SpeckHwangTopologyPairConfig,
    active_bits_for_topology_pair,
    evaluate_topology_pairs,
    select_family_candidates,
)


REQUIRED_FILES = (
    "SHA256SUMS",
    "source_expected_commit.txt",
    "git_revision.txt",
    "results.jsonl",
    "summary.json",
    "gate.json",
    "metadata.json",
    "progress.jsonl",
    "kernel_basis.csv",
    "keys.csv",
    "screen_parity_rows.npy",
    "validation_parity_rows.npy",
    "selected_candidates.json",
    "baseline_phase_c/anchor/parity_rows.npy",
    "baseline_phase_c/anchor/metadata.json",
    "baseline_phase_c/anchor/completed.npy",
    "baseline_phase_c/control/parity_rows.npy",
    "baseline_phase_c/control/metadata.json",
    "baseline_phase_c/control/completed.npy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a retrieved E27-N topology-pair archive."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation, gate = validate_archive(
        args.artifact_root, expected_source_commit=args.expected_source_commit
    )
    _write_json(args.artifact_root / "validation.local.json", validation)
    _write_json(args.artifact_root / "gate.local.json", gate)
    print(json.dumps(validation, sort_keys=True))
    return 0 if validation["status"] == "pass" else 1


def validate_archive(
    root: Path, *, expected_source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = [
        f"required file missing: {item}"
        for item in REQUIRED_FILES
        if not (root / item).is_file()
    ]
    if errors:
        return _failed(root, expected_source_commit, errors)
    manifest_errors = _verify_manifest(root)
    errors.extend(manifest_errors)
    source_commit = (root / "source_expected_commit.txt").read_text(encoding="utf-8").strip()
    if source_commit != expected_source_commit:
        errors.append("source commit mismatch")
    git_revision = (root / "git_revision.txt").read_text(encoding="utf-8").strip()
    if git_revision != expected_source_commit:
        errors.append("recorded git revision mismatch")
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    archived_gate = _read_json(root / "gate.json")
    config = SpeckHwangTopologyPairConfig(
        run_id=str(metadata["run_id"]),
        chunk_size=int(metadata["chunk_size"]),
        backend=str(metadata["backend"]),
        device=str(metadata["device"]),
    )
    keys = _read_keys(root / "keys.csv")
    if keys != make_phase_c_keys(config.phase_c_config()):
        errors.append("key sequence differs from Phase C")
    baseline_valid = True
    frozen = {
        root / "baseline_phase_c/anchor/parity_rows.npy": PHASE_C_ANCHOR_PARITY_SHA256,
        root / "baseline_phase_c/anchor/metadata.json": PHASE_C_ANCHOR_METADATA_SHA256,
        root / "baseline_phase_c/control/parity_rows.npy": PHASE_C_CONTROL_PARITY_SHA256,
        root / "baseline_phase_c/control/metadata.json": PHASE_C_CONTROL_METADATA_SHA256,
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            baseline_valid = False
            errors.append(f"Phase C frozen SHA mismatch: {path.relative_to(root)}")
    for role in ("anchor", "control"):
        parity = np.load(root / f"baseline_phase_c/{role}/parity_rows.npy")
        completed_baseline = np.load(
            root / f"baseline_phase_c/{role}/completed.npy"
        )
        if (
            completed_baseline.shape != parity.shape
            or completed_baseline.dtype != np.bool_
            or not bool(completed_baseline.all())
        ):
            baseline_valid = False
            errors.append(f"Phase C completion array invalid: {role}")
    screen = np.load(root / "screen_parity_rows.npy")
    validation_rows = np.load(root / "validation_parity_rows.npy")
    selected_json = _read_json(root / "selected_candidates.json")
    selected = {family: tuple(int(value) for value in selected_json[family]) for family in FAMILIES}
    selected_specs = tuple((family, lane) for family in FAMILIES for lane in selected[family])
    if screen.shape != (2, 16, 8) or screen.dtype != np.uint32:
        errors.append("screen aggregate has wrong shape or dtype")
    if (
        validation_rows.shape != (len(selected_specs), 56)
        or validation_rows.dtype != np.uint32
    ):
        errors.append("validation aggregate has wrong shape or dtype")
    paper_mask = hwang_speck_basis_masks(7)[0]
    candidates = {
        family: tuple(
            lane for lane in LANES
            if bool(mask_is_balanced(screen[index, lane], paper_mask).all())
        )
        for index, family in enumerate(FAMILIES)
    }
    if selected != select_family_candidates(candidates):
        errors.append("selected candidates differ from frozen per-family rule")
    cache_errors, completed, resumed = _verify_caches(
        root,
        config=config,
        keys=keys,
        screen=screen,
        validation=validation_rows,
        selected_specs=selected_specs,
        summary=summary,
    )
    errors.extend(cache_errors)
    timing_identities = _timing_identities(root / "progress.jsonl")
    expected_timing_identities = {
        (family, lane, "screen", key_index)
        for family in FAMILIES
        for lane in LANES
        for key_index in range(8)
    } | {
        (family, lane, "validation", key_index)
        for family, lane in selected_specs
        for key_index in range(56)
    }
    if timing_identities != expected_timing_identities:
        errors.append("timing identities differ from the frozen dynamic plan")
    timing_rows = len(timing_identities)
    recomputed = evaluate_topology_pairs(
        config,
        keys=keys,
        screen_parity_rows=screen,
        candidates=candidates,
        selected=selected,
        selected_specs=selected_specs,
        validation_parity_rows=validation_rows,
        baseline_valid=baseline_valid,
        caches_completed=completed,
        resume_rows_generated=resumed,
        cuda_available=True,
        device_count=1,
        timing_rows=timing_rows,
    )
    if _read_jsonl(root / "results.jsonl") != recomputed["rows"]:
        errors.append("results differ from local GF(2) recomputation")
    for field in ("status", "decision", "readiness_checks", "metrics", "claim_scope"):
        if archived_gate.get(field) != recomputed["gate"].get(field):
            errors.append(f"remote gate differs from local recomputation: {field}")
    status = "pass" if not errors else "fail"
    gate = recomputed["gate"]
    gate["local_validation"] = {"status": status, "errors": errors}
    if errors:
        gate["status"] = "fail"
        gate["decision"] = "innovation2_speck_topology_pair_protocol_invalid"
    return (
        {
            "status": status,
            "artifact_root": str(root),
            "expected_source_commit": expected_source_commit,
            "source_commit": source_commit,
            "git_revision": git_revision,
            "manifest_verified": not manifest_errors,
            "phase_c_baselines_verified": baseline_valid,
            "timing_rows": timing_rows,
            "expected_timing_rows": recomputed["metadata"]["expected_new_exact_rows"],
            "remote_gate_matches_recomputation": not any(
                error.startswith("remote gate differs") for error in errors
            ),
            "errors": errors,
        },
        gate,
    )


def _verify_caches(
    root: Path,
    *,
    config: SpeckHwangTopologyPairConfig,
    keys: tuple[int, ...],
    screen: np.ndarray,
    validation: np.ndarray,
    selected_specs: tuple[tuple[str, int], ...],
    summary: dict[str, Any],
) -> tuple[list[str], dict[str, bool], dict[str, int]]:
    errors: list[str] = []
    completed: dict[str, bool] = {}
    resumed = {str(key): int(value) for key, value in summary.get("resume_rows_generated", {}).items()}
    selected_lookup = {spec: index for index, spec in enumerate(selected_specs)}
    for family_index, family in enumerate(FAMILIES):
        for lane in LANES:
            phases = ("screen", "validation") if (family, lane) in selected_lookup else ("screen",)
            for phase in phases:
                role = f"{family}_lane{lane:02d}_{phase}"
                cache = root / "cache" / family / f"lane{lane:02d}" / phase
                paths = [cache / name for name in ("parity_rows.npy", "completed.npy", "metadata.json")]
                if any(not path.is_file() for path in paths):
                    errors.append(f"cache missing for {role}")
                    completed[role] = False
                    continue
                values = np.load(cache / "parity_rows.npy")
                done = np.load(cache / "completed.npy")
                expected = (
                    screen[family_index, lane]
                    if phase == "screen"
                    else validation[selected_lookup[(family, lane)]]
                )
                if values.shape != (1, expected.size) or not np.array_equal(values[0], expected):
                    errors.append(f"cache values differ for {role}")
                complete = done.shape == values.shape and done.dtype == np.bool_ and bool(done.all())
                completed[role] = complete
                if not complete:
                    errors.append(f"cache incomplete for {role}")
                metadata = _read_json(cache / "metadata.json")
                phase_keys = (
                    keys[: config.screen_keys]
                    if phase == "screen"
                    else keys[config.screen_keys :]
                )
                expected_metadata = {
                    "active_bits": list(active_bits_for_topology_pair(family, lane)),
                    "assignments_per_key": 1 << 30,
                    "backend": config.backend,
                    "chunk_size": config.chunk_size,
                    "cipher": "SPECK32/64",
                    "device": config.device,
                    "fixed_plaintext": "0x00000000",
                    "keys": [f"0x{key:016X}" for key in phase_keys],
                    "output_bit_order": "LSB-first",
                    "rounds": [7],
                    "run_id": f"{config.run_id}:{role}",
                }
                for field, expected_value in expected_metadata.items():
                    if metadata.get(field) != expected_value:
                        errors.append(f"cache metadata {field} differs for {role}")
    if set(resumed) != set(completed):
        errors.append("resume roles differ from cache roles")
    return errors, completed, resumed


def _timing_identities(path: Path) -> set[tuple[str, int, str, int]]:
    return {
        (str(row["family"]), int(row["lane"]), str(row["phase"]), int(row["key_index"]))
        for row in _read_jsonl(path)
        if row.get("event") == "speck_parity_row_done" and row.get("elapsed_seconds") is not None
    }


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
            errors.append(f"SHA mismatch or missing: {relative}")
    return errors


def _read_keys(path: Path) -> tuple[int, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["key_index"]))
    return tuple(int(row["key_hex"], 16) for row in rows)


def _failed(root: Path, commit: str, errors: list[str]):
    return (
        {"status": "fail", "artifact_root": str(root), "expected_source_commit": commit, "errors": errors},
        {"run_id": root.name, "status": "fail", "decision": "innovation2_speck_topology_pair_protocol_invalid", "local_validation": {"status": "fail", "errors": errors}},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
