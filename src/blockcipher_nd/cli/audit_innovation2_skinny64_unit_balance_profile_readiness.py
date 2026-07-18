from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    make_structures,
)
from blockcipher_nd.tasks.innovation2.skinny64_unit_balance_profile_readiness import (
    Skinny64ProfileConfig,
    Skinny64UnitProfileConfig,
    Skinny64UnitProfileTransitionConfig,
    build_skinny_checkerboard,
    build_skinny_unit_atlas,
    evaluate_skinny_unit_profile,
    result_rows_for_skinny_profile,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E81 SKINNY-64/64 r4 strict unit-profile label readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol", choices=("e81", "e82"), default="e81")
    parser.add_argument("--anchor-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config: Skinny64ProfileConfig
    if args.protocol == "e82":
        if args.anchor_root is None:
            raise ValueError("E82 requires --anchor-root pointing to completed E81")
        config = Skinny64UnitProfileTransitionConfig(run_id=args.run_id)
    else:
        config = Skinny64UnitProfileConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"training": False, "cipher": "SKINNY-64/64", "rounds": config.rounds},
    )

    structures = make_structures(config)
    _write_progress(progress, "structures_ready", {"structures": len(structures)})
    raw = build_skinny_unit_atlas(config, structures)
    raw_counts = {
        status: sum(row["status"] == status for row in raw["rows"])
        for status in ("positive", "negative", "unknown")
    }
    _write_progress(progress, "raw_atlas_complete", raw_counts)
    matched = build_skinny_checkerboard(
        raw["labels"], structures, attempts=config.match_attempts
    )
    _write_progress(
        progress, "matched_benchmark_complete", matched["split_metrics"]
    )
    anchor = None
    if args.protocol == "e82":
        anchor = _validate_e81_anchor(args.anchor_root, structures)
        _write_progress(progress, "e81_anchor_validated", anchor["checks"])
    gate = evaluate_skinny_unit_profile(
        config,
        structures,
        raw,
        matched,
        anchor_checks=None if anchor is None else anchor["checks"],
    )
    result_rows = result_rows_for_skinny_profile(config, gate)
    targets, observed = _matched_profile_arrays(raw["labels"].shape, matched["rows"])
    metadata = {
        "run_id": config.run_id,
        "task": (
            "innovation2_skinny64_r5_unit_balance_profile_transition"
            if args.protocol == "e82"
            else "innovation2_skinny64_unit_balance_profile_readiness"
        ),
        "experiment": args.protocol,
        "anchor_root": None if args.anchor_root is None else str(args.anchor_root),
        "anchor_checks": None if anchor is None else anchor["checks"],
        "anchor_hashes": None if anchor is None else anchor["hashes"],
        "same_budget_anchor": "i2_gift64_r4_unit_balance_profile_readiness_20260718",
        "config": serializable_config(config),
        "target": (
            "64 unit-output cube XOR coordinates are zero for every SKINNY-64/64 "
            "scheduled 64-bit key and inactive plaintext offset"
        ),
        "positive_semantics": (
            "full 8-variable cube monomial absent from a sound active-variable "
            "ANF-support over-approximation"
        ),
        "negative_semantics": (
            "a concrete scheduled 64-bit key and inactive offset produce unit XOR one"
        ),
        "unknown_semantics": "neither a sound positive certificate nor witness was found",
        "checkerboard_selection_uses_labels": True,
        "raw_atlas_training_forbidden": True,
        "split_frozen_before_matching": True,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "metrics": gate["metrics"],
        "gate": gate,
        "result_rows": result_rows,
    }

    np.save(args.output_root / "profile_targets.npy", targets)
    np.save(args.output_root / "profile_observed.npy", observed)
    np.save(args.output_root / "prefix_features.npy", raw["prefix_features"])
    _write_jsonl(args.output_root / "atlas.jsonl", raw["rows"])
    _write_csv(args.output_root / "matched_unit_contrast.csv", matched["rows"])
    _write_json(
        args.output_root / "structures.json",
        {
            "structures": [
                {
                    "index": structure.index,
                    "structure_id": structure.structure_id,
                    "role": structure.role,
                    "active_bits": list(structure.active_bits),
                    "active_mask_hex": f"0x{structure.active_mask:016X}",
                    "split": "validation" if not structure.index % 4 else "train",
                }
                for structure in structures
            ]
        },
    )
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)}, sort_keys=True
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _validate_e81_anchor(
    anchor_root: Path,
    structures: tuple[Any, ...],
) -> dict[str, Any]:
    gate_path = anchor_root / "gate.json"
    structures_path = anchor_root / "structures.json"
    metadata_path = anchor_root / "metadata.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    structure_payload = json.loads(structures_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_structures = [
        {
            "index": structure.index,
            "structure_id": structure.structure_id,
            "role": structure.role,
            "active_bits": list(structure.active_bits),
            "active_mask_hex": f"0x{structure.active_mask:016X}",
            "split": "validation" if not structure.index % 4 else "train",
        }
        for structure in structures
    ]
    config = metadata.get("config", {})
    checks = {
        "e81_anchor_run_id_matches": gate.get("run_id")
        == "i2_skinny64_r4_unit_balance_profile_readiness_20260719",
        "e81_anchor_status_is_hold": gate.get("status") == "hold",
        "e81_anchor_decision_matches": gate.get("decision")
        == "innovation2_skinny64_unit_balance_profile_not_ready",
        "e81_anchor_protocol_passes": bool(gate.get("protocol_checks"))
        and all(gate["protocol_checks"].values()),
        "e81_anchor_rounds_are_four": config.get("rounds") == 4,
        "e82_structures_replay_e81": structure_payload.get("structures")
        == expected_structures,
    }
    return {
        "checks": checks,
        "hashes": {
            "gate.json": _sha256(gate_path),
            "structures.json": _sha256(structures_path),
            "metadata.json": _sha256(metadata_path),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matched_profile_arrays(
    shape: tuple[int, int], rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.full(shape, -1, dtype=np.int8)
    observed = np.zeros(shape, dtype=np.bool_)
    for row in rows:
        structure_index = int(row["structure_index"])
        output_bit = int(row["output_bit"])
        if observed[structure_index, output_bit]:
            raise ValueError("duplicate matched profile edge")
        targets[structure_index, output_bit] = int(row["label"])
        observed[structure_index, output_bit] = True
    return targets, observed


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
