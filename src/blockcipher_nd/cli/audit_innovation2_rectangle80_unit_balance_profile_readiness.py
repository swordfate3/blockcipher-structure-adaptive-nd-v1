from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    make_structures,
)
from blockcipher_nd.tasks.innovation2.rectangle80_unit_balance_profile_readiness import (
    Rectangle80ProfileConfig,
    Rectangle80UnitProfileConfig,
    Rectangle80UnitProfileExpansionConfig,
    build_rectangle80_checkerboard,
    build_rectangle80_unit_atlas,
    evaluate_rectangle80_unit_profile,
    result_rows_for_rectangle80_profile,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E87 RECTANGLE-80 unit-output profile label readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol", choices=("e87", "e88"), default="e87")
    parser.add_argument("--anchor-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config: Rectangle80ProfileConfig
    if args.protocol == "e88":
        if args.anchor_root is None:
            raise ValueError(
                "E88 requires --anchor-root pointing to the completed E87 run"
            )
        config = Rectangle80UnitProfileExpansionConfig(run_id=args.run_id)
    else:
        config = Rectangle80UnitProfileConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"training": False, "cipher": "RECTANGLE-80", "rounds": config.rounds},
    )
    structures = make_structures(config)
    _write_progress(progress, "structures_ready", {"structures": len(structures)})
    raw = build_rectangle80_unit_atlas(config, structures)
    raw_counts = {
        status: sum(row["status"] == status for row in raw["rows"])
        for status in ("positive", "negative", "unknown")
    }
    _write_progress(progress, "raw_atlas_complete", raw_counts)
    matched = build_rectangle80_checkerboard(
        raw["labels"], structures, attempts=config.match_attempts
    )
    _write_progress(
        progress, "matched_benchmark_complete", matched["split_metrics"]
    )
    anchor_checks = None
    if args.protocol == "e88":
        anchor_checks = _validate_e87_anchor(args.anchor_root, structures, raw)
        _write_progress(progress, "e87_anchor_validated", anchor_checks)
    gate = evaluate_rectangle80_unit_profile(
        config, structures, raw, matched, anchor_checks=anchor_checks
    )
    result_rows = result_rows_for_rectangle80_profile(config, gate)
    targets, observed = _matched_profile_arrays(
        raw["labels"].shape, matched["rows"]
    )
    metadata = {
        "run_id": config.run_id,
        "task": (
            "innovation2_rectangle80_unit_balance_profile_expansion"
            if args.protocol == "e88"
            else "innovation2_rectangle80_unit_balance_profile_readiness"
        ),
        "experiment": args.protocol,
        "anchor_root": None if args.anchor_root is None else str(args.anchor_root),
        "anchor_checks": anchor_checks,
        "config": serializable_config(config),
        "specification": {
            "version": "final RECTANGLE-80 published in 2015, not REC-0",
            "primary_source": (
                "https://csrc.nist.gov/csrc/media/events/lightweight-cryptography-"
                "workshop-2015/documents/papers/session8-wentao-paper.pdf"
            ),
            "independent_source_commit": (
                "SalaQ/RECTANGLE@2a71673e1479365cd11c9d6bb75f84f6c7a4926a"
            ),
            "zero_vector": "0x0874E8B1E3542D96",
        },
        "target": (
            "64 unit-output cube XOR coordinates are zero for every RECTANGLE-80 "
            "80-bit key and inactive plaintext offset"
        ),
        "positive_semantics": (
            "full 8-variable cube monomial absent from a sound active-variable "
            "ANF-support over-approximation"
        ),
        "negative_semantics": (
            "a concrete scheduled 80-bit key and inactive offset produce unit XOR one"
        ),
        "unknown_semantics": (
            "neither a sound positive certificate nor a concrete witness was found"
        ),
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


def _validate_e87_anchor(
    anchor_root: Path,
    structures: tuple[Any, ...],
    raw: dict[str, Any],
) -> dict[str, bool]:
    gate = json.loads((anchor_root / "gate.json").read_text(encoding="utf-8"))
    structure_payload = json.loads(
        (anchor_root / "structures.json").read_text(encoding="utf-8")
    )
    anchor_structures = structure_payload["structures"]
    expected_structures = [
        {
            "index": structure.index,
            "structure_id": structure.structure_id,
            "role": structure.role,
            "active_bits": list(structure.active_bits),
            "active_mask_hex": f"0x{structure.active_mask:016X}",
            "split": "validation" if not structure.index % 4 else "train",
        }
        for structure in structures[:96]
    ]
    anchor_labels = np.full((96, 64), -1, dtype=np.int8)
    with (anchor_root / "atlas.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["label"] is not None:
                anchor_labels[
                    int(row["structure_index"]), int(row["output_bit"])
                ] = int(row["label"])
    anchor_prefix = np.load(
        anchor_root / "prefix_features.npy", allow_pickle=False
    )
    return {
        "e87_anchor_status_is_pass": gate.get("status") == "pass",
        "e87_anchor_decision_matches": gate.get("decision")
        == "innovation2_rectangle80_unit_profile_ready",
        "e87_anchor_shape_matches": anchor_labels.shape == (96, 64)
        and anchor_prefix.shape == (96, 64, 39),
        "first_96_structure_definitions_equal": anchor_structures
        == expected_structures,
        "first_96_ternary_labels_equal": np.array_equal(
            anchor_labels, raw["labels"][:96]
        ),
        "first_96_prefix_features_equal": np.array_equal(
            anchor_prefix, raw["prefix_features"][:96]
        ),
    }


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
