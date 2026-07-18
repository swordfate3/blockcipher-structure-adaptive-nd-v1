from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    UniversalBalanceAtlasConfig,
    build_checkerboard_benchmark,
    build_raw_atlas,
    evaluate_atlas,
    make_output_masks,
    make_structures,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build E43 PRESENT universal-balance certificate/witness atlas."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--structure-count", type=int, default=96)
    parser.add_argument("--witness-keys", type=int, default=16)
    parser.add_argument("--offsets-per-structure", type=int, default=8)
    parser.add_argument("--match-attempts", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = UniversalBalanceAtlasConfig(
        run_id=args.run_id,
        mode=args.mode,
        structure_count=args.structure_count,
        witness_keys=args.witness_keys,
        offsets_per_structure=args.offsets_per_structure,
        match_attempts=args.match_attempts,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": config.run_id, "mode": config.mode, "training": False},
    )

    structures = make_structures(config)
    masks = make_output_masks()
    _write_progress(
        progress_path,
        "fixture_ready",
        {"structures": len(structures), "masks": len(masks)},
    )
    raw = build_raw_atlas(config, structures, masks)
    raw_counts = {
        status: sum(row["status"] == status for row in raw["rows"])
        for status in ("positive", "negative", "unknown")
    }
    _write_progress(progress_path, "raw_atlas_complete", raw_counts)
    matched = build_checkerboard_benchmark(
        labels=raw["labels"],
        structures=structures,
        masks=masks,
        attempts=config.match_attempts,
    )
    _write_progress(
        progress_path,
        "matched_benchmark_complete",
        matched["split_metrics"],
    )
    evaluation = evaluate_atlas(config, structures, masks, raw, matched)
    gate = evaluation["gate"]

    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_universal_balance_atlas",
        "config": serializable_config(config),
        "target": (
            "masked cube XOR is zero for every PRESENT-80 key and inactive-bit "
            "offset"
        ),
        "positive_semantics": (
            "full active-variable cube monomial absent from a sound ANF-support "
            "over-approximation"
        ),
        "negative_semantics": "concrete key and inactive offset produce masked XOR one",
        "unknown_semantics": "neither positive certificate nor negative witness found",
        "checkerboard_selection_uses_labels": True,
        "split_frozen_before_matching": True,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "atlas.jsonl", raw["rows"])
    _write_csv(args.output_root / "matched_contrast.csv", matched["rows"])
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
    _write_json(
        args.output_root / "masks.json",
        {
            "masks": [
                {
                    "index": mask.index,
                    "mask_id": mask.mask_id,
                    "family": mask.family,
                    "mask_hex": f"0x{mask.value:016X}",
                    "bits": list(mask.bits),
                }
                for mask in masks
            ]
        },
    )
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress_path,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)}, sort_keys=True
        )
    )
    return 1 if gate["status"] == "fail" else 0


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
