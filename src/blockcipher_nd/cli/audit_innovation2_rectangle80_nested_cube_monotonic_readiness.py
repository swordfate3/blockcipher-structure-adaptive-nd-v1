from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.rectangle80_nested_cube_monotonic_readiness import (
    RUN_ID,
    Rectangle80NestedCubeConfig,
    build_nested_atlas,
    build_nested_checkerboard,
    evaluate_nested_cube_readiness,
    load_e88_anchor,
    make_nested_chains,
    result_rows_for_nested_cube,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit E94 RECTANGLE-80 r4 nested 7/8/9-bit cube monotonic labels."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--anchor-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Rectangle80NestedCubeConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"training": False, "cipher": "RECTANGLE-80", "rounds": 4},
    )
    anchor = load_e88_anchor(args.anchor_root)
    _write_progress(progress, "e88_anchor_loaded", anchor["checks"])
    chains = make_nested_chains(anchor["structures"])
    _write_progress(progress, "nested_chains_ready", {"chains": len(chains)})
    raw = build_nested_atlas(config, chains, anchor)
    _write_progress(
        progress,
        "nested_atlas_complete",
        {"monotonicity": raw["monotonicity"]},
    )
    matched = build_nested_checkerboard(
        raw["labels"], chains, attempts=config.match_attempts
    )
    _write_progress(
        progress,
        "matched_nested_contrast_complete",
        {"dimension_metrics": matched["dimension_metrics"]},
    )
    gate = evaluate_nested_cube_readiness(config, chains, anchor, raw, matched)
    result_rows = result_rows_for_nested_cube(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_nested_cube_monotonic_readiness",
        "experiment": "e94",
        "anchor_root": str(args.anchor_root),
        "anchor_hashes": anchor["hashes"],
        "config": serializable_config(config),
        "dimensions": [7, 8, 9],
        "chain_construction": {
            "removed_bit": "A8[chain_index mod 8]",
            "added_bit": "sorted([0..63] minus A8)[chain_index mod 56]",
            "split": "validation iff chain_index mod 4 == 0; otherwise train",
        },
        "positive_semantics": (
            "direct full-cube ANF-support absence certificate or sound upward "
            "inheritance from a certified cube subset"
        ),
        "negative_semantics": (
            "a concrete scheduled 80-bit key and inactive offset produce unit XOR one"
        ),
        "unknown_semantics": "neither a sound positive certificate nor a witness was found",
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

    _write_jsonl(args.output_root / "atlas.jsonl", raw["rows"])
    _write_json(
        args.output_root / "chains.json",
        {
            "chains": [
                {
                    "index": chain.index,
                    "chain_id": chain.chain_id,
                    "split": chain.split,
                    "active_bits_7": list(chain.active_bits_7),
                    "active_bits_8": list(chain.active_bits_8),
                    "active_bits_9": list(chain.active_bits_9),
                    "removed_bit": chain.removed_bit,
                    "added_bit": chain.added_bit,
                }
                for chain in chains
            ]
        },
    )
    _write_csv(args.output_root / "matched_nested_contrast.csv", matched["rows"])
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
