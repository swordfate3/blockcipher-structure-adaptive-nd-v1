from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_expanded_topology_labels import (
    ExpandedTopologyAuditConfig,
    evaluate_expanded_labels,
    fair_corrupt_player,
    run_cached_expanded_labels,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E37 expanded small-SPN topology benchmark audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExpandedTopologyAuditConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "cipher_variants": config.sbox_variants * config.player_variants,
            "rounds": list(config.rounds),
            "master_keys": config.keys,
            "training_performed": False,
        },
    )
    first = run_cached_expanded_labels(
        config, cache_root=args.output_root, progress_callback=callback
    )
    resumed = run_cached_expanded_labels(
        config, cache_root=args.output_root, progress_callback=callback
    )
    result = evaluate_expanded_labels(
        config,
        first,
        resume_generated_blocks=int(resumed["generated_blocks"]),
    )
    np.save(args.output_root / "selected_mask.npy", result["selected_mask"])
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "summary.json", result["summary"])
    _write_selected(args.output_root / "selected_cells.csv", result["selected_rows"])
    _write_structures(args.output_root / "structures.csv", first["structures"])
    _write_masks(args.output_root / "masks.csv", first["masks"])
    _write_variants(args.output_root / "variants.csv", first["variants"])
    _write_progress(
        progress_path,
        "run_done",
        {
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(
        json.dumps(
            {"gate": result["gate"], "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 1 if result["gate"]["status"] == "fail" else 0


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


def _write_selected(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round_index",
        "rounds",
        "structure_index",
        "structure_id",
        "mask_index",
        "mask_hex",
        "train_positive_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_structures(path: Path, structures: tuple[Any, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["structure_id", "active_nibbles", "active_bits", "dimension"],
        )
        writer.writeheader()
        for structure in structures:
            writer.writerow(
                {
                    "structure_id": structure.structure_id,
                    "active_nibbles": "-".join(map(str, structure.active_nibbles)),
                    "active_bits": "-".join(map(str, structure.active_bits)),
                    "dimension": structure.dimension,
                }
            )


def _write_masks(path: Path, masks: tuple[int, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mask_index", "mask_hex", "weight"])
        writer.writeheader()
        for index, mask in enumerate(masks):
            writer.writerow(
                {
                    "mask_index": index,
                    "mask_hex": f"0x{mask:04X}",
                    "weight": mask.bit_count(),
                }
            )


def _write_variants(path: Path, variants: tuple[Any, ...]) -> None:
    fieldnames = [
        "variant_id",
        "sbox_id",
        "player_id",
        "split",
        "player",
        "fair_corrupted_player",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            if variant.sbox_id < 3 and variant.player_id < 12:
                split = "train"
            elif variant.sbox_id == 3 and variant.player_id < 12:
                split = "unseen_sbox"
            elif variant.sbox_id < 3:
                split = "unseen_player"
            else:
                split = "dual_unseen"
            writer.writerow(
                {
                    "variant_id": variant.variant_id,
                    "sbox_id": variant.sbox_id,
                    "player_id": variant.player_id,
                    "split": split,
                    "player": "-".join(map(str, variant.player)),
                    "fair_corrupted_player": "-".join(
                        map(str, fair_corrupt_player(variant.player))
                    ),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
