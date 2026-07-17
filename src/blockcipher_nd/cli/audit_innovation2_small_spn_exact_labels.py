from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    SmallSpnAuditConfig,
    evaluate_exact_labels,
    run_cached_exact_labels,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E32 small-SPN exact all-key label-width audit."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--sbox-variants", type=int, default=4)
    parser.add_argument("--player-variants", type=int, default=4)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--keys", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SmallSpnAuditConfig(
        run_id=args.run_id,
        mode=args.mode,
        sbox_variants=args.sbox_variants,
        player_variants=args.player_variants,
        rounds=tuple(args.rounds),
        keys=args.keys,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "cipher_variants": config.sbox_variants * config.player_variants,
            "rounds": list(config.rounds),
            "master_keys": config.keys,
            "training_performed": False,
        },
    )
    first = run_cached_exact_labels(
        config, cache_root=args.output_root, progress_callback=callback
    )
    resumed = run_cached_exact_labels(
        config, cache_root=args.output_root, progress_callback=callback
    )
    result = evaluate_exact_labels(
        config,
        first,
        resume_generated_blocks=int(resumed["generated_blocks"]),
    )
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "summary.json", result["summary"])
    _write_structures(args.output_root / "structures.csv", first["structures"])
    _write_masks(args.output_root / "masks.csv", first["masks"])
    _write_progress(
        progress_path,
        "run_done",
        {
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(json.dumps({"gate": result["gate"], "output_root": str(args.output_root)}, sort_keys=True))
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
                {"mask_index": index, "mask_hex": f"0x{mask:04X}", "weight": mask.bit_count()}
            )


if __name__ == "__main__":
    raise SystemExit(main())
