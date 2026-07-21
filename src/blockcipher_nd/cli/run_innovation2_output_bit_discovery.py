from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.output_bit_discovery import (
    PAPER_RUN_ID,
    RUN_ID,
    OutputBitDiscoveryConfig,
    adjudicate_output_bit_discovery,
    build_bit_ranking,
    evaluate_output_bits,
    load_output_prediction_source,
    prepare_fresh_output_bit_data,
    select_discovery_candidates,
    serializable_config,
    validate_fresh_output_bit_data,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Innovation 2 OP10 discovery and fresh confirmation of easy "
            "PRESENT ciphertext output bits."
        )
    )
    parser.add_argument(
        "--mode", choices=("smoke", "fresh_confirmation"), default="smoke"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--source-output-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "fresh_confirmation":
        config = OutputBitDiscoveryConfig.fresh_confirmation(
            run_id=args.run_id or PAPER_RUN_ID,
            seed=args.seed,
            device=args.device or "cuda",
        )
    else:
        config = OutputBitDiscoveryConfig(
            run_id=args.run_id or RUN_ID,
            seed=args.seed,
            device=args.device or "cpu",
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "training": False,
            "sample_classification": False,
            "target": "selected_true_ciphertext_output_bits",
        },
    )
    source = load_output_prediction_source(args.source_output_root)
    if not all(source["checks"].values()):
        raise ValueError(f"invalid OP9 source bundle: {source['checks']}")
    progress("source_validated", source["checks"])
    discovery_rows = evaluate_output_bits(
        args.source_output_root,
        source,
        source["discovery_features"],
        source["discovery_targets"],
        split="discovery",
        batch_size=config.batch_size,
        device=config.device,
        progress=progress,
    )
    candidates = select_discovery_candidates(
        config,
        discovery_rows,
        source_run_id=source["metadata"]["run_id"],
    )
    candidate_path = args.output_root / "candidates.json"
    _write_json(candidate_path, candidates)
    candidate_sha256 = _sha256(candidate_path)
    (args.output_root / "candidates.sha256").write_text(
        f"{candidate_sha256}  candidates.json\n", encoding="ascii"
    )
    progress(
        "candidates_frozen_before_fresh_evaluation",
        {
            "candidate_count": len(candidates["candidates"]),
            "candidate_msb_indices": candidates["candidate_msb_indices"],
            "candidate_sha256": candidate_sha256,
        },
    )
    fresh = prepare_fresh_output_bit_data(
        config,
        source,
        args.output_root,
        candidate_sha256=candidate_sha256,
        progress=progress,
    )
    fresh_checks = validate_fresh_output_bit_data(
        config,
        source,
        fresh,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
    )
    progress("fresh_data_validated", fresh_checks)
    fresh_rows = evaluate_output_bits(
        args.source_output_root,
        source,
        fresh["features"],
        fresh["full_targets"],
        split="fresh_confirmation",
        batch_size=config.batch_size,
        device=config.device,
        progress=progress,
    )
    gate = adjudicate_output_bit_discovery(
        config,
        source["checks"],
        fresh_checks,
        discovery_rows,
        fresh_rows,
        candidates,
    )
    result_rows = discovery_rows + fresh_rows
    for row in result_rows:
        row["run_id"] = config.run_id
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    ranking = build_bit_ranking(
        discovery_rows, fresh_rows, candidates, gate
    )
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_prediction",
        "experiment": "op10_present_r3_easy_output_bit_discovery",
        "mode": config.mode,
        "cipher": "PRESENT-80",
        "rounds": 3,
        "source_run_id": source["metadata"]["run_id"],
        "source_output_root": str(args.source_output_root),
        "config": serializable_config(config),
        "input": "64 MSB-first plaintext bits",
        "target": "selected true ciphertext output bit values",
        "bit_order": "msb_first",
        "sample_classification": False,
        "training": False,
        "candidate_sha256": candidate_sha256,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source_checks": source["checks"],
        "fresh_checks": fresh_checks,
        "candidates": candidates,
        "ranking": ranking,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_csv(args.output_root / "ranking.csv", ranking)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    progress(
        "run_done", {"status": gate["status"], "decision": gate["decision"]}
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)}, sort_keys=True
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
