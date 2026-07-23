from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.gift64_output_prediction_data import (
    Gift64OutputPredictionDataConfig,
    prepare_gift64_fresh_data,
    prepare_gift64_source_data,
    serializable_config,
    validate_gift64_fresh_data,
    validate_gift64_source_data,
)
from blockcipher_nd.tasks.innovation2.gift64_output_prediction_discovery import (
    MODEL_NAMES,
    Gift64DiscoveryTrainingConfig,
    adjudicate_gift64_discovery,
    evaluate_gift64_output_split,
    freeze_gift64_candidates,
    select_gift64_discovery_candidates,
    serializable_training_config,
    train_gift64_discovery_matrix,
)


READINESS_RUN_ID = "i2_output_prediction_gx1_gift64_r3_full64_readiness_seed11_20260723"
FORMAL_RUN_ID = "i2_output_prediction_gx1_gift64_r3_full64_seed11_20260723"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Innovation 2 GX1 GIFT-64 r3 full-output discovery and "
            "fresh selected-bit confirmation."
        )
    )
    parser.add_argument("--mode", choices=("readiness", "formal"), default="readiness")
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "formal":
        run_id = args.run_id or FORMAL_RUN_ID
        data_config = Gift64OutputPredictionDataConfig.formal(run_id=run_id)
        training_config = Gift64DiscoveryTrainingConfig.formal(
            run_id=run_id,
            device=args.device or "cuda",
        )
    else:
        run_id = args.run_id or READINESS_RUN_ID
        data_config = Gift64OutputPredictionDataConfig(run_id=run_id)
        training_config = Gift64DiscoveryTrainingConfig(
            run_id=run_id,
            device=args.device or "cpu",
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": run_id,
            "mode": args.mode,
            "training": True,
            "sample_classification": False,
            "target": "full_64_bit_true_gift64_ciphertext_output",
        },
    )
    source = prepare_gift64_source_data(
        data_config,
        args.output_root,
        progress=progress,
    )
    source_checks = validate_gift64_source_data(data_config, source)
    progress("source_data_validated", source_checks)
    if not all(source_checks.values()):
        raise ValueError(f"invalid GX1 source data: {source_checks}")

    training = train_gift64_discovery_matrix(
        training_config,
        source,
        args.output_root,
        progress=progress,
    )
    discovery = evaluate_gift64_output_split(
        training_config,
        args.output_root,
        source["discovery_features"],
        source["discovery_targets"],
        split="discovery",
        progress=progress,
    )
    candidates = select_gift64_discovery_candidates(
        training_config,
        discovery["per_bit_rows"],
    )
    frozen = freeze_gift64_candidates(candidates, args.output_root)
    candidate_sha256 = str(frozen["candidate_sha256"])
    progress(
        str(frozen["event"]),
        {
            "candidate_count": len(candidates["candidates"]),
            "candidate_msb_indices": candidates["candidate_msb_indices"],
            "candidate_sha256": candidate_sha256,
        },
    )

    fresh = prepare_gift64_fresh_data(
        data_config,
        source,
        args.output_root,
        candidate_sha256=candidate_sha256,
        progress=progress,
    )
    fresh_checks = validate_gift64_fresh_data(
        data_config,
        source,
        fresh,
        candidate_sha256=candidate_sha256,
    )
    progress("fresh_data_validated", fresh_checks)
    fresh_evaluation = evaluate_gift64_output_split(
        training_config,
        args.output_root,
        fresh["features"],
        fresh["full_targets"],
        split="fresh_confirmation",
        progress=progress,
    )
    gate = adjudicate_gift64_discovery(
        training_config,
        source_checks,
        fresh_checks,
        training,
        discovery["per_bit_rows"],
        fresh_evaluation["per_bit_rows"],
        candidates,
        candidate_sha256=candidate_sha256,
    )
    result_rows = _result_rows(
        training,
        discovery,
        fresh_evaluation,
        gate,
    )
    ranking = _build_ranking(
        discovery["per_bit_rows"],
        fresh_evaluation["per_bit_rows"],
        candidates,
        gate,
    )
    metadata = {
        "run_id": run_id,
        "task": "innovation2_output_prediction",
        "experiment": "gx1_gift64_r3_full64_discovery",
        "mode": args.mode,
        "cipher": "GIFT-64",
        "rounds": data_config.rounds,
        "data_config": serializable_config(data_config),
        "training_config": serializable_training_config(training_config),
        "secret_key_hex": f"{int(source['secret_key']):032x}",
        "key_protocol": (
            "one fixed unknown 128-bit key; disjoint train, discovery, and "
            "post-freeze fresh plaintexts"
        ),
        "input": "64 MSB-first plaintext bits",
        "target": "64 MSB-first true GIFT-64 ciphertext bits",
        "candidate_target": "fresh-confirmed true ciphertext output bits",
        "bit_order": "msb_first",
        "sample_classification": False,
        "candidate_sha256": candidate_sha256,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": run_id,
        "metadata": metadata,
        "source_checks": source_checks,
        "fresh_checks": fresh_checks,
        "training_rows": training["rows"],
        "discovery_full_output_rows": discovery["full_output_rows"],
        "fresh_full_output_rows": fresh_evaluation["full_output_rows"],
        "candidates": candidates,
        "ranking": ranking,
        "checkpoint_manifest": training["checkpoints"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_csv(args.output_root / "ranking.csv", ranking)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "checkpoint_manifest.json",
        training["checkpoints"],
    )
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _result_rows(
    training: dict[str, Any],
    discovery: dict[str, Any],
    fresh: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = (
        ("training_summary", training["rows"]),
        ("full_output_metric", discovery["full_output_rows"]),
        ("per_bit_metric", discovery["per_bit_rows"]),
        ("full_output_metric", fresh["full_output_rows"]),
        ("per_bit_metric", fresh["per_bit_rows"]),
    )
    return [
        {
            **row,
            "row_type": row_type,
            "status": gate["status"],
            "decision": gate["decision"],
        }
        for row_type, rows in groups
        for row in rows
    ]


def _build_ranking(
    discovery_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    candidates: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    discovery_index = {
        (str(row["model"]), int(row["msb_index"])): row for row in discovery_rows
    }
    fresh_index = {
        (str(row["model"]), int(row["msb_index"])): row for row in fresh_rows
    }
    ranking_rows = candidates["all_64_discovery_ranking"]
    rank_by_bit = {
        int(row["msb_index"]): rank for rank, row in enumerate(ranking_rows, start=1)
    }
    selection_by_bit = {int(row["msb_index"]): row for row in ranking_rows}
    candidate_bits = {int(bit) for bit in candidates["candidate_msb_indices"]}
    confirmed_bits = {
        int(bit) for bit in gate["metrics"]["fresh_confirmed_msb_indices"]
    }
    rows: list[dict[str, Any]] = []
    for bit in range(64):
        selection = selection_by_bit[bit]
        discovery_true = discovery_index[(MODEL_NAMES[0], bit)]
        discovery_shuffle = discovery_index[(MODEL_NAMES[1], bit)]
        fresh_true = fresh_index[(MODEL_NAMES[0], bit)]
        fresh_shuffle = fresh_index[(MODEL_NAMES[1], bit)]
        rows.append(
            {
                "discovery_rank": rank_by_bit[bit],
                "msb_index": bit,
                "integer_bit": 63 - bit,
                "nibble_msb_index": bit // 4,
                "bit_in_nibble_msb": bit % 4,
                "selected_candidate": bit in candidate_bits,
                "fresh_confirmed": bit in confirmed_bits,
                "eligible_on_discovery": bool(selection["eligible"]),
                "discovery_selection_score": selection["selection_score"],
                "discovery_true_auc": discovery_true["auc"],
                "discovery_shuffle_auc": discovery_shuffle["auc"],
                "discovery_auc_minus_shuffle": (
                    float(discovery_true["auc"]) - float(discovery_shuffle["auc"])
                ),
                "discovery_true_accuracy_margin": discovery_true[
                    "accuracy_minus_majority"
                ],
                "fresh_true_auc": fresh_true["auc"],
                "fresh_shuffle_auc": fresh_shuffle["auc"],
                "fresh_auc_minus_shuffle": (
                    float(fresh_true["auc"]) - float(fresh_shuffle["auc"])
                ),
                "fresh_true_accuracy_margin": fresh_true["accuracy_minus_majority"],
            }
        )
    rows.sort(key=lambda row: int(row["discovery_rank"]))
    return rows


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
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
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


if __name__ == "__main__":
    raise SystemExit(main())
