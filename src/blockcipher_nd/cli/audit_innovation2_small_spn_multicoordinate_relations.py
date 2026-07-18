from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_multicoordinate_relations import (
    MASKS,
    SmallSpnRelationConfig,
    coordinate_parity_bits,
    evaluate_relation_benchmark,
    generate_candidate_pairs,
    label_candidate_pairs,
    load_source_metadata,
    select_relation_templates,
    selected_relation_labels_and_witnesses,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E62 small-SPN relation audit.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SmallSpnRelationConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "source_load_start", {"source_root": str(args.source_root)})
    source_gate, source_metadata = load_source_metadata(args.source_root)
    parity_words = np.load(args.source_root / "parity_words.npy", mmap_mode="r")
    completed = np.load(args.source_root / "completed.npy", mmap_mode="r")
    masks = np.asarray(
        [int(value, 16) for value in source_metadata["output_masks"]],
        dtype=np.uint16,
    )
    _write_progress(progress, "source_load_done", {"parity_shape": list(parity_words.shape)})

    candidate_pairs = generate_candidate_pairs(
        seed=config.candidate_seed, count=config.candidate_pairs
    )
    np.save(args.output_root / "candidate_pairs.npy", candidate_pairs)
    _write_progress(
        progress,
        "candidate_pairs_done",
        {"candidate_pairs": len(candidate_pairs)},
    )
    coordinate_bits = coordinate_parity_bits(parity_words, masks)
    candidate_labels = label_candidate_pairs(coordinate_bits, candidate_pairs)
    selected_rounds, selected_candidates, selected_train_counts = (
        select_relation_templates(
            candidate_labels,
            maximum_per_round=config.maximum_selected_per_round,
        )
    )
    relation_labels, witnesses, certificates_valid = (
        selected_relation_labels_and_witnesses(
            coordinate_bits,
            candidate_pairs,
            selected_rounds,
            selected_candidates,
        )
    )
    selected_pairs = candidate_pairs[selected_candidates]
    np.save(args.output_root / "selected_relation_pairs.npy", selected_pairs)
    np.save(args.output_root / "selected_round_indices.npy", selected_rounds)
    np.save(args.output_root / "selected_candidate_indices.npy", selected_candidates)
    np.save(args.output_root / "selected_train_positive_counts.npy", selected_train_counts)
    np.save(args.output_root / "relation_labels.npy", relation_labels)
    np.save(args.output_root / "witness_key_indices.npy", witnesses)
    np.save(args.output_root / "certificate_valid.npy", certificates_valid)
    _write_selected_csv(
        args.output_root / "selected_relations.csv",
        selected_rounds,
        selected_candidates,
        selected_pairs,
        selected_train_counts,
        source_metadata,
    )
    _write_progress(
        progress,
        "strict_relation_labels_done",
        {
            "selected_relations": len(selected_rounds),
            "positive_labels": int(relation_labels.sum()),
            "negative_labels": int(relation_labels.size - relation_labels.sum()),
        },
    )
    evaluation = evaluate_relation_benchmark(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        completed=completed,
        candidate_pairs=candidate_pairs,
        selected_rounds=selected_rounds,
        selected_candidates=selected_candidates,
        selected_train_counts=selected_train_counts,
        relation_labels=relation_labels,
        witness_key_indices=witnesses,
        certificates_valid=certificates_valid,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_multicoordinate_relation_readiness",
        "source_root": str(args.source_root),
        "source_run_id": source_gate.get("run_id"),
        "candidate_seed": config.candidate_seed,
        "candidate_pairs": config.candidate_pairs,
        "maximum_selected_per_round": config.maximum_selected_per_round,
        "relation_size": 2,
        "coordinate_definition": "active structure index x linear output mask index",
        "label_definition": (
            "1 iff two coordinate parity vectors XOR to zero for all 256 master keys"
        ),
        "strict_negative_certificate": "first master-key index with odd relation parity",
        "exact_key_vectors_are_model_hidden": True,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _write_selected_csv(
    path: Path,
    rounds: np.ndarray,
    candidates: np.ndarray,
    pairs: np.ndarray,
    train_counts: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    masks = [int(value, 16) for value in metadata["output_masks"]]
    structures = metadata["structures"]
    round_values = metadata["rounds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "relation_index",
                "candidate_index",
                "round_index",
                "rounds",
                "left_coordinate",
                "left_structure",
                "left_mask_hex",
                "right_coordinate",
                "right_structure",
                "right_mask_hex",
                "train_positive_count",
            ),
        )
        writer.writeheader()
        for relation_index, (round_index, candidate, pair, train_count) in enumerate(
            zip(rounds, candidates, pairs, train_counts, strict=True)
        ):
            left_structure, left_mask = divmod(int(pair[0]), MASKS)
            right_structure, right_mask = divmod(int(pair[1]), MASKS)
            writer.writerow(
                {
                    "relation_index": relation_index,
                    "candidate_index": int(candidate),
                    "round_index": int(round_index),
                    "rounds": int(round_values[int(round_index)]),
                    "left_coordinate": int(pair[0]),
                    "left_structure": structures[left_structure]["structure_id"],
                    "left_mask_hex": f"0x{masks[left_mask]:04X}",
                    "right_coordinate": int(pair[1]),
                    "right_structure": structures[right_structure]["structure_id"],
                    "right_mask_hex": f"0x{masks[right_mask]:04X}",
                    "train_positive_count": int(train_count),
                }
            )


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": event,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


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


if __name__ == "__main__":
    raise SystemExit(main())
