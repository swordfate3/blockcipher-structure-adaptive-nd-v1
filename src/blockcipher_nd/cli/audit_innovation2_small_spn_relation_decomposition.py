from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_multicoordinate_relations import (
    coordinate_parity_bits,
)
from blockcipher_nd.tasks.innovation2.small_spn_relation_decomposition import (
    RelationDecompositionConfig,
    decompose_relation_labels,
    evaluate_relation_decomposition,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E64 relation decomposition.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--e37-root", required=True, type=Path)
    parser.add_argument("--e62-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RelationDecompositionConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "source_load_start", {})
    e37_metadata = json.loads(
        (args.e37_root / "metadata.json").read_text(encoding="utf-8")
    )
    e62_gate = json.loads((args.e62_root / "gate.json").read_text(encoding="utf-8"))
    parity_words = np.load(args.e37_root / "parity_words.npy", mmap_mode="r")
    masks = np.asarray(
        [int(value, 16) for value in e37_metadata["output_masks"]],
        dtype=np.uint16,
    )
    coordinate_bits = coordinate_parity_bits(parity_words, masks)
    pairs = np.load(args.e62_root / "selected_relation_pairs.npy")
    rounds = np.load(args.e62_root / "selected_round_indices.npy")
    labels = np.load(args.e62_root / "relation_labels.npy")
    witnesses = np.load(args.e62_root / "witness_key_indices.npy")
    decomposition = decompose_relation_labels(
        coordinate_bits, pairs, rounds, labels, witnesses
    )
    np.save(args.output_root / "singleton_zero.npy", decomposition["singleton_zero"])
    np.save(args.output_root / "relation_categories.npy", decomposition["categories"])
    _write_progress(
        progress,
        "decomposition_done",
        {
            "all_relation_labels_recompute": decomposition[
                "all_relation_labels_recompute"
            ],
            "all_negative_witnesses_replay_odd": decomposition[
                "all_negative_witnesses_replay_odd"
            ],
        },
    )
    evaluation = evaluate_relation_decomposition(
        config,
        e62_gate=e62_gate,
        relation_labels=labels,
        singleton_zero=decomposition["singleton_zero"],
        categories=decomposition["categories"],
        all_relation_labels_recompute=bool(
            decomposition["all_relation_labels_recompute"]
        ),
        all_negative_witnesses_replay_odd=bool(
            decomposition["all_negative_witnesses_replay_odd"]
        ),
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_relation_decomposition",
        "e37_root": str(args.e37_root),
        "e62_root": str(args.e62_root),
        "category_codes": {
            "0": "trivial_positive_both_zero",
            "1": "nontrivial_positive_equal_nonzero",
            "2": "negative_exactly_one_zero",
            "3": "nontrivial_negative_both_nonzero_different",
        },
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
