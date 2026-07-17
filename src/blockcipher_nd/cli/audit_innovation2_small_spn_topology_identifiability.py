from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_topology_identifiability import (
    adjudicate_topology_identifiability,
    topology_identifiability_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E36 small-SPN topology-label identifiability."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--contrast-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(progress_path, "run_start", {"run_id": args.run_id})
    labels = np.load(args.label_root / "labels.npy")
    selected = np.load(args.contrast_root / "selected_mask.npy")
    label_metadata = _read_json(args.label_root / "metadata.json")
    label_gate = _read_json(args.label_root / "gate.json")
    contrast_metadata = _read_json(args.contrast_root / "metadata.json")
    contrast_gate = _read_json(args.contrast_root / "gate.json")
    variants = label_metadata.get("variants", [])
    readiness = {
        "label_source_decision_matches": label_gate.get("decision")
        == "innovation2_small_spn_exact_label_shortcut_dominated",
        "contrast_source_decision_matches": contrast_gate.get("decision")
        == "innovation2_small_spn_matched_contrast_ready",
        "label_shape_is_16x4x14x64": labels.shape == (16, 4, 14, 64),
        "selected_shape_is_4x14x64": selected.shape == (4, 14, 64),
        "selected_cells_are_589": int(selected.sum()) == 589,
        "variant_order_is_sbox_then_player": len(variants) == 16
        and all(
            int(variant["sbox_id"]) == index // 4
            and int(variant["player_id"]) == index % 4
            for index, variant in enumerate(variants)
        ),
        "selection_is_train_only": contrast_metadata.get(
            "heldout_labels_used_for_selection"
        )
        is False,
    }
    metrics = topology_identifiability_metrics(labels, selected)
    gate = adjudicate_topology_identifiability(
        run_id=args.run_id, metrics=metrics, readiness=readiness
    )
    rows = [
        {
            "run_id": args.run_id,
            "task": "innovation2_small_spn_topology_label_identifiability",
            "metric": key,
            "count": int(value),
            "fraction": metrics["fractions"].get(key),
            "training_performed": False,
        }
        for key, value in metrics["counts"].items()
    ]
    metadata = {
        "run_id": args.run_id,
        "task": "innovation2_small_spn_topology_label_identifiability",
        "label_source_run": label_gate.get("run_id"),
        "contrast_source_run": contrast_gate.get("run_id"),
        "heldout_labels_used_for_selection": False,
        "heldout_labels_used_for_audit": True,
        "training_performed": False,
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", {"run_id": args.run_id, "gate": gate, "rows": rows})
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress_path,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(path: Path, event: str, payload: dict) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
