from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.cross_spn_r3_profile_operator_synthesis import (
    adjudicate_method_synthesis,
    load_frozen_sources,
    source_hashes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E80 PRESENT/GIFT r3-only cross-SPN method synthesis."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--present-root", required=True, type=Path)
    parser.add_argument("--gift-root", required=True, type=Path)
    parser.add_argument("--skinny-r7-root", required=True, type=Path)
    parser.add_argument("--skinny-r8-root", required=True, type=Path)
    parser.add_argument("--skinny-adjacent-root", required=True, type=Path)
    parser.add_argument("--skinny-bottom-row-root", required=True, type=Path)
    parser.add_argument("--skinny-single-cell-root", required=True, type=Path)
    parser.add_argument("--real-spn-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    _write_progress(progress_path, "run_start", {"training": False})
    roots = {
        "present": args.present_root,
        "gift": args.gift_root,
        "skinny_r7": args.skinny_r7_root,
        "skinny_r8": args.skinny_r8_root,
        "skinny_adjacent": args.skinny_adjacent_root,
        "skinny_bottom_row": args.skinny_bottom_row_root,
        "skinny_single_cell": args.skinny_single_cell_root,
        "real_spn": args.real_spn_root,
    }
    sources = load_frozen_sources(roots)
    _write_progress(
        progress_path,
        "sources_loaded",
        {"source_run_ids": {name: source["gate"]["run_id"] for name, source in sources.items()}},
    )
    gate, rows = adjudicate_method_synthesis(args.run_id, sources)
    hashes = source_hashes(sources)
    summary = {
        "run_id": args.run_id,
        "training_performed": False,
        "source_roots": {name: str(root) for name, root in roots.items()},
        "source_hashes": hashes,
        "rows": rows,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_evidence_csv(args.output_root / "evidence_matrix.csv", rows)
    _write_json(args.output_root / "source_hashes.json", hashes)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_progress(
        progress_path,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
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
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def _write_evidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cipher",
        "rounds",
        "evidence_role",
        "structures",
        "observed_matched_edges",
        "strict_profile_labels_ready",
        "two_seed_neural_attribution_confirmed",
        "mean_true_auc",
        "mean_true_minus_independent",
        "mean_true_minus_corrupted",
        "input_dim",
        "parameter_count",
        "epochs",
        "ready_label_family_count",
        "training_performed",
        "direct_cross_cipher_auc_ranking_allowed",
        "status",
        "decision",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
