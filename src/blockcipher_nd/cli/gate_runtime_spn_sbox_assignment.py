from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_assignment import (
    adjudicate_uknit_sbox_assignment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate the uKNIT runtime-SPN S-box assignment U1 gate."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-context",
        choices=("late_cell", "edge_gate"),
        default="late_cell",
    )
    parser.add_argument(
        "--candidate-cell-input-mode",
        choices=(
            "difference_only",
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
        ),
        default=None,
    )
    parser.add_argument(
        "--anchor-context",
        choices=("late_pair", "edge_gate"),
        default="late_pair",
    )
    parser.add_argument(
        "--anchor-cell-input-mode",
        choices=(
            "difference_only",
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
        ),
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.run_root / "results.jsonl")
    gate = adjudicate_uknit_sbox_assignment(
        run_id=args.run_id,
        rows=rows,
        candidate_context=args.candidate_context,
        candidate_cell_input_mode=args.candidate_cell_input_mode,
        anchor_context=args.anchor_context,
        anchor_cell_input_mode=args.anchor_cell_input_mode,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {"results": str(args.run_root / "results.jsonl")},
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": gate["cipher"],
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 10,
        "seeds": [0, 1],
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
