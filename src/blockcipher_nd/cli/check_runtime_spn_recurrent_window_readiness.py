from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window_readiness import (
    build_recurrent_window_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and fail-closed audit a two-seed uKNIT recurrent-window "
            "Runtime-SPN experiment plan without generating data or training."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    manifests, gate = build_recurrent_window_readiness(
        run_id=args.run_id,
        tasks=tasks,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_root / "manifest.jsonl", manifests)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "validation.json",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "checks": gate["protocol_checks"],
            "plan": str(args.plan),
            "manifest_rows": len(manifests),
            "expected_rows": gate["expected_rows"],
            "training_performed": False,
        },
    )
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": args.run_id,
            "task": gate["task"],
            "status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    _append_progress(
        args.output_root / "progress.jsonl",
        {
            "event": "readiness_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "time": time.time(),
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] == "pass" else 4


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


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
