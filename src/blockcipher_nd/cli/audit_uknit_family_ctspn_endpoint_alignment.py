from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_endpoint_alignment import (
    PROBE_ROWS,
    RUN_ID,
    run_endpoint_alignment_audit,
    write_endpoint_alignment_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen CT-SPN K1 checkpoints and audit where native endpoint "
            "and transition-order information is lost, without training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1-gate", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--probe-rows", type=int, default=PROBE_ROWS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-A run_id must remain frozen as {RUN_ID}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("K1-A output root must be new or empty")
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    k1_gate = _read_json(args.k1_gate)
    checkpoint_manifest = _read_json(args.checkpoint_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    _progress(progress_path, "run_start", run_id=RUN_ID, probe_rows=args.probe_rows)
    payload = run_endpoint_alignment_audit(
        task_rows=tasks,
        k1_gate=k1_gate,
        checkpoint_manifest=checkpoint_manifest,
        probe_rows=args.probe_rows,
    )
    write_endpoint_alignment_artifacts(payload, args.output_root)
    gate = payload["gate"]
    _progress(
        progress_path,
        "run_done",
        run_id=RUN_ID,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if payload["validation"]["status"] == "pass" else 4


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
