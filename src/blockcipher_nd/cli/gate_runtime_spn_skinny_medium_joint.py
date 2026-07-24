from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    adjudicate_runtime_spn_skinny_medium_joint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly adjudicate two SKINNY runtime-topology seed gates."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed0-gate", required=True, type=Path)
    parser.add_argument("--seed1-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("rtg2a", "rtg2b"), default="rtg2a")
    parser.add_argument("--progress", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = (args.seed0_gate, args.seed1_gate)
    gates = [_read_json(path) for path in source_paths]
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=args.run_id,
        gates=gates,
        phase=args.phase,
    )
    source_evidence = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in source_paths
    ]
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "sources": source_evidence,
    }
    summary = {
        "run_id": args.run_id,
        "task": (
            "innovation1_rtg2a_skinny_general_gf2_medium_two_seed_synthesis"
            if args.phase == "rtg2a"
            else "innovation1_rtg2b_skinny_general_gf2_scale_two_seed_synthesis"
        ),
        "training_performed": False,
        "source_evidence": source_evidence,
        "gate": gate,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _append_progress(
        args.progress or args.output_root / "progress.jsonl",
        {
            "event": "joint_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().astimezone().isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
