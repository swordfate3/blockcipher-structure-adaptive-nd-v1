from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    READINESS_RUN_ID,
    build_k1b_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check zero-training readiness for CT-SPN K1-B native endpoints."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1-gate", required=True, type=Path)
    parser.add_argument("--k1a-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=READINESS_RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != READINESS_RUN_ID:
        raise ValueError(f"K1-B readiness run_id must be {READINESS_RUN_ID}")
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    manifests, gate = build_k1b_readiness(
        tasks=tasks,
        k1_gate=_read_json(args.k1_gate),
        k1a_gate=_read_json(args.k1a_gate),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_root / "results.jsonl", manifests)
    validation = {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) and all(gate["evidence_checks"].values()) else "fail",
        "checks": {**gate["protocol_checks"], **gate["evidence_checks"]},
        "result_rows": len(manifests),
        "expected_result_rows": 4,
    }
    summary = {
        "run_id": READINESS_RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": 0,
        "optimizer_steps": 0,
        "next_action": gate["next_action"],
    }
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "summary.json", summary)
    _write_jsonl(
        args.output_root / "progress.jsonl",
        [
            {"event": "run_start", "run_id": READINESS_RUN_ID, "time": time.time()},
            {
                "event": "run_done",
                "run_id": READINESS_RUN_ID,
                "status": gate["status"],
                "decision": gate["decision"],
                "time": time.time(),
            },
        ],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if validation["status"] == "pass" else 4


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
