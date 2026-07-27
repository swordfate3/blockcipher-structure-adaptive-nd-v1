from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    RUN_ID,
    build_ctspn_k1_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed zero-training readiness audit for the uKNIT/Dialga "
            "canonical-transition SPN K1 diagnostic."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k0-gate", required=True, type=Path)
    parser.add_argument("--k0-validation", required=True, type=Path)
    parser.add_argument("--present-gate", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
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
    present_gate = _read_json(args.present_gate) if args.present_gate else None
    manifests, gate = build_ctspn_k1_readiness(
        run_id=args.run_id,
        tasks=tasks,
        k0_gate=_read_json(args.k0_gate),
        k0_validation=_read_json(args.k0_validation),
        present_gate=present_gate,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_root / "manifest.jsonl", manifests)
    _write_json(args.output_root / "gate.json", gate)
    validation = {
        "run_id": args.run_id,
        "status": gate["status"],
        "checks": {**gate["protocol_checks"], **gate["evidence_checks"]},
        "plan": str(args.plan),
        "k0_gate": str(args.k0_gate),
        "k0_validation": str(args.k0_validation),
        "present_gate": None if args.present_gate is None else str(args.present_gate),
        "manifest_rows": len(manifests),
        "expected_rows": gate["expected_plan_rows"],
        "training_performed": False,
        "optimizer_steps": 0,
    }
    _write_json(args.output_root / "validation.json", validation)
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "status": gate["status"],
        "decision": gate["decision"],
        "implementation_ready": gate["implementation_ready"],
        "optimizer_step_authorized": gate["optimizer_step_authorized"],
        "training_performed": False,
        "claim_scope": gate["claim_scope"],
        "next_action": gate["next_action"],
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_jsonl(
        args.output_root / "results.jsonl",
        [
            {
                **summary,
                "training_rows": 0,
                "optimizer_steps": 0,
                "candidate_parameter_count": 438702,
                "anchor_parameter_count": 442466,
                "candidate_parameter_relative_delta": (438702 - 442466) / 442466,
            }
        ],
    )
    _append_progress(
        args.output_root / "progress.jsonl",
        {
            "event": "readiness_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "optimizer_step_authorized": gate["optimizer_step_authorized"],
            "time": time.time(),
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] == "pass" else 4


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
