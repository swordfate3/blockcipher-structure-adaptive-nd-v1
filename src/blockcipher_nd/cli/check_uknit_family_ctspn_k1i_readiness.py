from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1i import (
    READINESS_RUN_ID,
    build_k1i_readiness,
    validate_k1i_source_bindings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check zero-training readiness for K1-I exact GF(2) Boolean views."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1h-root", required=True, type=Path)
    parser.add_argument("--k1g-root", required=True, type=Path)
    parser.add_argument("--k1-root", required=True, type=Path)
    parser.add_argument("--k1-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=READINESS_RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != READINESS_RUN_ID:
        raise ValueError(f"K1-I readiness run_id must be {READINESS_RUN_ID}")
    require_fresh_output_root(args.output_root)
    candidate_tasks = tasks(args.plan)
    anchor_tasks = tasks(args.k1_plan)
    dataset_manifest = read_jsonl(args.k1g_root / "dataset_manifest.jsonl")
    source_checks = validate_k1i_source_bindings(
        candidate_tasks=candidate_tasks,
        dataset_manifest=dataset_manifest,
        anchor_tasks=anchor_tasks,
        anchor_results=read_jsonl(args.k1_root / "results.jsonl"),
        anchor_checkpoint_manifest=read_json(args.k1_root / "checkpoint_manifest.json"),
    )
    manifests, gate = build_k1i_readiness(
        tasks=candidate_tasks,
        k1h_gate=read_json(args.k1h_root / "gate.json"),
        source_checks=source_checks,
    )
    args.output_root.mkdir(parents=True)
    write_jsonl(args.output_root / "results.jsonl", manifests)
    checks = {**gate["protocol_checks"], **gate["evidence_checks"]}
    validation = {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if checks and all(checks.values()) else "fail",
        "checks": checks,
        "errors": gate["errors"],
        "result_rows": len(manifests),
        "expected_result_rows": 4,
        "training_rows": 0,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": READINESS_RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": 0,
        "optimizer_steps": 0,
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    preflight = {
        "run_id": READINESS_RUN_ID,
        "status": gate["status"],
        "plan": str(args.plan),
        "k1h_root": str(args.k1h_root),
        "k1g_root": str(args.k1g_root),
        "k1_root": str(args.k1_root),
        "k1_plan": str(args.k1_plan),
        "source_checks": source_checks,
    }
    write_json(args.output_root / "preflight.json", preflight)
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(args.output_root / "summary.json", summary)
    write_jsonl(
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


def tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def require_fresh_output_root(path: Path) -> None:
    protected = ("preflight.json", "results.jsonl", "gate.json", "validation.json")
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-I readiness output root already contains artifacts")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
