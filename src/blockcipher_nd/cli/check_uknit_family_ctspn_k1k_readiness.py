from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    EXPECTED_SOURCE_DIGESTS,
    READINESS_RUN_ID,
    build_k1k_readiness,
    validate_k1k_source_bindings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check zero-training readiness for the K1-K bounded topology "
            "edge residual over exact K1-I Boolean views."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1i-root", required=True, type=Path)
    parser.add_argument("--k1j-root", required=True, type=Path)
    parser.add_argument("--k1i-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=READINESS_RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != READINESS_RUN_ID:
        raise ValueError(f"K1-K readiness run_id must be {READINESS_RUN_ID}")
    require_fresh_output_root(args.output_root)
    candidate_tasks = tasks(args.plan)
    anchor_tasks = tasks(args.k1i_plan)
    dataset_manifest = read_jsonl(args.k1i_root / "dataset_manifest.jsonl")
    source_checks = validate_k1k_source_bindings(
        candidate_tasks=candidate_tasks,
        dataset_manifest=dataset_manifest,
        anchor_tasks=anchor_tasks,
        anchor_results=read_jsonl(args.k1i_root / "results.jsonl"),
        anchor_checkpoint_manifest=read_json(
            args.k1i_root / "checkpoint_manifest.json"
        ),
    )
    source_checks.update(
        {
            "k1i_gate_digest_exact": file_sha256(args.k1i_root / "gate.json")
            == EXPECTED_SOURCE_DIGESTS["k1i_gate"],
            "k1i_checkpoint_manifest_digest_exact": file_sha256(
                args.k1i_root / "checkpoint_manifest.json"
            )
            == EXPECTED_SOURCE_DIGESTS["k1i_checkpoint_manifest"],
            "k1i_dataset_manifest_digest_exact": file_sha256(
                args.k1i_root / "dataset_manifest.jsonl"
            )
            == EXPECTED_SOURCE_DIGESTS["k1i_dataset_manifest"],
            "k1j_gate_digest_exact": file_sha256(args.k1j_root / "gate.json")
            == EXPECTED_SOURCE_DIGESTS["k1j_gate"],
        }
    )
    manifests, gate = build_k1k_readiness(
        tasks=candidate_tasks,
        k1i_gate=read_json(args.k1i_root / "gate.json"),
        k1j_gate=read_json(args.k1j_root / "gate.json"),
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
        "k1i_root": str(args.k1i_root),
        "k1j_root": str(args.k1j_root),
        "k1i_plan": str(args.k1i_plan),
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
        raise ValueError("K1-K readiness output root already contains artifacts")


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
