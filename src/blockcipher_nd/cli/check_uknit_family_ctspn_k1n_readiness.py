from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    load_bound_datasets,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    EXPECTED_SOURCE_DIGESTS,
    K1M_DECISION,
    READINESS_RUN_ID,
    RUN_ID,
    build_k1n_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check K1-N exact inverse S-box/operator stages, controls, gradients, "
            "geometry, and K1-M source bindings before training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1m-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=READINESS_RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != READINESS_RUN_ID:
        raise ValueError(f"K1-N readiness run_id must remain {READINESS_RUN_ID}")
    require_fresh_output_root(args.output_root)
    paths = {
        "k1m_gate": args.k1m_root / "gate.json",
        "k1m_checkpoint_manifest": args.k1m_root / "checkpoint_manifest.json",
        "k1m_dataset_manifest": args.k1m_root / "dataset_manifest.jsonl",
        "k1m_controls": args.k1m_root / "controls.jsonl",
    }
    digests = {name: file_sha256(path) for name, path in paths.items()}
    k1m_gate = read_json(paths["k1m_gate"])
    dataset_manifest = read_jsonl(paths["k1m_dataset_manifest"])
    datasets = load_bound_datasets(dataset_manifest)
    source_checks = {
        "source_artifact_digests_exact": digests == EXPECTED_SOURCE_DIGESTS,
        "k1m_hold_decision_exact": (
            k1m_gate.get("status") == "hold"
            and k1m_gate.get("decision") == K1M_DECISION
            and all(k1m_gate.get("protocol_checks", {}).values())
        ),
        "k1m_four_sixty_rows_complete": (
            k1m_gate.get("protocol_checks", {}).get(
                "four_training_rows_complete"
            )
            is True
            and k1m_gate.get("protocol_checks", {}).get(
                "sixty_evaluation_rows_complete"
            )
            is True
        ),
    }
    tasks = read_tasks(args.plan)
    manifests, gate = build_k1n_readiness(
        tasks=tasks,
        datasets=datasets,
        source_checks=source_checks,
    )
    args.output_root.mkdir(parents=True)
    preflight = {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if all(source_checks.values()) else "fail",
        "execution_authorized": gate["execution_authorized"],
        "training_run_id": RUN_ID,
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "k1m_root": str(args.k1m_root),
        "source_digests": digests,
        "source_checks": source_checks,
        "training_rows": 0,
        "optimizer_steps": 0,
    }
    validation = {
        "run_id": READINESS_RUN_ID,
        "status": gate["status"],
        "checks": {**gate["protocol_checks"], **gate["evidence_checks"]},
        "errors": [
            *gate["failed_protocol_checks"],
            *gate["failed_evidence_checks"],
            *gate["errors"],
        ],
        "manifest_rows": len(manifests),
        "expected_manifest_rows": 4,
        "training_rows": 0,
        "optimizer_steps": 0,
    }
    write_json(args.output_root / "preflight.json", preflight)
    write_jsonl(args.output_root / "results.jsonl", manifests)
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        manifest_rows=len(manifests),
        training_rows=0,
        optimizer_steps=0,
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] == "pass" else 1


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in ("preflight.json", "results.jsonl", "gate.json")
    ):
        raise ValueError("K1-N readiness output already exists")


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


def progress(path: Path, event: str, **payload: Any) -> None:
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
