from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.engine.task_inputs import prepare_task_inputs
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_CONTROL_ROWS,
    EXPECTED_TRAINING_ROWS,
    RUN_ID,
    adjudicate_ctspn_k1,
    evaluate_frozen_control_panel,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    CANDIDATE_MODEL,
    build_ctspn_k1_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fail-closed uKNIT/Dialga CT-SPN K1 eight-row diagnostic "
            "and five-condition frozen-checkpoint attribution panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k0-gate", required=True, type=Path)
    parser.add_argument("--k0-validation", required=True, type=Path)
    parser.add_argument("--present-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--dataset-cache-chunk-size", type=int, default=1024)
    parser.add_argument("--dataset-cache-workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1 run_id must remain frozen as {RUN_ID}")
    if args.dataset_cache_chunk_size <= 0 or args.dataset_cache_workers <= 0:
        raise ValueError("K1 disk-cache chunk size and worker count must be positive")

    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    k0_gate = _read_json(args.k0_gate)
    k0_validation = _read_json(args.k0_validation)
    present_gate = _read_json(args.present_gate)
    manifests, readiness = build_ctspn_k1_readiness(
        run_id=args.run_id,
        tasks=tasks,
        k0_gate=k0_gate,
        k0_validation=k0_validation,
        present_gate=present_gate,
    )
    preflight = {
        "run_id": args.run_id,
        "status": "pass" if readiness["optimizer_step_authorized"] else "hold",
        "execution_authorized": readiness["optimizer_step_authorized"],
        "readiness": readiness,
        "manifest_rows": len(manifests),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "k0_gate": str(args.k0_gate),
        "k0_gate_sha256": file_sha256(args.k0_gate),
        "k0_validation": str(args.k0_validation),
        "k0_validation_sha256": file_sha256(args.k0_validation),
        "present_gate": str(args.present_gate),
        "present_gate_sha256": file_sha256(args.present_gate),
        "batch_size": EXPECTED_BATCH_SIZE,
    }
    if not readiness["optimizer_step_authorized"]:
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 4

    _require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    _write_json(args.output_root / "preflight.json", preflight)
    _write_jsonl(args.output_root / "manifest.jsonl", manifests)

    train_argv = _training_argv(args)
    train_args = parse_train_args(train_argv)
    parsed_tasks = build_tasks(train_args)
    if parsed_tasks != tasks:
        raise ValueError("K1 training parser drifted from the preflight task panel")
    train_main(train_argv)

    results_path = args.output_root / "results.jsonl"
    training_rows = _read_jsonl(results_path)
    if len(training_rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError(
            f"K1 expected {EXPECTED_TRAINING_ROWS} training rows, "
            f"got {len(training_rows)}"
        )
    validation_datasets = {}
    for task in tasks:
        if task["model_key"] != CANDIDATE_MODEL:
            continue
        key = (str(task["cipher_key"]), int(task["seed"]))
        inputs = prepare_task_inputs(task, train_args)
        validation_datasets[key] = inputs.validation_dataset

    _append_progress(
        args.output_root / "progress.jsonl",
        "frozen_control_evaluation_start",
        run_id=args.run_id,
        expected_rows=EXPECTED_CONTROL_ROWS,
    )
    control_rows = evaluate_frozen_control_panel(
        task_rows=tasks,
        training_rows=training_rows,
        validation_datasets=validation_datasets,
        batch_size=EXPECTED_BATCH_SIZE,
        device=args.device,
    )
    _write_jsonl(args.output_root / "controls.jsonl", control_rows)
    gate = adjudicate_ctspn_k1(
        run_id=args.run_id,
        task_rows=tasks,
        training_rows=training_rows,
        control_rows=control_rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "training_rows": len(training_rows),
        "expected_training_rows": EXPECTED_TRAINING_ROWS,
        "control_rows": len(control_rows),
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
        "plan_sha256": preflight["plan_sha256"],
        "preflight": str(args.output_root / "preflight.json"),
        "training_results": str(results_path),
        "frozen_controls": str(args.output_root / "controls.jsonl"),
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": True,
        "training_rows": len(training_rows),
        "control_rows": len(control_rows),
        "optimizer_steps_in_controls": 0,
        "seed_results": gate["seed_results"],
        "claim_scope": gate["claim_scope"],
        "next_action": gate["next_action"],
    }
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "summary.json", summary)
    _write_checkpoint_manifest(training_rows, args.output_root / "checkpoint_manifest.json")
    write_history_csv(results_path, args.output_root / "history.csv")
    _append_progress(
        args.output_root / "progress.jsonl",
        "k1_gate_done",
        run_id=args.run_id,
        status=gate["status"],
        decision=gate["decision"],
        control_rows=len(control_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _training_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        str(args.device),
        "--batch-size",
        str(EXPECTED_BATCH_SIZE),
        "--dataset-cache-root",
        str(args.output_root / "cache"),
        "--dataset-cache-chunk-size",
        str(args.dataset_cache_chunk_size),
        "--dataset-cache-workers",
        str(args.dataset_cache_workers),
        "--checkpoint-output-dir",
        str(args.output_root / "checkpoints"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "results.jsonl"),
    ]


def _require_fresh_output_root(path: Path) -> None:
    if not path.exists():
        return
    protected = (
        "preflight.json",
        "manifest.jsonl",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "validation.json",
        "summary.json",
        "checkpoint_manifest.json",
        "history.csv",
        "checkpoints",
        "cache",
    )
    existing = [name for name in protected if (path / name).exists()]
    if existing:
        raise ValueError(f"K1 output root already contains run artifacts: {existing}")


def _write_checkpoint_manifest(
    rows: list[dict[str, Any]], path: Path
) -> None:
    entries = []
    for row in rows:
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        entries.append(
            {
                "cipher_key": row["cipher_key"],
                "seed": row["seed"],
                "model": row["model"],
                "selected_checkpoint": row["training"]["selected_checkpoint"],
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            }
        )
    _write_json(
        path,
        {
            "run_id": RUN_ID,
            "status": "pass" if len(entries) == EXPECTED_TRAINING_ROWS else "fail",
            "entries": entries,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSONL objects: {path}")
        rows.append(payload)
    return rows


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


def _append_progress(path: Path, event: str, **payload: Any) -> None:
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
