from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.engine.task_inputs import prepare_task_inputs
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    CANDIDATE_MODEL,
    EXPECTED_BATCH_SIZE,
    EXPECTED_CONTROL_ROWS,
    EXPECTED_TRAINING_ROWS,
    RUN_ID,
    adjudicate_k1f,
    build_k1f_readiness,
    evaluate_k1f_controls,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fail-closed K1-F cell/path hypergraph diagnostic and "
            "six-condition frozen-checkpoint panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--k1d-gate", required=True, type=Path)
    parser.add_argument("--k1d-controls", required=True, type=Path)
    parser.add_argument("--k1e-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--dataset-cache-chunk-size", type=int, default=1024)
    parser.add_argument("--dataset-cache-workers", type=int, default=1)
    parser.add_argument(
        "--resume-controls",
        action="store_true",
        help="Reuse four completed K1-F checkpoints and resume frozen controls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-F run_id must remain frozen as {RUN_ID}")
    tasks = _tasks(args.plan)
    source_tasks = _tasks(args.source_plan)
    k1d_gate = _read_json(args.k1d_gate)
    k1e_gate = _read_json(args.k1e_gate)
    manifests, readiness = build_k1f_readiness(
        source_tasks=source_tasks,
        k1e_gate=k1e_gate,
    )
    if not readiness["optimizer_step_authorized"]:
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 4
    train_argv = _training_argv(args)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-F training parser drifted from the frozen plan")
    if args.resume_controls:
        _validate_resume_root(
            output_root=args.output_root,
            plan=args.plan,
            source_plan=args.source_plan,
            source_root=args.source_root,
        )
    else:
        _require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
        preflight = {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "readiness": readiness,
            "manifest_rows": len(manifests),
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "source_plan": str(args.source_plan),
            "source_plan_sha256": file_sha256(args.source_plan),
            "source_root": str(args.source_root),
            "source_cache_root": str(args.source_root / "cache"),
            "k1d_gate": str(args.k1d_gate),
            "k1d_controls": str(args.k1d_controls),
            "k1e_gate": str(args.k1e_gate),
        }
        _write_json(args.output_root / "preflight.json", preflight)
        _write_jsonl(args.output_root / "manifest.jsonl", manifests)
        train_main(train_argv)
    training_rows = _read_jsonl(args.output_root / "results.jsonl")
    if len(training_rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("K1-F did not produce four training rows")
    validation_datasets = {
        (str(task["cipher_key"]), int(task["seed"])): prepare_task_inputs(
            task,
            train_args,
        ).validation_dataset
        for task in tasks
        if task["model_key"] == CANDIDATE_MODEL
    }
    _progress(
        args.output_root / "progress.jsonl",
        "frozen_control_evaluation_start",
        expected_rows=EXPECTED_CONTROL_ROWS,
    )
    controls = evaluate_k1f_controls(
        tasks=tasks,
        training_rows=training_rows,
        validation_datasets=validation_datasets,
        k1d_controls=_read_jsonl(args.k1d_controls),
        device=args.device,
    )
    _write_jsonl(args.output_root / "controls.jsonl", controls)
    gate = adjudicate_k1f(
        tasks=tasks,
        training_rows=training_rows,
        control_rows=controls,
        readiness_gate=readiness,
        k1d_gate=k1d_gate,
        k1e_gate=k1e_gate,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "training_rows": len(training_rows),
        "expected_training_rows": EXPECTED_TRAINING_ROWS,
        "control_rows": len(controls),
        "expected_control_rows": EXPECTED_CONTROL_ROWS,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": len(training_rows),
        "control_rows": len(controls),
        "seed_results": gate["seed_results"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "summary.json", summary)
    _write_checkpoint_manifest(
        training_rows,
        args.output_root / "checkpoint_manifest.json",
    )
    write_history_csv(
        args.output_root / "results.jsonl",
        args.output_root / "history.csv",
    )
    _progress(
        args.output_root / "progress.jsonl",
        "k1f_gate_done",
        status=gate["status"],
        decision=gate["decision"],
        control_rows=len(controls),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _training_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        str(EXPECTED_BATCH_SIZE),
        "--dataset-cache-root",
        str(args.source_root / "cache"),
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
    protected = (
        "preflight.json",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-F output root already contains run artifacts")


def _validate_resume_root(
    *,
    output_root: Path,
    plan: Path,
    source_plan: Path,
    source_root: Path,
) -> None:
    preflight = _read_json(output_root / "preflight.json")
    rows = _read_jsonl(output_root / "results.jsonl")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("status") != "pass"
        or preflight.get("execution_authorized") is not True
        or preflight.get("plan_sha256") != file_sha256(plan)
        or preflight.get("source_plan_sha256") != file_sha256(source_plan)
        or preflight.get("source_cache_root") != str(source_root / "cache")
        or len(rows) != EXPECTED_TRAINING_ROWS
    ):
        raise ValueError("K1-F resume root does not match completed frozen training")
    for row in rows:
        checkpoint = Path(str(row.get("training", {}).get("checkpoint_output", "")))
        if row.get("model") != CANDIDATE_MODEL or not checkpoint.is_file():
            raise ValueError("K1-F resume root is missing a completed checkpoint")


def _write_checkpoint_manifest(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
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
    _write_json(path, {"run_id": RUN_ID, "status": "pass", "entries": entries})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


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
