from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1.runtime_spn_paired_runtime_objective_k1by14 import (
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_TRAINING_ROWS,
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    evaluate_checkpoints,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the K1-BY14 paired runtime-objective diagnostic."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-BY14 run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-BY14 plan does not match the frozen protocol")
    readiness = build_readiness(tasks=tasks)
    if readiness.get("optimizer_step_authorized") is not True:
        raise ValueError(f"K1-BY14 readiness failed: {readiness}")
    if not args.readiness_only and not str(args.device).startswith("cuda"):
        raise ValueError("K1-BY14 diagnostic training requires CUDA")

    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            **readiness,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "requested_device": args.device,
        },
    )
    if args.readiness_only:
        gate = {
            **readiness,
            "decision": "innovation1_runtime_spn_k1by14_readiness_authorized",
            "training_performed": False,
            "optimizer_steps": 0,
            "claim_scope": (
                "K1-BY14 zero-training implementation readiness only; no AUC, "
                "scale, transfer, universal-SPN or publication claim."
            ),
            "next_action": (
                "Run the frozen four-row CUDA diagnostic without changing the "
                "objective, controls, seeds, samples, pairs or epochs."
            ),
        }
        write_json(args.output_root / "gate.json", gate)
        write_json(
            args.output_root / "validation.json",
            {
                "run_id": RUN_ID,
                "status": readiness["status"],
                "checks": {
                    **readiness["protocol_checks"],
                    **readiness["evidence_checks"],
                },
                "errors": readiness["errors"],
                "optimizer_steps": 0,
            },
        )
        write_json(
            args.output_root / "summary.json",
            {
                "run_id": RUN_ID,
                "status": gate["status"],
                "decision": gate["decision"],
                "claim_scope": gate["claim_scope"],
                "next_action": gate["next_action"],
            },
        )
        progress(
            args.output_root / "progress.jsonl",
            "run_done",
            status=gate["status"],
            decision=gate["decision"],
        )
        print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
        return 0

    train_argv = training_argv(args)
    if build_tasks(parse_train_args(train_argv)) != tasks:
        raise ValueError("K1-BY14 training parser drifted from the frozen plan")
    train_main(train_argv)
    result_rows = read_jsonl(args.output_root / "results.jsonl")
    progress(
        args.output_root / "progress.jsonl",
        "same_checkpoint_evaluation_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    evaluation_rows = evaluate_checkpoints(
        tasks=tasks,
        result_rows=result_rows,
        checkpoint_root=args.output_root / "checkpoints",
        cache_root=args.output_root / "cache",
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    progress_rows = read_jsonl(args.output_root / "progress.jsonl")
    gate = adjudicate(
        tasks=tasks,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
        progress_rows=progress_rows,
        readiness=readiness,
        checkpoint_root=args.output_root / "checkpoints",
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(result_rows),
        "expected_training_rows": EXPECTED_TRAINING_ROWS,
        "evaluation_rows": len(evaluation_rows),
        "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_comparison_csv(
        args.output_root / "condition_comparison.csv",
        evaluation_rows,
    )
    write_history_csv(
        args.output_root / "results.jsonl",
        args.output_root / "history.csv",
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def training_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        "64",
        "--hidden-bits",
        "32",
        "--dataset-cache-root",
        str(args.output_root / "cache"),
        "--dataset-cache-chunk-size",
        "1024",
        "--dataset-cache-workers",
        "1",
        "--checkpoint-output-dir",
        str(args.output_root / "checkpoints"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "results.jsonl"),
    ]


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-BY14 output root already contains run artifacts")


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        "orientation",
        "runtime_condition",
        "auc",
        "accuracy",
        "best_accuracy",
        "checkpoint_sha256",
        "learned_parameter_fingerprint",
        "runtime_program_sha256",
        "dataset_cache_dir",
        "training_performed",
        "optimizer_steps",
    )
    records = [
        {
            **row,
            "auc": row["metrics"]["auc"],
            "accuracy": row["metrics"]["accuracy"],
            "best_accuracy": row["metrics"]["best_accuracy"],
        }
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_history_csv(results_path: Path, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in read_jsonl(results_path):
        for epoch in result.get("history", []):
            rows.append(
                {
                    "seed": result.get("seed"),
                    "model": result.get("model"),
                    **epoch,
                }
            )
    fields = (
        "seed",
        "model",
        "epoch",
        "train_loss",
        "train_auxiliary_loss",
        "train_runtime_primary_loss",
        "train_runtime_counterfactual_loss",
        "train_runtime_loss_gap",
        "train_runtime_margin_loss",
        "train_runtime_violation_rate",
        "train_auc",
        "val_auc",
        "val_accuracy",
        "learning_rate",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


__all__ = [
    "main",
    "parse_args",
    "read_jsonl",
    "require_fresh_output_root",
    "training_argv",
    "write_comparison_csv",
    "write_history_csv",
]
