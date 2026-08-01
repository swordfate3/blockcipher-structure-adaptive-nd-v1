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
from blockcipher_nd.tasks.innovation1.runtime_spn_trainable_post_expert_adapter_k1by13 import (
    CONDITIONS,
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    comparison_rows,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the K1-BY13 zero-initialized trainable post-expert adapter gate."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-BY13 run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-BY13 plan does not match the frozen protocol")
    readiness = build_readiness(tasks=tasks)
    if readiness.get("optimizer_step_authorized") is not True:
        raise ValueError(f"K1-BY13 readiness failed: {readiness}")
    if not args.readiness_only and args.device != "cuda":
        raise ValueError("K1-BY13 full training requires CUDA; local CPU is prohibited")

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
    progress(
        args.output_root / "progress.jsonl",
        "k1by13_preflight_complete",
        readiness_status=readiness["status"],
        optimizer_steps=0,
    )
    if args.readiness_only:
        readiness_gate = {
            **readiness,
            "decision": "innovation1_runtime_spn_k1by13_readiness_authorized",
            "training_performed": False,
            "claim_scope": (
                "K1-BY13 zero-training implementation readiness only; no AUC, "
                "scale, attack, transfer, universality, or publication claim."
            ),
            "next_action": (
                "Publish and verify the exact source commit, then run the frozen "
                "eight-row diagnostic on the remote A6000 because local CUDA is "
                "unavailable."
            ),
        }
        write_json(args.output_root / "gate.json", readiness_gate)
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
                "status": readiness["status"],
                "decision": readiness_gate["decision"],
                "claim_scope": readiness_gate["claim_scope"],
                "next_action": readiness_gate["next_action"],
            },
        )
        progress(
            args.output_root / "progress.jsonl",
            "run_done",
            status=readiness["status"],
            decision=readiness_gate["decision"],
        )
        print(json.dumps(readiness_gate, ensure_ascii=False, sort_keys=True))
        return 0

    train_argv = training_argv(args)
    if build_tasks(parse_train_args(train_argv)) != tasks:
        raise ValueError("K1-BY13 training parser drifted from the frozen plan")
    train_main(train_argv)
    result_rows = read_jsonl(args.output_root / "results.jsonl")
    progress_rows = read_jsonl(args.output_root / "progress.jsonl")
    checkpoint_root = args.output_root / "checkpoints"
    gate = adjudicate(
        tasks=tasks,
        result_rows=result_rows,
        progress_rows=progress_rows,
        readiness=readiness,
        checkpoint_root=checkpoint_root,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(result_rows),
        "expected_rows": 8,
        "optimizer_steps": 8 * 10 * 64,
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
            "parameter_counts": gate["parameter_counts"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_comparison_csv(
        args.output_root / "condition_comparison.csv",
        comparison_rows(gate),
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
        "progress.jsonl",
        "gate.json",
        "checkpoints",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-BY13 output root already contains run artifacts")


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        *(f"{name}_auc" for name in CONDITIONS),
        "correct_minus_anchor_correct",
        "correct_minus_adapter_affine",
        "correct_minus_adapter_shuffled",
        "adapter_correct_output_projection_l2",
        "adapter_affine_output_projection_l2",
        "adapter_shuffled_output_projection_l2",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
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


def write_history_csv(results_path: Path, output_path: Path) -> None:
    fields = (
        "run_index",
        "run_label",
        "cipher",
        "model",
        "selected_model",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    )
    records: list[dict[str, Any]] = []
    for run_index, row in enumerate(read_jsonl(results_path), start=1):
        label = (
            f"run{run_index}: {row.get('cipher', '')} r{row.get('rounds', '')} "
            f"{row.get('model', row.get('selected_model', ''))} "
            f"seed{row.get('seed', '')}"
        )
        history = row.get("history", [])
        if not isinstance(history, list):
            continue
        for item in history:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "run_index": run_index,
                    "run_label": label,
                    "cipher": row.get("cipher", ""),
                    "model": row.get("model", ""),
                    "selected_model": row.get("selected_model", ""),
                    "rounds": row.get("rounds", ""),
                    "seed": row.get("seed", ""),
                    "samples_per_class": row.get("samples_per_class", ""),
                    "pairs_per_sample": row.get("pairs_per_sample", ""),
                    **item,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "parse_args",
    "progress",
    "read_jsonl",
    "require_fresh_output_root",
    "training_argv",
    "write_comparison_csv",
    "write_history_csv",
    "write_json",
]
