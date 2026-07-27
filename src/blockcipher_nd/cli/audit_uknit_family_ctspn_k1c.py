from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.engine.task_inputs import prepare_task_inputs
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_BATCH_SIZE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1c import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate_k1c,
    evaluate_k1c,
    validate_k1c_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the four frozen K1-B checkpoints on their exact train and "
            "validation caches under five topology conditions, without training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-C run_id must remain frozen as {RUN_ID}")
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_results = _read_jsonl(args.source_root / "results.jsonl")
    source_controls = _read_jsonl(args.source_root / "controls.jsonl")
    checkpoint_manifest = _read_json(args.source_root / "checkpoint_manifest.json")
    source_preflight = _read_json(args.source_root / "preflight.json")
    source_checks = validate_k1c_source(
        tasks=tasks,
        source_gate=source_gate,
        source_results=source_results,
        source_controls=source_controls,
        checkpoint_manifest=checkpoint_manifest,
        source_preflight=source_preflight,
        plan_path=args.plan,
        source_root=args.source_root,
    )
    if not all(source_checks.values()):
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "status": "invalid",
                    "decision": "innovation1_uknit_family_ctspn_k1c_source_invalid",
                    "source_checks": source_checks,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 4

    _require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "source_root": str(args.source_root),
        "source_gate_sha256": file_sha256(args.source_root / "gate.json"),
        "source_results_sha256": file_sha256(args.source_root / "results.jsonl"),
        "source_controls_sha256": file_sha256(args.source_root / "controls.jsonl"),
        "source_checkpoint_manifest_sha256": file_sha256(
            args.source_root / "checkpoint_manifest.json"
        ),
        "source_checks": source_checks,
    }
    _write_json(args.output_root / "preflight.json", preflight)
    _progress(args.output_root / "progress.jsonl", "k1c_preflight_passed")

    train_argv = _cache_argv(args)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-C cache parser drifted from the frozen K1-B plan")
    datasets = {}
    for index, task in enumerate(tasks, start=1):
        inputs = prepare_task_inputs(
            task,
            train_args,
            progress_path=str(args.output_root / "progress.jsonl"),
            index=index,
            total=len(tasks),
        )
        key = (str(task["cipher_key"]), int(task["seed"]))
        datasets[(*key, "train")] = inputs.train_dataset
        datasets[(*key, "validation")] = inputs.validation_dataset

    cache_events = _read_jsonl(args.output_root / "progress.jsonl")
    source_checks = {
        **source_checks,
        "all_eight_dataset_caches_reused": sum(
            row.get("event") == "cache_reuse" for row in cache_events
        )
        == 8,
        "no_dataset_cache_generation": not any(
            row.get("event") == "cache_start" for row in cache_events
        ),
    }
    if not all(source_checks.values()):
        raise ValueError("K1-C source cache was not reused exactly")

    _progress(
        args.output_root / "progress.jsonl",
        "k1c_frozen_replay_start",
        expected_rows=EXPECTED_RESULT_ROWS,
    )
    rows = evaluate_k1c(
        tasks=tasks,
        source_results=source_results,
        source_controls=source_controls,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        device=args.device,
    )
    gate = adjudicate_k1c(rows=rows, source_checks=source_checks)
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": 0,
        "optimizer_steps": 0,
        "result_rows": len(rows),
        "attribution_summary": gate["attribution_summary"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_csv(args.output_root / "attribution.csv", rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "summary.json", summary)
    _progress(
        args.output_root / "progress.jsonl",
        "k1c_gate_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _cache_argv(args: argparse.Namespace) -> list[str]:
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
        "1024",
        "--dataset-cache-workers",
        "1",
        "--checkpoint-output-dir",
        str(args.source_root / "checkpoints"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "unused-training-results.jsonl"),
    ]


def _require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "attribution.csv",
        "progress.jsonl",
        "gate.json",
        "validation.json",
        "summary.json",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-C output root already contains audit artifacts")


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "cipher_key",
        "seed",
        "split",
        "condition",
        "rows",
        "auc",
        "correct_minus_condition_auc",
        "max_abs_probability_delta_from_correct",
        "mean_abs_probability_delta_from_correct",
        "dataset_sha256",
        "checkpoint_sha256",
        "state_dict_sha256",
        "training_performed",
        "optimizer_steps",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
