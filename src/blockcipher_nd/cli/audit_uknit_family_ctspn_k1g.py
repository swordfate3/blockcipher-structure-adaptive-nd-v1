from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.engine.task_config import (
    build_dataset_config,
    resolve_task_keys,
    validation_samples_per_class,
)
from blockcipher_nd.engine.task_inputs import prepare_task_inputs
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    EXPECTED_BATCH_SIZE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    SAME_KEY_SEED_OFFSET,
    adjudicate_k1g,
    evaluate_k1g,
    validate_k1g_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen K1-F checkpoints on train-seen, fresh-same-key, and "
            "original-cross-key data under six relation controls without training."
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
        raise ValueError(f"K1-G run_id must remain frozen as {RUN_ID}")
    tasks = _tasks(args.plan)
    source_gate = _read_json(args.source_root / "gate.json")
    source_results = _read_jsonl(args.source_root / "results.jsonl")
    source_controls = _read_jsonl(args.source_root / "controls.jsonl")
    checkpoint_manifest = _read_json(args.source_root / "checkpoint_manifest.json")
    source_preflight = _read_json(args.source_root / "preflight.json")
    source_checks = validate_k1g_source(
        tasks=tasks,
        source_gate=source_gate,
        source_results=source_results,
        source_controls=source_controls,
        checkpoint_manifest=checkpoint_manifest,
        source_preflight=source_preflight,
        plan_path=args.plan,
    )
    if not all(source_checks.values()):
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "status": "invalid",
                    "decision": "innovation1_uknit_family_ctspn_k1g_source_invalid",
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
        "same_key_seed_offset": SAME_KEY_SEED_OFFSET,
        "source_checks": source_checks,
    }
    _write_json(args.output_root / "preflight.json", preflight)
    _progress(args.output_root / "progress.jsonl", "k1g_preflight_passed")

    source_cache_root = Path(str(source_preflight["source_cache_root"]))
    train_argv = _cache_argv(args, source_cache_root=source_cache_root)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-G cache parser drifted from the frozen K1-F plan")

    datasets = {}
    manifest_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        inputs = prepare_task_inputs(
            task,
            train_args,
            progress_path=str(args.output_root / "progress.jsonl"),
            index=index,
            total=len(tasks),
        )
        cipher = str(task["cipher_key"])
        seed = int(task["seed"])
        datasets[(cipher, seed, "train_seen")] = inputs.train_dataset
        datasets[(cipher, seed, "cross_key_validation")] = inputs.validation_dataset

        train_key, _ = resolve_task_keys(task)
        same_key_task = {**task, "validation_key": train_key}
        same_key_cipher = build_cipher(
            cipher,
            int(task["rounds"]),
            key=train_key,
        )
        config = build_dataset_config(
            same_key_task,
            cipher=same_key_cipher,
            samples_per_class=validation_samples_per_class(task),
            seed=seed + SAME_KEY_SEED_OFFSET,
            split="validation",
        )
        fresh_args = argparse.Namespace(**vars(train_args))
        fresh_args.dataset_cache_root = str(args.output_root / "cache")
        same_key_dataset = make_task_dataset(
            config,
            fresh_args,
            same_key_task,
            split="same_key_fresh",
            progress_path=str(args.output_root / "progress.jsonl"),
            index=index,
            total=len(tasks),
        )
        datasets[(cipher, seed, "same_key_fresh")] = same_key_dataset
        for split, dataset, key_scope, dataset_seed in (
            ("train_seen", inputs.train_dataset, "train_key", seed),
            (
                "same_key_fresh",
                same_key_dataset,
                "train_key",
                seed + SAME_KEY_SEED_OFFSET,
            ),
            (
                "cross_key_validation",
                inputs.validation_dataset,
                "validation_key",
                seed + 10_000,
            ),
        ):
            manifest_rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "split": split,
                    "key_scope": key_scope,
                    "dataset_seed": dataset_seed,
                    "rows": int(dataset.features.shape[0]),
                    "dataset_sha256": differential_dataset_sha256(dataset),
                    "cache_dir": str(getattr(dataset, "cache_dir", "")),
                }
            )

    cache_events = _read_jsonl(args.output_root / "progress.jsonl")
    source_checks = {
        **source_checks,
        "all_eight_source_caches_reused": sum(
            row.get("event") == "cache_reuse"
            and row.get("split") in {"train", "validation"}
            for row in cache_events
        )
        == 8,
        "no_source_cache_regeneration": not any(
            row.get("event") == "cache_start"
            and row.get("split") in {"train", "validation"}
            for row in cache_events
        ),
        "four_fresh_same_key_caches_created": sum(
            row.get("event") == "cache_start" and row.get("split") == "same_key_fresh"
            for row in cache_events
        )
        == 4,
    }
    if not all(source_checks.values()):
        raise ValueError("K1-G dataset cache contract failed")

    _write_jsonl(args.output_root / "dataset_manifest.jsonl", manifest_rows)
    _progress(
        args.output_root / "progress.jsonl",
        "k1g_frozen_replay_start",
        expected_rows=EXPECTED_RESULT_ROWS,
    )
    rows = evaluate_k1g(
        tasks=tasks,
        source_results=source_results,
        source_controls=source_controls,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        device=args.device,
    )
    gate = adjudicate_k1g(rows=rows, source_checks=source_checks)
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_csv(args.output_root / "attribution.csv", rows)
    _write_decision_outputs(args.output_root, rows=rows, gate=gate)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _write_decision_outputs(
    output_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> None:
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
        "uknit_auc_floor_summary": gate["uknit_auc_floor_summary"],
        "replay_diagnostics": gate["replay_diagnostics"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _progress(
        output_root / "progress.jsonl",
        "k1g_gate_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(rows),
    )


def _tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _cache_argv(
    args: argparse.Namespace,
    *,
    source_cache_root: Path,
) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        str(EXPECTED_BATCH_SIZE),
        "--dataset-cache-root",
        str(source_cache_root),
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
        "dataset_manifest.jsonl",
        "results.jsonl",
        "attribution.csv",
        "progress.jsonl",
        "gate.json",
        "validation.json",
        "summary.json",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-G output root already contains audit artifacts")


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
        "key_scope",
        "dataset_seed",
        "same_key_train_overlap_rows",
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
