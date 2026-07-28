from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.run_uknit_family_midori64_k1ai import (
    load_k1ai_datasets,
)
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_canonical_walsh_k1an import (
    CONTROL_CONDITIONS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_TRAINING_ROWS,
    MODEL_TO_CONDITION,
    RUN_ID,
    adjudicate_k1an,
    build_control_checks,
    candidate_protocol_frozen,
    evaluate_k1an_panel,
    expected_condition_keys,
    expected_dataset_keys,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the K1-AN fixed-budget Midori64 r4 fixed canonical Walsh "
            "transition-residual panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1ak-root", required=True, type=Path)
    parser.add_argument("--k1am-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AN run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-AN plan does not match the frozen protocol")

    paths = {
        "k1ak_gate": args.k1ak_root / "gate.json",
        "k1ak_validation": args.k1ak_root / "validation.json",
        "k1ak_controls": args.k1ak_root / "controls.jsonl",
        "k1ak_dataset_manifest": args.k1ak_root / "dataset_manifest.jsonl",
        "k1am_gate": args.k1am_root / "gate.json",
        "k1am_validation": args.k1am_root / "validation.json",
        "k1am_results": args.k1am_root / "results.jsonl",
        "k1am_controls": args.k1am_root / "controls.jsonl",
    }
    source_digests = {name: file_sha256(path) for name, path in paths.items()}
    k1ak_gate = read_json(paths["k1ak_gate"])
    k1ak_validation = read_json(paths["k1ak_validation"])
    anchor_rows = read_jsonl(paths["k1ak_controls"])
    manifest_rows = read_jsonl(paths["k1ak_dataset_manifest"])
    k1am_gate = read_json(paths["k1am_gate"])
    k1am_validation = read_json(paths["k1am_validation"])
    k1am_results = read_jsonl(paths["k1am_results"])
    k1am_controls = read_jsonl(paths["k1am_controls"])
    source_checks = source_binding_checks(
        k1ak_gate=k1ak_gate,
        k1ak_validation=k1ak_validation,
        k1ak_controls=anchor_rows,
        dataset_manifest=manifest_rows,
        k1am_gate=k1am_gate,
        k1am_validation=k1am_validation,
        k1am_results=k1am_results,
        k1am_controls=k1am_controls,
        source_digests=source_digests,
    )
    control_checks = build_control_checks(tasks)
    datasets = load_k1ai_datasets(manifest_rows)
    source_checks["six_cache_payload_digests_verified"] = (
        set(datasets) == expected_dataset_keys()
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AN source binding failed: {source_checks}")
    if not all(control_checks.values()):
        raise ValueError(f"K1-AN structure-control binding failed: {control_checks}")

    if args.resume_evaluation:
        k1ak_preflight = read_json(args.k1ak_root / "preflight.json")
        source_cache_root = Path(str(k1ak_preflight["source_cache_root"]))
        validate_resume_root(args, source_cache_root)
    else:
        k1ak_preflight = read_json(args.k1ak_root / "preflight.json")
        source_cache_root = Path(str(k1ak_preflight["source_cache_root"]))
        require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
        initialization_manifest = build_initialization_manifest()
        write_json(
            args.output_root / "initialization_manifest.json",
            initialization_manifest,
        )
        train_argv = training_argv(args, source_cache_root)
        train_args = parse_train_args(train_argv)
        if build_tasks(train_args) != tasks:
            raise ValueError("K1-AN training parser drifted from the frozen plan")
        write_json(
            args.output_root / "preflight.json",
            {
                "run_id": RUN_ID,
                "status": "pass",
                "execution_authorized": True,
                "plan": str(args.plan),
                "plan_sha256": file_sha256(args.plan),
                "k1ak_root": str(args.k1ak_root),
                "k1am_root": str(args.k1am_root),
                "source_cache_root": str(source_cache_root),
                "source_digests": source_digests,
                "expected_source_digests": EXPECTED_SOURCE_DIGESTS,
                "source_checks": source_checks,
                "control_checks": control_checks,
            },
        )
        write_jsonl(args.output_root / "dataset_manifest.jsonl", manifest_rows)
        train_main(train_argv)

    training_rows = read_jsonl(args.output_root / "results.jsonl")
    if len(training_rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("K1-AN did not produce six training rows")
    cache_checks = cache_reuse_checks(read_jsonl(args.output_root / "progress.jsonl"))
    if not all(cache_checks.values()):
        raise ValueError(f"K1-AN source cache reuse failed: {cache_checks}")
    checkpoint_manifest = build_checkpoint_manifest(training_rows)
    write_json(args.output_root / "checkpoint_manifest.json", checkpoint_manifest)
    progress(
        args.output_root / "progress.jsonl",
        "k1an_three_split_panel_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    evaluation_rows = evaluate_k1an_panel(
        tasks=tasks,
        training_rows=training_rows,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    write_csv(args.output_root / "comparison.csv", evaluation_rows)
    gate = adjudicate_k1an(
        tasks=tasks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoint_manifest=checkpoint_manifest,
        anchor_rows=anchor_rows,
        source_checks=source_checks,
        control_checks=control_checks,
        cache_checks=cache_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(training_rows),
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
            "training_rows": len(training_rows),
            "evaluation_rows": len(evaluation_rows),
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_history_csv(
        args.output_root / "results.jsonl", args.output_root / "history.csv"
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        evaluation_rows=len(evaluation_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def training_argv(args: argparse.Namespace, source_cache_root: Path) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        str(EXPECTED_BATCH_SIZE),
        "--hidden-bits",
        "32",
        "--dataset-cache-root",
        str(source_cache_root),
        "--dataset-cache-chunk-size",
        "1024",
        "--dataset-cache-workers",
        "1",
        "--checkpoint-output-dir",
        str(args.output_root / "checkpoints"),
        "--initialization-manifest",
        str(args.output_root / "initialization_manifest.json"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "results.jsonl"),
    ]


def cache_reuse_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    source_events = [
        row
        for row in events
        if row.get("event") in {"cache_reuse", "cache_start"}
        and row.get("split") in {"train", "validation"}
    ]
    reuse_keys = {
        (int(row.get("seed", -1)), str(row.get("model")), str(row.get("split")))
        for row in source_events
        if row.get("event") == "cache_reuse"
    }
    expected_reuse_keys = {
        (seed, model, split)
        for seed in EXPECTED_SEEDS
        for model in MODEL_TO_CONDITION
        for split in ("train", "validation")
    }
    return {
        "twelve_training_validation_cache_reuses_exact": (
            len(source_events) == 12
            and len(reuse_keys) == 12
            and reuse_keys == expected_reuse_keys
        ),
        "no_training_or_validation_cache_regenerated": not any(
            row.get("event") == "cache_start" for row in source_events
        ),
    }


def build_checkpoint_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    for row in rows:
        model = str(row.get("model"))
        condition = MODEL_TO_CONDITION.get(model)
        if condition not in CONTROL_CONDITIONS:
            raise ValueError("K1-AN training row has an unknown model")
        key = (int(row["seed"]), condition)
        if key in observed:
            raise ValueError(f"duplicate K1-AN checkpoint source: {key}")
        observed.add(key)
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        if not checkpoint.is_file():
            raise ValueError(f"missing K1-AN checkpoint: {checkpoint}")
        entries.append(
            {
                "cipher_key": "midori64",
                "seed": int(row["seed"]),
                "condition": condition,
                "model": model,
                "selected_checkpoint": row["training"]["selected_checkpoint"],
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            }
        )
    if observed != expected_condition_keys():
        raise ValueError("K1-AN checkpoint sources are incomplete")
    return {"run_id": RUN_ID, "status": "pass", "entries": entries}


def build_initialization_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "targets": {
            model: {
                "kind": "scratch",
                "target_mapping": "aligned",
            }
            for model in MODEL_TO_CONDITION
        },
    }


def validate_resume_root(args: argparse.Namespace, source_cache_root: Path) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    rows = read_jsonl(args.output_root / "results.jsonl")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("source_cache_root") != str(source_cache_root)
        or len(rows) != EXPECTED_TRAINING_ROWS
        or read_json(args.output_root / "initialization_manifest.json")
        != build_initialization_manifest()
    ):
        raise ValueError("K1-AN resume root does not match the frozen training")
    build_checkpoint_manifest(rows)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "initialization_manifest.json",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-AN output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_manifest",
    "build_initialization_manifest",
    "cache_reuse_checks",
    "main",
    "parse_args",
    "read_tasks",
]
