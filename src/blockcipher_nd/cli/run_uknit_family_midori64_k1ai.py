from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    CONTROL_CONDITIONS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_TRAINING_ROWS,
    MODEL_TO_CONDITION,
    RUN_ID,
    adjudicate_k1ai,
    build_control_checks,
    candidate_protocol_frozen,
    evaluate_k1ai_panel,
    expected_condition_keys,
    expected_dataset_keys,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the K1-AI independently trained Midori64 r4 cell8 neural "
            "structure-attribution panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1ah-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AI run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-AI plan does not match the frozen protocol")

    source_gate_path = args.k1ah_root / "gate.json"
    source_manifest_path = args.k1ah_root / "dataset_manifest.jsonl"
    source_validation_path = args.k1ah_root / "validation.json"
    source_gate = read_json(source_gate_path)
    source_validation = read_json(source_validation_path)
    all_manifest_rows = read_jsonl(source_manifest_path)
    manifest_rows = [
        row
        for row in all_manifest_rows
        if row.get("phase") == "confirmation" and int(row.get("cell", -1)) == 8
    ]
    source_digests = {
        "gate": file_sha256(source_gate_path),
        "dataset_manifest": file_sha256(source_manifest_path),
        "validation": file_sha256(source_validation_path),
    }
    source_checks = source_binding_checks(
        gate=source_gate,
        validation=source_validation,
        source_digests=source_digests,
        manifest_rows=manifest_rows,
    )
    control_checks = build_control_checks(tasks)
    datasets = load_k1ai_datasets(manifest_rows)
    source_checks["six_cache_payload_digests_verified"] = (
        set(datasets) == expected_dataset_keys()
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AI K1-AH source binding failed: {source_checks}")
    if not all(control_checks.values()):
        raise ValueError(f"K1-AI structure-control binding failed: {control_checks}")

    source_cache_root = args.k1ah_root / "cache"
    train_argv = training_argv(args, source_cache_root)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-AI training parser drifted from the frozen plan")

    if args.resume_evaluation:
        validate_resume_root(args, source_cache_root)
    else:
        require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
        write_json(
            args.output_root / "preflight.json",
            {
                "run_id": RUN_ID,
                "status": "pass",
                "execution_authorized": True,
                "plan": str(args.plan),
                "plan_sha256": file_sha256(args.plan),
                "k1ah_root": str(args.k1ah_root),
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
        raise ValueError("K1-AI did not produce eight training rows")
    cache_checks = cache_reuse_checks(read_jsonl(args.output_root / "progress.jsonl"))
    if not all(cache_checks.values()):
        raise ValueError(f"K1-AI source cache reuse failed: {cache_checks}")
    checkpoint_manifest = build_checkpoint_manifest(training_rows)
    write_json(args.output_root / "checkpoint_manifest.json", checkpoint_manifest)
    progress(
        args.output_root / "progress.jsonl",
        "k1ai_three_split_panel_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    evaluation_rows = evaluate_k1ai_panel(
        tasks=tasks,
        training_rows=training_rows,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    write_csv(args.output_root / "split_attribution.csv", evaluation_rows)
    gate = adjudicate_k1ai(
        tasks=tasks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoint_manifest=checkpoint_manifest,
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
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "results.jsonl"),
    ]


def load_k1ai_datasets(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], DiskDifferentialDataset]:
    datasets: dict[tuple[int, str], DiskDifferentialDataset] = {}
    for row in manifest_rows:
        key = (int(row["seed"]), str(row["split"]))
        if key in datasets:
            raise ValueError(f"duplicate K1-AI source cache: {key}")
        cache_dir = Path(str(row["cache_dir"]))
        metadata_path = cache_dir / "metadata.json"
        features_path = cache_dir / "features.npy"
        labels_path = cache_dir / "labels.npy"
        if not all(
            path.is_file() for path in (metadata_path, features_path, labels_path)
        ):
            raise ValueError(f"missing K1-AI source cache payload: {cache_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset = DiskDifferentialDataset(
            features=np.load(features_path, mmap_mode="r"),
            labels=np.load(labels_path, mmap_mode="r"),
            metadata=metadata,
            cache_dir=cache_dir,
        )
        if int(dataset.features.shape[0]) != int(row["rows"]):
            raise ValueError(f"K1-AI source cache row count mismatch: {cache_dir}")
        if differential_dataset_sha256(dataset) != row.get("dataset_sha256"):
            raise ValueError(f"K1-AI source cache digest mismatch: {cache_dir}")
        datasets[key] = dataset
    if set(datasets) != expected_dataset_keys():
        raise ValueError("K1-AI requires exactly six cell8 K1-AH source caches")
    return datasets


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
        "sixteen_training_validation_cache_reuses_exact": (
            len(source_events) == 16
            and len(reuse_keys) == 16
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
            raise ValueError("K1-AI training row has an unknown model")
        key = (int(row["seed"]), condition)
        if key in observed:
            raise ValueError(f"duplicate K1-AI checkpoint source: {key}")
        observed.add(key)
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        if not checkpoint.is_file():
            raise ValueError(f"missing K1-AI checkpoint: {checkpoint}")
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
        raise ValueError("K1-AI checkpoint sources are incomplete")
    return {"run_id": RUN_ID, "status": "pass", "entries": entries}


def validate_resume_root(args: argparse.Namespace, source_cache_root: Path) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    rows = read_jsonl(args.output_root / "results.jsonl")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("source_cache_root") != str(source_cache_root)
        or len(rows) != EXPECTED_TRAINING_ROWS
    ):
        raise ValueError("K1-AI resume root does not match the frozen training")
    build_checkpoint_manifest(rows)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-AI output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_manifest",
    "cache_reuse_checks",
    "load_k1ai_datasets",
    "main",
    "parse_args",
    "read_tasks",
]
