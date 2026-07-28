from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1p import render_k1p_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    SAME_KEY_SEED_OFFSET,
    dataset_row_overlap_count,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    build_k1n_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1p import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    EXPECTED_SEEDS,
    LOWER_ROUNDS,
    RUN_ID,
    adjudicate_k1p,
    evaluate_lower_round,
    reuse_k1o_anchor,
    validate_k1p_source,
    validate_k1p_tasks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the frozen uKNIT 0x40 differential at r3/r4/r5 with "
            "disk-backed data and a closed-form scorer; perform no neural training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-P run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-P feature batch size is frozen at 256")
    tasks = read_tasks(args.plan)
    task_checks = validate_k1p_tasks(tasks)
    if not all(task_checks.values()):
        raise ValueError(f"K1-P task protocol is invalid: {task_checks}")

    source_gate = read_json(args.source_root / "gate.json")
    source_validation = read_json(args.source_root / "validation.json")
    source_results = read_jsonl(args.source_root / "results.jsonl")
    source_features = read_jsonl(args.source_root / "feature_manifest.jsonl")
    source_scorers = read_jsonl(args.source_root / "scorer_manifest.jsonl")
    source_dataset_manifest = read_jsonl(
        args.source_root / "dataset_manifest.jsonl"
    )
    source_checks = validate_k1p_source(
        source_root=args.source_root,
        source_gate=source_gate,
        source_validation=source_validation,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
    )
    if not all(source_checks.values()):
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "status": "invalid",
                    "decision": "innovation1_uknit_family_ctspn_k1p_source_invalid",
                    "source_checks": source_checks,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 4

    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    train_argv = cache_argv(args)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-P cache parser drifted from the frozen task plan")
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "training_authorized": False,
            "optimizer_steps_authorized": 0,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "source_root": str(args.source_root),
            "source_artifact_sha256": {
                name: file_sha256(args.source_root / name)
                for name in (
                    "gate.json",
                    "results.jsonl",
                    "validation.json",
                    "feature_manifest.jsonl",
                    "scorer_manifest.jsonl",
                )
            },
            "source_checks": source_checks,
            "task_checks": task_checks,
            "feature_batch_size": args.batch_size,
            "device": args.device,
            "training_rows": 0,
            "neural_parameter_count": 0,
            "optimizer_steps": 0,
            "epochs": 0,
        },
    )
    progress(args.output_root / "progress.jsonl", "k1p_preflight_passed")

    task_map = {
        (int(task["rounds"]), int(task["seed"])): task for task in tasks
    }
    datasets_by_round = {}
    manifest_rows: list[dict[str, Any]] = []
    for rounds in LOWER_ROUNDS:
        round_datasets = {}
        for seed in EXPECTED_SEEDS:
            task = task_map[(rounds, seed)]
            index = 1 + (rounds - min(LOWER_ROUNDS)) * len(EXPECTED_SEEDS) + seed
            inputs = prepare_task_inputs(
                task,
                train_args,
                progress_path=str(args.output_root / "progress.jsonl"),
                index=index,
                total=len(LOWER_ROUNDS) * len(EXPECTED_SEEDS),
            )
            train_key, _ = resolve_task_keys(task)
            same_key_task = {**task, "validation_key": train_key}
            same_key_cipher = build_cipher("uknit64", rounds, key=train_key)
            same_key_config = build_dataset_config(
                same_key_task,
                cipher=same_key_cipher,
                samples_per_class=validation_samples_per_class(task),
                seed=seed + SAME_KEY_SEED_OFFSET,
                split="validation",
            )
            same_key_dataset = make_task_dataset(
                same_key_config,
                train_args,
                same_key_task,
                split="same_key_fresh",
                progress_path=str(args.output_root / "progress.jsonl"),
                index=index,
                total=len(LOWER_ROUNDS) * len(EXPECTED_SEEDS),
            )
            split_datasets = {
                "train_seen": inputs.train_dataset,
                "same_key_fresh": same_key_dataset,
                "cross_key_validation": inputs.validation_dataset,
            }
            if dataset_row_overlap_count(
                inputs.train_dataset,
                same_key_dataset,
            ) != 0:
                raise ValueError("K1-P same-key fresh rows overlap training rows")
            for split, dataset in split_datasets.items():
                round_datasets[(seed, split)] = dataset
                manifest_rows.append(
                    dataset_manifest_row(
                        rounds=rounds,
                        seed=seed,
                        split=split,
                        dataset=dataset,
                    )
                )
        datasets_by_round[rounds] = round_datasets

    manifest_rows.extend(reused_r5_dataset_manifest(source_dataset_manifest))
    write_jsonl(args.output_root / "dataset_manifest.jsonl", manifest_rows)
    cache_checks = validate_lower_cache_contract(
        output_root=args.output_root,
        manifest_rows=manifest_rows,
    )
    if not all(cache_checks.values()):
        raise ValueError(f"K1-P lower-round cache contract failed: {cache_checks}")
    source_checks = {**source_checks, **cache_checks}

    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    progress(
        args.output_root / "progress.jsonl",
        "k1p_round_calibration_start",
        expected_feature_rows=EXPECTED_FEATURE_ROWS,
        expected_scorer_rows=EXPECTED_SCORER_ROWS,
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    for rounds in LOWER_ROUNDS:
        exact_structures = {}
        wrong_structures = {}
        for seed in EXPECTED_SEEDS:
            task = task_map[(rounds, seed)]
            input_bits = int(
                datasets_by_round[rounds][(seed, "train_seen")].features.shape[1]
            )
            exact_structures[seed] = build_k1n_control(
                task=task,
                condition="exact_composition",
                input_bits=input_bits,
            ).runtime_structure
            wrong_structures[seed] = build_k1n_control(
                task=task,
                condition="wrong_sbox_semantics",
                input_bits=input_bits,
            ).runtime_structure
        round_features, round_scorers, round_results = evaluate_lower_round(
            rounds=rounds,
            datasets=datasets_by_round[rounds],
            exact_structures=exact_structures,
            wrong_sbox_structures=wrong_structures,
            batch_size=args.batch_size,
        )
        feature_rows.extend(round_features)
        scorer_rows.extend(round_scorers)
        result_rows.extend(round_results)

    r5_features, r5_scorers, r5_results = reuse_k1o_anchor(
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
    )
    feature_rows.extend(r5_features)
    scorer_rows.extend(r5_scorers)
    result_rows.extend(r5_results)
    write_jsonl(args.output_root / "feature_manifest.jsonl", feature_rows)
    write_jsonl(args.output_root / "scorer_manifest.jsonl", scorer_rows)
    write_jsonl(args.output_root / "results.jsonl", result_rows)
    write_round_csv(args.output_root / "round_calibration.csv", result_rows)

    gate = adjudicate_k1p(
        tasks=tasks,
        result_rows=result_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks=source_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "dataset_rows": len(manifest_rows),
        "expected_dataset_rows": 18,
        "feature_rows": len(feature_rows),
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "scorer_rows": len(scorer_rows),
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "training_rows": 0,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "remote_scale": gate["remote_scale"],
            "round_pass": gate["round_pass"],
            "round_results": gate["round_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
            "training_rows": 0,
            "optimizer_steps": 0,
        },
    )
    plot_report = render_k1p_svg(gate, args.output_root / "curves.svg")
    write_json(args.output_root / "plot_report.json", plot_report)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
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


def cache_argv(args: argparse.Namespace) -> list[str]:
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
        str(args.output_root / "unused-checkpoints"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "unused-training-results.jsonl"),
    ]


def dataset_manifest_row(
    *,
    rounds: int,
    seed: int,
    split: str,
    dataset: Any,
) -> dict[str, Any]:
    dataset_seed = seed
    key_scope = "train_key"
    if split == "same_key_fresh":
        dataset_seed += SAME_KEY_SEED_OFFSET
    elif split == "cross_key_validation":
        dataset_seed += 10_000
        key_scope = "validation_key"
    return {
        "run_id": RUN_ID,
        "source_run_id": None,
        "source_artifact_reused": False,
        "cipher_key": "uknit64",
        "rounds": rounds,
        "seed": seed,
        "split": split,
        "key_scope": key_scope,
        "dataset_seed": dataset_seed,
        "rows": int(dataset.features.shape[0]),
        "dataset_sha256": differential_dataset_sha256(dataset),
        "cache_dir": str(getattr(dataset, "cache_dir", "")),
    }


def reused_r5_dataset_manifest(
    source_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for source in source_rows:
        if source.get("cipher_key") != "uknit64":
            continue
        row = dict(source)
        upstream = row.get("run_id")
        row.update(
            {
                "run_id": RUN_ID,
                "source_run_id": upstream,
                "source_artifact_reused": True,
                "rounds": 5,
            }
        )
        rows.append(row)
    if len(rows) != 6:
        raise ValueError("K1-P requires six reused K1-O r5 dataset rows")
    return rows


def validate_lower_cache_contract(
    *,
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    lower = [row for row in manifest_rows if int(row.get("rounds", -1)) in LOWER_ROUNDS]
    r5 = [row for row in manifest_rows if int(row.get("rounds", -1)) == 5]
    progress_rows = read_jsonl(output_root / "progress.jsonl")
    return {
        "twelve_lower_dataset_rows_complete": len(lower) == 12,
        "six_reused_r5_dataset_rows_complete": len(r5) == 6,
        "all_lower_cache_payloads_present": all(
            all(
                (Path(str(row.get("cache_dir", ""))) / name).is_file()
                for name in ("metadata.json", "features.npy", "labels.npy")
            )
            for row in lower
        ),
        "all_r5_cache_payloads_present": all(
            all(
                (Path(str(row.get("cache_dir", ""))) / name).is_file()
                for name in ("metadata.json", "features.npy", "labels.npy")
            )
            for row in r5
        ),
        "twelve_lower_caches_created": sum(
            row.get("event") == "cache_start"
            and row.get("split") in {"train", "validation", "same_key_fresh"}
            for row in progress_rows
        )
        == 12,
        "durable_cache_progress_recorded": any(
            row.get("event") in {"cache_positive_chunk", "cache_negative_chunk"}
            for row in progress_rows
        ),
    }


def write_round_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "rounds",
        "seed",
        "split",
        "view",
        "rows",
        "auc",
        "zero_threshold_accuracy",
        "feature_dim",
        "dataset_sha256",
        "feature_sha256",
        "scorer_sha256",
        "source_run_id",
        "source_artifact_reused",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "dataset_manifest.jsonl",
        "feature_manifest.jsonl",
        "scorer_manifest.jsonl",
        "results.jsonl",
        "progress.jsonl",
        "gate.json",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-P output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "cache_argv",
    "dataset_manifest_row",
    "main",
    "read_tasks",
    "reused_r5_dataset_manifest",
    "validate_lower_cache_contract",
]
