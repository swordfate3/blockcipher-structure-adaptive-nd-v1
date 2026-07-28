from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.audit_uknit_family_ctspn_k1q import (
    cache_argv,
    require_fresh_output_root,
    write_position_csv,
)
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1q import render_k1q_svg
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
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    SAME_KEY_SEED_OFFSET,
    dataset_row_overlap_count,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    CONFIRMATION_PHASE,
    DISCOVERY_PHASE,
    EXPECTED_SPLITS,
    candidate_bit_index,
    candidate_difference,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_difference_position_k1ah import (
    CONFIRMATION_SEEDS,
    RUN_ID,
    adjudicate_k1ah,
    bind_discovery_input_differences,
    build_confirmation_tasks,
    evaluate_position,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_qualification_k1ag import (
    CONFIG_PATH as K1AG_CONFIG_PATH,
    load_and_validate_config as load_k1ag_config,
    run_qualification,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the same Midori64 role-1 difference across sixteen native cells "
            "at r4, then confirm at most two candidates on untouched seeds and keys."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AH run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-AH feature batch size is frozen at 256")

    discovery_tasks = read_tasks(args.plan)
    task_checks = validate_discovery_tasks(discovery_tasks)
    if not all(task_checks.values()):
        raise ValueError(f"K1-AH discovery task protocol is invalid: {task_checks}")
    k1ag_payload = run_qualification(load_k1ag_config(K1AG_CONFIG_PATH))
    k1ag_gate = k1ag_payload["gate"]
    if k1ag_gate.get("status") != "pass":
        raise ValueError("K1-AH requires a live passing K1-AG qualification")

    if args.resume:
        validate_resume_root(args)
    else:
        require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
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
                "k1ag_config": str(K1AG_CONFIG_PATH),
                "k1ag_config_sha256": file_sha256(K1AG_CONFIG_PATH),
                "k1ag_decision": k1ag_gate["decision"],
                "task_checks": task_checks,
                "feature_batch_size": args.batch_size,
                "device": args.device,
                "training_rows": 0,
                "neural_parameter_count": 0,
                "optimizer_steps": 0,
                "epochs": 0,
            },
        )
        progress(args.output_root / "progress.jsonl", "k1ah_preflight_passed")

    train_args = parse_train_args(cache_argv(args))
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    progress(
        args.output_root / "progress.jsonl",
        "k1ah_discovery_start",
        candidate_count=len(discovery_tasks),
    )
    for index, task in enumerate(discovery_tasks, start=1):
        cell = int(task["model_options"]["active_cell"])
        datasets, manifests = prepare_position_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=DISCOVERY_PHASE,
            cell=cell,
            index=index,
            total=len(discovery_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_position(
            phase=DISCOVERY_PHASE,
            cell=cell,
            seed=int(task["seed"]),
            datasets=datasets,
            exact_structure=exact,
            wrong_sbox_structure=wrong,
            batch_size=args.batch_size,
        )
        feature_rows.extend(features)
        scorer_rows.extend(scorers)
        result_rows.extend(results)
        progress(
            args.output_root / "progress.jsonl",
            "k1ah_discovery_position_done",
            cell=cell,
            input_difference=f"0x{candidate_difference(cell):016x}",
            index=index,
            total=len(discovery_tasks),
        )

    selection = select_discovery_candidates(result_rows)
    write_json(args.output_root / "selection.json", selection)
    write_partial_artifacts(
        args.output_root,
        dataset_rows=dataset_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
    )
    progress(
        args.output_root / "progress.jsonl",
        "k1ah_discovery_done",
        selected_cells=selection["selected_cells"],
    )

    confirmation_tasks = build_confirmation_tasks(
        discovery_tasks,
        selection["selected_cells"],
    )
    confirmation_checks = validate_confirmation_tasks(
        confirmation_tasks,
        selection["selected_cells"],
    )
    if not all(confirmation_checks.values()):
        raise ValueError(
            f"K1-AH confirmation task protocol is invalid: {confirmation_checks}"
        )
    if confirmation_tasks:
        progress(
            args.output_root / "progress.jsonl",
            "k1ah_confirmation_start",
            task_count=len(confirmation_tasks),
            cells=sorted(
                {
                    int(task["model_options"]["active_cell"])
                    for task in confirmation_tasks
                }
            ),
        )
    for index, task in enumerate(confirmation_tasks, start=1):
        cell = int(task["model_options"]["active_cell"])
        datasets, manifests = prepare_position_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=CONFIRMATION_PHASE,
            cell=cell,
            index=index,
            total=len(confirmation_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_position(
            phase=CONFIRMATION_PHASE,
            cell=cell,
            seed=int(task["seed"]),
            datasets=datasets,
            exact_structure=exact,
            wrong_sbox_structure=wrong,
            batch_size=args.batch_size,
        )
        feature_rows.extend(features)
        scorer_rows.extend(scorers)
        result_rows.extend(results)
        write_partial_artifacts(
            args.output_root,
            dataset_rows=dataset_rows,
            feature_rows=feature_rows,
            scorer_rows=scorer_rows,
            result_rows=result_rows,
        )
        progress(
            args.output_root / "progress.jsonl",
            "k1ah_confirmation_task_done",
            cell=cell,
            seed=int(task["seed"]),
            index=index,
            total=len(confirmation_tasks),
        )

    progress_rows = read_jsonl(args.output_root / "progress.jsonl")
    source_checks = {
        "preflight_plan_bound": (
            read_json(args.output_root / "preflight.json").get("plan_sha256")
            == file_sha256(args.plan)
        ),
        "live_k1ag_qualification_exact": (
            k1ag_gate.get("decision")
            == "innovation1_uknit_family_midori64_k1ag_qualified"
            and bool(k1ag_gate.get("protocol_checks"))
            and all(k1ag_gate.get("protocol_checks", {}).values())
            and bool(k1ag_gate.get("research_checks"))
            and all(k1ag_gate.get("research_checks", {}).values())
        ),
        "confirmation_tasks_frozen": all(confirmation_checks.values()),
        "fresh_splits_disjoint_from_train": all(
            int(row.get("row_overlap_with_train", -1)) == 0
            for row in dataset_rows
            if row.get("split") != "train_seen"
        ),
        "durable_cache_progress_recorded": any(
            row.get("event")
            in {"cache_positive_chunk", "cache_negative_chunk", "cache_reuse"}
            for row in progress_rows
        ),
    }
    gate = adjudicate_k1ah(
        discovery_tasks=discovery_tasks,
        selection=selection,
        dataset_rows=dataset_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_checks=source_checks,
    )
    finalize(
        args.output_root,
        gate=gate,
        selection=selection,
        dataset_rows=dataset_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def read_tasks(path: Path) -> list[dict[str, Any]]:
    parsed = tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    return bind_discovery_input_differences(parsed)


def prepare_position_datasets(
    *,
    task: dict[str, Any],
    train_args: argparse.Namespace,
    output_root: Path,
    phase: str,
    cell: int,
    index: int,
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = prepare_task_inputs(
        task,
        train_args,
        progress_path=str(output_root / "progress.jsonl"),
        index=index,
        total=total,
    )
    train_key, validation_key = resolve_task_keys(task)
    same_key_task = {**task, "validation_key": train_key}
    same_key_cipher = build_cipher("midori64", 4, key=train_key)
    same_key_config = build_dataset_config(
        same_key_task,
        cipher=same_key_cipher,
        samples_per_class=validation_samples_per_class(task),
        seed=int(task["seed"]) + SAME_KEY_SEED_OFFSET,
        split="validation",
    )
    same_key_dataset = make_task_dataset(
        same_key_config,
        train_args,
        same_key_task,
        split="same_key_fresh",
        progress_path=str(output_root / "progress.jsonl"),
        index=index,
        total=total,
    )
    datasets = {
        "train_seen": inputs.train_dataset,
        "same_key_fresh": same_key_dataset,
        "cross_key_validation": inputs.validation_dataset,
    }
    train_dataset = datasets["train_seen"]
    manifests = []
    for split in EXPECTED_SPLITS:
        dataset = datasets[split]
        overlap = (
            0
            if split == "train_seen"
            else dataset_row_overlap_count(train_dataset, dataset)
        )
        key = train_key if split != "cross_key_validation" else validation_key
        cache_dir = Path(str(getattr(dataset, "cache_dir", "")))
        manifests.append(
            {
                "run_id": RUN_ID,
                "phase": phase,
                "cipher_key": "midori64",
                "rounds": 4,
                "cell": cell,
                "bit_index": candidate_bit_index(cell),
                "active_bit_role": 1,
                "input_difference": candidate_difference(cell),
                "input_difference_hex": f"0x{candidate_difference(cell):016x}",
                "seed": int(task["seed"]),
                "dataset_seed": dataset_seed(int(task["seed"]), split),
                "split": split,
                "key_scope": (
                    "validation_key" if split == "cross_key_validation" else "train_key"
                ),
                "key_hex": f"0x{int(key):032x}",
                "rows": int(dataset.features.shape[0]),
                "dataset_sha256": differential_dataset_sha256(dataset),
                "cache_dir": str(cache_dir),
                "cache_payloads_present": all(
                    (cache_dir / name).is_file()
                    for name in ("metadata.json", "features.npy", "labels.npy")
                ),
                "row_overlap_with_train": overlap,
            }
        )
    return datasets, manifests


def build_structures(
    task: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> tuple[Any, Any]:
    input_bits = int(datasets["train_seen"].features.shape[1])
    pair_bits = input_bits // int(task["pairs_per_sample"])
    models = []
    for model_key in (
        "runtime_spn_ct_k1n_exact_composition_true",
        "runtime_spn_ct_k1n_exact_composition_wrong_sbox",
    ):
        models.append(
            build_model(
                model_key,
                input_bits=input_bits,
                hidden_bits=32,
                pair_bits=pair_bits,
                structure="SPN",
                model_options=dict(task["model_options"]),
            ).runtime_structure
        )
    return models[0], models[1]


def dataset_seed(seed: int, split: str) -> int:
    if split == "same_key_fresh":
        return seed + SAME_KEY_SEED_OFFSET
    if split == "cross_key_validation":
        return seed + 10_000
    return seed


def write_partial_artifacts(
    output_root: Path,
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    write_jsonl(output_root / "feature_manifest.jsonl", feature_rows)
    write_jsonl(output_root / "scorer_manifest.jsonl", scorer_rows)
    write_jsonl(output_root / "results.jsonl", result_rows)
    write_position_csv(output_root / "difference_position.csv", result_rows)


def finalize(
    output_root: Path,
    *,
    gate: Mapping[str, Any],
    selection: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_partial_artifacts(
        output_root,
        dataset_rows=dataset_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "dataset_rows": len(dataset_rows),
        "feature_rows": len(feature_rows),
        "scorer_rows": len(scorer_rows),
        "result_rows": len(result_rows),
        "training_rows": 0,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    write_json(output_root / "gate.json", gate)
    write_json(output_root / "validation.json", validation)
    write_json(
        output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "remote_scale": gate["remote_scale"],
            "selected_cells": selection["selected_cells"],
            "confirmed_cells": gate["confirmed_cells"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
            "dataset_rows": len(dataset_rows),
            "feature_rows": len(feature_rows),
            "scorer_rows": len(scorer_rows),
            "result_rows": len(result_rows),
            "training_rows": 0,
            "optimizer_steps": 0,
        },
    )
    plot_report = render_k1q_svg(
        gate,
        output_root / "curves.svg",
        cipher_label="Midori64",
        rounds=4,
        confirmation_seeds=CONFIRMATION_SEEDS,
        anchor_cell=1,
    )
    write_json(output_root / "plot_report.json", plot_report)
    progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        selected_cells=selection["selected_cells"],
        confirmed_cells=gate["confirmed_cells"],
        result_rows=len(result_rows),
    )


def validate_resume_root(args: argparse.Namespace) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("plan") != str(args.plan)
    ):
        raise ValueError("K1-AH resume root does not match the frozen run")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_structures",
    "dataset_seed",
    "main",
    "parse_args",
    "prepare_position_datasets",
    "read_tasks",
]
