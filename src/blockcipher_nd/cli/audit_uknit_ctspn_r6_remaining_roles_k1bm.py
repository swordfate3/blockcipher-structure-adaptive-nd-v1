from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.audit_uknit_ctspn_r6_position_k1bl import (
    read_tasks as read_k1bl_tasks,
)
from blockcipher_nd.cli.audit_uknit_family_ctspn_k1q import (
    build_structures,
    cache_argv,
    prepare_position_datasets,
    write_position_csv,
)
from blockcipher_nd.cli.plot_uknit_ctspn_r6_remaining_roles_k1bm import (
    render_k1bm_svg,
)
from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_remaining_roles_k1bm import (
    ACTIVE_BIT_ROLES,
    CONFIRMATION_PHASE,
    DISCOVERY_PHASE,
    K1BL_REQUIRED_DECISION,
    ROUNDS,
    RUN_ID,
    adjudicate_k1bm,
    build_confirmation_tasks,
    build_discovery_tasks,
    candidate_bit_index,
    candidate_difference,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    evaluate_position,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_ctspn_r6_remaining_roles_k1bm_20260729.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan uKNIT r6 single-bit roles 0, 2, and 3 across all native cells."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-BM run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-BM feature batch size is frozen at 256")
    config = load_config(args.config)
    source_plan = ROOT / str(config["source_plan"])
    source_gate_path = ROOT / str(config["source_gate"])
    source_gate = read_json(source_gate_path)
    role1_templates = read_k1bl_tasks(source_plan)
    discovery_tasks = build_discovery_tasks(role1_templates)
    task_checks = validate_discovery_tasks(discovery_tasks)
    if not all(task_checks.values()):
        raise ValueError(f"K1-BM discovery protocol is invalid: {task_checks}")

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
                "config": str(args.config),
                "config_sha256": file_sha256(args.config),
                "source_plan": str(source_plan),
                "source_plan_sha256": file_sha256(source_plan),
                "source_gate": str(source_gate_path),
                "source_gate_decision": source_gate.get("decision"),
                "task_checks": task_checks,
                "rounds": ROUNDS,
                "active_bit_roles": list(ACTIVE_BIT_ROLES),
                "feature_batch_size": args.batch_size,
                "device": args.device,
            },
        )
        progress(args.output_root / "progress.jsonl", "k1bm_preflight_passed")

    train_args = parse_train_args(cache_arguments(args, source_plan))
    dataset_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    progress(
        args.output_root / "progress.jsonl",
        "k1bm_discovery_start",
        candidate_count=len(discovery_tasks),
        active_bit_roles=list(ACTIVE_BIT_ROLES),
    )
    for index, task in enumerate(discovery_tasks, start=1):
        datasets, manifests = prepare_k1bm_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=DISCOVERY_PHASE,
            index=index,
            total=len(discovery_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_k1bm_position(
            phase=DISCOVERY_PHASE,
            task=task,
            datasets=datasets,
            exact=exact,
            wrong=wrong,
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
            "k1bm_discovery_candidate_done",
            active_bit_role=int(task["model_options"]["active_bit_role"]),
            cell=int(task["model_options"]["active_cell"]),
            bit_index=_task_bit_index(task),
            index=index,
            total=len(discovery_tasks),
        )

    selection = select_discovery_candidates(result_rows)
    write_json(args.output_root / "selection.json", selection)
    progress(
        args.output_root / "progress.jsonl",
        "k1bm_discovery_done",
        selected_bit_indices=selection["selected_bit_indices"],
        selected_by_role=selection["selected_by_role"],
    )

    confirmation_tasks = build_confirmation_tasks(
        discovery_tasks, selection["selected_bit_indices"]
    )
    confirmation_checks = validate_confirmation_tasks(
        confirmation_tasks, selection["selected_bit_indices"]
    )
    if not all(confirmation_checks.values()):
        raise ValueError(f"K1-BM confirmation protocol is invalid: {confirmation_checks}")
    if confirmation_tasks:
        progress(
            args.output_root / "progress.jsonl",
            "k1bm_confirmation_start",
            task_count=len(confirmation_tasks),
            selected_bit_indices=selection["selected_bit_indices"],
        )
    for index, task in enumerate(confirmation_tasks, start=1):
        datasets, manifests = prepare_k1bm_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=CONFIRMATION_PHASE,
            index=index,
            total=len(confirmation_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_k1bm_position(
            phase=CONFIRMATION_PHASE,
            task=task,
            datasets=datasets,
            exact=exact,
            wrong=wrong,
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
            "k1bm_confirmation_task_done",
            bit_index=_task_bit_index(task),
            seed=int(task["seed"]),
            index=index,
            total=len(confirmation_tasks),
        )

    progress_rows = read_jsonl(args.output_root / "progress.jsonl")
    source_checks = {
        "preflight_config_bound": (
            read_json(args.output_root / "preflight.json").get("config_sha256")
            == file_sha256(args.config)
        ),
        "k1bl_role1_source_completed_hold": source_gate.get("status") == "hold"
        and source_gate.get("decision") == K1BL_REQUIRED_DECISION
        and not source_gate.get("failed_protocol_checks"),
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
    gate = adjudicate_k1bm(
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


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_id": RUN_ID,
        "cipher": "uknit64",
        "rounds": ROUNDS,
        "active_bit_roles": list(ACTIVE_BIT_ROLES),
        "cells": 16,
        "discovery_seed": 2,
        "confirmation_seeds": [3, 4],
        "discovery_samples_per_class": 1024,
        "confirmation_samples_per_class": 2048,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "runtime_round_start": 4,
        "runtime_rounds": 2,
        "max_selected_per_role": 1,
        "auc_floor": 0.55,
        "raw_margin": 0.01,
        "label_shuffle_margin": 0.03,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"K1-BM config is not frozen: {mismatches}")
    for field in ("source_plan", "source_gate"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"K1-BM config requires {field}")
    return payload


def cache_arguments(args: argparse.Namespace, source_plan: Path) -> list[str]:
    proxy = argparse.Namespace(**vars(args))
    proxy.plan = source_plan
    return cache_argv(proxy)


def prepare_k1bm_datasets(
    *,
    task: dict[str, Any],
    train_args: argparse.Namespace,
    output_root: Path,
    phase: str,
    index: int,
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell = int(task["model_options"]["active_cell"])
    role = int(task["model_options"]["active_bit_role"])
    return prepare_position_datasets(
        task=task,
        train_args=train_args,
        output_root=output_root,
        phase=phase,
        cell=cell,
        index=index,
        total=total,
        run_id=RUN_ID,
        rounds=ROUNDS,
        bit_index=candidate_bit_index(cell, role),
        active_bit_role=role,
        input_difference=candidate_difference(cell, role),
    )


def evaluate_k1bm_position(
    *,
    phase: str,
    task: Mapping[str, Any],
    datasets: Mapping[str, Any],
    exact: Any,
    wrong: Any,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cell = int(task["model_options"]["active_cell"])
    role = int(task["model_options"]["active_bit_role"])
    bit_index = candidate_bit_index(cell, role)
    return evaluate_position(
        phase=phase,
        cell=cell,
        seed=int(task["seed"]),
        datasets=datasets,
        exact_structure=exact,
        wrong_sbox_structure=wrong,
        batch_size=batch_size,
        run_id=RUN_ID,
        rounds=ROUNDS,
        bit_index=bit_index,
        active_bit_role=role,
        input_difference=1 << bit_index,
        label_shuffle_seed_base=20260729,
    )


def _task_bit_index(task: Mapping[str, Any]) -> int:
    return candidate_bit_index(
        int(task["model_options"]["active_cell"]),
        int(task["model_options"]["active_bit_role"]),
    )


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
            "selected_bit_indices": selection["selected_bit_indices"],
            "confirmed_bit_indices": gate["confirmed_bit_indices"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    plot_report = render_k1bm_svg(gate, output_root / "curves.svg")
    write_json(output_root / "plot_report.json", plot_report)
    progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        selected_bit_indices=selection["selected_bit_indices"],
        confirmed_bit_indices=gate["confirmed_bit_indices"],
        result_rows=len(result_rows),
    )


def validate_resume_root(args: argparse.Namespace) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("config_sha256") != file_sha256(args.config)
        or preflight.get("config") != str(args.config)
    ):
        raise ValueError("K1-BM resume root does not match the frozen run")


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "selection.json",
        "dataset_manifest.jsonl",
        "feature_manifest.jsonl",
        "scorer_manifest.jsonl",
        "results.jsonl",
        "progress.jsonl",
        "gate.json",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-BM output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_config", "main", "parse_args"]
