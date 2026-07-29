from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.audit_uknit_family_ctspn_k1q import (
    build_structures,
    cache_argv,
    prepare_position_datasets,
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
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_position_k1bl import (
    ACTIVE_BIT_ROLE,
    ANCHOR_CELL,
    CONFIRMATION_PHASE,
    DISCOVERY_PHASE,
    ROUNDS,
    RUN_ID,
    adjudicate_k1bl,
    bind_discovery_input_differences,
    build_confirmation_tasks,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    candidate_bit_index,
    candidate_difference,
    evaluate_position,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the uKNIT r6 role-1 input bit across all native cells, then "
            "confirm the r5 cell11 anchor and at most two new candidates."
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
        raise ValueError(f"K1-BL run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-BL feature batch size is frozen at 256")

    discovery_tasks = read_tasks(args.plan)
    task_checks = validate_discovery_tasks(discovery_tasks)
    if not all(task_checks.values()):
        raise ValueError(f"K1-BL discovery protocol is invalid: {task_checks}")

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
                "task_checks": task_checks,
                "rounds": ROUNDS,
                "feature_batch_size": args.batch_size,
                "device": args.device,
            },
        )
        progress(args.output_root / "progress.jsonl", "k1bl_preflight_passed")

    train_args = parse_train_args(cache_argv(args))
    dataset_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    progress(
        args.output_root / "progress.jsonl",
        "k1bl_discovery_start",
        candidate_count=len(discovery_tasks),
    )
    for index, task in enumerate(discovery_tasks, start=1):
        cell = int(task["model_options"]["active_cell"])
        datasets, manifests = prepare_k1bl_datasets(
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
        features, scorers, results = evaluate_k1bl_position(
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
            "k1bl_discovery_position_done",
            cell=cell,
            input_difference=f"0x{candidate_difference(cell):016x}",
            index=index,
            total=len(discovery_tasks),
        )

    selection = select_discovery_candidates(result_rows)
    write_json(args.output_root / "selection.json", selection)
    progress(
        args.output_root / "progress.jsonl",
        "k1bl_discovery_done",
        anchor_passes_discovery=selection["anchor_passes_discovery"],
        selected_cells=selection["selected_cells"],
    )

    confirmation_tasks = build_confirmation_tasks(
        discovery_tasks, selection["selected_cells"]
    )
    confirmation_checks = validate_confirmation_tasks(
        confirmation_tasks, selection["selected_cells"]
    )
    if not all(confirmation_checks.values()):
        raise ValueError(f"K1-BL confirmation protocol is invalid: {confirmation_checks}")
    progress(
        args.output_root / "progress.jsonl",
        "k1bl_confirmation_start",
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
        datasets, manifests = prepare_k1bl_datasets(
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
        features, scorers, results = evaluate_k1bl_position(
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
            "k1bl_confirmation_task_done",
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
    gate = adjudicate_k1bl(
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
    tasks = tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    return bind_discovery_input_differences(tasks)


def prepare_k1bl_datasets(
    *,
    task: dict[str, Any],
    train_args: argparse.Namespace,
    output_root: Path,
    phase: str,
    cell: int,
    index: int,
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        bit_index=candidate_bit_index(cell),
        active_bit_role=ACTIVE_BIT_ROLE,
        input_difference=candidate_difference(cell),
    )


def evaluate_k1bl_position(
    *,
    phase: str,
    task: Mapping[str, Any],
    datasets: Mapping[str, Any],
    exact: Any,
    wrong: Any,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cell = int(task["model_options"]["active_cell"])
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
        bit_index=candidate_bit_index(cell),
        active_bit_role=ACTIVE_BIT_ROLE,
        input_difference=candidate_difference(cell),
        label_shuffle_seed_base=20260729,
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
            "selected_cells": selection["selected_cells"],
            "confirmed_cells": gate["confirmed_cells"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    plot_report = render_k1q_svg(
        gate,
        output_root / "curves.svg",
        cipher_label="uKNIT",
        rounds=ROUNDS,
        anchor_cell=ANCHOR_CELL,
        anchor_label="r5最强位置",
        always_show_anchor=True,
        left_margin=0.13,
        subtitle=(
            "固定 bit_role=1、四对密文、严格负样本和精确五阶段特征；"
            "只把轮数从 r5 推到 r6，并在 16 个原生 cell 之间移动差分。"
        ),
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
        raise ValueError("K1-BL resume root does not match the frozen run")


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
        raise ValueError("K1-BL output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "read_tasks"]
