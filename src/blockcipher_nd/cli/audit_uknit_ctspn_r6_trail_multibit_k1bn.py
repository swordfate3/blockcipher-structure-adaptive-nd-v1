from __future__ import annotations

import argparse
import csv
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
)
from blockcipher_nd.cli.plot_uknit_ctspn_r6_trail_multibit_k1bn import (
    render_k1bn_svg,
)
from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_trail_multibit_k1bn import (
    BEAM_WIDTH,
    CANDIDATE_FAMILIES,
    CONFIRMATION_PHASE,
    DISCOVERY_PHASE,
    K1BM_REQUIRED_DECISION,
    OUTCOMES_PER_ACTIVE_CELL,
    ROUNDS,
    RUN_ID,
    SELECTED_PER_FAMILY,
    TRAIL_ROUNDS,
    TWO_CELL_PREFILTER_SIZE,
    adjudicate_k1bn,
    build_candidate_manifest,
    build_confirmation_tasks,
    build_discovery_tasks,
    select_discovery_candidates,
    validate_candidate_manifest,
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
    "innovation1_uknit_ctspn_r6_trail_multibit_k1bn_20260729.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen DDT/trail-guided multibit differences at uKNIT r6."
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
        raise ValueError(f"K1-BN run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-BN feature batch size is frozen at 256")
    config = load_config(args.config)
    source_plan = ROOT / str(config["source_plan"])
    source_gate_path = ROOT / str(config["source_gate"])
    source_gate = read_json(source_gate_path)
    template = read_k1bl_tasks(source_plan)[0]
    candidate_manifest = build_candidate_manifest()
    manifest_checks = validate_candidate_manifest(candidate_manifest)
    candidates = candidate_manifest["selected_candidates"]
    discovery_tasks = build_discovery_tasks(template, candidates)
    task_checks = validate_discovery_tasks(discovery_tasks, candidates)
    if not all(manifest_checks.values()) or not all(task_checks.values()):
        raise ValueError(
            f"K1-BN frozen candidates are invalid: {manifest_checks} {task_checks}"
        )

    if args.resume:
        validate_resume_root(args)
    else:
        require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
        write_json(args.output_root / "candidate_manifest.json", candidate_manifest)
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
                "candidate_manifest_sha256": file_sha256(
                    args.output_root / "candidate_manifest.json"
                ),
                "candidate_manifest_checks": manifest_checks,
                "task_checks": task_checks,
                "rounds": ROUNDS,
                "feature_batch_size": args.batch_size,
                "device": args.device,
            },
        )
        progress(args.output_root / "progress.jsonl", "k1bn_preflight_passed")

    train_args = parse_train_args(cache_arguments(args, source_plan))
    dataset_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    progress(
        args.output_root / "progress.jsonl",
        "k1bn_discovery_start",
        candidate_count=len(discovery_tasks),
        candidate_families=list(CANDIDATE_FAMILIES),
    )
    for index, task in enumerate(discovery_tasks, start=1):
        datasets, manifests = prepare_candidate_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=DISCOVERY_PHASE,
            index=index,
            total=len(discovery_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_candidate(
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
            "k1bn_discovery_candidate_done",
            candidate_id=task["candidate_id"],
            candidate_family=task["candidate_family"],
            index=index,
            total=len(discovery_tasks),
        )

    selection = select_discovery_candidates(result_rows, candidates)
    write_json(args.output_root / "selection.json", selection)
    progress(
        args.output_root / "progress.jsonl",
        "k1bn_discovery_done",
        selected_candidate_ids=selection["selected_candidate_ids"],
        selected_by_family=selection["selected_by_family"],
    )

    confirmation_tasks = build_confirmation_tasks(
        discovery_tasks, selection["selected_candidate_ids"]
    )
    confirmation_checks = validate_confirmation_tasks(
        confirmation_tasks, selection["selected_candidate_ids"]
    )
    if not all(confirmation_checks.values()):
        raise ValueError(f"K1-BN confirmation protocol is invalid: {confirmation_checks}")
    if confirmation_tasks:
        progress(
            args.output_root / "progress.jsonl",
            "k1bn_confirmation_start",
            task_count=len(confirmation_tasks),
            selected_candidate_ids=selection["selected_candidate_ids"],
        )
    for index, task in enumerate(confirmation_tasks, start=1):
        datasets, manifests = prepare_candidate_datasets(
            task=task,
            train_args=train_args,
            output_root=args.output_root,
            phase=CONFIRMATION_PHASE,
            index=index,
            total=len(confirmation_tasks),
        )
        dataset_rows.extend(manifests)
        exact, wrong = build_structures(task, datasets)
        features, scorers, results = evaluate_candidate(
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
            "k1bn_confirmation_task_done",
            candidate_id=task["candidate_id"],
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
        "candidate_manifest_bound": (
            read_json(args.output_root / "preflight.json").get(
                "candidate_manifest_sha256"
            )
            == file_sha256(args.output_root / "candidate_manifest.json")
        ),
        "k1bm_single_bit_source_completed_hold": source_gate.get("status") == "hold"
        and source_gate.get("decision") == K1BM_REQUIRED_DECISION
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
    gate = adjudicate_k1bn(
        candidate_manifest=candidate_manifest,
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
        "candidate_families": list(CANDIDATE_FAMILIES),
        "cell_local_pool_size": 176,
        "two_cell_prefilter_size": TWO_CELL_PREFILTER_SIZE,
        "selected_per_family": SELECTED_PER_FAMILY,
        "beam_width": BEAM_WIDTH,
        "outcomes_per_active_cell": OUTCOMES_PER_ACTIVE_CELL,
        "trail_rounds": TRAIL_ROUNDS,
        "discovery_seed": 2,
        "confirmation_seeds": [3, 4],
        "discovery_samples_per_class": 1024,
        "confirmation_samples_per_class": 2048,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "runtime_round_start": 4,
        "runtime_rounds": 2,
        "max_selected_per_family": 1,
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
        raise ValueError(f"K1-BN config is not frozen: {mismatches}")
    for field in ("source_plan", "source_gate"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"K1-BN config requires {field}")
    return payload


def cache_arguments(args: argparse.Namespace, source_plan: Path) -> list[str]:
    proxy = argparse.Namespace(**vars(args))
    proxy.plan = source_plan
    return cache_argv(proxy)


def prepare_candidate_datasets(
    *,
    task: dict[str, Any],
    train_args: argparse.Namespace,
    output_root: Path,
    phase: str,
    index: int,
    total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_index = int(task["candidate_index"])
    datasets, rows = prepare_position_datasets(
        task=task,
        train_args=train_args,
        output_root=output_root,
        phase=phase,
        cell=candidate_index,
        index=index,
        total=total,
        run_id=RUN_ID,
        rounds=ROUNDS,
        bit_index=candidate_index,
        active_bit_role=-1,
        input_difference=int(task["input_difference"]),
    )
    return datasets, [_normalize_candidate_row(row, task) for row in rows]


def evaluate_candidate(
    *,
    phase: str,
    task: Mapping[str, Any],
    datasets: Mapping[str, Any],
    exact: Any,
    wrong: Any,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_index = int(task["candidate_index"])
    feature_rows, scorer_rows, result_rows = evaluate_position(
        phase=phase,
        cell=candidate_index,
        seed=int(task["seed"]),
        datasets=datasets,
        exact_structure=exact,
        wrong_sbox_structure=wrong,
        batch_size=batch_size,
        run_id=RUN_ID,
        rounds=ROUNDS,
        bit_index=candidate_index,
        active_bit_role=-1,
        input_difference=int(task["input_difference"]),
        label_shuffle_seed_base=20260729,
    )
    return (
        [_normalize_candidate_row(row, task) for row in feature_rows],
        [_normalize_candidate_row(row, task) for row in scorer_rows],
        [_normalize_candidate_row(row, task) for row in result_rows],
    )


def _normalize_candidate_row(
    row: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(row)
    normalized.pop("cell", None)
    normalized.pop("bit_index", None)
    normalized.pop("active_bit_role", None)
    normalized.update(
        {
            "candidate_id": task["candidate_id"],
            "candidate_index": task["candidate_index"],
            "candidate_family": task["candidate_family"],
            "candidate_family_rank": task["candidate_family_rank"],
            "candidate_source_cells": task["candidate_source_cells"],
            "candidate_source_nibbles": task["candidate_source_nibbles"],
            "input_weight": int(task["input_difference"]).bit_count(),
            "trail_log2_probability": task["trail_log2_probability"],
            "trail_total_active_sboxes": task["trail_total_active_sboxes"],
        }
    )
    return normalized


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
    write_candidate_csv(output_root / "difference_candidates.csv", result_rows)


def write_candidate_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "phase",
        "candidate_id",
        "candidate_family",
        "candidate_family_rank",
        "input_difference_hex",
        "input_weight",
        "trail_log2_probability",
        "trail_total_active_sboxes",
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
        "fit_rows",
        "pairs_per_sample",
        "negative_mode",
        "training_performed",
        "neural_parameter_count",
        "optimizer_steps",
        "epochs",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
            "selected_candidate_ids": selection["selected_candidate_ids"],
            "confirmed_candidate_ids": gate["confirmed_candidate_ids"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    plot_report = render_k1bn_svg(gate, output_root / "curves.svg")
    write_json(output_root / "plot_report.json", plot_report)
    progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        selected_candidate_ids=selection["selected_candidate_ids"],
        confirmed_candidate_ids=gate["confirmed_candidate_ids"],
        result_rows=len(result_rows),
    )


def validate_resume_root(args: argparse.Namespace) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("config_sha256") != file_sha256(args.config)
        or preflight.get("config") != str(args.config)
    ):
        raise ValueError("K1-BN resume root does not match the frozen run")


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "candidate_manifest.json",
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
        raise ValueError("K1-BN output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_candidate",
    "load_config",
    "main",
    "parse_args",
    "prepare_candidate_datasets",
]
