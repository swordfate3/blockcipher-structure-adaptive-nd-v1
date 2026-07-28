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
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import (
    load_k1r_datasets,
    read_tasks,
)
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import (
    source_binding_checks as k1s_source_binding_checks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_TRAINING_ROWS,
    MODEL_TO_CONDITION,
    RUN_ID,
    adjudicate_k1t,
    build_k1t_readiness,
    candidate_protocol_frozen,
    evaluate_k1t_panel,
    expected_training_keys,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the uKNIT K1-T deterministic position-residual diagnostic."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1q-root", required=True, type=Path)
    parser.add_argument("--k1r-root", required=True, type=Path)
    parser.add_argument("--k1r-plan", required=True, type=Path)
    parser.add_argument("--k1s-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help="Verify frozen sources and zero-step model readiness, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-T run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-T plan does not match the frozen protocol")
    require_fresh_output_root(args.output_root)

    source_digests = {
        name: file_sha256(path) for name, path in source_artifact_paths(args).items()
    }
    k1q_gate = read_json(args.k1q_root / "gate.json")
    k1q_validation = read_json(args.k1q_root / "validation.json")
    k1r_gate = read_json(args.k1r_root / "gate.json")
    k1r_validation = read_json(args.k1r_root / "validation.json")
    k1s_gate = read_json(args.k1s_root / "gate.json")
    k1s_validation = read_json(args.k1s_root / "validation.json")
    dataset_manifest = [
        row
        for row in read_jsonl(args.k1q_root / "dataset_manifest.jsonl")
        if row.get("phase") == "confirmation" and int(row.get("cell", -1)) == 11
    ]
    k1r_checkpoints = read_json(args.k1r_root / "checkpoint_manifest.json")
    bound_k1s_checks = k1s_source_binding_checks(
        source_digests={
            name.removeprefix("bound_"): digest
            for name, digest in source_digests.items()
            if name.startswith("bound_")
        },
        k1q_gate=k1q_gate,
        k1q_validation=k1q_validation,
        k1r_gate=k1r_gate,
        k1r_validation=k1r_validation,
        dataset_manifest=dataset_manifest,
        checkpoint_entries=k1r_checkpoints.get("entries", []),
    )
    source_checks = source_binding_checks(
        source_digests=source_digests,
        k1s_gate=k1s_gate,
        k1s_validation=k1s_validation,
        bound_source_checks=bound_k1s_checks,
    )
    datasets = load_k1r_datasets(dataset_manifest)
    source_checks["six_cache_payload_digests_verified"] = len(datasets) == 6
    if not all(source_checks.values()):
        raise ValueError(f"K1-T source binding failed: {source_checks}")

    readiness = build_k1t_readiness(
        tasks=tasks,
        datasets=datasets,
        source_checks=source_checks,
    )
    if readiness.get("optimizer_step_authorized") is not True:
        raise ValueError(f"K1-T readiness failed: {readiness}")
    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            **readiness,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "source_digests": source_digests,
            "expected_source_digests": EXPECTED_SOURCE_DIGESTS,
        },
    )
    write_jsonl(args.output_root / "dataset_manifest.jsonl", dataset_manifest)
    if args.readiness_only:
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0
    train_argv = training_argv(args, args.k1q_root / "cache")
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != tasks:
        raise ValueError("K1-T training parser drifted from the frozen plan")
    train_main(train_argv)

    training_rows = read_jsonl(args.output_root / "results.jsonl")
    if len(training_rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("K1-T did not produce six training rows")
    cache_checks = cache_reuse_checks(read_jsonl(args.output_root / "progress.jsonl"))
    if not all(cache_checks.values()):
        raise ValueError(f"K1-T source cache reuse failed: {cache_checks}")
    checkpoint_manifest = build_checkpoint_manifest(training_rows)
    write_json(args.output_root / "checkpoint_manifest.json", checkpoint_manifest)
    progress(
        args.output_root / "progress.jsonl",
        "k1t_three_split_panel_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    k1r_anchor_rows = [
        row
        for row in read_jsonl(args.k1r_root / "controls.jsonl")
        if row.get("condition") == "exact_composition"
    ]
    evaluation_rows = evaluate_k1t_panel(
        tasks=tasks,
        training_rows=training_rows,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        k1r_anchor_rows=k1r_anchor_rows,
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    write_csv(args.output_root / "split_attribution.csv", evaluation_rows)
    gate = adjudicate_k1t(
        tasks=tasks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoint_manifest=checkpoint_manifest,
        readiness=readiness,
        source_checks=source_checks,
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
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
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
        evaluation_rows=len(evaluation_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def source_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    bound_paths = {
        "k1q_gate": args.k1q_root / "gate.json",
        "k1q_dataset_manifest": args.k1q_root / "dataset_manifest.jsonl",
        "k1q_results": args.k1q_root / "results.jsonl",
        "k1q_feature_manifest": args.k1q_root / "feature_manifest.jsonl",
        "k1q_scorer_manifest": args.k1q_root / "scorer_manifest.jsonl",
        "k1q_validation": args.k1q_root / "validation.json",
        "k1r_plan": args.k1r_plan,
        "k1r_gate": args.k1r_root / "gate.json",
        "k1r_checkpoint_manifest": args.k1r_root / "checkpoint_manifest.json",
        "k1r_results": args.k1r_root / "results.jsonl",
        "k1r_controls": args.k1r_root / "controls.jsonl",
        "k1r_validation": args.k1r_root / "validation.json",
    }
    return {
        **{f"bound_{name}": path for name, path in bound_paths.items()},
        "k1s_gate": args.k1s_root / "gate.json",
        "k1s_validation": args.k1s_root / "validation.json",
        "k1s_results": args.k1s_root / "results.jsonl",
        "k1s_feature_manifest": args.k1s_root / "feature_manifest.jsonl",
        "k1s_scorer_manifest": args.k1s_root / "scorer_manifest.jsonl",
        "k1s_checkpoint_manifest": args.k1s_root / "checkpoint_manifest.json",
    }


def training_argv(args: argparse.Namespace, source_cache_root: Path) -> list[str]:
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
    expected = {
        (seed, model, split)
        for seed in EXPECTED_SEEDS
        for model in MODEL_TO_CONDITION
        for split in ("train", "validation")
    }
    return {
        "twelve_training_validation_cache_reuses_exact": (
            len(source_events) == 12 and reuse_keys == expected
        ),
        "no_training_or_validation_cache_regenerated": not any(
            row.get("event") == "cache_start" for row in source_events
        ),
    }


def build_checkpoint_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    for row in rows:
        model = str(row.get("model"))
        condition = MODEL_TO_CONDITION.get(model)
        if condition not in CONTROL_MODELS:
            raise ValueError("K1-T training row has an unknown model")
        key = (int(row["seed"]), condition)
        if key in observed:
            raise ValueError(f"duplicate K1-T checkpoint source: {key}")
        observed.add(key)
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        if not checkpoint.is_file():
            raise ValueError(f"missing K1-T checkpoint: {checkpoint}")
        entries.append(
            {
                "cipher_key": "uknit64",
                "seed": int(row["seed"]),
                "condition": condition,
                "model": model,
                "selected_checkpoint": row["training"]["selected_checkpoint"],
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            }
        )
    if observed != expected_training_keys():
        raise ValueError("K1-T checkpoint sources are incomplete")
    return {"run_id": RUN_ID, "status": "pass", "entries": entries}


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
        raise ValueError("K1-T output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_manifest",
    "cache_reuse_checks",
    "main",
    "parse_args",
    "source_artifact_paths",
]
