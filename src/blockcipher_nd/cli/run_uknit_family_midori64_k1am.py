from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.run_uknit_family_midori64_k1ai import load_k1ai_datasets
from blockcipher_nd.cli.run_uknit_family_midori64_k1ak import read_tasks
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_semantic_contrast_k1am import (
    CONTRAST_MARGIN,
    CONTRAST_SCALE,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_TRAINING_ROWS,
    ORIENTATIONS,
    ORIENTATION_MODELS,
    ORIENTATION_OPTIONS,
    RUN_ID,
    adjudicate_k1am,
    build_k1am_model,
    build_model_checks,
    candidate_protocol_frozen,
    evaluate_k1am_panel,
    expected_dataset_keys,
    expected_training_keys,
    source_binding_checks,
    task_map,
    training_map,
    training_protocol_frozen,
)
from blockcipher_nd.training import TrainingConfig, train_binary_classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the K1-AM fixed-budget correct-versus-swapped S-box semantic "
            "contrast panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--k1ak-root", required=True, type=Path)
    parser.add_argument("--k1al-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-training", action="store_true")
    resume.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AM run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-AM plan does not match the frozen protocol")

    source_paths = {
        "k1ak_gate": args.k1ak_root / "gate.json",
        "k1ak_validation": args.k1ak_root / "validation.json",
        "k1ak_controls": args.k1ak_root / "controls.jsonl",
        "k1ak_dataset_manifest": args.k1ak_root / "dataset_manifest.jsonl",
        "k1al_gate": args.k1al_root / "gate.json",
        "k1al_validation": args.k1al_root / "validation.json",
        "k1al_results": args.k1al_root / "results.jsonl",
    }
    if not all(path.is_file() for path in source_paths.values()):
        raise ValueError("K1-AM source evidence is incomplete")
    source_digests = {name: file_sha256(path) for name, path in source_paths.items()}
    k1ak_gate = read_json(source_paths["k1ak_gate"])
    k1ak_validation = read_json(source_paths["k1ak_validation"])
    k1ak_controls = read_jsonl(source_paths["k1ak_controls"])
    dataset_manifest = read_jsonl(source_paths["k1ak_dataset_manifest"])
    k1al_gate = read_json(source_paths["k1al_gate"])
    k1al_validation = read_json(source_paths["k1al_validation"])
    k1al_results = read_jsonl(source_paths["k1al_results"])
    source_checks = source_binding_checks(
        k1ak_gate=k1ak_gate,
        k1ak_validation=k1ak_validation,
        k1ak_controls=k1ak_controls,
        dataset_manifest=dataset_manifest,
        k1al_gate=k1al_gate,
        k1al_validation=k1al_validation,
        k1al_results=k1al_results,
        source_digests=source_digests,
    )
    model_checks = build_model_checks(tasks)
    datasets = load_k1ai_datasets(dataset_manifest)
    source_checks["six_dataset_payload_digests_verified"] = (
        set(datasets) == expected_dataset_keys()
    )
    source_checks["source_artifact_hashes_live"] = (
        source_digests == EXPECTED_SOURCE_DIGESTS
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AM source binding failed: {source_checks}")
    if not all(model_checks.values()):
        raise ValueError(f"K1-AM model binding failed: {model_checks}")

    if args.resume_evaluation:
        validate_resume_root(args, source_digests)
        training_rows = read_jsonl(args.output_root / "results.jsonl")
        checkpoint_manifest = read_json(args.output_root / "checkpoint_manifest.json")
    elif args.resume_training:
        validate_training_resume_root(
            args,
            source_digests=source_digests,
            dataset_manifest=dataset_manifest,
        )
        progress(
            args.output_root / "progress.jsonl",
            "training_resume_start",
            expected_training_rows=EXPECTED_TRAINING_ROWS,
        )
        training_rows = train_panel(
            tasks=tasks,
            datasets=datasets,
            output_root=args.output_root,
            device=args.device,
            resume_training=True,
        )
        checkpoint_manifest = build_checkpoint_manifest(training_rows)
        write_json(
            args.output_root / "checkpoint_manifest.json",
            checkpoint_manifest,
        )
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
                "k1ak_root": str(args.k1ak_root),
                "k1al_root": str(args.k1al_root),
                "source_digests": source_digests,
                "expected_source_digests": EXPECTED_SOURCE_DIGESTS,
                "source_checks": source_checks,
                "model_checks": model_checks,
                "dataset_generation_performed": False,
                "expected_training_rows": EXPECTED_TRAINING_ROWS,
                "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
            },
        )
        write_jsonl(args.output_root / "dataset_manifest.jsonl", dataset_manifest)
        progress(
            args.output_root / "progress.jsonl",
            "run_start",
            expected_training_rows=EXPECTED_TRAINING_ROWS,
            expected_evaluation_rows=EXPECTED_EVALUATION_ROWS,
        )
        training_rows = train_panel(
            tasks=tasks,
            datasets=datasets,
            output_root=args.output_root,
            device=args.device,
        )
        write_jsonl(args.output_root / "results.jsonl", training_rows)
        checkpoint_manifest = build_checkpoint_manifest(training_rows)
        write_json(
            args.output_root / "checkpoint_manifest.json",
            checkpoint_manifest,
        )

    if not training_protocol_frozen(training_rows):
        raise ValueError("K1-AM completed training rows drifted from the protocol")
    progress(
        args.output_root / "progress.jsonl",
        "three_split_panel_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    evaluation_rows = evaluate_k1am_panel(
        tasks=tasks,
        training_rows=training_rows,
        checkpoint_manifest=checkpoint_manifest,
        datasets=datasets,
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    write_comparison_csv(args.output_root / "comparison.csv", evaluation_rows)
    gate = adjudicate_k1am(
        tasks=tasks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoint_manifest=checkpoint_manifest,
        k1ak_controls=k1ak_controls,
        source_checks=source_checks,
        model_checks=model_checks,
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
        args.output_root / "results.jsonl",
        args.output_root / "history.csv",
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        training_rows=len(training_rows),
        evaluation_rows=len(evaluation_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def train_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[int, str], Any],
    output_root: Path,
    device: str,
    resume_training: bool = False,
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    existing_rows = (
        read_jsonl(output_root / "results.jsonl")
        if resume_training and (output_root / "results.jsonl").is_file()
        else []
    )
    existing_by_key = training_map(existing_rows, fail_closed=False)
    if len(existing_by_key) != len(existing_rows):
        raise ValueError("K1-AM partial training rows contain unsupported entries")
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for orientation in ORIENTATIONS:
            key = (seed, orientation)
            task = tasks_by_key[(seed, orientation)]
            model = build_k1am_model(task=task, orientation=orientation)
            initial_state_sha256 = tensor_mapping_sha256(model.state_dict())
            checkpoint = output_root / "checkpoints" / f"seed{seed}_{orientation}.pt"
            if resume_training and checkpoint.is_file():
                row = training_row_from_checkpoint(
                    model=model,
                    task=task,
                    datasets=datasets,
                    checkpoint=checkpoint,
                    seed=seed,
                    orientation=orientation,
                    initial_state_sha256=initial_state_sha256,
                )
                existing = existing_by_key.get(key)
                if existing is not None and existing != row:
                    raise ValueError(
                        f"K1-AM recovered row differs from existing row: {key}"
                    )
                rows.append(row)
                write_jsonl(output_root / "results.jsonl", rows)
                progress(
                    output_root / "progress.jsonl",
                    "training_row_recovered",
                    seed=seed,
                    orientation=orientation,
                    auc=row["metrics"]["auc"],
                    optimizer_steps=row["training"]["optimizer_steps"],
                    checkpoint=str(checkpoint),
                )
                continue
            if key in existing_by_key:
                raise ValueError(
                    f"K1-AM partial row has no recoverable checkpoint: {key}"
                )
            progress(
                output_root / "progress.jsonl",
                "training_row_start",
                seed=seed,
                orientation=orientation,
                initial_state_sha256=initial_state_sha256,
            )

            def callback(
                event: str,
                payload: dict[str, Any],
                *,
                _seed: int = seed,
                _orientation: str = orientation,
            ) -> None:
                progress(
                    output_root / "progress.jsonl",
                    f"training_{event}",
                    seed=_seed,
                    orientation=_orientation,
                    **training_progress_payload(payload),
                )

            train_binary_classifier(
                model,
                datasets[(seed, "train_seen")],
                datasets[(seed, "cross_key_validation")],
                TrainingConfig(
                    epochs=10,
                    batch_size=64,
                    learning_rate=1e-4,
                    seed=seed,
                    device=device,
                    optimizer="adam",
                    weight_decay=1e-5,
                    lr_scheduler="none",
                    checkpoint_metric="val_auc",
                    restore_best_checkpoint=True,
                    loss="mse",
                    checkpoint_output=checkpoint,
                ),
                progress_callback=callback,
            )
            row = training_row_from_checkpoint(
                model=model,
                task=task,
                datasets=datasets,
                checkpoint=checkpoint,
                seed=seed,
                orientation=orientation,
                initial_state_sha256=initial_state_sha256,
            )
            optimizer_steps = int(row["training"]["optimizer_steps"])
            rows.append(row)
            write_jsonl(output_root / "results.jsonl", rows)
            progress(
                output_root / "progress.jsonl",
                "training_row_done",
                seed=seed,
                orientation=orientation,
                auc=row["metrics"]["auc"],
                optimizer_steps=optimizer_steps,
                checkpoint=str(checkpoint),
            )
    return rows


def training_progress_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    safe_payload = dict(payload)
    if "path" in safe_payload:
        safe_payload["checkpoint_path"] = safe_payload.pop("path")
    return safe_payload


def training_row_from_checkpoint(
    *,
    model: torch.nn.Module,
    task: Mapping[str, Any],
    datasets: Mapping[tuple[int, str], Any],
    checkpoint: Path,
    seed: int,
    orientation: str,
    initial_state_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not {
        "state_dict",
        "history",
        "final_metrics",
        "metadata",
    }.issubset(payload):
        raise ValueError("K1-AM checkpoint payload is incomplete")
    state = payload["state_dict"]
    history = payload["history"]
    final_metrics = payload["final_metrics"]
    metadata = payload["metadata"]
    if (
        not isinstance(state, Mapping)
        or not isinstance(history, list)
        or not isinstance(final_metrics, Mapping)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("K1-AM checkpoint payload types are invalid")
    validate_training_checkpoint(
        history=history,
        final_metrics=final_metrics,
        metadata=metadata,
        checkpoint=checkpoint,
        seed=seed,
    )
    model.load_state_dict(state, strict=True)
    selected_state_sha256 = tensor_mapping_sha256(model.state_dict())
    if selected_state_sha256 != tensor_mapping_sha256(state):
        raise ValueError("K1-AM checkpoint strict load changed selected state")
    optimizer_steps = int(metadata["optimizer_state_step_after"])
    return {
        "run_id": RUN_ID,
        "cipher": "Midori64",
        "cipher_key": "midori64",
        "structure": "SPN",
        "model": ORIENTATION_MODELS[orientation],
        "selected_model": ORIENTATION_MODELS[orientation],
        "orientation": orientation,
        "rounds": 4,
        "seed": seed,
        "samples_per_class": 2048,
        "pairs_per_sample": 4,
        "input_difference": int(task["input_difference"]),
        "difference_profile": task["difference_profile"],
        "negative_mode": task["negative_mode"],
        "sample_structure": task["sample_structure"],
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "initial_state_sha256": initial_state_sha256,
        "selected_state_sha256": selected_state_sha256,
        "semantic_contrast_orientation": ORIENTATION_OPTIONS[orientation],
        "semantic_contrast_scale": CONTRAST_SCALE,
        "semantic_contrast_margin": CONTRAST_MARGIN,
        "metrics": dict(final_metrics),
        "history": history,
        "training": {
            **dict(metadata),
            "optimizer_steps": optimizer_steps,
            "samples_total": int(datasets[(seed, "train_seen")].features.shape[0]),
        },
        "validation": {
            "samples_total": int(
                datasets[(seed, "cross_key_validation")].features.shape[0]
            ),
        },
        "train_dataset_sha256": differential_dataset_sha256(
            datasets[(seed, "train_seen")]
        ),
        "validation_dataset_sha256": differential_dataset_sha256(
            datasets[(seed, "cross_key_validation")]
        ),
        "residual_gate": float(torch.tanh(model.backbone.residual_gate.detach())),
        "transition_gate": float(torch.tanh(model.backbone.transition_gate.detach())),
    }


def validate_training_checkpoint(
    *,
    history: Sequence[Mapping[str, Any]],
    final_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    checkpoint: Path,
    seed: int,
) -> None:
    expected_metadata = {
        "epochs": 10,
        "epochs_ran": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "optimizer_state_reused": False,
        "optimizer_state_step_before": 0,
        "optimizer_state_step_after": 640,
        "optimizer_session_call": 1,
        "weight_decay": 1e-5,
        "lr_scheduler": "none",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "loss": "mse",
        "selected_checkpoint": "best",
        "seed": seed,
        "device": "cpu",
        "checkpoint_output": str(checkpoint),
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("K1-AM checkpoint metadata drifted from the frozen protocol")
    if len(history) != 10 or [int(row.get("epoch", -1)) for row in history] != list(
        range(1, 11)
    ):
        raise ValueError("K1-AM checkpoint history is incomplete")
    required_history = {
        "train_auxiliary_loss",
        "train_semantic_loss_gap",
        "val_auc",
    }
    if any(
        not required_history.issubset(row)
        or not all(math.isfinite(float(row[name])) for name in required_history)
        for row in history
    ):
        raise ValueError("K1-AM checkpoint history metrics are invalid")
    if not all(math.isfinite(float(value)) for value in final_metrics.values()):
        raise ValueError("K1-AM checkpoint final metrics are invalid")
    best_auc = max(float(row["val_auc"]) for row in history)
    if (
        abs(float(final_metrics.get("auc", math.nan)) - best_auc) > 1e-7
        or abs(float(metadata.get("best_checkpoint_metric", math.nan)) - best_auc)
        > 1e-7
        or int(metadata.get("best_epoch", -1))
        != max(history, key=lambda row: float(row["val_auc"]))["epoch"]
    ):
        raise ValueError("K1-AM checkpoint best-selection metadata is inconsistent")


def build_checkpoint_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    for row in rows:
        seed = int(row["seed"])
        orientation = str(row["orientation"])
        key = (seed, orientation)
        if key in observed:
            raise ValueError(f"duplicate K1-AM checkpoint source: {key}")
        observed.add(key)
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        if not checkpoint.is_file():
            raise ValueError(f"missing K1-AM checkpoint: {checkpoint}")
        entries.append(
            {
                "cipher_key": "midori64",
                "seed": seed,
                "orientation": orientation,
                "model": row["model"],
                "selected_checkpoint": row["training"]["selected_checkpoint"],
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "state_dict_sha256": row["selected_state_sha256"],
                "initial_state_sha256": row["initial_state_sha256"],
            }
        )
    if observed != expected_training_keys():
        raise ValueError("K1-AM checkpoint sources are incomplete")
    return {"run_id": RUN_ID, "status": "pass", "entries": entries}


def validate_resume_root(
    args: argparse.Namespace,
    source_digests: Mapping[str, str],
) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    rows = read_jsonl(args.output_root / "results.jsonl")
    manifest = read_json(args.output_root / "checkpoint_manifest.json")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("source_digests") != dict(source_digests)
        or len(rows) != EXPECTED_TRAINING_ROWS
        or not training_protocol_frozen(rows)
        or manifest.get("run_id") != RUN_ID
    ):
        raise ValueError("K1-AM resume root does not match the frozen training")
    build_checkpoint_manifest(rows)


def validate_training_resume_root(
    args: argparse.Namespace,
    *,
    source_digests: Mapping[str, str],
    dataset_manifest: Sequence[Mapping[str, Any]],
) -> None:
    preflight_path = args.output_root / "preflight.json"
    local_manifest_path = args.output_root / "dataset_manifest.jsonl"
    if not preflight_path.is_file() or not local_manifest_path.is_file():
        raise ValueError("K1-AM training resume root lacks bound preflight evidence")
    preflight = read_json(preflight_path)
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("source_digests") != dict(source_digests)
        or int(preflight.get("expected_training_rows", -1)) != EXPECTED_TRAINING_ROWS
        or int(preflight.get("expected_evaluation_rows", -1))
        != EXPECTED_EVALUATION_ROWS
        or read_jsonl(local_manifest_path) != list(dataset_manifest)
    ):
        raise ValueError("K1-AM training resume root does not match the frozen run")
    completed_artifacts = (
        "controls.jsonl",
        "gate.json",
        "validation.json",
        "summary.json",
    )
    if any((args.output_root / name).exists() for name in completed_artifacts):
        raise ValueError("K1-AM completed output cannot use training recovery")


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
        raise ValueError("K1-AM output root already contains run artifacts")


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        "orientation",
        "split",
        "condition",
        "rows",
        "auc",
        "correct_minus_condition_auc",
        "max_abs_probability_delta_from_correct",
        "checkpoint_sha256",
        "state_dict_sha256",
        "dataset_sha256",
        "training_performed",
        "optimizer_steps",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_manifest",
    "main",
    "parse_args",
    "require_fresh_output_root",
    "train_panel",
    "training_progress_payload",
    "training_row_from_checkpoint",
    "validate_resume_root",
    "validate_training_checkpoint",
    "validate_training_resume_root",
]
