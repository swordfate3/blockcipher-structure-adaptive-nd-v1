from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.modeling import configure_structure_aware_model
from blockcipher_nd.engine.progress import progress_callback, task_progress_payload, write_progress
from blockcipher_nd.engine.task_config import build_dataset_config, build_training_config, resolve_task_keys
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.training import TrainingResult, train_binary_classifier


def run_optional_pretraining(
    model,
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    pair_bits: int | None,
    progress_path: str | None,
    index: int | None,
    total: int | None,
) -> TrainingResult | None:
    pretrain_epochs = int(
        task.get("pretrain_epochs")
        if task.get("pretrain_epochs") is not None
        else args.pretrain_epochs
    )
    round_sequence = resolve_pretrain_round_sequence(task, args)
    if pretrain_epochs <= 0 or not round_sequence:
        return None

    stage_results: list[tuple[int, TrainingResult]] = []
    for stage_index, pretrain_rounds in enumerate(round_sequence):
        pretrain_task = {
            **task,
            "rounds": pretrain_rounds,
            "pretrain_rounds": pretrain_rounds,
        }
        configure_structure_aware_model(model, task["cipher_key"], pretrain_rounds)
        result = run_pretraining_stage(
            model,
            pretrain_task,
            args,
            pretrain_epochs=pretrain_epochs,
            target_rounds=int(task["rounds"]),
            stage_index=stage_index,
            stage_total=len(round_sequence),
            pair_bits=pair_bits,
            progress_path=progress_path,
            index=index,
            total=total,
        )
        stage_results.append((pretrain_rounds, result))

    final_result = stage_results[-1][1]
    stages = [
        curriculum_stage_metadata(rounds, result)
        for rounds, result in stage_results
    ]
    return TrainingResult(
        history=final_result.history,
        final_metrics=final_result.final_metrics,
        metadata={
            **final_result.metadata,
            "round_sequence": list(round_sequence),
            "curriculum_stages": stages,
        },
    )


def resolve_pretrain_round_sequence(
    task: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, ...]:
    task_sequence = tuple(task.get("pretrain_round_sequence") or ())
    cli_sequence = tuple(getattr(args, "pretrain_round_sequence", ()) or ())
    sequence = task_sequence or cli_sequence
    scalar = task.get("pretrain_rounds")
    if scalar is None:
        scalar = args.pretrain_rounds
    if sequence and scalar is not None:
        raise ValueError("use either pretrain_round_sequence or pretrain_rounds, not both")
    if not sequence and scalar is not None:
        sequence = (int(scalar),)
    if not sequence:
        return ()
    if any(rounds <= 0 or rounds >= int(task["rounds"]) for rounds in sequence):
        raise ValueError("pretrain_round_sequence rounds must be lower than target rounds")
    if any(current >= following for current, following in zip(sequence, sequence[1:])):
        raise ValueError("pretrain_round_sequence must be strictly increasing")
    return sequence


def run_pretraining_stage(
    model,
    pretrain_task: dict[str, Any],
    args: argparse.Namespace,
    *,
    pretrain_epochs: int,
    target_rounds: int,
    stage_index: int,
    stage_total: int,
    pair_bits: int | None,
    progress_path: str | None,
    index: int | None,
    total: int | None,
) -> TrainingResult:
    pretrain_rounds = int(pretrain_task["rounds"])
    pretrain_train_key, pretrain_validation_key = resolve_task_keys(pretrain_task)
    pretrain_cipher = build_cipher(
        pretrain_task["cipher_key"],
        pretrain_rounds,
        key=pretrain_train_key,
    )
    pretrain_validation_cipher = build_cipher(
        pretrain_task["cipher_key"],
        pretrain_rounds,
        key=pretrain_validation_key,
    )
    pretrain_dataset = make_task_dataset(
        build_dataset_config(
            pretrain_task,
            cipher=pretrain_cipher,
            samples_per_class=pretrain_task["samples_per_class"],
            seed=pretrain_task["seed"] + 20_000 + stage_index,
        ),
        args,
        pretrain_task,
        split="pretrain_train",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    pretrain_validation_dataset = make_task_dataset(
        build_dataset_config(
            pretrain_task,
            cipher=pretrain_validation_cipher,
            samples_per_class=max(8, pretrain_task["samples_per_class"] // 2),
            seed=pretrain_task["seed"] + 30_000 + stage_index,
        ),
        args,
        pretrain_task,
        split="pretrain_validation",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    expected_input_bits = int(pretrain_dataset.features.shape[1])
    if pair_bits is not None and expected_input_bits % pair_bits != 0:
        raise ValueError("pretraining feature width is incompatible with target pair_bits")
    write_progress(
        progress_path,
        "pretrain_cache_ready",
        {
            "index": index,
            "total": total,
            "target_rounds": target_rounds,
            "pretrain_rounds": pretrain_rounds,
            "pretrain_epochs": pretrain_epochs,
            "curriculum_stage_index": stage_index + 1,
            "curriculum_stage_total": stage_total,
            "train_rows": int(pretrain_dataset.features.shape[0]),
            "validation_rows": int(pretrain_validation_dataset.features.shape[0]),
            "input_bits": expected_input_bits,
            **task_progress_payload(pretrain_task),
        },
    )
    return train_binary_classifier(
        model,
        pretrain_dataset,
        pretrain_validation_dataset,
        build_training_config(
            pretrain_task,
            args,
            epochs=pretrain_epochs,
            seed=pretrain_task["seed"] + 40_000 + stage_index,
        ),
        progress_callback=progress_callback(
            progress_path,
            "pretraining",
            pretrain_task,
            index=index,
            total=total,
        ),
    )


def curriculum_stage_metadata(rounds: int, result: TrainingResult) -> dict[str, Any]:
    return {
        "rounds": rounds,
        "metrics": result.final_metrics,
        "history": result.history,
        "epochs": result.metadata.get("epochs"),
        "epochs_ran": result.metadata.get("epochs_ran"),
        "best_epoch": result.metadata.get("best_epoch"),
        "best_checkpoint_metric": result.metadata.get("best_checkpoint_metric"),
        "selected_checkpoint": result.metadata.get("selected_checkpoint"),
        "stopped_epoch": result.metadata.get("stopped_epoch"),
        "seed": result.metadata.get("seed"),
    }


def pretraining_metadata(result: TrainingResult | None) -> dict[str, Any]:
    if result is None:
        return {"enabled": False}
    metadata = {
        "enabled": True,
        "metrics": result.final_metrics,
        "epochs_ran": result.metadata.get("epochs_ran"),
        "best_epoch": result.metadata.get("best_epoch"),
        "best_checkpoint_metric": result.metadata.get("best_checkpoint_metric"),
        "selected_checkpoint": result.metadata.get("selected_checkpoint"),
    }
    if "round_sequence" in result.metadata:
        metadata["round_sequence"] = result.metadata["round_sequence"]
        metadata["curriculum_stages"] = result.metadata["curriculum_stages"]
    return metadata
