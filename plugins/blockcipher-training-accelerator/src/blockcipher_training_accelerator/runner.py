from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.engine.modeling import (
    configure_structure_aware_model,
    infer_pair_bits,
    select_model_key,
)
from blockcipher_nd.engine.pretraining import run_optional_pretraining
from blockcipher_nd.engine.progress import progress_callback, reset_progress, task_progress_payload, write_progress
from blockcipher_nd.engine.results import build_task_result
from blockcipher_nd.engine.task_config import build_dataset_config, build_training_config, resolve_task_keys
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model

from blockcipher_training_accelerator.profiles import resolve_profile
from blockcipher_training_accelerator.trainer import train_binary_classifier_accelerated


def run_accelerated_matrix(argv: list[str]) -> list[dict[str, Any]]:
    args = parse_accelerated_args(argv)
    profile = resolve_profile(args.speed_profile)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    reset_progress(args.progress_output)
    write_progress(
        args.progress_output,
        "accelerated_run_start",
        {
            "total": len(tasks),
            "output": str(output),
            "dataset_cache_root": args.dataset_cache_root,
            "speed_profile": profile.to_json_dict(),
        },
    )

    rows: list[dict[str, Any]] = []
    try:
        with output.open("w", encoding="utf-8") as handle:
            for index, task in enumerate(tasks, start=1):
                write_progress(
                    args.progress_output,
                    "row_start",
                    {
                        "index": index,
                        "total": len(tasks),
                        **task_progress_payload(task),
                    },
                )
                row = run_accelerated_task(
                    task,
                    args,
                    profile_name=args.speed_profile,
                    progress_path=args.progress_output,
                    index=index,
                    total=len(tasks),
                )
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                write_progress(
                    args.progress_output,
                    "row_done",
                    {
                        "index": index,
                        "total": len(tasks),
                        "accuracy": row["metrics"]["accuracy"],
                        "selected_model": row["selected_model"],
                        **task_progress_payload(task),
                    },
                )
    except Exception as exc:
        write_progress(
            args.progress_output,
            "accelerated_run_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    return rows


def parse_accelerated_args(argv: list[str]):
    speed_profile = "baseline"
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--speed-profile":
            speed_profile = argv[index + 1]
            index += 2
            continue
        if item.startswith("--speed-profile="):
            speed_profile = item.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(item)
        index += 1
    args = parse_train_args(cleaned)
    args.speed_profile = speed_profile
    return args


def run_accelerated_task(
    task: dict[str, Any],
    args,
    *,
    profile_name: str,
    progress_path: str | None = None,
    index: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    row_started = time.perf_counter()
    profile = resolve_profile(profile_name)
    train_key, validation_key = resolve_task_keys(task)
    train_cipher = build_cipher(task["cipher_key"], task["rounds"], key=train_key)
    validation_cipher = build_cipher(task["cipher_key"], task["rounds"], key=validation_key)
    model_key = select_model_key(
        task["model_key"],
        train_cipher.structure,
        task["pairs_per_sample"],
    )
    pair_bits = infer_pair_bits(train_cipher.block_bits, task["feature_encoding"])

    train_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=train_cipher,
            samples_per_class=task["samples_per_class"],
            seed=task["seed"],
        ),
        args,
        task,
        split="train",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    validation_dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=validation_cipher,
            samples_per_class=max(8, task["samples_per_class"] // 2),
            seed=task["seed"] + 10_000,
        ),
        args,
        task,
        split="validation",
        progress_path=progress_path,
        index=index,
        total=total,
    )
    write_progress(
        progress_path,
        "cache_ready",
        {
            "index": index,
            "total": total,
            "dataset_cache_enabled": bool(args.dataset_cache_root),
            "train_rows": int(train_dataset.features.shape[0]),
            "validation_rows": int(validation_dataset.features.shape[0]),
            "input_bits": int(train_dataset.features.shape[1]),
            **task_progress_payload(task),
        },
    )

    model = build_model(
        model_key,
        input_bits=train_dataset.features.shape[1],
        hidden_bits=args.hidden_bits,
        pair_bits=pair_bits,
        structure=train_cipher.structure,
        model_options=task.get("model_options"),
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    pretrain_result = run_optional_pretraining(
        model,
        task,
        args,
        pair_bits=pair_bits,
        progress_path=progress_path,
        index=index,
        total=total,
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    training_result = train_binary_classifier_accelerated(
        model,
        train_dataset,
        validation_dataset,
        build_training_config(task, args, epochs=args.epochs, seed=task["seed"]),
        profile=profile,
        progress_callback=progress_callback(
            progress_path,
            "accelerated_training",
            task,
            index=index,
            total=total,
        ),
    )
    row = build_task_result(
        task=task,
        args=args,
        train_cipher=train_cipher,
        validation_cipher=validation_cipher,
        train_key=train_key,
        validation_key=validation_key,
        model=model,
        model_key=model_key,
        pair_bits=pair_bits,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        training_result=training_result,
        pretrain_result=pretrain_result,
    )
    row["training"]["accelerator"]["row_duration_seconds"] = round(time.perf_counter() - row_started, 6)
    return row
