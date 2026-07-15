from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from blockcipher_nd.engine.task_config import resolve_final_test_key, resolve_task_keys


def reset_progress(path: str | None) -> None:
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")


def write_progress(
    path: str | None, event: str, payload: dict[str, Any] | None = None
) -> None:
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "time": time.time(),
        **(payload or {}),
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def progress_callback(
    path: str | None,
    stage: str,
    task: dict[str, Any],
    *,
    index: int | None,
    total: int | None,
    split: str | None = None,
) -> Callable[[str, dict[str, Any]], None]:
    def callback(event: str, payload: dict[str, Any]) -> None:
        record = {
            "stage": stage,
            "index": index,
            "total": total,
            **task_progress_payload(task),
            **payload,
        }
        if split is not None:
            record["split"] = split
        write_progress(path, event, record)

    return callback


def task_progress_payload(task: dict[str, Any]) -> dict[str, Any]:
    train_key, validation_key = resolve_task_keys(task)
    return {
        "cipher_key": task["cipher_key"],
        "model": task["model_key"],
        "architecture": task["architecture"],
        "rounds": task["rounds"],
        "seed": task["seed"],
        "samples_per_class": task["samples_per_class"],
        "train_samples_total": task.get("train_samples_total"),
        "validation_samples_total": task.get("validation_samples_total"),
        "final_test_samples_total": task.get("final_test_samples_total"),
        "final_test_repeats": task.get("final_test_repeats", 0),
        "train_key": train_key,
        "validation_key": validation_key,
        "final_test_key": resolve_final_test_key(task),
        "dataset_label_mode": task.get("dataset_label_mode", "balanced_per_class"),
        "pairs_per_sample": task["pairs_per_sample"],
        "feature_encoding": task["feature_encoding"],
        "negative_mode": task["negative_mode"],
        "key_rotation_interval": task["key_rotation_interval"],
        "difference_profile": task.get("difference_profile", ""),
        "difference_member": task.get("difference_member", ""),
        "sample_structure": task["sample_structure"],
        "integral_active_nibble": task["integral_active_nibble"],
        "selected_bit_indices": task["selected_bit_indices"],
        "loss": task.get("loss", ""),
        "optimizer_state_transition": task.get(
            "optimizer_state_transition", "reset_each_stage"
        ),
        "target_epochs": task.get("target_epochs"),
        "pretrain_rounds": task.get("pretrain_rounds"),
        "pretrain_round_sequence": list(task.get("pretrain_round_sequence", ())),
        "pretrain_epochs": task.get("pretrain_epochs"),
    }
