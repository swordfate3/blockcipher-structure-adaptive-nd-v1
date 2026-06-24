from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.engine.pretraining import pretraining_metadata
from blockcipher_nd.training import TrainingResult


def build_task_result(
    *,
    task: dict[str, Any],
    args: argparse.Namespace,
    train_cipher,
    validation_cipher,
    train_key: int | None,
    validation_key: int | None,
    model,
    model_key: str,
    pair_bits: int | None,
    train_dataset,
    validation_dataset,
    training_result: TrainingResult,
    pretrain_result: TrainingResult | None,
) -> dict[str, Any]:
    return {
        "cipher": train_cipher.name,
        "cipher_key": task["cipher_key"],
        "structure": train_cipher.structure,
        "model": task["model_key"],
        "selected_model": model_key,
        "architecture": task["architecture"],
        "architecture_rank": task.get("architecture_rank"),
        "matching_score": task.get("matching_score"),
        "matching_evidence": task.get("matching_evidence", ""),
        "literature": task.get("literature", ""),
        "rounds": task["rounds"],
        "seed": task["seed"],
        "train_key": train_key,
        "validation_key": validation_key,
        "input_difference": task["input_difference"],
        "difference_profile": task.get("difference_profile", ""),
        "difference_member": task.get("difference_member", ""),
        "difference_source": task.get("difference_source", ""),
        "samples_per_class": task["samples_per_class"],
        "pairs_per_sample": task["pairs_per_sample"],
        "feature_encoding": task["feature_encoding"],
        "negative_mode": task["negative_mode"],
        "key_rotation_interval": task["key_rotation_interval"],
        "sample_structure": task["sample_structure"],
        "integral_active_nibble": task["integral_active_nibble"],
        "metrics": training_result.final_metrics,
        "history": training_result.history,
        "training": {
            **training_result.metadata,
            "dataset_cache_root": args.dataset_cache_root,
            "dataset_cache_chunk_size": args.dataset_cache_chunk_size if args.dataset_cache_root else None,
            "input_bits": int(train_dataset.features.shape[1]),
            "feature_encoding": task["feature_encoding"],
            "pairs_per_sample": task["pairs_per_sample"],
            "pair_bits": pair_bits,
            "key_rotation_interval": task["key_rotation_interval"],
            "sample_structure": task["sample_structure"],
            "integral_active_nibble": task["integral_active_nibble"],
            "model_options": task.get("model_options", {}),
            "selected_bit_indices": task["selected_bit_indices"],
            "pretraining": pretraining_metadata(pretrain_result),
        },
        **model_metadata(model),
        "validation": {
            "cipher": validation_cipher.name,
            "structure": validation_cipher.structure,
            "rounds": validation_cipher.rounds,
            "feature_encoding": validation_dataset.metadata["feature_encoding"],
            "negative_mode": validation_dataset.metadata["negative_mode"],
            "pairs_per_sample": validation_dataset.metadata["pairs_per_sample"],
            "samples_per_class": validation_dataset.metadata["samples_per_class"],
            "key_rotation_interval": validation_dataset.metadata["key_rotation_interval"],
            "sample_structure": validation_dataset.metadata["sample_structure"],
            "integral_active_nibble": validation_dataset.metadata["integral_active_nibble"],
        },
    }
