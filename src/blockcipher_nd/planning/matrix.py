from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.registry.difference_profiles import (
    difference_for_profile,
    literature_difference_profiles,
)
from blockcipher_nd.registry.cipher_factory import default_difference


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.plan:
        return tasks_from_plan(
            Path(args.plan),
            feature_encoding=args.feature_encoding,
            pairs_per_sample=args.pairs_per_sample,
            difference_profile=args.difference_profile,
            difference_member=args.difference_member,
            key_rotation_interval=args.key_rotation_interval,
            sample_structure=args.sample_structure,
            integral_active_nibble=args.integral_active_nibble,
        )

    tasks: list[dict[str, Any]] = []
    for cipher_key in args.ciphers:
        for rounds in args.rounds:
            for seed in args.seeds:
                for model_key in args.models:
                    tasks.append(
                        {
                            "cipher_key": cipher_key,
                            "model_key": model_key,
                            "architecture": model_key,
                            "rounds": rounds,
                            "seed": seed,
                            "samples_per_class": args.samples_per_class,
                            "train_samples_total": args.train_samples_total,
                            "validation_samples_total": args.validation_samples_total,
                            "final_test_samples_total": args.final_test_samples_total,
                            "final_test_repeats": args.final_test_repeats,
                            "dataset_label_mode": args.dataset_label_mode,
                            "pairs_per_sample": args.pairs_per_sample,
                            "feature_encoding": args.feature_encoding,
                            "negative_mode": args.negative_mode,
                            "key_rotation_interval": args.key_rotation_interval,
                            "sample_structure": args.sample_structure,
                            "integral_active_nibble": args.integral_active_nibble,
                            "integral_active_nibbles": (),
                            "validation_integral_active_nibbles": (),
                            "selected_bit_indices": (),
                            "loss": args.loss,
                            "learning_rate": args.learning_rate,
                            "optimizer": args.optimizer,
                            "optimizer_state_transition": args.optimizer_state_transition,
                            "weight_decay": args.weight_decay,
                            "lr_scheduler": args.lr_scheduler,
                            "max_learning_rate": args.max_learning_rate,
                            "checkpoint_metric": args.checkpoint_metric,
                            "restore_best_checkpoint": args.restore_best_checkpoint,
                            "early_stopping_patience": args.early_stopping_patience,
                            "early_stopping_min_delta": args.early_stopping_min_delta,
                            "target_epochs": args.epochs,
                            "pretrain_rounds": args.pretrain_rounds,
                            "pretrain_round_sequence": args.pretrain_round_sequence,
                            "pretrain_epochs": args.pretrain_epochs,
                            "model_options": {},
                            "train_key": None,
                            "validation_key": None,
                            "final_test_key": None,
                            **difference_metadata(
                                cipher_key,
                                args.difference_profile,
                                args.difference_member,
                            ),
                        }
                    )
    return tasks


def tasks_from_plan(
    path: Path,
    feature_encoding: str,
    pairs_per_sample: int,
    difference_profile: str | None,
    difference_member: int,
    key_rotation_interval: int = 0,
    sample_structure: str = "independent_pairs",
    integral_active_nibble: int = 0,
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        plan_task(
            row,
            feature_encoding,
            pairs_per_sample,
            difference_profile,
            difference_member,
            key_rotation_interval,
            sample_structure,
            integral_active_nibble,
        )
        for row in rows
    ]


def plan_task(
    row: dict[str, str],
    feature_encoding: str,
    pairs_per_sample: int,
    difference_profile: str | None,
    difference_member: int,
    key_rotation_interval: int = 0,
    sample_structure: str = "independent_pairs",
    integral_active_nibble: int = 0,
) -> dict[str, Any]:
    cipher_key = cipher_key_from_name(row["cipher"])
    task = {
        "cipher_key": cipher_key,
        "model_key": row["model_key"],
        "architecture": row["network"],
        "architecture_rank": int(row["architecture_rank"]),
        "matching_score": int(row["score"]),
        "matching_evidence": row.get("evidence", ""),
        "literature": row.get("literature", ""),
        "rounds": int(row["rounds"]),
        "seed": int(row["seed"]),
        "samples_per_class": int(row["samples_per_class"]),
        "train_samples_total": optional_int(row.get("train_samples_total")),
        "validation_samples_total": optional_int(row.get("validation_samples_total")),
        "final_test_samples_total": optional_int(row.get("final_test_samples_total")),
        "final_test_repeats": optional_int(row.get("final_test_repeats")) or 0,
        "dataset_label_mode": row.get("dataset_label_mode") or "balanced_per_class",
        "pairs_per_sample": int(row.get("pairs_per_sample") or pairs_per_sample),
        "feature_encoding": row.get("feature_encoding") or feature_encoding,
        "negative_mode": row.get("negative_mode") or "random_ciphertext",
        "key_rotation_interval": optional_int(row.get("key_rotation_interval"))
        if row.get("key_rotation_interval") not in {None, ""}
        else key_rotation_interval,
        "sample_structure": row.get("sample_structure") or sample_structure,
        "integral_active_nibble": optional_int(row.get("integral_active_nibble"))
        if row.get("integral_active_nibble") not in {None, ""}
        else integral_active_nibble,
        "integral_active_nibbles": optional_int_tuple(
            row.get("integral_active_nibbles")
        ),
        "validation_integral_active_nibbles": optional_int_tuple(
            row.get("validation_integral_active_nibbles")
        ),
        "loss": row.get("loss") or "bce",
        "learning_rate": optional_float(row.get("learning_rate")),
        "optimizer": row.get("optimizer") or None,
        "optimizer_state_transition": row.get("optimizer_state_transition")
        or "reset_each_stage",
        "weight_decay": optional_float(row.get("weight_decay")),
        "lr_scheduler": row.get("lr_scheduler") or None,
        "max_learning_rate": optional_float(row.get("max_learning_rate")),
        "checkpoint_metric": row.get("checkpoint_metric") or None,
        "restore_best_checkpoint": optional_bool(row.get("restore_best_checkpoint")),
        "early_stopping_patience": optional_int(row.get("early_stopping_patience")),
        "early_stopping_min_delta": optional_float(row.get("early_stopping_min_delta")),
        "target_epochs": optional_int(row.get("target_epochs")),
        "pretrain_rounds": optional_int(row.get("pretrain_rounds")),
        "pretrain_round_sequence": optional_int_tuple(
            row.get("pretrain_round_sequence"),
            field_name="pretrain_round_sequence",
        ),
        "pretrain_epochs": optional_int(row.get("pretrain_epochs")),
        "model_options": optional_json(row.get("model_options")),
        "selected_bit_indices": optional_int_tuple(row.get("selected_bit_indices")),
        "train_key": optional_int(row.get("train_key")),
        "validation_key": optional_int(row.get("validation_key")),
        "final_test_key": optional_int(row.get("final_test_key")),
    }
    task.update(
        difference_metadata(
            cipher_key,
            row.get("difference_profile") or difference_profile,
            int(row.get("difference_member") or difference_member),
        )
    )
    return task


def difference_metadata(
    cipher_key: str,
    profile_name: str | None,
    member_index: int,
) -> dict[str, Any]:
    if not profile_name:
        return {
            "input_difference": default_difference(cipher_key),
            "difference_profile": "",
            "difference_member": "",
            "difference_source": "",
        }
    profile = literature_difference_profiles()[profile_name]
    if profile.cipher != cipher_key:
        raise ValueError(
            f"difference profile {profile_name} is for {profile.cipher}, not {cipher_key}"
        )
    return {
        "input_difference": difference_for_profile(profile_name, member_index),
        "difference_profile": profile_name,
        "difference_member": member_index,
        "difference_source": profile.source,
    }


def optional_json(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    value = value.strip()
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("model_options must be a JSON object")
    return parsed


def optional_int_tuple(
    value: str | None,
    *,
    field_name: str = "selected_bit_indices",
) -> tuple[int, ...]:
    if value is None:
        return ()
    value = value.strip()
    if not value:
        return ()
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, int) for item in parsed
    ):
        raise ValueError(f"{field_name} must be a JSON list of integers")
    return tuple(parsed)


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"unsupported boolean value: {value}")


def optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value, 0)


def cipher_key_from_name(cipher_name: str) -> str:
    mapping = {
        "SPECK32/64": "speck32",
        "PRESENT-80": "present80",
        "GIFT-64": "gift64",
        "SKINNY-64/64": "skinny64",
        "uKNIT-BC": "uknit64",
        "DES": "des",
        "SM4": "sm4",
        "SIMON64/128": "simon64",
        "Simeck64/128": "simeck64",
    }
    try:
        return mapping[cipher_name]
    except KeyError as exc:
        raise ValueError(f"unsupported cipher in plan: {cipher_name}") from exc
