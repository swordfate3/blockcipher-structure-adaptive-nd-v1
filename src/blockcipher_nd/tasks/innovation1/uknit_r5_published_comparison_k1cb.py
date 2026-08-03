from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.data.differential.metadata import dataset_metadata
from blockcipher_nd.engine.datasets import dataset_cache_dir
from blockcipher_nd.engine.task_config import (
    build_dataset_config,
    resolve_task_keys,
    validation_samples_per_class,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca import (
    RUN_ID as K1CA_RUN_ID,
    result_protocol_frozen as k1ca_result_protocol_frozen,
)


RUN_ID = "i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803"
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 4
EXPECTED_INPUT_BITS = 512
EXPECTED_SAMPLES_PER_CLASS = 262_144
EXPECTED_TRAIN_ROWS = 524_288
EXPECTED_VALIDATION_ROWS = 131_072
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_RESULT_ROWS = 6
EXPECTED_CACHE_REUSES = 12

ARCHITECTURES = {
    "zhang_wang_mcnd": "spn_zhang_wang_mcnd_adapter",
    "liu_case3_conv2d": "spn_liu_case3_conv2d_adapter",
    "gohr_style_resnet": "spn_gohr_style_resnet_pairset_adapter",
}
MODEL_TO_ARCHITECTURE = {model: name for name, model in ARCHITECTURES.items()}
EXPECTED_PARAMETER_COUNTS = {
    "zhang_wang_mcnd": 650_177,
    "liu_case3_conv2d": 130_945,
    "gohr_style_resnet": 191_937,
}


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=EXPECTED_PAIRS,
        difference_profile=None,
        difference_member=0,
    )


def expected_keys() -> set[tuple[int, str]]:
    return {
        (seed, architecture)
        for seed in EXPECTED_SEEDS
        for architecture in ARCHITECTURES
    }


def task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        architecture = MODEL_TO_ARCHITECTURE.get(str(task.get("model_key")))
        if architecture is None:
            continue
        key = (int(task.get("seed", -1)), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-CB task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-CB task matrix is incomplete")
    return mapped


def plan_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and task.get("final_test_samples_total") is None
            and int(task.get("final_test_repeats", -1)) == 0
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == CONFIRMATION_KEYS[seed][0]
            and int(task.get("validation_key", -1)) == CONFIRMATION_KEYS[seed][1]
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and task.get("optimizer_state_transition") == "reset_each_stage"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("lr_scheduler") == "none"
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            for (seed, _), task in mapped.items()
        )
    )


def audit_source_caches(
    tasks: Sequence[Mapping[str, Any]], source_cache_root: Path
) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    entries: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    payloads_complete = True
    metadata_match = True
    arrays_match = True
    generation_contract_match = True

    for seed in EXPECTED_SEEDS:
        task = mapped.get((seed, "zhang_wang_mcnd"))
        if task is None:
            payloads_complete = False
            continue
        train_key, validation_key = resolve_task_keys(dict(task))
        for split, key, samples_per_class, dataset_seed in (
            ("train", train_key, EXPECTED_SAMPLES_PER_CLASS, seed),
            (
                "validation",
                validation_key,
                validation_samples_per_class(dict(task)),
                seed + 10_000,
            ),
        ):
            cipher = build_cipher("uknit64", 5, key=key)
            config = build_dataset_config(
                dict(task),
                cipher=cipher,
                samples_per_class=samples_per_class,
                samples_total=(
                    task.get("train_samples_total")
                    if split == "train"
                    else task.get("validation_samples_total")
                ),
                seed=dataset_seed,
                split=split,
            )
            cache_path = dataset_cache_dir(source_cache_root, dict(task), config, split)
            expected_paths.add(cache_path)
            entry = _audit_cache_path(cache_path, config=config, split=split, seed=seed)
            entries.append(entry)
            payloads_complete = payloads_complete and entry["payload_complete"]
            metadata_match = metadata_match and entry["metadata_match"]
            arrays_match = arrays_match and entry["arrays_match"]
            generation_contract_match = (
                generation_contract_match and entry["generation_contract_match"]
            )

    discovered_metadata = (
        set(path.parent for path in source_cache_root.rglob("metadata.json"))
        if source_cache_root.is_dir()
        else set()
    )
    checks = {
        "six_row_plan_frozen": plan_protocol_frozen(tasks),
        "source_cache_root_exists": source_cache_root.is_dir(),
        "four_expected_cache_paths_exact": len(expected_paths) == 4,
        "four_cache_payloads_complete": len(entries) == 4 and payloads_complete,
        "cache_metadata_matches_k1cb_protocol": len(entries) == 4 and metadata_match,
        "cache_array_shapes_and_dtypes_match": len(entries) == 4 and arrays_match,
        "cache_generation_contract_matches": (
            len(entries) == 4 and generation_contract_match
        ),
        "no_extra_source_cache_metadata": discovered_metadata == expected_paths,
    }
    return {
        "run_id": RUN_ID,
        "source_run_id": K1CA_RUN_ID,
        "status": "pass" if checks and all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "entries": entries,
        "cache_generation_authorized": False,
        "training_authorized": bool(checks) and all(checks.values()),
    }


def _audit_cache_path(
    cache_path: Path, *, config: Any, split: str, seed: int
) -> dict[str, Any]:
    features_path = cache_path / "features.npy"
    labels_path = cache_path / "labels.npy"
    metadata_path = cache_path / "metadata.json"
    payload_complete = all(
        path.is_file() and path.stat().st_size > 0
        for path in (features_path, labels_path, metadata_path)
    )
    expected_metadata = dataset_metadata(config)
    metadata: dict[str, Any] = {}
    metadata_match = False
    generation_contract_match = False
    arrays_match = False
    if payload_complete:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_match = all(
                metadata.get(key) == value for key, value in expected_metadata.items()
            )
            generation_contract_match = (
                metadata.get("generation_chunk_size") == 1024
                and metadata.get("generation_workers") == 1
                and metadata.get("physical_shuffle") is False
                and metadata.get("training_shuffle") is True
            )
            features = np.load(features_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            rows = int(expected_metadata["samples_total"])
            arrays_match = (
                features.shape == (rows, EXPECTED_INPUT_BITS)
                and labels.shape == (rows,)
                and features.dtype == np.uint8
                and labels.dtype == np.uint8
                and int(metadata.get("input_bits", -1)) == EXPECTED_INPUT_BITS
                and int(metadata.get("total_rows", -1)) == rows
            )
        except (OSError, ValueError, json.JSONDecodeError):
            metadata_match = False
            generation_contract_match = False
            arrays_match = False
    return {
        "seed": seed,
        "split": split,
        "cache_path": str(cache_path),
        "payload_complete": payload_complete,
        "metadata_match": metadata_match,
        "arrays_match": arrays_match,
        "generation_contract_match": generation_contract_match,
        "rows": expected_metadata["samples_total"],
        "input_bits": EXPECTED_INPUT_BITS,
    }


def result_map(
    rows: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        architecture = MODEL_TO_ARCHITECTURE.get(str(row.get("model")))
        if architecture is None:
            continue
        key = (int(row.get("seed", -1)), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-CB result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-CB result matrix is incomplete")
    return mapped


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    mapped = result_map(rows, fail_closed=False)
    checkpoints = {
        str(row.get("training", {}).get("checkpoint_output", "")) for row in rows
    }
    return (
        len(rows) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and len(checkpoints) == EXPECTED_RESULT_ROWS
        and all(
            _result_row_frozen(row, seed=seed, architecture=architecture)
            for (seed, architecture), row in mapped.items()
        )
    )


def _result_row_frozen(row: Mapping[str, Any], *, seed: int, architecture: str) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    metrics = row.get("metrics", {})
    history = row.get("history", [])
    checkpoint = str(training.get("checkpoint_output", ""))
    return (
        int(row.get("rounds", -1)) == 5
        and int(row.get("seed", -1)) == seed
        and int(row.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("final_test_repeats", -1)) == 0
        and row.get("final_test_samples_total") is None
        and int(row.get("trainable_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNTS[architecture]
        and int(training.get("input_bits", -1)) == EXPECTED_INPUT_BITS
        and int(training.get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(training.get("validation_rows", -1)) == EXPECTED_VALIDATION_ROWS
        and int(training.get("train_positive_rows", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(training.get("train_negative_rows", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and training.get("train_dataset_storage") == "disk"
        and training.get("validation_dataset_storage") == "disk"
        and _is_source_cache_root(training.get("dataset_cache_root"))
        and _is_run_path(checkpoint)
        and checkpoint.endswith(".pt")
        and int(training.get("dataset_cache_chunk_size", -1)) == 1024
        and int(training.get("dataset_cache_workers", -1)) == 1
        and str(training.get("device", "")).startswith("cuda")
        and int(training.get("batch_size", -1)) == EXPECTED_BATCH_SIZE
        and int(training.get("epochs", -1)) == EXPECTED_EPOCHS
        and int(training.get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and training.get("checkpoint_metric") == "val_auc"
        and training.get("restore_best_checkpoint") is True
        and training.get("selected_checkpoint") == "best"
        and len(history) == EXPECTED_EPOCHS
        and [int(item.get("epoch", -1)) for item in history]
        == list(range(1, EXPECTED_EPOCHS + 1))
        and int(validation.get("samples_total", -1)) == EXPECTED_VALIDATION_ROWS
        and all(
            math.isfinite(float(value))
            for value in (
                metrics.get("auc", math.nan),
                metrics.get("accuracy", math.nan),
                metrics.get("loss", math.nan),
                training.get("best_checkpoint_metric", math.nan),
            )
        )
    )


def cache_progress_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    cache_events = [
        event for event in events if str(event.get("event", "")).startswith("cache_")
    ]
    reuses = [event for event in cache_events if event.get("event") == "cache_reuse"]
    observed = {
        (
            int(event.get("seed", -1)),
            str(event.get("model")),
            str(event.get("split")),
        )
        for event in reuses
    }
    expected = {
        (seed, model, split)
        for seed in EXPECTED_SEEDS
        for model in ARCHITECTURES.values()
        for split in ("train", "validation")
    }
    generated = [
        event
        for event in cache_events
        if event.get("event") in {"cache_start", "cache_done"}
    ]
    final_test = [
        event
        for event in cache_events
        if str(event.get("split", "")).startswith("final_test")
    ]
    return {
        "twelve_source_cache_reuses_exact": (
            len(reuses) == EXPECTED_CACHE_REUSES and observed == expected
        ),
        "zero_cache_generation_events": not generated,
        "all_cache_reuses_under_k1ca_root": bool(reuses)
        and all(_is_source_cache_path(event.get("cache_path")) for event in reuses),
        "cache_chunks_and_workers_frozen": all(
            int(event.get("chunk_size", -1)) == 1024
            and int(event.get("workers", -1)) == 1
            for event in reuses
        ),
        "zero_final_test_cache_events": not final_test,
        "run_done_present": any(event.get("event") == "run_done" for event in events),
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    source_cache_audit: Mapping[str, Any],
    source_k1ca_gate: Mapping[str, Any],
    source_k1ca_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    k1ca_protocol_checks = source_k1ca_gate.get("protocol_checks", {})
    protocol_checks = {
        **dict(source_checks),
        "six_frozen_tasks_exact": plan_protocol_frozen(tasks),
        "source_cache_audit_pass": source_cache_audit.get("status") == "pass"
        and all(source_cache_audit.get("checks", {}).values()),
        "source_k1ca_gate_protocol_valid": (
            source_k1ca_gate.get("run_id") == K1CA_RUN_ID
            and source_k1ca_gate.get("status") in {"pass", "hold"}
            and bool(k1ca_protocol_checks)
            and all(k1ca_protocol_checks.values())
        ),
        "source_k1ca_results_protocol_frozen": k1ca_result_protocol_frozen(
            source_k1ca_rows
        ),
        "six_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS
        and set(rows) == expected_keys(),
        "result_protocol_frozen": result_protocol_frozen(result_rows),
        **cache_progress_checks(progress_events),
    }

    k1ca_auc: dict[int, dict[str, float]] = {}
    for row in source_k1ca_rows:
        seed = int(row.get("seed", -1))
        model = str(row.get("model", ""))
        if seed not in EXPECTED_SEEDS:
            continue
        if model == "runtime_spn_ct_k1t_position_histogram_invariant":
            name = "invariant_structure_expert"
        elif model == "autond_dbitnet2023":
            name = "autond_dbitnet"
        else:
            continue
        k1ca_auc.setdefault(seed, {})[name] = float(row["metrics"]["auc"])

    seed_results: dict[str, Any] = {}
    if set(rows) == expected_keys() and all(
        set(k1ca_auc.get(seed, {})) == {"invariant_structure_expert", "autond_dbitnet"}
        for seed in EXPECTED_SEEDS
    ):
        for seed in EXPECTED_SEEDS:
            candidate = k1ca_auc[seed]["invariant_structure_expert"]
            aucs = {
                **k1ca_auc[seed],
                **{
                    architecture: float(rows[(seed, architecture)]["metrics"]["auc"])
                    for architecture in ARCHITECTURES
                },
            }
            seed_results[str(seed)] = {
                "auc_by_architecture": aucs,
                "candidate_minus_baseline": {
                    name: candidate - auc
                    for name, auc in aucs.items()
                    if name != "invariant_structure_expert"
                },
            }

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if protocol_valid:
        status = "pass"
        decision = "innovation1_uknit_k1cb_published_comparison_complete"
        next_action = (
            "integrate all five architectures and both seeds into the paper main "
            "comparison table and visually checked figure, then close training"
        )
    else:
        status = "invalid"
        decision = "innovation1_uknit_k1cb_published_comparison_protocol_invalid"
        next_action = (
            "repair only the failed source-cache, source-result, plan, checkpoint, "
            "or result binding without generating data or changing the matrix"
        )
    return {
        "run_id": RUN_ID,
        "source_run_id": K1CA_RUN_ID,
        "status": status,
        "comparison_status": "complete" if protocol_valid else "invalid",
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "performance_gate": "none; all protocol-valid observations are reported",
        "next_action": next_action,
        "claim_scope": (
            "same-data two-seed 262144/class uKNIT-BC r5 project-protocol "
            "comparison against AutoND, MCND, Liu Conv2D and Gohr-style ResNet; "
            "architecture adaptations rather than exact reproductions or exhaustive "
            "hyperparameter searches; not formal million-scale, full-round, SOTA, "
            "attack-breakthrough or universal-SPN evidence"
        ),
        "blocked_actions": [
            "generating or replacing any K1-CA source cache",
            "dropping an unfavorable architecture or seed",
            "post-hoc tuning after seeing K1-CB results",
            "adding seeds, data, epochs, pairs, rounds or final tests",
            "million-scale or later mechanical scale-up",
        ],
    }


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").rstrip("\\").lower()


def _source_cache_root() -> str:
    return f"g:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{K1CA_RUN_ID}\\cache"


def _is_source_cache_root(value: Any) -> bool:
    return _normalized_path(value) == _normalized_path(_source_cache_root())


def _is_source_cache_path(value: Any) -> bool:
    return _normalized_path(value).startswith(
        _normalized_path(_source_cache_root()) + "\\"
    )


def _is_run_path(value: Any) -> bool:
    prefix = f"g:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{RUN_ID}\\"
    return _normalized_path(value).startswith(_normalized_path(prefix))


__all__ = [
    "ARCHITECTURES",
    "EXPECTED_CACHE_REUSES",
    "EXPECTED_PARAMETER_COUNTS",
    "EXPECTED_RESULT_ROWS",
    "RUN_ID",
    "adjudicate",
    "audit_source_caches",
    "cache_progress_checks",
    "plan_protocol_frozen",
    "read_tasks",
]
