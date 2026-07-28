from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    EXPECTED_PAIRS,
    EXPECTED_SEEDS,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    MODEL_TO_CONDITION,
)


RUN_ID = (
    "i1_uknit_family_ctspn_position_residual_"
    "k1u_medium_65536_seed3_seed4_20260728"
)
EXPECTED_SAMPLES_PER_CLASS = 65_536
EXPECTED_TRAIN_ROWS = 131_072
EXPECTED_VALIDATION_ROWS = 65_536
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_MODELS)
EXPECTED_CACHE_CREATIONS = len(EXPECTED_SEEDS) * 2
EXPECTED_CACHE_REUSES = len(EXPECTED_SEEDS) * (len(CONTROL_MODELS) - 1) * 2
AUC_FLOOR = 0.600
WRONG_SBOX_MARGIN = 0.010
INVARIANT_MARGIN = 0.030


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        condition = MODEL_TO_CONDITION.get(str(task.get("model_key")))
        if condition is None:
            continue
        key = (int(task.get("seed", -1)), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-U task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-U task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1))
            == EXPECTED_SAMPLES_PER_CLASS
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
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
            and int(task.get("model_options", {}).get("runtime_round_start", -1))
            == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("histogram_value_dim", -1))
            == 8
            for (seed, _), task in mapped.items()
        )
    )


def adjudicate_k1u(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        **dict(source_checks),
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_result_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS and set(rows) == expected_keys()
        ),
        "result_protocol_frozen": result_protocol_frozen(result_rows),
        **cache_protocol_checks(progress_events),
    }
    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    if set(rows) == expected_keys():
        for seed in EXPECTED_SEEDS:
            aucs = {
                condition: float(rows[(seed, condition)]["metrics"]["auc"])
                for condition in CONTROL_MODELS
            }
            exact = aucs["exact_position_histogram_residual"]
            result = {
                **{f"{condition}_auc": auc for condition, auc in aucs.items()},
                "exact_minus_wrong_sbox": (
                    exact - aucs["wrong_sbox_position_histogram_residual"]
                ),
                "exact_minus_invariant": (
                    exact - aucs["invariant_histogram_residual"]
                ),
            }
            seed_results[str(seed)] = result
            research_checks[f"seed{seed}_exact_auc_floor"] = exact >= AUC_FLOOR
            research_checks[f"seed{seed}_beats_wrong_sbox"] = (
                result["exact_minus_wrong_sbox"] >= WRONG_SBOX_MARGIN
            )
            research_checks[f"seed{seed}_beats_invariant"] = (
                result["exact_minus_invariant"] >= INVARIANT_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    exact_signal = bool(research_checks) and all(
        research_checks[f"seed{seed}_exact_auc_floor"] for seed in EXPECTED_SEEDS
    )
    semantic_pass = bool(research_checks) and all(
        research_checks[f"seed{seed}_beats_wrong_sbox"] for seed in EXPECTED_SEEDS
    )
    invariant_pass = bool(research_checks) and all(
        research_checks[f"seed{seed}_beats_invariant"] for seed in EXPECTED_SEEDS
    )
    per_seed_pass = {
        seed: all(
            research_checks.get(f"seed{seed}_{suffix}", False)
            for suffix in ("exact_auc_floor", "beats_wrong_sbox", "beats_invariant")
        )
        for seed in EXPECTED_SEEDS
    }

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1u_protocol_invalid"
        next_action = (
            "repair only the failed source, cache, checkpoint, result, or artifact "
            "binding and rerun the unchanged medium matrix"
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1u_medium_position_residual_supported"
        )
        next_action = (
            "hold mechanical uKNIT scale; replace the fixed sixteen-cell flatten "
            "projection with a runtime-cell-count shared aggregator and preregister "
            "one same-budget compatible-SPN transfer diagnostic"
        )
    elif exact_signal and not semantic_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1u_medium_signal_without_wrong_sbox_"
            "attribution"
        )
        next_action = (
            "hold scale and isolate the five deterministic stage contributions at "
            "the completed medium checkpoints"
        )
    elif exact_signal and not invariant_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1u_medium_signal_without_position_"
            "necessity"
        )
        next_action = (
            "hold scale and replace the candidate by the simpler invariant branch"
        )
    elif sum(per_seed_pass.values()) == 1:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1u_medium_seed_key_instability"
        )
        next_action = (
            "inspect the failed seed history and restored checkpoint without adding "
            "seeds, data, capacity, pairs, epochs, differences, or rounds"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1u_medium_position_residual_not_supported"
        )
        next_action = (
            "audit K1-T versus K1-U training dynamics and cache equivalence; do not "
            "mechanically scale or change the architecture"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no_mechanical_scale",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_wrong_sbox": WRONG_SBOX_MARGIN,
            "exact_minus_invariant": INVARIANT_MARGIN,
        },
        "descriptive_diagnostics": {
            "exact_signal_both_seeds": exact_signal,
            "wrong_sbox_attribution_both_seeds": semantic_pass,
            "position_necessity_both_seeds": invariant_pass,
            "per_seed_full_gate": {str(seed): passed for seed, passed in per_seed_pass.items()},
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed remote 65536/class uKNIT r5 cell11 medium diagnostic; not "
            "formal, paper-scale, attack, SOTA, transfer, or universal-SPN evidence"
        ),
        "blocked_actions": [
            "local execution or 262144/class mechanical scale",
            "more epochs, pairs, differences, rounds, seeds, or keys",
            "MoE, DDT/trails, cipher identity, or another network family",
            "family-transfer claims before retrieved evidence passes",
        ],
    }


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_RESULT_ROWS:
        return False
    try:
        mapped = result_map(rows)
    except ValueError:
        return False
    return all(
        _row_protocol_frozen(mapped[(seed, condition)], seed=seed)
        for seed, condition in expected_keys()
    )


def _row_protocol_frozen(row: Mapping[str, Any], *, seed: int) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    metrics = row.get("metrics", {})
    history = row.get("history", [])
    cache_root = str(training.get("dataset_cache_root", ""))
    checkpoint = str(training.get("checkpoint_output", ""))
    return (
        row.get("cipher_key") == "uknit64"
        and int(row.get("rounds", -1)) == 5
        and int(row.get("seed", -1)) == seed
        and int(row.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and row.get("runtime_structure_descriptor_sha256")
        == "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2"
        and int(row.get("runtime_structure_round_start", -1)) == 3
        and int(row.get("runtime_structure_loaded_rounds", -1)) == 2
        and int(training.get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(training.get("validation_rows", -1)) == EXPECTED_VALIDATION_ROWS
        and int(training.get("train_positive_rows", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(training.get("train_negative_rows", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(training.get("validation_positive_rows", -1))
        == EXPECTED_VALIDATION_ROWS // 2
        and int(training.get("validation_negative_rows", -1))
        == EXPECTED_VALIDATION_ROWS // 2
        and training.get("train_dataset_storage") == "disk"
        and training.get("validation_dataset_storage") == "disk"
        and _is_remote_project_path(cache_root)
        and _is_remote_project_path(checkpoint)
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
        and int(validation.get("samples_per_class", -1))
        == EXPECTED_VALIDATION_ROWS // 2
        and all(
            math.isfinite(float(value))
            for value in (
                metrics.get("auc", math.nan),
                metrics.get("accuracy", math.nan),
                metrics.get("loss", math.nan),
                training.get("best_checkpoint_metric", math.nan),
            )
        )
        and abs(
            float(metrics.get("auc", math.nan))
            - float(training.get("best_checkpoint_metric", math.nan))
        )
        <= 1e-12
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    starts = [event for event in events if event.get("event") == "cache_start"]
    dones = [event for event in events if event.get("event") == "cache_done"]
    reuses = [event for event in events if event.get("event") == "cache_reuse"]
    flushes = [
        event for event in events if event.get("event") == "cache_flush_start"
    ]
    positive_chunks = [
        event for event in events if event.get("event") == "cache_positive_chunk"
    ]
    negative_chunks = [
        event for event in events if event.get("event") == "cache_negative_chunk"
    ]
    created_paths = {_normalized_path(event.get("cache_path")) for event in starts}
    done_paths = {_normalized_path(event.get("cache_path")) for event in dones}
    flush_paths = {_normalized_path(event.get("cache_path")) for event in flushes}
    positive_paths = {
        _normalized_path(event.get("cache_path")) for event in positive_chunks
    }
    negative_paths = {
        _normalized_path(event.get("cache_path")) for event in negative_chunks
    }
    expected_created = {
        (seed, split)
        for seed in EXPECTED_SEEDS
        for split in ("train", "validation")
    }
    observed_created = {
        (int(event.get("seed", -1)), str(event.get("split"))) for event in starts
    }
    expected_reused = {
        (seed, model, split)
        for seed in EXPECTED_SEEDS
        for model in (
            CONTROL_MODELS["wrong_sbox_position_histogram_residual"],
            CONTROL_MODELS["invariant_histogram_residual"],
        )
        for split in ("train", "validation")
    }
    observed_reused = {
        (
            int(event.get("seed", -1)),
            str(event.get("model")),
            str(event.get("split")),
        )
        for event in reuses
    }
    all_paths = created_paths | done_paths | {
        _normalized_path(event.get("cache_path")) for event in reuses
    }
    return {
        "four_medium_cache_creations_exact": (
            len(starts) == EXPECTED_CACHE_CREATIONS
            and observed_created == expected_created
            and len(created_paths) == EXPECTED_CACHE_CREATIONS
        ),
        "four_medium_cache_completions_exact": (
            len(dones) == EXPECTED_CACHE_CREATIONS and done_paths == created_paths
        ),
        "all_created_caches_flushed_with_both_classes": (
            flush_paths == created_paths
            and positive_paths == created_paths
            and negative_paths == created_paths
        ),
        "eight_control_cache_reuses_exact": (
            len(reuses) == EXPECTED_CACHE_REUSES
            and observed_reused == expected_reused
        ),
        "all_cache_paths_under_remote_root": (
            bool(all_paths) and all(_is_remote_project_path(path) for path in all_paths)
        ),
        "cache_chunks_and_workers_frozen": all(
            int(event.get("chunk_size", -1)) == 1024
            and int(event.get("workers", -1)) == 1
            for event in (*starts, *reuses)
        ),
        "run_done_present": any(event.get("event") == "run_done" for event in events),
    }


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (int(row.get("seed", -1)), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-U result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-U results are incomplete")
    return mapped


def expected_keys() -> set[tuple[int, str]]:
    return {(seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_MODELS}


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def _is_remote_project_path(value: Any) -> bool:
    return _normalized_path(value).startswith("g:\\lxy\\")


__all__ = [
    "RUN_ID",
    "adjudicate_k1u",
    "cache_protocol_checks",
    "candidate_protocol_frozen",
    "expected_keys",
    "result_protocol_frozen",
    "task_map",
]
