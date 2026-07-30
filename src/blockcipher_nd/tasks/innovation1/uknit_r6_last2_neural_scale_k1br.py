from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    MODEL_TO_CONDITION,
)


RUN_ID = "i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730"
EXPECTED_SEED = 3
EXPECTED_SAMPLES_PER_CLASS = 262_144
EXPECTED_TRAIN_ROWS = 524_288
EXPECTED_VALIDATION_ROWS = 131_072
EXPECTED_PAIRS = 4
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_RESULT_ROWS = 3
EXPECTED_CACHE_CREATIONS = 2
EXPECTED_CACHE_REUSES = 4
INPUT_DIFFERENCE = 0x0000400000000000
DIFFERENCE_PROFILE = "uknit64_k1q_cell11_r5"
TRAIN_KEY = int("44" * 16, 16)
VALIDATION_KEY = int("55" * 16, 16)
WEAK_AUC_FLOOR = 0.51
STRONG_AUC_FLOOR = 0.55
WEAK_ATTRIBUTION_MARGIN = 0.005
STRONG_ATTRIBUTION_MARGIN = 0.01
CHECKPOINT_AUC_REPLAY_TOLERANCE = 1e-6
EXPECTED_DESCRIPTOR_SHA256S = frozenset(
    {
        "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2",
        "0b39c38d6eb7b02b86cdd8822466da63648d75a02a0688a2f23f27e2ecef81f8",
    }
)


def expected_keys() -> set[str]:
    return set(CONTROL_MODELS)


def task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        condition = MODEL_TO_CONDITION.get(str(task.get("model_key")))
        if condition is None:
            continue
        if condition in mapped:
            raise ValueError(f"duplicate K1-BR task: {condition}")
        mapped[condition] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BR task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 6
            and int(task.get("seed", -1)) == EXPECTED_SEED
            and int(task.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == TRAIN_KEY
            and int(task.get("validation_key", -1)) == VALIDATION_KEY
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and task.get("optimizer_state_transition") == "reset_each_stage"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("lr_scheduler") == "none"
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 4
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("histogram_value_dim", -1)) == 8
            for task in mapped.values()
        )
    )


def result_map(
    rows: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        if condition in mapped:
            raise ValueError(f"duplicate K1-BR result: {condition}")
        mapped[condition] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BR result matrix is incomplete")
    return mapped


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_RESULT_ROWS:
        return False
    try:
        mapped = result_map(rows)
    except ValueError:
        return False
    return all(_row_protocol_frozen(row) for row in mapped.values())


def _row_protocol_frozen(row: Mapping[str, Any]) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    metrics = row.get("metrics", {})
    history = row.get("history", [])
    return (
        row.get("cipher_key") == "uknit64"
        and int(row.get("rounds", -1)) == 6
        and int(row.get("seed", -1)) == EXPECTED_SEED
        and int(row.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and row.get("runtime_structure_descriptor_sha256")
        in EXPECTED_DESCRIPTOR_SHA256S
        and int(row.get("runtime_structure_round_start", -1)) == 4
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
        and _is_remote_path(training.get("dataset_cache_root"))
        and _is_remote_path(training.get("checkpoint_output"))
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
        <= CHECKPOINT_AUC_REPLAY_TOLERANCE
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    starts = [row for row in events if row.get("event") == "cache_start"]
    dones = [row for row in events if row.get("event") == "cache_done"]
    reuses = [row for row in events if row.get("event") == "cache_reuse"]
    flushes = [row for row in events if row.get("event") == "cache_flush_start"]
    positives = [row for row in events if row.get("event") == "cache_positive_chunk"]
    negatives = [row for row in events if row.get("event") == "cache_negative_chunk"]
    start_paths = {_normal_path(row.get("cache_path")) for row in starts}
    expected_reuses = {
        (model, split)
        for model in (
            CONTROL_MODELS["wrong_sbox_position_histogram_residual"],
            CONTROL_MODELS["invariant_histogram_residual"],
        )
        for split in ("train", "validation")
    }
    observed_reuses = {(str(row.get("model")), str(row.get("split"))) for row in reuses}
    return {
        "two_cache_creations_exact": (
            len(starts) == EXPECTED_CACHE_CREATIONS
            and {str(row.get("split")) for row in starts} == {"train", "validation"}
            and len(start_paths) == EXPECTED_CACHE_CREATIONS
        ),
        "two_cache_completions_exact": (
            len(dones) == EXPECTED_CACHE_CREATIONS
            and {_normal_path(row.get("cache_path")) for row in dones} == start_paths
        ),
        "durable_chunks_for_both_classes": (
            {_normal_path(row.get("cache_path")) for row in flushes} == start_paths
            and {_normal_path(row.get("cache_path")) for row in positives}
            == start_paths
            and {_normal_path(row.get("cache_path")) for row in negatives}
            == start_paths
        ),
        "four_control_cache_reuses_exact": (
            len(reuses) == EXPECTED_CACHE_REUSES and observed_reuses == expected_reuses
        ),
        "all_cache_paths_remote": bool(start_paths)
        and all(_is_remote_path(path) for path in start_paths),
        "cache_chunk_workers_frozen": all(
            int(row.get("chunk_size", -1)) == 1024 and int(row.get("workers", -1)) == 1
            for row in (*starts, *reuses)
        ),
        "run_done_present": any(row.get("event") == "run_done" for row in events),
    }


def adjudicate_k1br(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        **dict(source_checks),
        "three_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "three_result_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS and set(rows) == expected_keys()
        ),
        "result_protocol_frozen": result_protocol_frozen(result_rows),
        **cache_protocol_checks(progress_events),
    }
    metrics: dict[str, float] = {}
    if set(rows) == expected_keys():
        metrics = {
            condition: float(rows[condition]["metrics"]["auc"])
            for condition in CONTROL_MODELS
        }
    exact = metrics.get("exact_position_histogram_residual", math.nan)
    invariant = metrics.get("invariant_histogram_residual", math.nan)
    wrong = metrics.get("wrong_sbox_position_histogram_residual", math.nan)
    candidate = max(exact, invariant)
    candidate_condition = (
        "exact_position_histogram_residual"
        if exact >= invariant
        else "invariant_histogram_residual"
    )
    margin = candidate - wrong
    research_checks = {
        "candidate_weak_auc_floor": candidate >= WEAK_AUC_FLOOR,
        "candidate_strong_auc_floor": candidate >= STRONG_AUC_FLOOR,
        "weak_wrong_sbox_attribution": margin >= WEAK_ATTRIBUTION_MARGIN,
        "strong_wrong_sbox_attribution": margin >= STRONG_ATTRIBUTION_MARGIN,
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_r6_k1br_protocol_invalid"
        tier = "invalid"
        next_action = (
            "repair only the failed protocol or artifact binding and rerun unchanged"
        )
    elif (
        research_checks["candidate_strong_auc_floor"]
        and research_checks["strong_wrong_sbox_attribution"]
    ):
        status = "pass"
        decision = "innovation1_uknit_r6_k1br_strong_attributed_candidate"
        tier = "strong_attributed"
        next_action = "repeat the identical 262144/class matrix with seed4 before any formal scale"
    elif (
        research_checks["candidate_weak_auc_floor"]
        and research_checks["weak_wrong_sbox_attribution"]
    ):
        status = "pass"
        decision = "innovation1_uknit_r6_k1br_weak_attributed_signal"
        tier = "weak_attributed"
        next_action = "repeat the identical 262144/class matrix with seed4; do not jump to 1M/class"
    elif research_checks["candidate_weak_auc_floor"]:
        status = "hold"
        decision = "innovation1_uknit_r6_k1br_weak_unattributed_signal"
        tier = "weak_unattributed"
        next_action = "hold scale and diagnose representation attribution against the wrong-S-box control"
    else:
        status = "hold"
        decision = "innovation1_uknit_r6_k1br_no_positive_signal_at_scale"
        tier = "no_supported_positive_signal"
        next_action = "hold scale and redesign the r6 representation or difference before more data"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "tier": tier,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "aucs": metrics,
        "best_candidate_condition": candidate_condition,
        "best_candidate_auc": candidate,
        "wrong_sbox_auc": wrong,
        "attribution_margin": margin,
        "thresholds": {
            "weak_auc": WEAK_AUC_FLOOR,
            "strong_auc": STRONG_AUC_FLOOR,
            "weak_attribution_margin": WEAK_ATTRIBUTION_MARGIN,
            "strong_attribution_margin": STRONG_ATTRIBUTION_MARGIN,
        },
        "next_action": next_action,
        "remote_scale": "single_seed_diagnostic_only",
        "claim_scope": (
            "single-seed remote 262144/class uKNIT r6 larger diagnostic; not "
            "formal, paper-scale, attack, SOTA, breakthrough, transfer, route-"
            "ceiling, or universal-SPN evidence"
        ),
        "blocked_actions": [
            "call this formal training or definitive r6 failure",
            "jump directly to 1M/class",
            "change pairs, epochs, capacity, difference, or window in this gate",
        ],
    }


def _normal_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def _is_remote_path(value: Any) -> bool:
    return _normal_path(value).startswith("g:\\lxy\\")


__all__ = [
    "EXPECTED_RESULT_ROWS",
    "RUN_ID",
    "adjudicate_k1br",
    "cache_protocol_checks",
    "candidate_protocol_frozen",
    "result_protocol_frozen",
]
