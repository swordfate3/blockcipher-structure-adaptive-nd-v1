from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    INPUT_DIFFERENCE,
)


RUN_ID = "i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803"
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 4
EXPECTED_INPUT_BITS = 512
EXPECTED_SAMPLES_PER_CLASS = 262_144
EXPECTED_TRAIN_ROWS = 524_288
EXPECTED_VALIDATION_ROWS = 131_072
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_CACHE_CREATIONS = 4
EXPECTED_CACHE_REUSES = 4
EXPECTED_RESULT_ROWS = 4

CANDIDATE_AUC_FLOOR = 0.900
CANDIDATE_AUTOND_MARGIN = 0.100

ARCHITECTURES = {
    "invariant_structure_expert": "runtime_spn_ct_k1t_position_histogram_invariant",
    "autond_dbitnet": "autond_dbitnet2023",
}
MODEL_TO_ARCHITECTURE = {model: name for name, model in ARCHITECTURES.items()}
EXPECTED_PARAMETER_COUNTS = {
    "invariant_structure_expert": 214_316,
    "autond_dbitnet": 636_513,
}
EXPECTED_DESCRIPTOR_SHA256S = frozenset(
    {
        # Git may check out the pinned JSON with LF or CRLF on Windows.
        "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2",
        "0b39c38d6eb7b02b86cdd8822466da63648d75a02a0688a2f23f27e2ecef81f8",
    }
)


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
            raise ValueError(f"duplicate K1-CA task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-CA task matrix is incomplete")
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
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1)) == 128
            and int(task.get("model_options", {}).get("histogram_value_dim", -1)) == 8
            for (seed, _), task in mapped.items()
        )
    )


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
            raise ValueError(f"duplicate K1-CA result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-CA result matrix is incomplete")
    return mapped


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    mapped = result_map(rows, fail_closed=False)
    checkpoint_paths = {
        str(row.get("training", {}).get("checkpoint_output", "")) for row in rows
    }
    return (
        len(rows) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and len(checkpoint_paths) == EXPECTED_RESULT_ROWS
        and all(
            _row_protocol_frozen(row, seed=seed, architecture=architecture)
            for (seed, architecture), row in mapped.items()
        )
    )


def _row_protocol_frozen(
    row: Mapping[str, Any], *, seed: int, architecture: str
) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    metrics = row.get("metrics", {})
    history = row.get("history", [])
    checkpoint = str(training.get("checkpoint_output", ""))
    common = (
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
        and int(training.get("validation_positive_rows", -1))
        == EXPECTED_VALIDATION_ROWS // 2
        and int(training.get("validation_negative_rows", -1))
        == EXPECTED_VALIDATION_ROWS // 2
        and training.get("train_dataset_storage") == "disk"
        and training.get("validation_dataset_storage") == "disk"
        and _is_run_path(training.get("dataset_cache_root"))
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
    )
    if not common:
        return False
    if architecture == "autond_dbitnet":
        return True
    return (
        row.get("runtime_structure_descriptor_sha256") in EXPECTED_DESCRIPTOR_SHA256S
        and int(row.get("runtime_structure_round_start", -1)) == 3
        and int(row.get("runtime_structure_loaded_rounds", -1)) == 2
        and row.get("runtime_structure_mode") == "invariant"
        and row.get("runtime_structure_window_control") == "invariant"
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    relevant = [
        event
        for event in events
        if event.get("event") in {"cache_start", "cache_done", "cache_reuse"}
        and event.get("split") in {"train", "validation"}
    ]
    starts = [event for event in relevant if event.get("event") == "cache_start"]
    dones = [event for event in relevant if event.get("event") == "cache_done"]
    reuses = [event for event in relevant if event.get("event") == "cache_reuse"]
    created_paths = {_normalized_path(event.get("cache_path")) for event in starts}
    expected_seed_splits = {
        (seed, split) for seed in EXPECTED_SEEDS for split in ("train", "validation")
    }
    observed_starts = {
        (int(event.get("seed", -1)), str(event.get("split"))) for event in starts
    }
    observed_reuses = {
        (
            int(event.get("seed", -1)),
            str(event.get("model")),
            str(event.get("split")),
        )
        for event in reuses
    }
    expected_reuses = {
        (seed, ARCHITECTURES["autond_dbitnet"], split)
        for seed in EXPECTED_SEEDS
        for split in ("train", "validation")
    }
    final_test_events = [
        event
        for event in events
        if str(event.get("split", "")).startswith("final_test")
        and str(event.get("event", "")).startswith("cache_")
    ]
    return {
        "four_train_validation_cache_creations_exact": (
            len(starts) == EXPECTED_CACHE_CREATIONS
            and observed_starts == expected_seed_splits
            and len(created_paths) == EXPECTED_CACHE_CREATIONS
        ),
        "four_train_validation_cache_completions_exact": (
            len(dones) == EXPECTED_CACHE_CREATIONS
            and {_normalized_path(event.get("cache_path")) for event in dones}
            == created_paths
        ),
        "four_autond_cache_reuses_exact": (
            len(reuses) == EXPECTED_CACHE_REUSES and observed_reuses == expected_reuses
        ),
        "all_cache_paths_under_unique_run_root": bool(created_paths)
        and all(_is_run_path(path) for path in created_paths)
        and all(_is_run_path(event.get("cache_path")) for event in reuses),
        "cache_chunks_and_workers_frozen": all(
            int(event.get("chunk_size", -1)) == 1024
            and int(event.get("workers", -1)) == 1
            for event in (*starts, *reuses)
        ),
        "zero_final_test_cache_events": not final_test_events,
        "run_done_present": any(event.get("event") == "run_done" for event in events),
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        **dict(source_checks),
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "four_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS
        and set(rows) == expected_keys(),
        "result_protocol_frozen": result_protocol_frozen(result_rows),
        **cache_protocol_checks(progress_events),
    }
    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    if set(rows) == expected_keys():
        for seed in EXPECTED_SEEDS:
            candidate = float(
                rows[(seed, "invariant_structure_expert")]["metrics"]["auc"]
            )
            autond = float(rows[(seed, "autond_dbitnet")]["metrics"]["auc"])
            margin = candidate - autond
            seed_results[str(seed)] = {
                "auc_by_architecture": {
                    "invariant_structure_expert": candidate,
                    "autond_dbitnet": autond,
                },
                "candidate_minus_autond": margin,
            }
            research_checks[f"seed{seed}_candidate_auc_floor"] = (
                candidate >= CANDIDATE_AUC_FLOOR
            )
            research_checks[f"seed{seed}_candidate_autond_margin"] = (
                margin >= CANDIDATE_AUTOND_MARGIN
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_k1ca_closeout_protocol_invalid"
        next_action = (
            "repair only the failed source, cache, checkpoint, result, or archive "
            "binding and rerun the unchanged four-row closeout matrix"
        )
    elif research_pass:
        status = "pass"
        decision = "innovation1_uknit_k1ca_invariant_advantage_supported"
        next_action = (
            "close the K1-CA candidate-versus-AutoND stage, then run the frozen "
            "cache-reusing K1-CB published-network paper comparison before final "
            "table and figure integration"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_k1ca_invariant_advantage_not_supported"
        next_action = (
            "close the K1-CA candidate-versus-AutoND stage, report its frozen "
            "result, and still run the independently preregistered cache-reusing "
            "K1-CB paper comparison without changing K1-CA"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "experiment_stage_after_valid_result": "closed",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "thresholds": {
            "candidate_auc": CANDIDATE_AUC_FLOOR,
            "candidate_minus_autond": CANDIDATE_AUTOND_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed remote 262144/class uKNIT-BC r5 project-protocol paper "
            "closeout; not an exact AutoND public-code reproduction, formal "
            "million-scale benchmark, full-round result, attack breakthrough, "
            "SOTA claim, or universal-SPN network claim"
        ),
        "blocked_actions": [
            "changing or adding any K1-CA model-seed row",
            "using K1-CB to rescue or retune the K1-CA gate",
            "additional seeds, data, rounds, pairs, epochs, differences, or keys",
            "final-test repeats or final-test caches",
            "million-scale or later mechanical scale-up",
        ],
    }


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def _is_run_path(value: Any) -> bool:
    prefix = f"g:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{RUN_ID}\\"
    return _normalized_path(value).startswith(prefix)


__all__ = [
    "ARCHITECTURES",
    "EXPECTED_CACHE_CREATIONS",
    "EXPECTED_PARAMETER_COUNTS",
    "EXPECTED_RESULT_ROWS",
    "RUN_ID",
    "adjudicate",
    "candidate_protocol_frozen",
    "read_tasks",
]
