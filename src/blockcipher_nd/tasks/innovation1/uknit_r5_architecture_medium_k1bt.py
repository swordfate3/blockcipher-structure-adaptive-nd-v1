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
from blockcipher_nd.tasks.innovation1.uknit_r5_architecture_ablation_k1bs import (
    EXPECTED_PARAMETER_COUNTS as K1BS_PARAMETER_COUNTS,
)


RUN_ID = (
    "i1_uknit_r5_neural_architecture_medium_k1bt_16pair_"
    "65536_seed3_seed4_20260731"
)
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 16
PAIR_BITS = 128
EXPECTED_INPUT_BITS = EXPECTED_PAIRS * PAIR_BITS
EXPECTED_SAMPLES_PER_CLASS = 65_536
EXPECTED_TRAIN_ROWS = 131_072
EXPECTED_VALIDATION_ROWS = 32_768
EXPECTED_EPOCHS = 10
EXPERT_SIGNAL_FLOOR = 0.550
EXPERT_MARGIN = 0.010
ARCHITECTURES = {
    "uknit_structure_expert": "runtime_spn_ct_k1t_position_histogram_true",
    "autond_dbitnet": "autond_dbitnet2023",
}
MODEL_TO_ARCHITECTURE = {model: name for name, model in ARCHITECTURES.items()}
EXPECTED_PARAMETER_COUNTS = {
    name: K1BS_PARAMETER_COUNTS[name] for name in ARCHITECTURES
}
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(ARCHITECTURES)
EXPECTED_CACHE_CREATIONS = len(EXPECTED_SEEDS) * 2
EXPECTED_CACHE_REUSES = len(EXPECTED_SEEDS) * 2


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
            raise ValueError(f"duplicate K1-BT task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BT task matrix is incomplete")
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
            raise ValueError(f"duplicate K1-BT result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BT result matrix is incomplete")
    return mapped


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        row.get("model") in MODEL_TO_ARCHITECTURE
        and int(row.get("rounds", -1)) == 5
        and int(row.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNTS[MODEL_TO_ARCHITECTURE[str(row.get("model"))]]
        and int(row.get("training", {}).get("input_bits", -1)) == EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and Path(str(row.get("training", {}).get("checkpoint_output", ""))).is_file()
        for row in rows
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    relevant = [
        row
        for row in events
        if row.get("event") in {"cache_start", "cache_reuse"}
        and row.get("split") in {"train", "validation"}
    ]
    creations = [row for row in relevant if row.get("event") == "cache_start"]
    reuses = [row for row in relevant if row.get("event") == "cache_reuse"]
    return {
        "four_disk_caches_created": len(creations) == EXPECTED_CACHE_CREATIONS,
        "four_parameter_matched_cache_reuses": len(reuses) == EXPECTED_CACHE_REUSES,
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
        "four_result_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS and set(rows) == expected_keys()
        ),
        "result_protocol_frozen": result_protocol_frozen(result_rows),
        **cache_protocol_checks(progress_events),
    }
    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    if set(rows) == expected_keys():
        for seed in EXPECTED_SEEDS:
            expert = float(rows[(seed, "uknit_structure_expert")]["metrics"]["auc"])
            baseline = float(rows[(seed, "autond_dbitnet")]["metrics"]["auc"])
            margin = expert - baseline
            seed_results[str(seed)] = {
                "auc_by_architecture": {
                    "uknit_structure_expert": expert,
                    "autond_dbitnet": baseline,
                },
                "expert_minus_autond": margin,
            }
            research_checks[f"seed{seed}_expert_signal"] = expert >= EXPERT_SIGNAL_FLOOR
            research_checks[f"seed{seed}_expert_margin"] = margin >= EXPERT_MARGIN

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_k1bt_medium_protocol_invalid"
        scale = "no"
        next_action = "repair only the failed source, plan, cache, checkpoint, or result binding and rerun K1-BT unchanged"
    elif research_pass:
        status = "pass"
        decision = "innovation1_uknit_k1bt_medium_structure_expert_supported"
        scale = "authorized_262144_per_class"
        next_action = "run K1-BU remotely at 262144/class with the same two models, seeds, data protocol, pairs, epochs, and gates"
    else:
        status = "hold"
        decision = "innovation1_uknit_k1bt_medium_structure_expert_not_supported"
        scale = "no"
        next_action = "inspect failed-seed restored-best history and cache equivalence; do not mechanically scale or change the architecture"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": scale,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(name for name, ok in protocol_checks.items() if not ok),
        "research_checks": research_checks,
        "failed_research_checks": sorted(name for name, ok in research_checks.items() if not ok),
        "seed_results": seed_results,
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "thresholds": {"expert_auc": EXPERT_SIGNAL_FLOOR, "expert_minus_autond": EXPERT_MARGIN},
        "next_action": next_action,
        "claim_scope": "two-seed remote 65536/class uKNIT r5 medium architecture confirmation; not formal, paper-scale, attack, SOTA, transfer, universal-SPN, or capacity-matched topology-causal evidence",
        "blocked_actions": [
            "calling 65536/class formal or paper-scale evidence",
            "advancing to 262144/class unless both seeds pass",
            "changing data, difference, keys, pairs, epochs, labels, negatives, or model capacity",
        ],
    }


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
