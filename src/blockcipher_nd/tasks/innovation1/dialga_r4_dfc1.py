from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.planning.matrix import tasks_from_plan


RUN_ID = "i1_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1_20260802"
EXPECTED_SEEDS = (0, 1)
EXPECTED_PAIRS = 4
EXPECTED_INPUT_BITS = 1024
EXPECTED_SAMPLES_PER_CLASS = 1_000_000
EXPECTED_TRAIN_ROWS = 2_000_000
EXPECTED_VALIDATION_ROWS = 500_000
EXPECTED_FINAL_TEST_ROWS = 1_000_000
EXPECTED_FINAL_TEST_REPEATS = 5
EXPECTED_EPOCHS = 10
INPUT_DIFFERENCE = 0x40
FINAL_TEST_KEY = int("22" * 32, 16)
CORRECT_AUC_FLOOR = 0.900
TOPOLOGY_MARGIN = 0.005
AUTOND_MARGIN = 0.010
ARCHITECTURES = {
    "correct": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
    "autond": "autond_dbitnet2023",
}
MODEL_TO_ARCHITECTURE = {model: name for name, model in ARCHITECTURES.items()}
EXPECTED_PARAMETER_COUNTS = {
    "correct": 442_466,
    "corrupted": 442_466,
    "autond": 797_633,
}
EXPECTED_RESULT_ROWS = 6
EXPECTED_CACHE_CREATIONS = len(EXPECTED_SEEDS) * (
    2 + EXPECTED_FINAL_TEST_REPEATS
)
EXPECTED_CACHE_REUSES = EXPECTED_CACHE_CREATIONS * (len(ARCHITECTURES) - 1)


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


def _map_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_field: str,
    fail_closed: bool,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        architecture = MODEL_TO_ARCHITECTURE.get(str(row.get(model_field)))
        if architecture is None:
            continue
        key = (int(row.get("seed", -1)), architecture)
        if key in mapped:
            raise ValueError(f"duplicate DFC1 row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("DFC1 matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _map_rows(tasks, model_field="model_key", fail_closed=False)
    if len(tasks) != EXPECTED_RESULT_ROWS or set(mapped) != expected_keys():
        return False
    for (seed, architecture), task in mapped.items():
        options = task.get("model_options", {})
        common = (
            task.get("cipher_key") == "dialga128"
            and int(task.get("rounds", -1)) == 4
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1))
            == EXPECTED_SAMPLES_PER_CLASS
            and int(task.get("train_samples_total", -1)) == EXPECTED_TRAIN_ROWS
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("final_test_repeats", -1))
            == EXPECTED_FINAL_TEST_REPEATS
            and int(task.get("final_test_samples_total", -1))
            == EXPECTED_FINAL_TEST_ROWS
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == 0
            and int(task.get("validation_key", -1)) == int("11" * 32, 16)
            and int(task.get("final_test_key", -1)) == FINAL_TEST_KEY
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and task.get("optimizer_state_transition") == "reset_each_stage"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("lr_scheduler") == "none"
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
        )
        if not common:
            return False
        if architecture == "autond":
            if options:
                return False
            continue
        if not (
            options.get("runtime_structure_path")
            == "configs/runtime/spn/dialga128.json"
            and int(options.get("runtime_round_start", -1)) == 2
            and int(options.get("runtime_rounds", -1)) == 2
            and int(options.get("processor_steps", -1)) == 2
            and int(options.get("pair_embedding_dim", -1)) == 128
            and float(options.get("dropout", math.nan)) == 0.0
            and options.get("sbox_context_mode") == "edge_gate"
            and options.get("cell_input_mode") == "state_triplet"
            and options.get("round_window_mode") == "recurrent_window"
            and options.get("runtime_structure_window_control") == "full"
        ):
            return False
        if architecture == "corrupted" and int(
            options.get("topology_corruption_seed", -1)
        ) != 20260725:
            return False
    return True


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _map_rows(rows, model_field="model", fail_closed=False)
    checkpoint_paths = {
        str(row.get("training", {}).get("checkpoint_output", "")) for row in rows
    }
    return (
        len(rows) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and len(checkpoint_paths) == EXPECTED_RESULT_ROWS
        and all(path.endswith(".pt") for path in checkpoint_paths)
        and all(
            int(row.get("rounds", -1)) == 4
            and int(row.get("samples_per_class", -1))
            == EXPECTED_SAMPLES_PER_CLASS
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and int(row.get("final_test_repeats", -1))
            == EXPECTED_FINAL_TEST_REPEATS
            and int(row.get("final_test_samples_total", -1))
            == EXPECTED_FINAL_TEST_ROWS
            and int(row.get("final_test_key", -1)) == FINAL_TEST_KEY
            and int(row.get("trainable_parameter_count", -1))
            == EXPECTED_PARAMETER_COUNTS[architecture]
            and int(row.get("training", {}).get("input_bits", -1))
            == EXPECTED_INPUT_BITS
            and int(row.get("training", {}).get("train_rows", -1))
            == EXPECTED_TRAIN_ROWS
            and int(row.get("training", {}).get("validation_rows", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("epochs_ran", -1))
            == EXPECTED_EPOCHS
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and _final_evaluation_protocol_frozen(row, seed=seed)
            for (seed, architecture), row in mapped.items()
        )
    )


def _final_evaluation_protocol_frozen(
    row: Mapping[str, Any], *, seed: int
) -> bool:
    evaluation = row.get("final_evaluation", {})
    repeats = evaluation.get("metrics_by_repeat", [])
    return (
        int(evaluation.get("repeats", -1)) == EXPECTED_FINAL_TEST_REPEATS
        and int(evaluation.get("samples_total_per_repeat", -1))
        == EXPECTED_FINAL_TEST_ROWS
        and int(evaluation.get("final_test_key", -1)) == FINAL_TEST_KEY
        and evaluation.get("seeds")
        == [seed + 50_000 + index for index in range(EXPECTED_FINAL_TEST_REPEATS)]
        and isinstance(repeats, list)
        and len(repeats) == EXPECTED_FINAL_TEST_REPEATS
        and all(
            int(item.get("repeat", -1)) == index + 1
            and int(item.get("seed", -1)) == seed + 50_000 + index
            and int(item.get("final_test_key", -1)) == FINAL_TEST_KEY
            and int(item.get("samples_total", -1)) == EXPECTED_FINAL_TEST_ROWS
            and int(item.get("positive_rows", -1)) == EXPECTED_FINAL_TEST_ROWS // 2
            and int(item.get("negative_rows", -1)) == EXPECTED_FINAL_TEST_ROWS // 2
            and math.isfinite(float(item.get("auc", math.nan)))
            and math.isfinite(float(item.get("accuracy", math.nan)))
            for index, item in enumerate(repeats)
        )
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    relevant = [
        event
        for event in events
        if event.get("event") in {"cache_start", "cache_reuse"}
        and (
            event.get("split") in {"train", "validation"}
            or str(event.get("split", "")).startswith("final_test_")
        )
    ]
    creations = [event for event in relevant if event.get("event") == "cache_start"]
    reuses = [event for event in relevant if event.get("event") == "cache_reuse"]
    return {
        "fourteen_disk_caches_created": len(creations) == EXPECTED_CACHE_CREATIONS,
        "twenty_eight_parameter_matched_cache_reuses": len(reuses)
        == EXPECTED_CACHE_REUSES,
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = _map_rows(result_rows, model_field="model", fail_closed=False)
    results_frozen = result_protocol_frozen(result_rows)
    protocol_checks = {
        **dict(source_checks),
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS
        and set(rows) == expected_keys(),
        "result_protocol_frozen": results_frozen,
        **cache_protocol_checks(progress_events),
    }
    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    if set(rows) == expected_keys() and results_frozen:
        for seed in EXPECTED_SEEDS:
            aucs = {
                name: float(rows[(seed, name)]["metrics"]["auc"])
                for name in ARCHITECTURES
            }
            topology_margin = aucs["correct"] - aucs["corrupted"]
            autond_margin = aucs["correct"] - aucs["autond"]
            seed_results[str(seed)] = {
                "auc_by_architecture": aucs,
                "correct_minus_corrupted": topology_margin,
                "correct_minus_autond": autond_margin,
                "final_test_repeats": {},
            }
            research_checks[f"seed{seed}_correct_auc"] = (
                aucs["correct"] >= CORRECT_AUC_FLOOR
            )
            research_checks[f"seed{seed}_topology_margin"] = (
                topology_margin >= TOPOLOGY_MARGIN
            )
            research_checks[f"seed{seed}_autond_margin"] = (
                autond_margin >= AUTOND_MARGIN
            )
            for repeat_index in range(EXPECTED_FINAL_TEST_REPEATS):
                repeat = repeat_index + 1
                final_aucs = {
                    name: float(
                        rows[(seed, name)]["final_evaluation"]["metrics_by_repeat"][
                            repeat_index
                        ]["auc"]
                    )
                    for name in ARCHITECTURES
                }
                final_topology_margin = (
                    final_aucs["correct"] - final_aucs["corrupted"]
                )
                final_autond_margin = final_aucs["correct"] - final_aucs["autond"]
                seed_results[str(seed)]["final_test_repeats"][str(repeat)] = {
                    "auc_by_architecture": final_aucs,
                    "correct_minus_corrupted": final_topology_margin,
                    "correct_minus_autond": final_autond_margin,
                }
                prefix = f"seed{seed}_final{repeat}"
                research_checks[f"{prefix}_correct_auc"] = (
                    final_aucs["correct"] >= CORRECT_AUC_FLOOR
                )
                research_checks[f"{prefix}_topology_margin"] = (
                    final_topology_margin >= TOPOLOGY_MARGIN
                )
                research_checks[f"{prefix}_autond_margin"] = (
                    final_autond_margin >= AUTOND_MARGIN
                )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_dialga_dfc1_formal_protocol_invalid"
        next_action = (
            "repair only the failed source, plan, cache, checkpoint, or result "
            "binding and rerun DFC1 unchanged under a new unique run id"
        )
    elif research_pass:
        status = "pass"
        decision = "innovation1_dialga_dfc1_formal_topology_supported"
        next_action = (
            "freeze the exact Dialga prefix-r4 project-formal row and update the "
            "manuscript; do not mechanically scale data or rounds"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_dfc1_formal_topology_not_supported"
        next_action = (
            "inspect restored-best histories and cache equivalence; stop "
            "mechanical scale-up and inspect the failed seed/repeat"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "formal_scale": "project_formal_supported" if status == "pass" else "no",
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
            "correct_auc": CORRECT_AUC_FLOOR,
            "correct_minus_corrupted": TOPOLOGY_MARGIN,
            "correct_minus_autond": AUTOND_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed remote 1000000/class Dialga prefix-r4 project-formal "
            "evaluation with five fresh third-key final tests per model/seed; "
            "not full-round, key-recovery, SOTA, paper reproduction, or "
            "universal-SPN evidence"
        ),
        "blocked_actions": [
            "calling a pass full-round, key-recovery, SOTA, or paper reproduction",
            "averaging away a failed seed or final-test repeat",
            "mechanically increasing data after DFC1",
            "mechanically scaling Dialga r5",
        ],
    }


__all__ = [
    "ARCHITECTURES",
    "EXPECTED_CACHE_CREATIONS",
    "EXPECTED_CACHE_REUSES",
    "EXPECTED_FINAL_TEST_REPEATS",
    "EXPECTED_FINAL_TEST_ROWS",
    "EXPECTED_PARAMETER_COUNTS",
    "EXPECTED_RESULT_ROWS",
    "RUN_ID",
    "adjudicate",
    "candidate_protocol_frozen",
    "read_tasks",
]
