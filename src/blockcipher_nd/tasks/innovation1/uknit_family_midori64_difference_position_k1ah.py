from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    VARIANCE_FLOOR,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    CONFIRMATION_PHASE,
    CONFIRMATION_VIEWS,
    DISCOVERY_PHASE,
    DISCOVERY_VIEWS,
    EXPECTED_CELLS,
    EXPECTED_SPLITS,
    FRESH_SPLITS,
    _feature_map,
    _result_map,
    _scorer_map,
    _split_summary,
    candidate_difference,
    evaluate_position as evaluate_k1q_position,
)


RUN_ID = "i1_uknit_family_midori64_difference_position_k1ah_20260729"
CIPHER_KEY = "midori64"
ROUNDS = 4
DISCOVERY_SEED = 5
CONFIRMATION_SEEDS = (6, 7)
ACTIVE_BIT_ROLE = 1
ANCHOR_CELL = 1
ANCHOR_DIFFERENCE = 0x40
DISCOVERY_SAMPLES_PER_CLASS = 1024
DISCOVERY_HOLDOUT_PER_CLASS = 512
CONFIRMATION_SAMPLES_PER_CLASS = 2048
CONFIRMATION_HOLDOUT_PER_CLASS = 1024
EXPECTED_PAIRS = 4
RUNTIME_ROUND_START = 0
RUNTIME_ROUNDS = 2
CIPHER_ROUND_WINDOW_START = 2
AUC_FLOOR = 0.550
RAW_MARGIN = 0.010
LABEL_SHUFFLE_MARGIN = 0.030
MAX_SELECTED_CANDIDATES = 2
DISCOVERY_TRAIN_KEY = int("8" * 32, 16)
DISCOVERY_VALIDATION_KEY = int("9" * 32, 16)
CONFIRMATION_KEYS = {
    6: (int("a" * 32, 16), int("b" * 32, 16)),
    7: (int("c" * 32, 16), int("d" * 32, 16)),
}


def bind_discovery_input_differences(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    for source in tasks:
        task = dict(source)
        options = dict(task.get("model_options", {}))
        cell = int(options.get("active_cell", -1))
        role = int(options.get("active_bit_role", -1))
        expected = candidate_difference(cell)
        if role != ACTIVE_BIT_ROLE:
            raise ValueError("K1-AH discovery must keep active_bit_role=1")
        if int(str(options.get("input_difference_hex", "")), 0) != expected:
            raise ValueError("K1-AH declared input difference does not match cell role")
        task.update(
            {
                "input_difference": expected,
                "difference_profile": "",
                "difference_member": "",
                "difference_source": (
                    "K1-AH preregistered Midori64 native-cell role-1 scan"
                ),
            }
        )
        bound.append(task)
    return bound


def validate_discovery_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    cells = [
        int(task.get("model_options", {}).get("active_cell", -1)) for task in tasks
    ]
    return {
        "sixteen_discovery_tasks_complete": (
            len(tasks) == len(EXPECTED_CELLS) and sorted(cells) == list(EXPECTED_CELLS)
        ),
        "midori64_r4_seed5_only": all(
            task.get("cipher_key") == CIPHER_KEY
            and int(task.get("rounds", -1)) == ROUNDS
            and int(task.get("seed", -1)) == DISCOVERY_SEED
            for task in tasks
        ),
        "candidate_geometry_exact": all(
            int(task.get("input_difference", -1)) == candidate_difference(cell)
            and int(task.get("model_options", {}).get("active_bit_role", -1))
            == ACTIVE_BIT_ROLE
            for task, cell in zip(tasks, cells, strict=True)
        ),
        "discovery_budget_exact": all(
            int(task.get("samples_per_class", -1)) == DISCOVERY_SAMPLES_PER_CLASS
            and task.get("validation_samples_total") is None
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            for task in tasks
        ),
        "strict_data_protocol_exact": all(
            task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == DISCOVERY_TRAIN_KEY
            and int(task.get("validation_key", -1)) == DISCOVERY_VALIDATION_KEY
            for task in tasks
        ),
        "runtime_window_exact": all(
            task.get("model_options", {}).get("runtime_structure_path")
            == "configs/runtime/spn/midori64.json"
            and int(task.get("model_options", {}).get("runtime_round_start", -1))
            == RUNTIME_ROUND_START
            and int(task.get("model_options", {}).get("runtime_rounds", -1))
            == RUNTIME_ROUNDS
            and int(task.get("model_options", {}).get("cipher_round_window_start", -1))
            == CIPHER_ROUND_WINDOW_START
            for task in tasks
        ),
    }


def build_confirmation_tasks(
    discovery_tasks: Sequence[Mapping[str, Any]],
    selected_cells: Sequence[int],
) -> list[dict[str, Any]]:
    selected = tuple(int(cell) for cell in selected_cells)
    if len(selected) > MAX_SELECTED_CANDIDATES:
        raise ValueError("K1-AH may confirm at most two discovered candidates")
    if ANCHOR_CELL in selected or len(set(selected)) != len(selected):
        raise ValueError("K1-AH selected cells must be unique non-anchor candidates")
    task_by_cell = {
        int(task.get("model_options", {}).get("active_cell", -1)): task
        for task in discovery_tasks
    }
    if set(task_by_cell) != set(EXPECTED_CELLS):
        raise ValueError("K1-AH confirmation requires all discovery templates")
    if not selected:
        return []

    tasks: list[dict[str, Any]] = []
    for cell in (ANCHOR_CELL, *selected):
        for seed in CONFIRMATION_SEEDS:
            train_key, validation_key = CONFIRMATION_KEYS[seed]
            task = dict(task_by_cell[cell])
            task.update(
                {
                    "architecture": (
                        f"Midori64-CTSPN-K1AH-Confirm-Cell{cell:02d}-Seed{seed}"
                    ),
                    "seed": seed,
                    "samples_per_class": CONFIRMATION_SAMPLES_PER_CLASS,
                    "validation_samples_total": None,
                    "train_key": train_key,
                    "validation_key": validation_key,
                    "model_options": dict(task["model_options"]),
                }
            )
            tasks.append(task)
    return tasks


def validate_confirmation_tasks(
    tasks: Sequence[Mapping[str, Any]],
    selected_cells: Sequence[int],
) -> dict[str, bool]:
    selected = tuple(int(cell) for cell in selected_cells)
    expected_cells = (ANCHOR_CELL, *selected) if selected else ()
    expected_keys = {
        (cell, seed) for cell in expected_cells for seed in CONFIRMATION_SEEDS
    }
    observed_keys = {
        (
            int(task.get("model_options", {}).get("active_cell", -1)),
            int(task.get("seed", -1)),
        )
        for task in tasks
    }
    return {
        "confirmation_task_matrix_exact": (
            len(tasks) == len(expected_keys) and observed_keys == expected_keys
        ),
        "confirmation_budget_exact": all(
            int(task.get("samples_per_class", -1)) == CONFIRMATION_SAMPLES_PER_CLASS
            and task.get("validation_samples_total") is None
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            for task in tasks
        ),
        "confirmation_keys_exact": all(
            (int(task.get("train_key", -1)), int(task.get("validation_key", -1)))
            == CONFIRMATION_KEYS.get(int(task.get("seed", -1)))
            for task in tasks
        ),
        "confirmation_geometry_exact": all(
            int(task.get("input_difference", -1))
            == candidate_difference(
                int(task.get("model_options", {}).get("active_cell", -1))
            )
            and int(task.get("model_options", {}).get("active_bit_role", -1))
            == ACTIVE_BIT_ROLE
            for task in tasks
        ),
        "confirmation_data_protocol_exact": all(
            task.get("cipher_key") == CIPHER_KEY
            and int(task.get("rounds", -1)) == ROUNDS
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            for task in tasks
        ),
    }


def evaluate_position(
    *,
    phase: str,
    cell: int,
    seed: int,
    datasets: Mapping[str, DifferentialDataset],
    exact_structure: RuntimeSpnStructure,
    wrong_sbox_structure: RuntimeSpnStructure,
    batch_size: int = 256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows, scorer_rows, result_rows = evaluate_k1q_position(
        phase=phase,
        cell=cell,
        seed=seed,
        datasets=datasets,
        exact_structure=exact_structure,
        wrong_sbox_structure=wrong_sbox_structure,
        batch_size=batch_size,
    )
    for row in (*feature_rows, *scorer_rows, *result_rows):
        row.update({"run_id": RUN_ID, "cipher_key": CIPHER_KEY, "rounds": ROUNDS})
    return feature_rows, scorer_rows, result_rows


def select_discovery_candidates(
    result_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _result_map(result_rows)
    rankings: list[dict[str, Any]] = []
    for cell in EXPECTED_CELLS:
        summaries = [
            _split_summary(results, DISCOVERY_PHASE, cell, DISCOVERY_SEED, split)
            for split in FRESH_SPLITS
        ]
        minimum_exact = min(row["exact_auc"] for row in summaries)
        minimum_margin = min(row["exact_minus_raw"] for row in summaries)
        rankings.append(
            {
                "cell": cell,
                "bit_index": 4 * cell + 2,
                "input_difference": candidate_difference(cell),
                "input_difference_hex": f"0x{candidate_difference(cell):016x}",
                "is_anchor": cell == ANCHOR_CELL,
                "minimum_fresh_exact_auc": minimum_exact,
                "minimum_fresh_exact_minus_raw": minimum_margin,
                "eligible": (
                    cell != ANCHOR_CELL
                    and minimum_exact >= AUC_FLOOR
                    and minimum_margin >= RAW_MARGIN
                ),
                "fresh_splits": {
                    split: summary
                    for split, summary in zip(FRESH_SPLITS, summaries, strict=True)
                },
            }
        )
    ranked = sorted(
        rankings,
        key=lambda row: (
            -float(row["minimum_fresh_exact_auc"]),
            -float(row["minimum_fresh_exact_minus_raw"]),
            int(row["cell"]),
        ),
    )
    selected = [int(row["cell"]) for row in ranked if row["eligible"]][
        :MAX_SELECTED_CANDIDATES
    ]
    return {
        "run_id": RUN_ID,
        "status": "selected" if selected else "no_candidate",
        "anchor_cell": ANCHOR_CELL,
        "anchor_input_difference": ANCHOR_DIFFERENCE,
        "selected_cells": selected,
        "selected_input_differences": [candidate_difference(cell) for cell in selected],
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
        },
        "ranking": ranked,
    }


def adjudicate_k1ah(
    *,
    discovery_tasks: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    selected = tuple(int(cell) for cell in selection.get("selected_cells", ()))
    confirmation_cells = (ANCHOR_CELL, *selected) if selected else ()
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    scorers = _scorer_map(scorer_rows)
    expected_result_keys = _expected_result_keys(confirmation_cells)
    expected_scorer_keys = _expected_scorer_keys(confirmation_cells)
    expected_dataset_keys = _expected_dataset_keys(confirmation_cells)
    observed_dataset_keys = {
        (
            str(row.get("phase")),
            int(row.get("cell", -1)),
            int(row.get("seed", -1)),
            str(row.get("split")),
        )
        for row in dataset_rows
    }
    recomputed_selection = (
        select_discovery_candidates(result_rows)
        if _discovery_result_keys().issubset(results)
        else None
    )
    protocol_checks = {
        **dict(source_checks),
        **validate_discovery_tasks(discovery_tasks),
        "selection_recomputed_exactly": (
            recomputed_selection is not None
            and selection.get("selected_cells")
            == recomputed_selection.get("selected_cells")
            and selection.get("ranking") == recomputed_selection.get("ranking")
        ),
        "result_rows_complete": (
            len(result_rows) == len(expected_result_keys)
            and set(results) == expected_result_keys
        ),
        "feature_rows_complete": (
            len(feature_rows) == len(expected_result_keys)
            and set(features) == expected_result_keys
        ),
        "scorer_rows_complete": (
            len(scorer_rows) == len(expected_scorer_keys)
            and set(scorers) == expected_scorer_keys
        ),
        "dataset_rows_complete": (
            len(dataset_rows) == len(expected_dataset_keys)
            and observed_dataset_keys == expected_dataset_keys
        ),
        "run_geometry_exact": all(
            row.get("run_id") == RUN_ID
            and row.get("cipher_key") == CIPHER_KEY
            and int(row.get("rounds", -1)) == ROUNDS
            and int(row.get("active_bit_role", -1)) == ACTIVE_BIT_ROLE
            and int(row.get("input_difference", -1))
            == candidate_difference(int(row.get("cell", -1)))
            for row in (*result_rows, *feature_rows, *scorer_rows)
        ),
        "row_counts_exact": all(_row_count_exact(row) for row in result_rows),
        "feature_dimensions_exact": all(
            int(row.get("feature_dim", -1))
            == EXPECTED_FEATURE_DIMS.get(str(row.get("view")), -2)
            for row in (*result_rows, *feature_rows, *scorer_rows)
        ),
        "histograms_normalized_and_finite": all(
            row.get("finite") is True
            and row.get("nonnegative") is True
            and row.get("normalized") is True
            for row in feature_rows
        ),
        "same_dataset_per_position_seed_split": all(
            len(
                {
                    features[key].get("dataset_sha256")
                    for key in expected_result_keys
                    if key[:4] == prefix
                }
            )
            == 1
            for prefix in expected_dataset_keys
        )
        if set(features) == expected_result_keys
        else False,
        "exact_and_label_shuffle_features_identical": all(
            features[(CONFIRMATION_PHASE, cell, seed, split, CANDIDATE_VIEW)].get(
                "feature_sha256"
            )
            == features[
                (CONFIRMATION_PHASE, cell, seed, split, LABEL_SHUFFLE_VIEW)
            ].get("feature_sha256")
            for cell in confirmation_cells
            for seed in CONFIRMATION_SEEDS
            for split in EXPECTED_SPLITS
        )
        if set(features) == expected_result_keys and confirmation_cells
        else not confirmation_cells,
        "closed_form_only_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("neural_parameter_count", -1)) == 0
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            and float(row.get("variance_floor", math.nan)) == VARIANCE_FLOOR
            for row in (*result_rows, *scorer_rows)
        ),
        "all_metrics_finite": all(
            all(
                math.isfinite(float(row.get(name, math.nan)))
                for name in (
                    "auc",
                    "zero_threshold_accuracy",
                    "score_mean",
                    "score_std",
                    "score_min",
                    "score_max",
                )
            )
            for row in result_rows
        ),
        "cache_payloads_present": all(
            row.get("cache_payloads_present") is True for row in dataset_rows
        ),
    }
    discovery_summary = (
        {
            str(cell): {
                split: _split_summary(
                    results, DISCOVERY_PHASE, cell, DISCOVERY_SEED, split
                )
                for split in FRESH_SPLITS
            }
            for cell in EXPECTED_CELLS
        }
        if _discovery_result_keys().issubset(results)
        else {}
    )

    confirmation_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    research_checks: dict[str, bool] = {}
    confirmed_cells: list[int] = []
    if set(results) == expected_result_keys:
        for cell in confirmation_cells:
            confirmation_summary[str(cell)] = {}
            cell_passes: list[bool] = []
            for seed in CONFIRMATION_SEEDS:
                confirmation_summary[str(cell)][str(seed)] = {}
                for split in FRESH_SPLITS:
                    summary = _split_summary(
                        results, CONFIRMATION_PHASE, cell, seed, split
                    )
                    confirmation_summary[str(cell)][str(seed)][split] = summary
                    if cell != ANCHOR_CELL:
                        prefix = f"cell{cell}_seed{seed}_{split}"
                        checks = {
                            f"{prefix}_exact_auc_floor": (
                                summary["exact_auc"] >= AUC_FLOOR
                            ),
                            f"{prefix}_beats_raw": (
                                summary["exact_minus_raw"] >= RAW_MARGIN
                            ),
                            f"{prefix}_beats_label_shuffle": (
                                summary["exact_minus_label_shuffle"]
                                >= LABEL_SHUFFLE_MARGIN
                            ),
                        }
                        research_checks.update(checks)
                        cell_passes.extend(checks.values())
            if cell != ANCHOR_CELL and cell_passes and all(cell_passes):
                confirmed_cells.append(cell)

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1ah_protocol_invalid"
        next_action = (
            "repair only the failed K1-AH plan, cache, split, feature, scorer, "
            "selection, source qualification, or artifact invariant and rerun unchanged"
        )
    elif not selected:
        status = "hold"
        decision = "innovation1_uknit_family_midori64_k1ah_no_r4_position_discovered"
        next_action = (
            "stop mechanical r4 cell scans and preregister a lower-round boundary "
            "or trail-guided difference-value audit; keep trail data outside the network"
        )
    elif confirmed_cells:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1ah_confirmed_r4_position_supported"
        )
        next_action = (
            "run one local same-budget K1-AI neural attribution matrix on the strongest "
            "confirmed difference with correct, wrong-S-box, wrong-linear, and no-structure rows"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_midori64_k1ah_discovery_not_confirmed"
        next_action = (
            "treat discovery as seed/key-specific selection noise, stop position scanning, "
            "and preregister a lower-round boundary or trail-guided difference-value audit"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
            "exact_minus_label_shuffle": LABEL_SHUFFLE_MARGIN,
        },
        "selection": dict(selection),
        "confirmed_cells": confirmed_cells,
        "confirmed_input_differences": [
            candidate_difference(cell) for cell in confirmed_cells
        ],
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "discovery_summary": discovery_summary,
        "confirmation_summary": confirmation_summary,
        "next_action": next_action,
        "claim_scope": (
            "local zero-neural-training Midori64 r4 role-1 difference-position "
            "discovery and untouched-seed/key confirmation; not formal scale, attack, "
            "SOTA, transfer, neural distinguisher, high-round result, or Midori64 ceiling"
        ),
        "blocked_actions": [
            "remote scale or more positions, bit roles, pairs, samples, seeds, or keys",
            "neural training, MoE, trail inputs, or post-result threshold changes",
            "advancing a discovery-only candidate without seed6/7 confirmation",
        ],
    }


def _expected_result_keys(
    confirmation_cells: Sequence[int],
) -> set[tuple[str, int, int, str, str]]:
    keys = _discovery_result_keys()
    keys.update(
        (CONFIRMATION_PHASE, cell, seed, split, view)
        for cell in confirmation_cells
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _discovery_result_keys() -> set[tuple[str, int, int, str, str]]:
    return {
        (DISCOVERY_PHASE, cell, DISCOVERY_SEED, split, view)
        for cell in EXPECTED_CELLS
        for split in EXPECTED_SPLITS
        for view in DISCOVERY_VIEWS
    }


def _expected_scorer_keys(
    confirmation_cells: Sequence[int],
) -> set[tuple[str, int, int, str]]:
    keys = {
        (DISCOVERY_PHASE, cell, DISCOVERY_SEED, view)
        for cell in EXPECTED_CELLS
        for view in DISCOVERY_VIEWS
    }
    keys.update(
        (CONFIRMATION_PHASE, cell, seed, view)
        for cell in confirmation_cells
        for seed in CONFIRMATION_SEEDS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _expected_dataset_keys(
    confirmation_cells: Sequence[int],
) -> set[tuple[str, int, int, str]]:
    keys = {
        (DISCOVERY_PHASE, cell, DISCOVERY_SEED, split)
        for cell in EXPECTED_CELLS
        for split in EXPECTED_SPLITS
    }
    keys.update(
        (CONFIRMATION_PHASE, cell, seed, split)
        for cell in confirmation_cells
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
    )
    return keys


def _row_count_exact(row: Mapping[str, Any]) -> bool:
    if str(row.get("phase")) == DISCOVERY_PHASE:
        expected = (
            DISCOVERY_SAMPLES_PER_CLASS * 2
            if row.get("split") == "train_seen"
            else DISCOVERY_HOLDOUT_PER_CLASS * 2
        )
    else:
        expected = (
            CONFIRMATION_SAMPLES_PER_CLASS * 2
            if row.get("split") == "train_seen"
            else CONFIRMATION_HOLDOUT_PER_CLASS * 2
        )
    return int(row.get("rows", -1)) == expected


__all__ = [
    "ANCHOR_CELL",
    "CONFIRMATION_SEEDS",
    "DISCOVERY_SEED",
    "RUN_ID",
    "adjudicate_k1ah",
    "bind_discovery_input_differences",
    "build_confirmation_tasks",
    "evaluate_position",
    "select_discovery_candidates",
    "validate_confirmation_tasks",
    "validate_discovery_tasks",
]
