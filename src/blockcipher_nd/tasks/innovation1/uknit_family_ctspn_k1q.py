from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
    VARIANCE_FLOOR,
    deterministic_label_shuffle,
    extract_k1o_feature_views,
    fit_diagonal_fisher,
    numpy_array_sha256,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = (
    "i1_uknit_family_ctspn_difference_position_discovery_"
    "k1q_seed2_confirm_seed3_seed4_20260728"
)
DISCOVERY_PHASE = "discovery"
CONFIRMATION_PHASE = "confirmation"
DISCOVERY_SEED = 2
CONFIRMATION_SEEDS = (3, 4)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
DISCOVERY_VIEWS = (CANDIDATE_VIEW, RAW_VIEW)
CONFIRMATION_VIEWS = (CANDIDATE_VIEW, RAW_VIEW, LABEL_SHUFFLE_VIEW)
EXPECTED_CELLS = tuple(range(16))
ACTIVE_BIT_ROLE = 1
ANCHOR_CELL = 1
ANCHOR_DIFFERENCE = 0x40
DISCOVERY_SAMPLES_PER_CLASS = 1024
DISCOVERY_HOLDOUT_PER_CLASS = 512
CONFIRMATION_SAMPLES_PER_CLASS = 2048
CONFIRMATION_HOLDOUT_PER_CLASS = 1024
EXPECTED_PAIRS = 4
AUC_FLOOR = 0.550
RAW_MARGIN = 0.010
LABEL_SHUFFLE_MARGIN = 0.030
MAX_SELECTED_CANDIDATES = 2
DISCOVERY_TRAIN_KEY = int("2" * 32, 16)
DISCOVERY_VALIDATION_KEY = int("3" * 32, 16)
CONFIRMATION_KEYS = {
    3: (int("4" * 32, 16), int("5" * 32, 16)),
    4: (int("6" * 32, 16), int("7" * 32, 16)),
}


def candidate_bit_index(cell: int) -> int:
    if cell not in EXPECTED_CELLS:
        raise ValueError("K1-Q native cell must be in [0, 15]")
    return 4 * cell + 2


def candidate_difference(cell: int) -> int:
    return 1 << candidate_bit_index(cell)


def bind_discovery_input_differences(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    for source in tasks:
        task = dict(source)
        options = dict(task.get("model_options", {}))
        cell = int(options.get("active_cell", -1))
        role = int(options.get("active_bit_role", -1))
        declared_hex = str(options.get("input_difference_hex", ""))
        expected = candidate_difference(cell)
        if role != ACTIVE_BIT_ROLE:
            raise ValueError("K1-Q discovery must keep active_bit_role=1")
        if int(declared_hex, 0) != expected:
            raise ValueError("K1-Q declared input difference does not match cell role")
        task.update(
            {
                "input_difference": expected,
                "difference_profile": "",
                "difference_member": "",
                "difference_source": "K1-Q preregistered native-cell role-1 scan",
            }
        )
        bound.append(task)
    return bound


def validate_discovery_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    cells = [int(task.get("model_options", {}).get("active_cell", -1)) for task in tasks]
    return {
        "sixteen_discovery_tasks_complete": (
            len(tasks) == len(EXPECTED_CELLS) and sorted(cells) == list(EXPECTED_CELLS)
        ),
        "uknit_r5_seed2_only": all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
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
            int(task.get("samples_per_class", -1))
            == DISCOVERY_SAMPLES_PER_CLASS
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
            int(task.get("model_options", {}).get("runtime_round_start", -1)) == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            for task in tasks
        ),
    }


def build_confirmation_tasks(
    discovery_tasks: Sequence[Mapping[str, Any]],
    selected_cells: Sequence[int],
) -> list[dict[str, Any]]:
    selected = tuple(int(cell) for cell in selected_cells)
    if len(selected) > MAX_SELECTED_CANDIDATES:
        raise ValueError("K1-Q may confirm at most two discovered candidates")
    if ANCHOR_CELL in selected or len(set(selected)) != len(selected):
        raise ValueError("K1-Q selected cells must be unique non-anchor candidates")
    task_by_cell = {
        int(task.get("model_options", {}).get("active_cell", -1)): task
        for task in discovery_tasks
    }
    if set(task_by_cell) != set(EXPECTED_CELLS):
        raise ValueError("K1-Q confirmation requires all discovery task templates")
    if not selected:
        return []

    tasks: list[dict[str, Any]] = []
    for cell in (ANCHOR_CELL, *selected):
        for seed in CONFIRMATION_SEEDS:
            train_key, validation_key = CONFIRMATION_KEYS[seed]
            task = dict(task_by_cell[cell])
            options = dict(task["model_options"])
            task.update(
                {
                    "architecture": f"uKNIT64-CTSPN-K1Q-Confirm-Cell{cell:02d}-Seed{seed}",
                    "seed": seed,
                    "samples_per_class": CONFIRMATION_SAMPLES_PER_CLASS,
                    "validation_samples_total": None,
                    "train_key": train_key,
                    "validation_key": validation_key,
                    "model_options": options,
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
            int(task.get("samples_per_class", -1))
            == CONFIRMATION_SAMPLES_PER_CLASS
            and task.get("validation_samples_total") is None
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            for task in tasks
        ),
        "confirmation_keys_exact": all(
            (
                int(task.get("train_key", -1)),
                int(task.get("validation_key", -1)),
            )
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
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
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
    views = DISCOVERY_VIEWS if phase == DISCOVERY_PHASE else CONFIRMATION_VIEWS
    if phase not in {DISCOVERY_PHASE, CONFIRMATION_PHASE}:
        raise ValueError("K1-Q phase must be discovery or confirmation")
    if set(datasets) != set(EXPECTED_SPLITS):
        raise ValueError("K1-Q requires train, same-key fresh, and cross-key splits")

    input_difference = candidate_difference(cell)
    split_views: dict[str, dict[str, np.ndarray]] = {}
    feature_rows: list[dict[str, Any]] = []
    for split in EXPECTED_SPLITS:
        dataset = datasets[split]
        extracted, manifests = extract_k1o_feature_views(
            dataset,
            exact_structure=exact_structure,
            wrong_sbox_structure=wrong_sbox_structure,
            batch_size=batch_size,
        )
        split_views[split] = extracted
        dataset_sha = differential_dataset_sha256(dataset)
        for view in views:
            feature_rows.append(
                {
                    "run_id": RUN_ID,
                    "phase": phase,
                    "cipher_key": "uknit64",
                    "rounds": 5,
                    "cell": cell,
                    "bit_index": candidate_bit_index(cell),
                    "active_bit_role": ACTIVE_BIT_ROLE,
                    "input_difference": input_difference,
                    "input_difference_hex": f"0x{input_difference:016x}",
                    "seed": seed,
                    "split": split,
                    "view": view,
                    "rows": int(extracted[view].shape[0]),
                    "feature_dim": int(extracted[view].shape[1]),
                    "dataset_sha256": dataset_sha,
                    **manifests[view],
                }
            )

    train = datasets["train_seen"]
    train_labels = np.asarray(train.labels, dtype=np.uint8)
    shuffled_labels, permutation_sha = deterministic_label_shuffle(
        train_labels,
        seed=20260728 + seed * 100 + cell,
    )
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for view in views:
        fit_labels = shuffled_labels if view == LABEL_SHUFFLE_VIEW else train_labels
        scorer = fit_diagonal_fisher(split_views["train_seen"][view], fit_labels)
        scorer_rows.append(
            {
                "run_id": RUN_ID,
                "phase": phase,
                "cipher_key": "uknit64",
                "rounds": 5,
                "cell": cell,
                "bit_index": candidate_bit_index(cell),
                "active_bit_role": ACTIVE_BIT_ROLE,
                "input_difference": input_difference,
                "input_difference_hex": f"0x{input_difference:016x}",
                "seed": seed,
                "view": view,
                "fit_split": "train_seen",
                "fit_rows": int(len(train_labels)),
                "feature_dim": int(scorer.weights.shape[0]),
                "variance_floor": scorer.variance_floor,
                "class0_rows": scorer.class_counts[0],
                "class1_rows": scorer.class_counts[1],
                "weight_l2_norm": float(np.linalg.norm(scorer.weights)),
                "nonzero_weight_count": int(np.count_nonzero(scorer.weights)),
                "scorer_sha256": scorer.sha256,
                "label_permutation_sha256": (
                    permutation_sha if view == LABEL_SHUFFLE_VIEW else None
                ),
                "training_performed": False,
                "neural_parameter_count": 0,
                "optimizer_steps": 0,
                "epochs": 0,
            }
        )
        for split in EXPECTED_SPLITS:
            dataset = datasets[split]
            labels = np.asarray(dataset.labels, dtype=np.uint8)
            feature_values = split_views[split][view]
            scores = scorer.score(feature_values)
            result_rows.append(
                {
                    "run_id": RUN_ID,
                    "phase": phase,
                    "cipher_key": "uknit64",
                    "rounds": 5,
                    "cell": cell,
                    "bit_index": candidate_bit_index(cell),
                    "active_bit_role": ACTIVE_BIT_ROLE,
                    "input_difference": input_difference,
                    "input_difference_hex": f"0x{input_difference:016x}",
                    "seed": seed,
                    "split": split,
                    "view": view,
                    "rows": int(len(labels)),
                    "auc": binary_auc(labels, scores),
                    "zero_threshold_accuracy": float(
                        ((scores >= 0.0).astype(np.uint8) == labels).mean()
                    ),
                    "score_mean": float(scores.mean()),
                    "score_std": float(scores.std()),
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "feature_dim": int(feature_values.shape[1]),
                    "feature_sha256": numpy_array_sha256(feature_values),
                    "dataset_sha256": differential_dataset_sha256(dataset),
                    "scorer_sha256": scorer.sha256,
                    "fit_split": "train_seen",
                    "fit_rows": int(len(train_labels)),
                    "pairs_per_sample": EXPECTED_PAIRS,
                    "negative_mode": "encrypted_random_plaintexts",
                    "variance_floor": VARIANCE_FLOOR,
                    "training_performed": False,
                    "neural_parameter_count": 0,
                    "optimizer_steps": 0,
                    "epochs": 0,
                }
            )
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
        eligible = (
            cell != ANCHOR_CELL
            and minimum_exact >= AUC_FLOOR
            and minimum_margin >= RAW_MARGIN
        )
        rankings.append(
            {
                "cell": cell,
                "bit_index": candidate_bit_index(cell),
                "input_difference": candidate_difference(cell),
                "input_difference_hex": f"0x{candidate_difference(cell):016x}",
                "is_anchor": cell == ANCHOR_CELL,
                "minimum_fresh_exact_auc": minimum_exact,
                "minimum_fresh_exact_minus_raw": minimum_margin,
                "eligible": eligible,
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
    selected = [
        int(row["cell"])
        for row in ranked
        if row["eligible"]
    ][:MAX_SELECTED_CANDIDATES]
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


def adjudicate_k1q(
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
    task_checks = validate_discovery_tasks(discovery_tasks)
    protocol_checks = {
        **dict(source_checks),
        **task_checks,
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
            and row.get("cipher_key") == "uknit64"
            and int(row.get("rounds", -1)) == 5
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

    discovery_summary = {
        str(cell): {
            split: _split_summary(
                results, DISCOVERY_PHASE, cell, DISCOVERY_SEED, split
            )
            for split in FRESH_SPLITS
        }
        for cell in EXPECTED_CELLS
    } if _discovery_result_keys().issubset(results) else {}

    confirmation_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    research_checks: dict[str, bool] = {}
    confirmed_cells: list[int] = []
    if set(results) == expected_result_keys:
        for cell in confirmation_cells:
            confirmation_summary[str(cell)] = {}
            cell_passes = []
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
        decision = "innovation1_uknit_family_ctspn_k1q_protocol_invalid"
        next_action = (
            "repair only the failed K1-Q plan, cache, split, feature, scorer, "
            "selection, or artifact invariant and rerun the frozen audit"
        )
    elif not selected:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1q_no_r5_difference_position_discovered"
        )
        next_action = (
            "stop mechanical position scans and preregister a DDT/trail-guided "
            "input-difference ranking audit; keep trail data outside the network"
        )
    elif confirmed_cells:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1q_confirmed_r5_difference_position_supported"
        )
        next_action = (
            "run one same-budget K1-N-derived neural attribution matrix on the "
            "confirmed difference with exact, wrong-S-box, no-S-box, and no-topology rows"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1q_discovery_not_confirmed"
        )
        next_action = (
            "treat discovery as selection noise or key/seed instability, stop "
            "position scanning, and preregister DDT/trail-guided difference ranking"
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
            "local zero-neural-training uKNIT r5 role-1 difference-position discovery "
            "and untouched-seed confirmation; not formal scale, attack, SOTA, transfer, "
            "neural distinguisher, or uKNIT ceiling"
        ),
        "blocked_actions": [
            "remote scale or more positions, bit roles, pairs, samples, seeds, or keys",
            "neural training, MoE, DDT/trail inputs, or post-result threshold changes",
            "advancing a discovery-only candidate without seed3/4 confirmation",
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


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str, str], Mapping[str, Any]]:
    return {
        (
            str(row.get("phase")),
            int(row.get("cell", -1)),
            int(row.get("seed", -1)),
            str(row.get("split")),
            str(row.get("view")),
        ): row
        for row in rows
    }


def _feature_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str, str], Mapping[str, Any]]:
    return _result_map(rows)


def _scorer_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], Mapping[str, Any]]:
    return {
        (
            str(row.get("phase")),
            int(row.get("cell", -1)),
            int(row.get("seed", -1)),
            str(row.get("view")),
        ): row
        for row in rows
    }


def _split_summary(
    results: Mapping[tuple[str, int, int, str, str], Mapping[str, Any]],
    phase: str,
    cell: int,
    seed: int,
    split: str,
) -> dict[str, float]:
    exact = float(results[(phase, cell, seed, split, CANDIDATE_VIEW)]["auc"])
    raw = float(results[(phase, cell, seed, split, RAW_VIEW)]["auc"])
    summary = {
        "exact_auc": exact,
        "raw_auc": raw,
        "exact_minus_raw": exact - raw,
    }
    label_key = (phase, cell, seed, split, LABEL_SHUFFLE_VIEW)
    if label_key in results:
        label = float(results[label_key]["auc"])
        summary.update(
            {
                "label_shuffle_auc": label,
                "exact_minus_label_shuffle": exact - label,
            }
        )
    return summary


def _row_count_exact(row: Mapping[str, Any]) -> bool:
    phase = str(row.get("phase"))
    split = str(row.get("split"))
    if phase == DISCOVERY_PHASE:
        expected = (
            DISCOVERY_SAMPLES_PER_CLASS * 2
            if split == "train_seen"
            else DISCOVERY_HOLDOUT_PER_CLASS * 2
        )
    else:
        expected = (
            CONFIRMATION_SAMPLES_PER_CLASS * 2
            if split == "train_seen"
            else CONFIRMATION_HOLDOUT_PER_CLASS * 2
        )
    return int(row.get("rows", -1)) == expected


__all__ = [
    "ANCHOR_CELL",
    "CONFIRMATION_PHASE",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_VIEWS",
    "DISCOVERY_PHASE",
    "DISCOVERY_SEED",
    "DISCOVERY_VIEWS",
    "EXPECTED_CELLS",
    "FRESH_SPLITS",
    "RUN_ID",
    "adjudicate_k1q",
    "bind_discovery_input_differences",
    "build_confirmation_tasks",
    "candidate_bit_index",
    "candidate_difference",
    "evaluate_position",
    "select_discovery_candidates",
    "validate_confirmation_tasks",
    "validate_discovery_tasks",
]
