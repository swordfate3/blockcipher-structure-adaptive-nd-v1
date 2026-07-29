from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
    VARIANCE_FLOOR,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    CONFIRMATION_PHASE,
    DISCOVERY_PHASE,
    EXPECTED_SPLITS,
    FRESH_SPLITS,
    candidate_bit_index,
    candidate_difference,
)


RUN_ID = "i1_uknit_ctspn_r6_role1_position_k1bl_seed2_confirm_seed3_seed4_20260729"
ROUNDS = 6
RUNTIME_ROUND_START = 4
RUNTIME_ROUNDS = 2
DISCOVERY_SEED = 2
CONFIRMATION_SEEDS = (3, 4)
EXPECTED_CELLS = tuple(range(16))
ACTIVE_BIT_ROLE = 1
ANCHOR_CELL = 11
ANCHOR_DIFFERENCE = 0x0000400000000000
DISCOVERY_SAMPLES_PER_CLASS = 1024
DISCOVERY_HOLDOUT_PER_CLASS = 512
CONFIRMATION_SAMPLES_PER_CLASS = 2048
CONFIRMATION_HOLDOUT_PER_CLASS = 1024
EXPECTED_PAIRS = 4
AUC_FLOOR = 0.550
RAW_MARGIN = 0.010
LABEL_SHUFFLE_MARGIN = 0.030
MAX_SELECTED_CANDIDATES = 2
DISCOVERY_VIEWS = (CANDIDATE_VIEW, RAW_VIEW)
CONFIRMATION_VIEWS = (CANDIDATE_VIEW, RAW_VIEW, LABEL_SHUFFLE_VIEW)
DISCOVERY_TRAIN_KEY = int("2" * 32, 16)
DISCOVERY_VALIDATION_KEY = int("3" * 32, 16)
CONFIRMATION_KEYS = {
    3: (int("4" * 32, 16), int("5" * 32, 16)),
    4: (int("6" * 32, 16), int("7" * 32, 16)),
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
            raise ValueError("K1-BL must keep active_bit_role=1")
        if int(str(options.get("input_difference_hex", "")), 0) != expected:
            raise ValueError("K1-BL input difference does not match cell role")
        task.update(
            {
                "input_difference": expected,
                "difference_profile": "",
                "difference_member": "",
                "difference_source": (
                    "K1-BL preregistered uKNIT r6 native-cell role-1 scan"
                ),
            }
        )
        bound.append(task)
    return bound


def validate_discovery_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    cells = [int(task.get("model_options", {}).get("active_cell", -1)) for task in tasks]
    return {
        "sixteen_discovery_tasks_complete": (
            len(tasks) == len(EXPECTED_CELLS) and sorted(cells) == list(EXPECTED_CELLS)
        ),
        "uknit_r6_seed2_only": all(
            task.get("cipher_key") == "uknit64"
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
            int(task.get("model_options", {}).get("runtime_round_start", -1))
            == RUNTIME_ROUND_START
            and int(task.get("model_options", {}).get("runtime_rounds", -1))
            == RUNTIME_ROUNDS
            for task in tasks
        ),
    }


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
        passes_discovery = (
            minimum_exact >= AUC_FLOOR and minimum_margin >= RAW_MARGIN
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
                "passes_discovery": passes_discovery,
                "eligible": passes_discovery and cell != ANCHOR_CELL,
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
        int(row["cell"]) for row in ranked if row["eligible"]
    ][:MAX_SELECTED_CANDIDATES]
    anchor = next(row for row in rankings if int(row["cell"]) == ANCHOR_CELL)
    return {
        "run_id": RUN_ID,
        "status": "selected" if selected else "anchor_only",
        "anchor_cell": ANCHOR_CELL,
        "anchor_input_difference": ANCHOR_DIFFERENCE,
        "anchor_passes_discovery": bool(anchor["passes_discovery"]),
        "selected_cells": selected,
        "selected_input_differences": [candidate_difference(cell) for cell in selected],
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
        },
        "ranking": ranked,
    }


def build_confirmation_tasks(
    discovery_tasks: Sequence[Mapping[str, Any]],
    selected_cells: Sequence[int],
) -> list[dict[str, Any]]:
    selected = tuple(int(cell) for cell in selected_cells)
    if len(selected) > MAX_SELECTED_CANDIDATES:
        raise ValueError("K1-BL may confirm at most two non-anchor candidates")
    if ANCHOR_CELL in selected or len(set(selected)) != len(selected):
        raise ValueError("K1-BL selected cells must be unique and non-anchor")
    task_by_cell = {
        int(task.get("model_options", {}).get("active_cell", -1)): task
        for task in discovery_tasks
    }
    if set(task_by_cell) != set(EXPECTED_CELLS):
        raise ValueError("K1-BL confirmation requires all discovery templates")

    tasks: list[dict[str, Any]] = []
    for cell in (ANCHOR_CELL, *selected):
        for seed in CONFIRMATION_SEEDS:
            train_key, validation_key = CONFIRMATION_KEYS[seed]
            task = dict(task_by_cell[cell])
            task.update(
                {
                    "architecture": f"uKNIT64-CTSPN-K1BL-R6-Cell{cell:02d}-Seed{seed}",
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
    cells = (ANCHOR_CELL, *(int(cell) for cell in selected_cells))
    expected = {(cell, seed) for cell in cells for seed in CONFIRMATION_SEEDS}
    observed = {
        (
            int(task.get("model_options", {}).get("active_cell", -1)),
            int(task.get("seed", -1)),
        )
        for task in tasks
    }
    return {
        "confirmation_task_matrix_exact": len(tasks) == len(expected)
        and observed == expected,
        "confirmation_budget_exact": all(
            int(task.get("samples_per_class", -1))
            == CONFIRMATION_SAMPLES_PER_CLASS
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
        "confirmation_protocol_exact": all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == ROUNDS
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("input_difference", -1))
            == candidate_difference(
                int(task.get("model_options", {}).get("active_cell", -1))
            )
            for task in tasks
        ),
    }


def adjudicate_k1bl(
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
    confirmation_cells = (ANCHOR_CELL, *selected)
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
    recomputed = (
        select_discovery_candidates(result_rows)
        if _discovery_result_keys().issubset(results)
        else None
    )
    protocol_checks = {
        **dict(source_checks),
        **validate_discovery_tasks(discovery_tasks),
        "selection_recomputed_exactly": recomputed is not None
        and selection.get("selected_cells") == recomputed.get("selected_cells")
        and selection.get("ranking") == recomputed.get("ranking"),
        "result_rows_complete": len(result_rows) == len(expected_result_keys)
        and set(results) == expected_result_keys,
        "feature_rows_complete": len(feature_rows) == len(expected_result_keys)
        and set(features) == expected_result_keys,
        "scorer_rows_complete": len(scorer_rows) == len(expected_scorer_keys)
        and set(scorers) == expected_scorer_keys,
        "dataset_rows_complete": len(dataset_rows) == len(expected_dataset_keys)
        and observed_dataset_keys == expected_dataset_keys,
        "run_geometry_exact": all(
            row.get("run_id") == RUN_ID
            and row.get("cipher_key") == "uknit64"
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
        if set(features) == expected_result_keys
        else False,
        "closed_form_only_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("neural_parameter_count", -1)) == 0
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            and float(row.get("variance_floor", math.nan)) == VARIANCE_FLOOR
            for row in (*result_rows, *scorer_rows)
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row.get("auc", math.nan))) for row in result_rows
        ),
        "cache_payloads_present": all(
            row.get("cache_payloads_present") is True for row in dataset_rows
        ),
    }

    confirmation_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    research_checks: dict[str, bool] = {}
    confirmed_cells: list[int] = []
    if set(results) == expected_result_keys:
        for cell in confirmation_cells:
            confirmation_summary[str(cell)] = {}
            cell_checks: list[bool] = []
            for seed in CONFIRMATION_SEEDS:
                confirmation_summary[str(cell)][str(seed)] = {}
                for split in FRESH_SPLITS:
                    summary = _split_summary(
                        results, CONFIRMATION_PHASE, cell, seed, split
                    )
                    confirmation_summary[str(cell)][str(seed)][split] = summary
                    prefix = f"cell{cell}_seed{seed}_{split}"
                    checks = {
                        f"{prefix}_exact_auc_floor": summary["exact_auc"]
                        >= AUC_FLOOR,
                        f"{prefix}_beats_raw": summary["exact_minus_raw"]
                        >= RAW_MARGIN,
                        f"{prefix}_beats_label_shuffle": summary[
                            "exact_minus_label_shuffle"
                        ]
                        >= LABEL_SHUFFLE_MARGIN,
                    }
                    research_checks.update(checks)
                    cell_checks.extend(checks.values())
            if cell_checks and all(cell_checks):
                confirmed_cells.append(cell)

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_ctspn_k1bl_protocol_invalid"
        decision_text_zh = "协议无效：当前指标不能解释，修复后按原计划重跑。"
        next_action = (
            "repair only the failed K1-BL plan, cache, split, feature, scorer, "
            "selection, or artifact invariant and rerun unchanged"
        )
    elif confirmed_cells:
        status = "pass"
        decision = "innovation1_uknit_ctspn_k1bl_confirmed_r6_role1_difference"
        decision_text_zh = (
            f"结论：r6 role1 的 cell {confirmed_cells} 通过未见 seed/密钥确认，"
            "可进入16-pair专用神经网络归因。"
        )
        next_action = (
            "freeze the strongest confirmed r6 difference and train the uKNIT-only "
            "16-pair exact, wrong-S-box, and invariant neural matrix at 2048/class"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_ctspn_k1bl_no_confirmed_r6_role1_difference"
        decision_text_zh = (
            "结论：r6 的16个 role1 位置均未确认；只停止 role1，继续扫描 "
            "role0/2/3。"
        )
        next_action = (
            "do not call r6 random; scan the remaining three single-bit roles under "
            "the same discovery and untouched-confirmation protocol"
        )

    discovery_summary = {
        str(cell): {
            split: _split_summary(
                results, DISCOVERY_PHASE, cell, DISCOVERY_SEED, split
            )
            for split in FRESH_SPLITS
        }
        for cell in EXPECTED_CELLS
    } if _discovery_result_keys().issubset(results) else {}
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "decision_text_zh": decision_text_zh,
        "remote_scale": "no",
        "rounds": ROUNDS,
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
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
            "exact_minus_label_shuffle": LABEL_SHUFFLE_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "local zero-neural-training uKNIT r6 role-1 position discovery and "
            "untouched-seed confirmation; not neural, formal, attack, SOTA, or a "
            "complete r6 random-boundary claim"
        ),
        "blocked_actions": [
            "calling r6 random before all bit roles and trail-guided candidates are audited",
            "remote scale before a local neural exact-versus-control gate passes",
            "changing negatives, keys, pairs, thresholds, or validation after results",
        ],
    }


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
                "label_shuffled_auc": label,
                "exact_minus_label_shuffle": exact - label,
            }
        )
    return summary


def _discovery_result_keys() -> set[tuple[str, int, int, str, str]]:
    return {
        (DISCOVERY_PHASE, cell, DISCOVERY_SEED, split, view)
        for cell in EXPECTED_CELLS
        for split in EXPECTED_SPLITS
        for view in DISCOVERY_VIEWS
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
    "ACTIVE_BIT_ROLE",
    "ANCHOR_CELL",
    "CONFIRMATION_KEYS",
    "CONFIRMATION_SEEDS",
    "DISCOVERY_SEED",
    "EXPECTED_CELLS",
    "ROUNDS",
    "RUN_ID",
    "adjudicate_k1bl",
    "bind_discovery_input_differences",
    "build_confirmation_tasks",
    "select_discovery_candidates",
    "validate_confirmation_tasks",
    "validate_discovery_tasks",
]
