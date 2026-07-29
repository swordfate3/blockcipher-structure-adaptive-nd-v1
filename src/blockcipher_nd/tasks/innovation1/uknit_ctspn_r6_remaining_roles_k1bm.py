from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_position_k1bl import (
    AUC_FLOOR,
    CONFIRMATION_HOLDOUT_PER_CLASS,
    CONFIRMATION_KEYS,
    CONFIRMATION_SAMPLES_PER_CLASS,
    CONFIRMATION_SEEDS,
    DISCOVERY_HOLDOUT_PER_CLASS,
    DISCOVERY_SAMPLES_PER_CLASS,
    DISCOVERY_SEED,
    DISCOVERY_TRAIN_KEY,
    DISCOVERY_VALIDATION_KEY,
    EXPECTED_CELLS,
    EXPECTED_PAIRS,
    LABEL_SHUFFLE_MARGIN,
    RAW_MARGIN,
    ROUNDS,
    RUNTIME_ROUND_START,
    RUNTIME_ROUNDS,
)
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
)


RUN_ID = "i1_uknit_ctspn_r6_remaining_roles_k1bm_seed2_confirm_seed3_seed4_20260729"
ACTIVE_BIT_ROLES = (0, 2, 3)
DISCOVERY_VIEWS = (CANDIDATE_VIEW, RAW_VIEW)
CONFIRMATION_VIEWS = (CANDIDATE_VIEW, RAW_VIEW, LABEL_SHUFFLE_VIEW)
MAX_SELECTED_PER_ROLE = 1
EXPECTED_DISCOVERY_TASKS = len(ACTIVE_BIT_ROLES) * len(EXPECTED_CELLS)
K1BL_REQUIRED_DECISION = "innovation1_uknit_ctspn_k1bl_no_confirmed_r6_role1_difference"


def candidate_bit_index(cell: int, role: int) -> int:
    if cell not in EXPECTED_CELLS:
        raise ValueError("K1-BM native cell must be in [0, 15]")
    if role not in ACTIVE_BIT_ROLES:
        raise ValueError("K1-BM role must be one of 0, 2, 3")
    return 4 * cell + (3 - role)


def candidate_difference(cell: int, role: int) -> int:
    return 1 << candidate_bit_index(cell, role)


def build_discovery_tasks(
    role1_templates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    template_by_cell = {
        int(task.get("model_options", {}).get("active_cell", -1)): task
        for task in role1_templates
    }
    if set(template_by_cell) != set(EXPECTED_CELLS):
        raise ValueError("K1-BM requires all sixteen K1-BL cell templates")
    tasks: list[dict[str, Any]] = []
    for role in ACTIVE_BIT_ROLES:
        for cell in EXPECTED_CELLS:
            task = dict(template_by_cell[cell])
            options = dict(task["model_options"])
            difference = candidate_difference(cell, role)
            options.update(
                {
                    "active_bit_role": role,
                    "input_difference_hex": f"0x{difference:016x}",
                }
            )
            task.update(
                {
                    "architecture": (
                        f"uKNIT64-CTSPN-K1BM-R6-Role{role}-Cell{cell:02d}"
                    ),
                    "network": (
                        f"uKNIT64-CTSPN-K1BM-R6-Role{role}-Cell{cell:02d}"
                    ),
                    "family": "uknit_ctspn_r6_remaining_roles_k1bm",
                    "input_difference": difference,
                    "difference_profile": "",
                    "difference_member": "",
                    "difference_source": (
                        "K1-BM preregistered uKNIT r6 remaining single-bit roles"
                    ),
                    "model_options": options,
                }
            )
            tasks.append(task)
    return tasks


def validate_discovery_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    observed = {
        (
            int(task.get("model_options", {}).get("active_bit_role", -1)),
            int(task.get("model_options", {}).get("active_cell", -1)),
        )
        for task in tasks
    }
    expected = {
        (role, cell) for role in ACTIVE_BIT_ROLES for cell in EXPECTED_CELLS
    }
    return {
        "forty_eight_discovery_tasks_complete": len(tasks)
        == EXPECTED_DISCOVERY_TASKS
        and observed == expected,
        "uknit_r6_seed2_only": all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == ROUNDS
            and int(task.get("seed", -1)) == DISCOVERY_SEED
            for task in tasks
        ),
        "candidate_geometry_exact": all(
            int(task.get("input_difference", -1))
            == candidate_difference(
                int(task.get("model_options", {}).get("active_cell", -1)),
                int(task.get("model_options", {}).get("active_bit_role", -1)),
            )
            for task in tasks
        ),
        "discovery_budget_exact": all(
            int(task.get("samples_per_class", -1))
            == DISCOVERY_SAMPLES_PER_CLASS
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
    selected: list[int] = []
    selected_by_role: dict[str, int] = {}
    for role in ACTIVE_BIT_ROLES:
        role_rows: list[dict[str, Any]] = []
        for cell in EXPECTED_CELLS:
            bit_index = candidate_bit_index(cell, role)
            summaries = [
                _split_summary(
                    results,
                    DISCOVERY_PHASE,
                    bit_index,
                    DISCOVERY_SEED,
                    split,
                )
                for split in FRESH_SPLITS
            ]
            minimum_exact = min(row["exact_auc"] for row in summaries)
            minimum_margin = min(row["exact_minus_raw"] for row in summaries)
            row = {
                "role": role,
                "cell": cell,
                "bit_index": bit_index,
                "input_difference": candidate_difference(cell, role),
                "input_difference_hex": f"0x{candidate_difference(cell, role):016x}",
                "minimum_fresh_exact_auc": minimum_exact,
                "minimum_fresh_exact_minus_raw": minimum_margin,
                "eligible": minimum_exact >= AUC_FLOOR
                and minimum_margin >= RAW_MARGIN,
                "fresh_splits": {
                    split: summary
                    for split, summary in zip(FRESH_SPLITS, summaries, strict=True)
                },
            }
            rankings.append(row)
            role_rows.append(row)
        ranked_role = sorted(
            role_rows,
            key=lambda row: (
                -float(row["minimum_fresh_exact_auc"]),
                -float(row["minimum_fresh_exact_minus_raw"]),
                int(row["cell"]),
            ),
        )
        eligible = [row for row in ranked_role if row["eligible"]]
        if eligible:
            chosen = int(eligible[0]["bit_index"])
            selected.append(chosen)
            selected_by_role[str(role)] = chosen
    ranked = sorted(
        rankings,
        key=lambda row: (
            int(row["role"]),
            -float(row["minimum_fresh_exact_auc"]),
            -float(row["minimum_fresh_exact_minus_raw"]),
            int(row["cell"]),
        ),
    )
    return {
        "run_id": RUN_ID,
        "status": "selected" if selected else "no_candidate",
        "selected_bit_indices": selected,
        "selected_by_role": selected_by_role,
        "selected_input_differences": [1 << bit for bit in selected],
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
        },
        "ranking": ranked,
    }


def build_confirmation_tasks(
    discovery_tasks: Sequence[Mapping[str, Any]],
    selected_bit_indices: Sequence[int],
) -> list[dict[str, Any]]:
    selected = tuple(int(bit) for bit in selected_bit_indices)
    if len(selected) > len(ACTIVE_BIT_ROLES) or len(set(selected)) != len(selected):
        raise ValueError("K1-BM confirmation selection is invalid")
    task_by_bit = {
        candidate_bit_index(
            int(task["model_options"]["active_cell"]),
            int(task["model_options"]["active_bit_role"]),
        ): task
        for task in discovery_tasks
    }
    if set(selected) - set(task_by_bit):
        raise ValueError("K1-BM selected bit is outside the discovery matrix")
    tasks: list[dict[str, Any]] = []
    for bit_index in selected:
        template = task_by_bit[bit_index]
        cell = int(template["model_options"]["active_cell"])
        role = int(template["model_options"]["active_bit_role"])
        for seed in CONFIRMATION_SEEDS:
            train_key, validation_key = CONFIRMATION_KEYS[seed]
            task = dict(template)
            task.update(
                {
                    "architecture": (
                        f"uKNIT64-CTSPN-K1BM-R6-Role{role}-Cell{cell:02d}-Seed{seed}"
                    ),
                    "seed": seed,
                    "samples_per_class": CONFIRMATION_SAMPLES_PER_CLASS,
                    "validation_samples_total": None,
                    "train_key": train_key,
                    "validation_key": validation_key,
                    "model_options": dict(template["model_options"]),
                }
            )
            tasks.append(task)
    return tasks


def validate_confirmation_tasks(
    tasks: Sequence[Mapping[str, Any]],
    selected_bit_indices: Sequence[int],
) -> dict[str, bool]:
    selected = tuple(int(bit) for bit in selected_bit_indices)
    expected = {(bit, seed) for bit in selected for seed in CONFIRMATION_SEEDS}
    observed = {
        (
            candidate_bit_index(
                int(task["model_options"]["active_cell"]),
                int(task["model_options"]["active_bit_role"]),
            ),
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
            for task in tasks
        ),
    }


def adjudicate_k1bm(
    *,
    discovery_tasks: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    selected = tuple(int(bit) for bit in selection.get("selected_bit_indices", ()))
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    scorers = _scorer_map(scorer_rows)
    expected_result_keys = _expected_result_keys(selected)
    expected_scorer_keys = _expected_scorer_keys(selected)
    expected_dataset_keys = _expected_dataset_keys(selected)
    observed_dataset_keys = {
        (
            str(row.get("phase")),
            int(row.get("bit_index", -1)),
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
        and selection.get("selected_bit_indices")
        == recomputed.get("selected_bit_indices")
        and selection.get("ranking") == recomputed.get("ranking"),
        "result_rows_complete": len(result_rows) == len(expected_result_keys)
        and set(results) == expected_result_keys,
        "feature_rows_complete": len(feature_rows) == len(expected_result_keys)
        and set(features) == expected_result_keys,
        "scorer_rows_complete": len(scorer_rows) == len(expected_scorer_keys)
        and set(scorers) == expected_scorer_keys,
        "dataset_rows_complete": len(dataset_rows) == len(expected_dataset_keys)
        and observed_dataset_keys == expected_dataset_keys,
        "run_geometry_exact": all(_row_geometry_exact(row) for row in (
            *result_rows,
            *feature_rows,
            *scorer_rows,
        )),
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
            features[(CONFIRMATION_PHASE, bit, seed, split, CANDIDATE_VIEW)].get(
                "feature_sha256"
            )
            == features[
                (CONFIRMATION_PHASE, bit, seed, split, LABEL_SHUFFLE_VIEW)
            ].get("feature_sha256")
            for bit in selected
            for seed in CONFIRMATION_SEEDS
            for split in EXPECTED_SPLITS
        )
        if selected and set(features) == expected_result_keys
        else not selected,
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
    confirmed_bits: list[int] = []
    if selected and set(results) == expected_result_keys:
        for bit in selected:
            confirmation_summary[str(bit)] = {}
            bit_checks: list[bool] = []
            for seed in CONFIRMATION_SEEDS:
                confirmation_summary[str(bit)][str(seed)] = {}
                for split in FRESH_SPLITS:
                    summary = _split_summary(
                        results, CONFIRMATION_PHASE, bit, seed, split
                    )
                    confirmation_summary[str(bit)][str(seed)][split] = summary
                    prefix = f"bit{bit}_seed{seed}_{split}"
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
                    bit_checks.extend(checks.values())
            if bit_checks and all(bit_checks):
                confirmed_bits.append(bit)

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_ctspn_k1bm_protocol_invalid"
        decision_text_zh = "协议无效：当前指标不能解释，修复后按原计划重跑。"
        next_action = (
            "repair only the failed K1-BM plan, cache, split, feature, scorer, "
            "selection, or artifact invariant and rerun unchanged"
        )
    elif confirmed_bits:
        status = "pass"
        decision = "innovation1_uknit_ctspn_k1bm_confirmed_r6_single_bit_difference"
        decision_text_zh = (
            f"结论：r6 单 bit {confirmed_bits} 通过未见 seed/密钥确认，"
            "进入16-pair uKNIT专用网络归因。"
        )
        next_action = (
            "freeze the strongest confirmed r6 difference and train the uKNIT-only "
            "16-pair exact, wrong-S-box, and invariant neural matrix at 2048/class"
        )
    elif selected:
        status = "hold"
        decision = "innovation1_uknit_ctspn_k1bm_discovery_not_confirmed"
        decision_text_zh = (
            "结论：r6 剩余单 bit 角色出现发现候选，但未通过全部未见 seed/密钥确认。"
        )
        next_action = (
            "treat the selected rows as discovery noise and move to a preregistered "
            "DDT/trail-guided multi-bit difference ranking"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_ctspn_k1bm_no_r6_single_bit_candidate"
        decision_text_zh = (
            "结论：结合K1-BL，r6的64个单 bit 输入差分均无确认候选；"
            "下一步转轨迹引导的多 bit 差分。"
        )
        next_action = (
            "preregister a DDT/trail-guided multi-bit input-difference ranking; do "
            "not train or remotely scale r6 yet and do not call every r6 route random"
        )

    discovery_summary = {
        str(bit): {
            split: _split_summary(
                results, DISCOVERY_PHASE, bit, DISCOVERY_SEED, split
            )
            for split in FRESH_SPLITS
        }
        for bit in _expected_discovery_bits()
    } if _discovery_result_keys().issubset(results) else {}
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "decision_text_zh": decision_text_zh,
        "remote_scale": "no",
        "rounds": ROUNDS,
        "selection": dict(selection),
        "confirmed_bit_indices": confirmed_bits,
        "confirmed_input_differences": [1 << bit for bit in confirmed_bits],
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
            "local zero-neural-training uKNIT r6 roles0/2/3 single-bit discovery "
            "and untouched confirmation, interpreted with K1-BL role1; not neural, "
            "formal, attack, SOTA, or a universal r6 random-boundary claim"
        ),
        "blocked_actions": [
            "calling every r6 difference random before trail-guided multi-bit search",
            "remote scale before a local neural exact-versus-control gate passes",
            "post-result expansion of the frozen one-candidate-per-role confirmation",
        ],
    }


def _expected_discovery_bits() -> tuple[int, ...]:
    return tuple(
        candidate_bit_index(cell, role)
        for role in ACTIVE_BIT_ROLES
        for cell in EXPECTED_CELLS
    )


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str, str], Mapping[str, Any]]:
    return {
        (
            str(row.get("phase")),
            int(row.get("bit_index", -1)),
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
            int(row.get("bit_index", -1)),
            int(row.get("seed", -1)),
            str(row.get("view")),
        ): row
        for row in rows
    }


def _split_summary(
    results: Mapping[tuple[str, int, int, str, str], Mapping[str, Any]],
    phase: str,
    bit_index: int,
    seed: int,
    split: str,
) -> dict[str, float]:
    exact = float(results[(phase, bit_index, seed, split, CANDIDATE_VIEW)]["auc"])
    raw = float(results[(phase, bit_index, seed, split, RAW_VIEW)]["auc"])
    summary = {
        "exact_auc": exact,
        "raw_auc": raw,
        "exact_minus_raw": exact - raw,
    }
    label_key = (phase, bit_index, seed, split, LABEL_SHUFFLE_VIEW)
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
        (DISCOVERY_PHASE, bit, DISCOVERY_SEED, split, view)
        for bit in _expected_discovery_bits()
        for split in EXPECTED_SPLITS
        for view in DISCOVERY_VIEWS
    }


def _expected_result_keys(
    selected_bits: Sequence[int],
) -> set[tuple[str, int, int, str, str]]:
    keys = _discovery_result_keys()
    keys.update(
        (CONFIRMATION_PHASE, bit, seed, split, view)
        for bit in selected_bits
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _expected_scorer_keys(
    selected_bits: Sequence[int],
) -> set[tuple[str, int, int, str]]:
    keys = {
        (DISCOVERY_PHASE, bit, DISCOVERY_SEED, view)
        for bit in _expected_discovery_bits()
        for view in DISCOVERY_VIEWS
    }
    keys.update(
        (CONFIRMATION_PHASE, bit, seed, view)
        for bit in selected_bits
        for seed in CONFIRMATION_SEEDS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _expected_dataset_keys(
    selected_bits: Sequence[int],
) -> set[tuple[str, int, int, str]]:
    keys = {
        (DISCOVERY_PHASE, bit, DISCOVERY_SEED, split)
        for bit in _expected_discovery_bits()
        for split in EXPECTED_SPLITS
    }
    keys.update(
        (CONFIRMATION_PHASE, bit, seed, split)
        for bit in selected_bits
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
    )
    return keys


def _row_geometry_exact(row: Mapping[str, Any]) -> bool:
    cell = int(row.get("cell", -1))
    role = int(row.get("active_bit_role", -1))
    return (
        row.get("run_id") == RUN_ID
        and row.get("cipher_key") == "uknit64"
        and int(row.get("rounds", -1)) == ROUNDS
        and role in ACTIVE_BIT_ROLES
        and int(row.get("bit_index", -1)) == candidate_bit_index(cell, role)
        and int(row.get("input_difference", -1)) == candidate_difference(cell, role)
    )


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
    "ACTIVE_BIT_ROLES",
    "CONFIRMATION_SEEDS",
    "DISCOVERY_SEED",
    "ROUNDS",
    "RUN_ID",
    "adjudicate_k1bm",
    "build_confirmation_tasks",
    "build_discovery_tasks",
    "candidate_bit_index",
    "candidate_difference",
    "select_discovery_candidates",
    "validate_confirmation_tasks",
    "validate_discovery_tasks",
]
