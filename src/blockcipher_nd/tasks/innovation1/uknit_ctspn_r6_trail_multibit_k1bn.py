from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from blockcipher_nd.ciphers.spn.uknit import (
    UKNIT_LINEAR_TARGET_SOURCES,
    UKNIT_SBOX_TABLES,
)
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


RUN_ID = "i1_uknit_ctspn_r6_trail_multibit_k1bn_seed2_confirm_seed3_seed4_20260729"
FAMILY_CELL_LOCAL = "cell_local_multibit"
FAMILY_TWO_CELL = "two_cell_low_spread"
CANDIDATE_FAMILIES = (FAMILY_CELL_LOCAL, FAMILY_TWO_CELL)
CELL_LOCAL_POOL_SIZE = 16 * 11
TWO_CELL_PREFILTER_SIZE = 256
SELECTED_PER_FAMILY = 24
EXPECTED_DISCOVERY_CANDIDATES = SELECTED_PER_FAMILY * len(CANDIDATE_FAMILIES)
CONFIRMATION_PER_FAMILY = 1
BEAM_WIDTH = 16
OUTCOMES_PER_ACTIVE_CELL = 4
TRAIL_ROUNDS = 6
K1BM_REQUIRED_DECISION = "innovation1_uknit_ctspn_k1bm_no_r6_single_bit_candidate"
DISCOVERY_VIEWS = (CANDIDATE_VIEW, RAW_VIEW)
CONFIRMATION_VIEWS = (CANDIDATE_VIEW, RAW_VIEW, LABEL_SHUFFLE_VIEW)


def build_candidate_manifest() -> dict[str, Any]:
    cell_local_inputs = [
        {
            "family": FAMILY_CELL_LOCAL,
            "input_difference": nibble << (4 * cell),
            "source_cells": [cell],
            "source_nibbles": [nibble],
            "prefilter": None,
        }
        for cell in range(16)
        for nibble in range(1, 16)
        if nibble.bit_count() >= 2
    ]
    pair_pool = _two_cell_prefilter_pool()
    ranked_by_family = {
        FAMILY_CELL_LOCAL: _rank_trail_candidates(cell_local_inputs),
        FAMILY_TWO_CELL: _rank_trail_candidates(pair_pool),
    }
    selected: list[dict[str, Any]] = []
    pool_ranking: list[dict[str, Any]] = []
    for family in CANDIDATE_FAMILIES:
        ranked = ranked_by_family[family]
        for family_rank, row in enumerate(ranked, start=1):
            row["family_rank"] = family_rank
            row["selected_for_data_gate"] = family_rank <= SELECTED_PER_FAMILY
            pool_ranking.append(row)
            if family_rank <= SELECTED_PER_FAMILY:
                selected.append(dict(row))
    selected.sort(
        key=lambda row: (
            CANDIDATE_FAMILIES.index(str(row["family"])),
            int(row["family_rank"]),
        )
    )
    for candidate_index, row in enumerate(selected):
        row["candidate_index"] = candidate_index
        row["candidate_id"] = candidate_id(row)
    return {
        "run_id": RUN_ID,
        "status": "frozen",
        "generator": {
            "trail_rounds": TRAIL_ROUNDS,
            "beam_width": BEAM_WIDTH,
            "outcomes_per_active_cell": OUTCOMES_PER_ACTIVE_CELL,
            "two_cell_prefilter_size": TWO_CELL_PREFILTER_SIZE,
            "selected_per_family": SELECTED_PER_FAMILY,
            "ranking_metric": "best_characteristic_log2_probability_then_activity",
            "ddt_usage": "candidate_selection_only_not_neural_input",
        },
        "source_hashes": {
            "uknit_sbox_tables_sha256": _canonical_sha256(UKNIT_SBOX_TABLES),
            "uknit_linear_target_sources_sha256": _canonical_sha256(
                UKNIT_LINEAR_TARGET_SOURCES
            ),
        },
        "pool_counts": {
            FAMILY_CELL_LOCAL: len(ranked_by_family[FAMILY_CELL_LOCAL]),
            FAMILY_TWO_CELL: len(ranked_by_family[FAMILY_TWO_CELL]),
        },
        "selected_candidates": selected,
        "pool_ranking": pool_ranking,
    }


def validate_candidate_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    selected = list(manifest.get("selected_candidates", []))
    pool = list(manifest.get("pool_ranking", []))
    selected_ids = [str(row.get("candidate_id")) for row in selected]
    expected_hashes = {
        "uknit_sbox_tables_sha256": _canonical_sha256(UKNIT_SBOX_TABLES),
        "uknit_linear_target_sources_sha256": _canonical_sha256(
            UKNIT_LINEAR_TARGET_SOURCES
        ),
    }
    return {
        "candidate_source_hashes_exact": manifest.get("source_hashes")
        == expected_hashes,
        "candidate_pool_counts_exact": manifest.get("pool_counts")
        == {
            FAMILY_CELL_LOCAL: CELL_LOCAL_POOL_SIZE,
            FAMILY_TWO_CELL: TWO_CELL_PREFILTER_SIZE,
        },
        "forty_eight_candidates_frozen": len(selected)
        == EXPECTED_DISCOVERY_CANDIDATES
        and len(set(selected_ids)) == EXPECTED_DISCOVERY_CANDIDATES,
        "family_quotas_exact": all(
            sum(row.get("family") == family for row in selected)
            == SELECTED_PER_FAMILY
            for family in CANDIDATE_FAMILIES
        ),
        "selected_prefix_matches_pool": all(
            [
                row.get("input_difference")
                for row in selected
                if row.get("family") == family
            ]
            == [
                row.get("input_difference")
                for row in pool
                if row.get("family") == family
            ][:SELECTED_PER_FAMILY]
            for family in CANDIDATE_FAMILIES
        ),
        "all_candidates_are_multibit": all(
            int(row.get("input_difference", 0)).bit_count() >= 2 for row in selected
        ),
        "trail_scores_finite": all(
            math.isfinite(float(row.get("trail_log2_probability", math.nan)))
            for row in pool
        ),
        "candidate_indices_contiguous": [
            int(row.get("candidate_index", -1)) for row in selected
        ]
        == list(range(EXPECTED_DISCOVERY_CANDIDATES)),
    }


def build_discovery_tasks(
    template: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _candidate_task(template, candidate, seed=DISCOVERY_SEED, confirmation=False)
        for candidate in candidates
    ]


def validate_discovery_tasks(
    tasks: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    by_id = {str(row["candidate_id"]): row for row in candidates}
    observed = {str(task.get("candidate_id")) for task in tasks}
    return {
        "discovery_task_matrix_exact": len(tasks) == len(candidates)
        == EXPECTED_DISCOVERY_CANDIDATES
        and observed == set(by_id),
        "candidate_differences_exact": all(
            int(task.get("input_difference", -1))
            == int(by_id[str(task.get("candidate_id"))]["input_difference"])
            for task in tasks
        ),
        "uknit_r6_seed2_only": all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == ROUNDS
            and int(task.get("seed", -1)) == DISCOVERY_SEED
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
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results = _result_map(result_rows)
    rankings: list[dict[str, Any]] = []
    selected: list[str] = []
    selected_by_family: dict[str, str] = {}
    for candidate in candidates:
        candidate_id_value = str(candidate["candidate_id"])
        summaries = [
            _split_summary(
                results,
                DISCOVERY_PHASE,
                candidate_id_value,
                DISCOVERY_SEED,
                split,
            )
            for split in FRESH_SPLITS
        ]
        minimum_exact = min(row["exact_auc"] for row in summaries)
        minimum_margin = min(row["exact_minus_raw"] for row in summaries)
        rankings.append(
            {
                **dict(candidate),
                "minimum_fresh_exact_auc": minimum_exact,
                "minimum_fresh_exact_minus_raw": minimum_margin,
                "eligible": minimum_exact >= AUC_FLOOR
                and minimum_margin >= RAW_MARGIN,
                "fresh_splits": {
                    split: summary
                    for split, summary in zip(FRESH_SPLITS, summaries, strict=True)
                },
            }
        )
    for family in CANDIDATE_FAMILIES:
        family_rows = sorted(
            (row for row in rankings if row["family"] == family),
            key=lambda row: (
                -float(row["minimum_fresh_exact_auc"]),
                -float(row["minimum_fresh_exact_minus_raw"]),
                int(row["family_rank"]),
            ),
        )
        eligible = [row for row in family_rows if row["eligible"]]
        if eligible:
            chosen = str(eligible[0]["candidate_id"])
            selected.append(chosen)
            selected_by_family[family] = chosen
    return {
        "run_id": RUN_ID,
        "status": "selected" if selected else "no_candidate",
        "selected_candidate_ids": selected,
        "selected_by_family": selected_by_family,
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
        },
        "ranking": sorted(
            rankings,
            key=lambda row: (
                CANDIDATE_FAMILIES.index(str(row["family"])),
                int(row["family_rank"]),
            ),
        ),
    }


def build_confirmation_tasks(
    discovery_tasks: Sequence[Mapping[str, Any]],
    selected_candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    selected = tuple(str(candidate_id) for candidate_id in selected_candidate_ids)
    task_by_id = {str(task["candidate_id"]): task for task in discovery_tasks}
    if len(selected) > len(CANDIDATE_FAMILIES) or len(set(selected)) != len(selected):
        raise ValueError("K1-BN confirmation selection is invalid")
    if set(selected) - set(task_by_id):
        raise ValueError("K1-BN selected candidate is outside the discovery matrix")
    families = [str(task_by_id[candidate_id]["candidate_family"]) for candidate_id in selected]
    if len(set(families)) != len(families):
        raise ValueError("K1-BN confirms at most one candidate per family")
    tasks: list[dict[str, Any]] = []
    for candidate_id_value in selected:
        template = task_by_id[candidate_id_value]
        candidate = {
            "candidate_id": candidate_id_value,
            "candidate_index": template["candidate_index"],
            "family": template["candidate_family"],
            "family_rank": template["candidate_family_rank"],
            "input_difference": template["input_difference"],
            "input_weight": int(template["input_difference"]).bit_count(),
            "source_cells": template["candidate_source_cells"],
            "source_nibbles": template["candidate_source_nibbles"],
            "trail_log2_probability": template["trail_log2_probability"],
            "trail_total_active_sboxes": template["trail_total_active_sboxes"],
        }
        for seed in CONFIRMATION_SEEDS:
            tasks.append(_candidate_task(template, candidate, seed=seed, confirmation=True))
    return tasks


def validate_confirmation_tasks(
    tasks: Sequence[Mapping[str, Any]],
    selected_candidate_ids: Sequence[str],
) -> dict[str, bool]:
    selected = tuple(str(candidate_id) for candidate_id in selected_candidate_ids)
    expected = {
        (candidate_id, seed)
        for candidate_id in selected
        for seed in CONFIRMATION_SEEDS
    }
    observed = {
        (str(task.get("candidate_id")), int(task.get("seed", -1))) for task in tasks
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


def adjudicate_k1bn(
    *,
    candidate_manifest: Mapping[str, Any],
    discovery_tasks: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    candidates = list(candidate_manifest.get("selected_candidates", []))
    selected = tuple(str(value) for value in selection.get("selected_candidate_ids", ()))
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    expected_result_keys = _expected_result_keys(candidates, selected)
    expected_scorer_keys = _expected_scorer_keys(candidates, selected)
    expected_dataset_keys = _expected_dataset_keys(candidates, selected)
    observed_dataset_keys = {
        (
            str(row.get("phase")),
            str(row.get("candidate_id")),
            int(row.get("seed", -1)),
            str(row.get("split")),
        )
        for row in dataset_rows
    }
    recomputed = (
        select_discovery_candidates(result_rows, candidates)
        if _discovery_result_keys(candidates).issubset(results)
        else None
    )
    protocol_checks = {
        **dict(source_checks),
        **validate_candidate_manifest(candidate_manifest),
        **validate_discovery_tasks(discovery_tasks, candidates),
        "selection_recomputed_exactly": recomputed is not None
        and selection.get("selected_candidate_ids")
        == recomputed.get("selected_candidate_ids")
        and selection.get("ranking") == recomputed.get("ranking"),
        "result_rows_complete": len(result_rows) == len(expected_result_keys)
        and set(results) == expected_result_keys,
        "feature_rows_complete": len(feature_rows) == len(expected_result_keys)
        and set(features) == expected_result_keys,
        "scorer_rows_complete": len(scorer_rows) == len(expected_scorer_keys)
        and {_scorer_key(row) for row in scorer_rows} == expected_scorer_keys,
        "dataset_rows_complete": len(dataset_rows) == len(expected_dataset_keys)
        and observed_dataset_keys == expected_dataset_keys,
        "run_geometry_exact": all(
            _row_geometry_exact(row, candidates)
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
            features[
                (CONFIRMATION_PHASE, candidate_id_value, seed, split, CANDIDATE_VIEW)
            ].get("feature_sha256")
            == features[
                (
                    CONFIRMATION_PHASE,
                    candidate_id_value,
                    seed,
                    split,
                    LABEL_SHUFFLE_VIEW,
                )
            ].get("feature_sha256")
            for candidate_id_value in selected
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

    confirmation_summary: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    confirmed: list[str] = []
    if selected and set(results) == expected_result_keys:
        for candidate_id_value in selected:
            confirmation_summary[candidate_id_value] = {}
            candidate_checks: list[bool] = []
            for seed in CONFIRMATION_SEEDS:
                confirmation_summary[candidate_id_value][str(seed)] = {}
                for split in FRESH_SPLITS:
                    summary = _split_summary(
                        results,
                        CONFIRMATION_PHASE,
                        candidate_id_value,
                        seed,
                        split,
                    )
                    confirmation_summary[candidate_id_value][str(seed)][split] = summary
                    prefix = f"{candidate_id_value}_seed{seed}_{split}"
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
                    candidate_checks.extend(checks.values())
            if candidate_checks and all(candidate_checks):
                confirmed.append(candidate_id_value)

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_ctspn_k1bn_protocol_invalid"
        decision_text_zh = "协议无效：修复失败项后按冻结候选原样重跑。"
        next_action = "repair only failed K1-BN protocol invariants and rerun unchanged"
    elif confirmed:
        status = "pass"
        decision = "innovation1_uknit_ctspn_k1bn_confirmed_r6_multibit_difference"
        decision_text_zh = (
            f"结论：r6 多 bit 候选 {confirmed} 通过未见 seed/密钥确认；"
            "进入uKNIT专用16-pair神经归因。"
        )
        next_action = (
            "freeze the strongest confirmed multi-bit difference and train the "
            "uKNIT-only r6 16-pair exact, wrong-Sbox, and invariant matrix at 2048/class"
        )
    elif selected:
        status = "hold"
        decision = "innovation1_uknit_ctspn_k1bn_multibit_discovery_not_confirmed"
        decision_text_zh = "结论：多 bit 发现候选未通过全部未见 seed/密钥确认。"
        next_action = (
            "record r5-to-r6 as the observed boundary for the searched single-bit "
            "and frozen DDT-guided multi-bit families; do not claim universal randomness"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_ctspn_k1bn_no_r6_multibit_candidate"
        decision_text_zh = (
            "结论：48个DDT/轨迹优先多 bit 差分均无发现候选；"
            "r5是已搜索差分族的最后稳定轮。"
        )
        next_action = (
            "record r5-to-r6 as the observed boundary for all 64 single-bit and the "
            "frozen DDT-guided cell-local/two-cell multi-bit families; do not claim "
            "that every possible r6 distinguisher is random"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "decision_text_zh": decision_text_zh,
        "remote_scale": "no",
        "rounds": ROUNDS,
        "selection": dict(selection),
        "confirmed_candidate_ids": confirmed,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "confirmation_summary": confirmation_summary,
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
            "exact_minus_label_shuffle": LABEL_SHUFFLE_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "local zero-neural-training uKNIT r6 DDT-guided multibit discovery and "
            "untouched confirmation after a complete single-bit hold; not neural, "
            "formal, attack, SOTA, or a universal r6 random-boundary proof"
        ),
        "blocked_actions": [
            "calling every possible r6 distinguisher random",
            "remote scale before a local neural exact-versus-control gate passes",
            "feeding trail/DDT metadata into the neural input",
            "post-result expansion of confirmation candidates",
        ],
    }


def candidate_id(row: Mapping[str, Any]) -> str:
    family = str(row["family"])
    cells = [int(value) for value in row["source_cells"]]
    nibbles = [int(value) for value in row["source_nibbles"]]
    if family == FAMILY_CELL_LOCAL:
        return f"cm_c{cells[0]:02d}_d{nibbles[0]:x}"
    return f"tc_c{cells[0]:02d}d{nibbles[0]:x}_c{cells[1]:02d}d{nibbles[1]:x}"


def _candidate_task(
    template: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    confirmation: bool,
) -> dict[str, Any]:
    task = dict(template)
    options = dict(task["model_options"])
    difference = int(candidate["input_difference"])
    candidate_id_value = str(candidate["candidate_id"])
    options["input_difference_hex"] = f"0x{difference:016x}"
    train_key, validation_key = (
        CONFIRMATION_KEYS[seed]
        if confirmation
        else (DISCOVERY_TRAIN_KEY, DISCOVERY_VALIDATION_KEY)
    )
    task.update(
        {
            "architecture": f"uKNIT64-CTSPN-K1BN-{candidate_id_value}-Seed{seed}",
            "network": f"uKNIT64-CTSPN-K1BN-{candidate_id_value}-Seed{seed}",
            "family": "uknit_ctspn_r6_trail_multibit_k1bn",
            "seed": seed,
            "samples_per_class": (
                CONFIRMATION_SAMPLES_PER_CLASS
                if confirmation
                else DISCOVERY_SAMPLES_PER_CLASS
            ),
            "validation_samples_total": None,
            "train_key": train_key,
            "validation_key": validation_key,
            "input_difference": difference,
            "difference_profile": "",
            "difference_member": "",
            "difference_source": "K1-BN frozen uKNIT r6 DDT/trail ranking",
            "model_options": options,
            "candidate_id": candidate_id_value,
            "candidate_index": int(candidate["candidate_index"]),
            "candidate_family": str(candidate["family"]),
            "candidate_family_rank": int(candidate["family_rank"]),
            "candidate_source_cells": list(candidate["source_cells"]),
            "candidate_source_nibbles": list(candidate["source_nibbles"]),
            "trail_log2_probability": float(candidate["trail_log2_probability"]),
            "trail_total_active_sboxes": int(candidate["trail_total_active_sboxes"]),
        }
    )
    return task


def _two_cell_prefilter_pool() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_cell in range(15):
        for right_cell in range(left_cell + 1, 16):
            for left_nibble in range(1, 16):
                left_outcomes = _ddt_outcomes(0, left_cell, left_nibble)[
                    :OUTCOMES_PER_ACTIVE_CELL
                ]
                for right_nibble in range(1, 16):
                    right_outcomes = _ddt_outcomes(0, right_cell, right_nibble)[
                        :OUTCOMES_PER_ACTIVE_CELL
                    ]
                    best: tuple[float, int, int, int] | None = None
                    for left_output, left_count in left_outcomes:
                        for right_output, right_count in right_outcomes:
                            after_sbox = (left_output << (4 * left_cell)) | (
                                right_output << (4 * right_cell)
                            )
                            after_linear = _linear_difference(after_sbox, 0)
                            score = math.log2(left_count / 16.0) + math.log2(
                                right_count / 16.0
                            )
                            key = (
                                score,
                                -_active_cell_count(after_linear),
                                -after_linear.bit_count(),
                                -after_linear,
                            )
                            if best is None or key > best:
                                best = key
                    assert best is not None
                    rows.append(
                        {
                            "family": FAMILY_TWO_CELL,
                            "input_difference": (left_nibble << (4 * left_cell))
                            | (right_nibble << (4 * right_cell)),
                            "source_cells": [left_cell, right_cell],
                            "source_nibbles": [left_nibble, right_nibble],
                            "prefilter": {
                                "round0_log2_probability": best[0],
                                "round0_active_output_cells": -best[1],
                                "round0_output_bit_weight": -best[2],
                            },
                        }
                    )
    rows.sort(
        key=lambda row: (
            -float(row["prefilter"]["round0_log2_probability"]),
            int(row["prefilter"]["round0_active_output_cells"]),
            int(row["prefilter"]["round0_output_bit_weight"]),
            int(row["input_difference"]),
        )
    )
    return rows[:TWO_CELL_PREFILTER_SIZE]


def _rank_trail_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        trail = _best_characteristic(int(candidate["input_difference"]))
        ranked.append(
            {
                **dict(candidate),
                "input_difference_hex": f"0x{int(candidate['input_difference']):016x}",
                "input_weight": int(candidate["input_difference"]).bit_count(),
                **trail,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["trail_log2_probability"]),
            int(row["trail_total_active_sboxes"]),
            int(row["trail_final_active_cells"]),
            int(row["trail_final_bit_weight"]),
            int(row["input_difference"]),
        )
    )
    return ranked


def _best_characteristic(input_difference: int) -> dict[str, Any]:
    beam: list[tuple[float, int, tuple[dict[str, Any], ...], int]] = [
        (0.0, input_difference, (), 0)
    ]
    for round_index in range(TRAIL_ROUNDS):
        next_by_state: dict[int, tuple[float, tuple[dict[str, Any], ...], int]] = {}
        for score, state, path, total_active in beam:
            active = _active_cell_count(state)
            for increment, after_sbox in _expand_sbox_difference(state, round_index):
                next_state = _linear_difference(after_sbox, round_index)
                next_path = path + (
                    {
                        "round": round_index,
                        "input_difference_hex": f"0x{state:016x}",
                        "sbox_output_difference_hex": f"0x{after_sbox:016x}",
                        "linear_output_difference_hex": f"0x{next_state:016x}",
                        "active_sboxes": active,
                        "round_log2_probability": increment,
                    },
                )
                candidate = (score + increment, next_path, total_active + active)
                previous = next_by_state.get(next_state)
                if previous is None or _trail_state_key(candidate, next_state) > _trail_state_key(
                    previous, next_state
                ):
                    next_by_state[next_state] = candidate
        beam = [
            (score, state, path, total_active)
            for state, (score, path, total_active) in next_by_state.items()
        ]
        beam.sort(
            key=lambda row: (-row[0], row[3], _active_cell_count(row[1]), row[1])
        )
        beam = beam[:BEAM_WIDTH]
    score, final_state, path, total_active = beam[0]
    return {
        "trail_log2_probability": score,
        "trail_total_active_sboxes": total_active,
        "trail_final_difference_hex": f"0x{final_state:016x}",
        "trail_final_active_cells": _active_cell_count(final_state),
        "trail_final_bit_weight": final_state.bit_count(),
        "trail_path": list(path),
    }


def _expand_sbox_difference(
    state: int,
    round_index: int,
) -> list[tuple[float, int]]:
    partial: list[tuple[float, int]] = [(0.0, 0)]
    for cell in range(16):
        input_difference = (state >> (4 * cell)) & 0xF
        if input_difference == 0:
            continue
        expanded: dict[int, float] = {}
        for score, output_state in partial:
            for output_difference, count in _ddt_outcomes(
                round_index, cell, input_difference
            )[:OUTCOMES_PER_ACTIVE_CELL]:
                candidate_state = output_state | (output_difference << (4 * cell))
                candidate_score = score + math.log2(count / 16.0)
                expanded[candidate_state] = max(
                    candidate_score,
                    expanded.get(candidate_state, -math.inf),
                )
        partial = sorted(
            ((score, output_state) for output_state, score in expanded.items()),
            key=lambda row: (-row[0], row[1].bit_count(), row[1]),
        )[:BEAM_WIDTH]
    return partial


@lru_cache(maxsize=None)
def _ddt_outcomes(
    round_index: int,
    cell: int,
    input_difference: int,
) -> tuple[tuple[int, int], ...]:
    table = UKNIT_SBOX_TABLES[round_index][cell]
    counts = [0] * 16
    for value in range(16):
        counts[table[value] ^ table[value ^ input_difference]] += 1
    return tuple(
        sorted(
            ((output, count) for output, count in enumerate(counts) if count),
            key=lambda row: (-row[1], row[0]),
        )
    )


@lru_cache(maxsize=None)
def _linear_basis(round_index: int) -> tuple[int, ...]:
    outputs = [0] * 64
    for target, sources in enumerate(UKNIT_LINEAR_TARGET_SOURCES[round_index]):
        for source in sources:
            outputs[source] |= 1 << target
    return tuple(outputs)


def _linear_difference(state: int, round_index: int) -> int:
    result = 0
    basis = _linear_basis(round_index)
    remaining = state
    while remaining:
        least_bit = remaining & -remaining
        result ^= basis[least_bit.bit_length() - 1]
        remaining ^= least_bit
    return result


def _active_cell_count(state: int) -> int:
    return sum(((state >> (4 * cell)) & 0xF) != 0 for cell in range(16))


def _trail_state_key(
    row: tuple[float, tuple[dict[str, Any], ...], int],
    state: int,
) -> tuple[float, int, int, int]:
    score, _, total_active = row
    return score, -total_active, -_active_cell_count(state), -state


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str], Mapping[str, Any]]:
    return {
        (
            str(row.get("phase")),
            str(row.get("candidate_id")),
            int(row.get("seed", -1)),
            str(row.get("split")),
            str(row.get("view")),
        ): row
        for row in rows
    }


def _feature_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str], Mapping[str, Any]]:
    return _result_map(rows)


def _scorer_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("phase")),
        str(row.get("candidate_id")),
        int(row.get("seed", -1)),
        str(row.get("view")),
    )


def _split_summary(
    results: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    phase: str,
    candidate_id_value: str,
    seed: int,
    split: str,
) -> dict[str, float]:
    exact = float(
        results[(phase, candidate_id_value, seed, split, CANDIDATE_VIEW)]["auc"]
    )
    raw = float(results[(phase, candidate_id_value, seed, split, RAW_VIEW)]["auc"])
    summary = {
        "exact_auc": exact,
        "raw_auc": raw,
        "exact_minus_raw": exact - raw,
    }
    label_key = (phase, candidate_id_value, seed, split, LABEL_SHUFFLE_VIEW)
    if label_key in results:
        label = float(results[label_key]["auc"])
        summary.update(
            {
                "label_shuffled_auc": label,
                "exact_minus_label_shuffle": exact - label,
            }
        )
    return summary


def _discovery_result_keys(
    candidates: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, int, str, str]]:
    return {
        (DISCOVERY_PHASE, str(candidate["candidate_id"]), DISCOVERY_SEED, split, view)
        for candidate in candidates
        for split in EXPECTED_SPLITS
        for view in DISCOVERY_VIEWS
    }


def _expected_result_keys(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
) -> set[tuple[str, str, int, str, str]]:
    keys = _discovery_result_keys(candidates)
    keys.update(
        (CONFIRMATION_PHASE, candidate_id_value, seed, split, view)
        for candidate_id_value in selected
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _expected_scorer_keys(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
) -> set[tuple[str, str, int, str]]:
    keys = {
        (DISCOVERY_PHASE, str(candidate["candidate_id"]), DISCOVERY_SEED, view)
        for candidate in candidates
        for view in DISCOVERY_VIEWS
    }
    keys.update(
        (CONFIRMATION_PHASE, candidate_id_value, seed, view)
        for candidate_id_value in selected
        for seed in CONFIRMATION_SEEDS
        for view in CONFIRMATION_VIEWS
    )
    return keys


def _expected_dataset_keys(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
) -> set[tuple[str, str, int, str]]:
    keys = {
        (DISCOVERY_PHASE, str(candidate["candidate_id"]), DISCOVERY_SEED, split)
        for candidate in candidates
        for split in EXPECTED_SPLITS
    }
    keys.update(
        (CONFIRMATION_PHASE, candidate_id_value, seed, split)
        for candidate_id_value in selected
        for seed in CONFIRMATION_SEEDS
        for split in EXPECTED_SPLITS
    )
    return keys


def _row_geometry_exact(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> bool:
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    candidate = by_id.get(str(row.get("candidate_id")))
    return candidate is not None and (
        row.get("run_id") == RUN_ID
        and row.get("cipher_key") == "uknit64"
        and int(row.get("rounds", -1)) == ROUNDS
        and int(row.get("candidate_index", -1))
        == int(candidate["candidate_index"])
        and row.get("candidate_family") == candidate["family"]
        and int(row.get("input_difference", -1))
        == int(candidate["input_difference"])
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
    "BEAM_WIDTH",
    "CANDIDATE_FAMILIES",
    "CONFIRMATION_PHASE",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_VIEWS",
    "DISCOVERY_PHASE",
    "DISCOVERY_SEED",
    "DISCOVERY_VIEWS",
    "EXPECTED_DISCOVERY_CANDIDATES",
    "EXPECTED_SPLITS",
    "FAMILY_CELL_LOCAL",
    "FAMILY_TWO_CELL",
    "RUN_ID",
    "SELECTED_PER_FAMILY",
    "adjudicate_k1bn",
    "build_candidate_manifest",
    "build_confirmation_tasks",
    "build_discovery_tasks",
    "select_discovery_candidates",
    "validate_candidate_manifest",
    "validate_confirmation_tasks",
    "validate_discovery_tasks",
]
