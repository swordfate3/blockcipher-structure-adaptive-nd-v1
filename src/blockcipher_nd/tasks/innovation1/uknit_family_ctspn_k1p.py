from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
    RUN_ID as K1O_RUN_ID,
    evaluate_k1o,
)


RUN_ID = (
    "i1_uknit_family_ctspn_partial_state_round_calibration_"
    "k1p_r3_r4_r5_seed0_seed1_20260728"
)
K1O_DECISION = (
    "innovation1_uknit_family_ctspn_k1o_"
    "current_differential_signal_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate.json": "e4da0e2c02404cd8a65457f4c07d8f0b7b8767f17faf6cba48c08faad2d031f1",
    "results.jsonl": "345572ef9a311144ba42dbaf6e856f2c78242e620adc38f8340639ecfe842c25",
    "validation.json": "9883037941c8b281f014ea95709f08d8365f3e042709d6dc0957e3d95b3457f3",
    "feature_manifest.jsonl": (
        "b6a8fe931c3a37a626d5897c4feba066b40913526fdaa1fc850f60e1e8a2a6af"
    ),
    "scorer_manifest.jsonl": (
        "55b6af9c99d4b80b68455a7ac11779e3f08b6ecae37f09166f3540685ac81ec5"
    ),
}
EXPECTED_ROUNDS = (3, 4, 5)
LOWER_ROUNDS = (3, 4)
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
VIEW_NAMES = (CANDIDATE_VIEW, RAW_VIEW, LABEL_SHUFFLE_VIEW)
EXPECTED_RESULT_ROWS = (
    len(EXPECTED_ROUNDS)
    * len(EXPECTED_SEEDS)
    * len(EXPECTED_SPLITS)
    * len(VIEW_NAMES)
)
EXPECTED_FEATURE_ROWS = EXPECTED_RESULT_ROWS
EXPECTED_SCORER_ROWS = (
    len(EXPECTED_ROUNDS) * len(EXPECTED_SEEDS) * len(VIEW_NAMES)
)
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
EXPECTED_PAIRS = 4
VARIANCE_FLOOR = 1e-6
AUC_FLOOR = 0.550
RAW_MARGIN = 0.010
LABEL_SHUFFLE_MARGIN = 0.030
EXPECTED_RUNTIME_START = {3: 1, 4: 2, 5: 3}


def validate_k1p_source(
    *,
    source_root: Path,
    source_gate: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    source_results: Sequence[Mapping[str, Any]],
    source_features: Sequence[Mapping[str, Any]],
    source_scorers: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    return {
        "k1o_artifact_digests_exact": all(
            (source_root / name).is_file()
            and file_sha256(source_root / name) == digest
            for name, digest in EXPECTED_SOURCE_DIGESTS.items()
        ),
        "k1o_clean_hold_exact": (
            source_gate.get("run_id") == K1O_RUN_ID
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == K1O_DECISION
            and bool(source_gate.get("protocol_checks"))
            and all(source_gate.get("protocol_checks", {}).values())
        ),
        "k1o_validation_passed": (
            source_validation.get("run_id") == K1O_RUN_ID
            and source_validation.get("status") == "pass"
            and not source_validation.get("errors")
        ),
        "k1o_result_rows_complete": len(source_results) == 36,
        "k1o_feature_rows_complete": len(source_features) == 36,
        "k1o_scorer_rows_complete": len(source_scorers) == 12,
    }


def validate_k1p_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    keys = {
        (int(task.get("rounds", -1)), int(task.get("seed", -1)))
        for task in tasks
    }
    expected = {
        (rounds, seed) for rounds in EXPECTED_ROUNDS for seed in EXPECTED_SEEDS
    }
    return {
        "six_frozen_tasks_complete": len(tasks) == 6 and keys == expected,
        "uknit_only": all(task.get("cipher_key") == "uknit64" for task in tasks),
        "round_window_alignment_exact": all(
            int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(
                task.get("model_options", {}).get("runtime_round_start", -1)
            )
            == EXPECTED_RUNTIME_START.get(int(task.get("rounds", -1)), -2)
            for task in tasks
        ),
        "data_protocol_frozen": all(
            task.get("samples_per_class") == 2048
            and task.get("pairs_per_sample") == EXPECTED_PAIRS
            and task.get("input_difference") == 0x40
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("sample_structure") == "independent_pairs"
            and task.get("key_rotation_interval") == 0
            and task.get("train_key") == 0
            and task.get("validation_key")
            == int("1" * 32, 16)
            for task in tasks
        ),
    }


def evaluate_lower_round(
    *,
    rounds: int,
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    exact_structures: Mapping[int, RuntimeSpnStructure],
    wrong_sbox_structures: Mapping[int, RuntimeSpnStructure],
    batch_size: int = 256,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if rounds not in LOWER_ROUNDS:
        raise ValueError("K1-P lower-round evaluation supports only r3 and r4")
    feature_rows, scorer_rows, result_rows = evaluate_k1o(
        datasets=datasets,
        exact_structures=exact_structures,
        wrong_sbox_structures=wrong_sbox_structures,
        batch_size=batch_size,
    )
    return (
        [
            _retag_generated(row, rounds=rounds)
            for row in feature_rows
            if row.get("view") in VIEW_NAMES
        ],
        [
            _retag_generated(row, rounds=rounds)
            for row in scorer_rows
            if row.get("view") in VIEW_NAMES
        ],
        [
            _retag_generated(row, rounds=rounds)
            for row in result_rows
            if row.get("view") in VIEW_NAMES
        ],
    )


def reuse_k1o_anchor(
    *,
    source_results: Sequence[Mapping[str, Any]],
    source_features: Sequence[Mapping[str, Any]],
    source_scorers: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    return (
        [
            _retag_reused(row, rounds=5)
            for row in source_features
            if row.get("view") in VIEW_NAMES
        ],
        [
            _retag_reused(row, rounds=5)
            for row in source_scorers
            if row.get("view") in VIEW_NAMES
        ],
        [
            _retag_reused(row, rounds=5)
            for row in source_results
            if row.get("view") in VIEW_NAMES
        ],
    )


def adjudicate_k1p(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    source_results: Sequence[Mapping[str, Any]],
    source_features: Sequence[Mapping[str, Any]],
    source_scorers: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    scorers = _scorer_map(scorer_rows)
    expected_results = {
        (rounds, seed, split, view)
        for rounds in EXPECTED_ROUNDS
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for view in VIEW_NAMES
    }
    expected_scorers = {
        (rounds, seed, view)
        for rounds in EXPECTED_ROUNDS
        for seed in EXPECTED_SEEDS
        for view in VIEW_NAMES
    }
    task_checks = validate_k1p_tasks(tasks)
    protocol_checks = {
        **dict(source_checks),
        **task_checks,
        "fifty_four_results_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(results) == expected_results
        ),
        "fifty_four_feature_rows_complete": (
            len(feature_rows) == EXPECTED_FEATURE_ROWS
            and set(features) == expected_results
        ),
        "eighteen_scorers_complete": (
            len(scorer_rows) == EXPECTED_SCORER_ROWS
            and set(scorers) == expected_scorers
        ),
        "run_cipher_rounds_exact": all(
            row.get("run_id") == RUN_ID
            and row.get("cipher_key") == "uknit64"
            and int(row.get("rounds", -1)) in EXPECTED_ROUNDS
            for row in result_rows
        ),
        "split_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in result_rows
        ),
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
        "exact_and_label_shuffle_features_identical": (
            set(features) == expected_results
            and all(
                features[(rounds, seed, split, CANDIDATE_VIEW)].get(
                    "feature_sha256"
                )
                == features[(rounds, seed, split, LABEL_SHUFFLE_VIEW)].get(
                    "feature_sha256"
                )
                for rounds in EXPECTED_ROUNDS
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
            )
        ),
        "same_dataset_per_round_seed_split": (
            set(features) == expected_results
            and all(
                len(
                    {
                        features[(rounds, seed, split, view)].get(
                            "dataset_sha256"
                        )
                        for view in VIEW_NAMES
                    }
                )
                == 1
                for rounds in EXPECTED_ROUNDS
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
            )
        ),
        "closed_form_only_zero_training": all(
            row.get("fit_split") == "train_seen"
            and row.get("fit_rows") == EXPECTED_TRAIN_ROWS
            and row.get("variance_floor") == VARIANCE_FLOOR
            and row.get("training_performed") is False
            and row.get("neural_parameter_count") == 0
            and row.get("optimizer_steps") == 0
            and row.get("epochs") == 0
            for row in (*result_rows, *scorer_rows)
        ),
        "label_shuffles_nonidentity_and_seed_bound": (
            set(scorers) == expected_scorers
            and all(
                bool(
                    scorers[(rounds, seed, LABEL_SHUFFLE_VIEW)].get(
                        "label_permutation_sha256"
                    )
                )
                and scorers[(rounds, 0, LABEL_SHUFFLE_VIEW)].get(
                    "label_permutation_sha256"
                )
                != scorers[(rounds, 1, LABEL_SHUFFLE_VIEW)].get(
                    "label_permutation_sha256"
                )
                for rounds in EXPECTED_ROUNDS
                for seed in EXPECTED_SEEDS
            )
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
        "r5_results_replay_k1o_exactly": _anchor_replays(
            result_rows,
            source_results,
            key_fields=("seed", "split", "view"),
        ),
        "r5_features_replay_k1o_exactly": _anchor_replays(
            feature_rows,
            source_features,
            key_fields=("seed", "split", "view"),
        ),
        "r5_scorers_replay_k1o_exactly": _anchor_replays(
            scorer_rows,
            source_scorers,
            key_fields=("seed", "view"),
        ),
    }

    round_results: dict[str, dict[str, dict[str, Any]]] = {}
    research_checks: dict[str, bool] = {}
    if set(results) == expected_results:
        for rounds in EXPECTED_ROUNDS:
            round_results[str(rounds)] = {}
            for seed in EXPECTED_SEEDS:
                round_results[str(rounds)][str(seed)] = {}
                for split in EXPECTED_SPLITS:
                    summary = _split_summary(results, rounds, seed, split)
                    round_results[str(rounds)][str(seed)][split] = summary
                    if rounds in LOWER_ROUNDS and split in FRESH_SPLITS:
                        prefix = f"r{rounds}_seed{seed}_{split}"
                        research_checks[f"{prefix}_exact_auc_floor"] = (
                            summary["exact_auc"] >= AUC_FLOOR
                        )
                        research_checks[f"{prefix}_beats_raw"] = (
                            summary["exact_minus_raw"] >= RAW_MARGIN
                        )
                        research_checks[f"{prefix}_beats_label_shuffle"] = (
                            summary["exact_minus_label_shuffle"]
                            >= LABEL_SHUFFLE_MARGIN
                        )

    round_pass = {
        str(rounds): bool(research_checks)
        and all(
            passed
            for name, passed in research_checks.items()
            if name.startswith(f"r{rounds}_")
        )
        for rounds in LOWER_ROUNDS
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    lower_fresh = [
        round_results[str(rounds)][str(seed)][split]
        for rounds in LOWER_ROUNDS
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
        if str(rounds) in round_results
    ]
    all_lower_exact_below_floor = bool(lower_fresh) and all(
        row["exact_auc"] < AUC_FLOOR for row in lower_fresh
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1p_protocol_invalid"
        next_action = (
            "repair only the failed K1-P source, task, cache, feature, scorer, "
            "or artifact invariant and rerun the frozen calibration unchanged"
        )
    elif round_pass["3"] and round_pass["4"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1p_"
            "lower_round_signal_supported_r5_loss_boundary"
        )
        next_action = (
            "preregister K1-Q as an r5-only input-difference discovery and "
            "untouched-seed confirmation; keep the K1-P scorer, splits, pairs, "
            "strict negatives, and r5 K1-O anchor unchanged"
        )
    elif round_pass["3"] and not round_pass["4"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1p_"
            "r3_signal_supported_boundary_before_r4"
        )
        next_action = (
            "preregister K1-Q as an r4-only input-difference discovery and "
            "confirmation before returning to any r5 neural architecture"
        )
    elif not round_pass["3"] and round_pass["4"]:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1p_"
            "nonmonotonic_round_or_split_instability"
        )
        next_action = (
            "audit r3/r4 round invocation, bit ordering, keys, cache metadata, "
            "and runtime-window alignment before difference or model search"
        )
    elif all_lower_exact_below_floor:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1p_"
            "current_difference_unresolved_from_r3"
        )
        next_action = (
            "audit the 0x40 plaintext-XOR construction, integer/bit ordering, "
            "uKNIT round invocation, key binding, and runtime-window alignment "
            "against deterministic cipher test vectors before difference search"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1p_"
            "lower_round_signal_or_attribution_unstable"
        )
        next_action = (
            "localize the failed seed, key scope, and exact-vs-control margin at "
            "r3/r4 before any scale, difference search, or neural redesign"
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
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "round_pass": round_pass,
        "round_results": round_results,
        "next_action": next_action,
        "claim_scope": (
            "two-seed local uKNIT r3/r4/r5 2048/class deterministic round "
            "calibration; not neural training, formal scale, attack, SOTA, "
            "transfer, difference discovery, or ceiling"
        ),
        "blocked_actions": [
            "another neural architecture, MoE, or expert before K1-P decision",
            "remote scale or more samples, pairs, epochs, seeds, or keys",
            "DDT/trail features or difference search inside the calibration",
            "averaging over a failed seed or fresh split",
        ],
    }


def _retag_generated(row: Mapping[str, Any], *, rounds: int) -> dict[str, Any]:
    tagged = dict(row)
    tagged.update(
        {
            "run_id": RUN_ID,
            "rounds": rounds,
            "source_run_id": None,
            "source_artifact_reused": False,
        }
    )
    return tagged


def _retag_reused(row: Mapping[str, Any], *, rounds: int) -> dict[str, Any]:
    tagged = dict(row)
    upstream = tagged.get("source_run_id")
    tagged.update(
        {
            "run_id": RUN_ID,
            "rounds": rounds,
            "source_run_id": K1O_RUN_ID,
            "upstream_source_run_id": upstream,
            "source_artifact_reused": True,
        }
    )
    return tagged


def _anchor_replays(
    current_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> bool:
    source = {
        tuple(row.get(field) for field in key_fields): row
        for row in source_rows
        if row.get("view") in VIEW_NAMES
    }
    current = {
        tuple(row.get(field) for field in key_fields): row
        for row in current_rows
        if int(row.get("rounds", -1)) == 5
    }
    if len(source) != len(current) or set(source) != set(current):
        return False
    ignored = {
        "run_id",
        "rounds",
        "source_run_id",
        "upstream_source_run_id",
        "source_artifact_reused",
    }
    return all(
        all(current[key].get(name) == value for name, value in row.items() if name not in ignored)
        for key, row in source.items()
    )


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["rounds"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["view"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-P result row: {key}")
        mapped[key] = row
    return mapped


def _feature_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["rounds"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["view"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-P feature row: {key}")
        mapped[key] = row
    return mapped


def _scorer_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["rounds"]), int(row["seed"]), str(row["view"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-P scorer row: {key}")
        mapped[key] = row
    return mapped


def _split_summary(
    rows: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    rounds: int,
    seed: int,
    split: str,
) -> dict[str, float]:
    exact = float(rows[(rounds, seed, split, CANDIDATE_VIEW)]["auc"])
    raw = float(rows[(rounds, seed, split, RAW_VIEW)]["auc"])
    shuffled = float(rows[(rounds, seed, split, LABEL_SHUFFLE_VIEW)]["auc"])
    return {
        "exact_auc": exact,
        "raw_auc": raw,
        "label_shuffle_auc": shuffled,
        "exact_minus_raw": exact - raw,
        "exact_minus_label_shuffle": exact - shuffled,
    }


__all__ = [
    "AUC_FLOOR",
    "EXPECTED_FEATURE_ROWS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_ROUNDS",
    "EXPECTED_SCORER_ROWS",
    "EXPECTED_RUNTIME_START",
    "LABEL_SHUFFLE_MARGIN",
    "LOWER_ROUNDS",
    "RAW_MARGIN",
    "RUN_ID",
    "VIEW_NAMES",
    "adjudicate_k1p",
    "evaluate_lower_round",
    "reuse_k1o_anchor",
    "validate_k1p_source",
    "validate_k1p_tasks",
]
