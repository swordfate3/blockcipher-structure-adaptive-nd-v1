from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    deterministic_position_histogram,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import project_features
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    VARIANCE_FLOOR,
    deterministic_label_shuffle,
    fit_diagonal_fisher,
    numpy_array_sha256,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i1_uknit_r6_public_window_depth_k1bo_seed3_seed4_20260730"
EXPECTED_SEEDS = (3, 4)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
EXPECTED_PAIRS = 4
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048

R5_EXACT2_POSITION = "r5_exact2_position"
R6_EXACT2_POSITION = "r6_exact2_position"
R6_EXACT3_POSITION = "r6_exact3_position"
R6_WRONG3_POSITION = "r6_wrong3_position"
R6_SHUFFLE3_POSITION = "r6_shuffle3_position"
R6_EXACT3_INVARIANT = "r6_exact3_invariant"
R6_WRONG3_INVARIANT = "r6_wrong3_invariant"
R6_SHUFFLE3_INVARIANT = "r6_shuffle3_invariant"
R6_RAW = "r6_raw"

R5_VIEWS = (R5_EXACT2_POSITION,)
R6_VIEWS = (
    R6_EXACT2_POSITION,
    R6_EXACT3_POSITION,
    R6_WRONG3_POSITION,
    R6_SHUFFLE3_POSITION,
    R6_EXACT3_INVARIANT,
    R6_WRONG3_INVARIANT,
    R6_SHUFFLE3_INVARIANT,
    R6_RAW,
)
VIEW_NAMES = (*R5_VIEWS, *R6_VIEWS)
LABEL_SHUFFLE_VIEWS = (R6_SHUFFLE3_POSITION, R6_SHUFFLE3_INVARIANT)
EXPECTED_FEATURE_DIMS = {
    R5_EXACT2_POSITION: 5 * 16 * 16,
    R6_EXACT2_POSITION: 5 * 16 * 16,
    R6_EXACT3_POSITION: 7 * 16 * 16,
    R6_WRONG3_POSITION: 7 * 16 * 16,
    R6_SHUFFLE3_POSITION: 7 * 16 * 16,
    R6_EXACT3_INVARIANT: 7 * 16,
    R6_WRONG3_INVARIANT: 7 * 16,
    R6_SHUFFLE3_INVARIANT: 7 * 16,
    R6_RAW: 16 * 16,
}


def extract_window_depth_views(
    dataset: DifferentialDataset,
    *,
    exact_two: RuntimeSpnStructure,
    exact_three: RuntimeSpnStructure | None = None,
    wrong_three: RuntimeSpnStructure | None = None,
    rounds: int,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], bool]:
    if rounds not in (5, 6):
        raise ValueError("K1-BO supports only the frozen r5/r6 prefixes")
    if exact_two.rounds != 2 or exact_two.block_bits != 64 or exact_two.cells != 16:
        raise ValueError("K1-BO two-transition structure is invalid")
    if batch_size <= 0:
        raise ValueError("K1-BO batch_size must be positive")
    features = np.asarray(dataset.features)
    if features.ndim != 2 or features.shape[1] != EXPECTED_PAIRS * 2 * 64:
        raise ValueError("K1-BO requires four 64-bit ciphertext pairs per sample")

    if rounds == 5:
        names = R5_VIEWS
    else:
        if exact_three is None or wrong_three is None:
            raise ValueError("K1-BO r6 requires exact and wrong three-transition views")
        if exact_three.rounds != 3 or wrong_three.rounds != 3:
            raise ValueError("K1-BO r6 three-transition structures are invalid")
        if not torch.equal(exact_three.linear_matrices, wrong_three.linear_matrices):
            raise ValueError("K1-BO wrong S-box must preserve all linear matrices")
        if torch.equal(exact_three.sbox_truth_bits, wrong_three.sbox_truth_bits):
            raise ValueError("K1-BO wrong S-box must change S-box semantics")
        names = R6_VIEWS

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    prefix_equal = True
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch = torch.as_tensor(
            np.asarray(features[start:stop]).copy(),
            dtype=torch.float32,
        )
        two_runtime = project_features(batch, exact_two)
        two = deterministic_position_histogram(two_runtime, exact_two)
        if rounds == 5:
            chunks[R5_EXACT2_POSITION].append(_flatten(two))
            continue

        assert exact_three is not None and wrong_three is not None
        three_runtime = project_features(batch, exact_three)
        exact = deterministic_position_histogram(three_runtime, exact_three)
        wrong = deterministic_position_histogram(
            project_features(batch, wrong_three),
            wrong_three,
        )
        prefix_equal = prefix_equal and torch.equal(exact[:, :5], two)
        exact_position = _flatten(exact)
        wrong_position = _flatten(wrong)
        exact_invariant = _flatten(exact.mean(dim=2))
        wrong_invariant = _flatten(wrong.mean(dim=2))
        chunks[R6_EXACT2_POSITION].append(_flatten(two))
        chunks[R6_EXACT3_POSITION].append(exact_position)
        chunks[R6_WRONG3_POSITION].append(wrong_position)
        chunks[R6_SHUFFLE3_POSITION].append(exact_position)
        chunks[R6_EXACT3_INVARIANT].append(exact_invariant)
        chunks[R6_WRONG3_INVARIANT].append(wrong_invariant)
        chunks[R6_SHUFFLE3_INVARIANT].append(exact_invariant)
        chunks[R6_RAW].append(_flatten(exact[:, :1]))

    views = {
        name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for name, values in chunks.items()
    }
    manifests = {
        name: _feature_manifest(name, values)
        for name, values in views.items()
    }
    if rounds == 6:
        if numpy_array_sha256(views[R6_EXACT3_POSITION]) != numpy_array_sha256(
            views[R6_SHUFFLE3_POSITION]
        ):
            raise ValueError("K1-BO position label shuffle changed exact features")
        if numpy_array_sha256(views[R6_EXACT3_INVARIANT]) != numpy_array_sha256(
            views[R6_SHUFFLE3_INVARIANT]
        ):
            raise ValueError("K1-BO invariant label shuffle changed exact features")
        if numpy_array_sha256(views[R6_EXACT3_POSITION]) == numpy_array_sha256(
            views[R6_WRONG3_POSITION]
        ):
            raise ValueError("K1-BO wrong S-box did not change position features")
    return views, manifests, prefix_equal


def evaluate_k1bo(
    *,
    datasets: Mapping[tuple[int, int, str], DifferentialDataset],
    r5_exact_two: RuntimeSpnStructure,
    r6_exact_two: RuntimeSpnStructure,
    r6_exact_three: RuntimeSpnStructure,
    r6_wrong_three: RuntimeSpnStructure,
    batch_size: int = 256,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    expected_datasets = {
        (rounds, seed, split)
        for rounds in (5, 6)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-BO requires all twelve frozen source datasets")

    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    split_views: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    prefix_checks: dict[str, bool] = {}

    for rounds in (5, 6):
        views_for_round = R5_VIEWS if rounds == 5 else R6_VIEWS
        for seed in EXPECTED_SEEDS:
            for split in EXPECTED_SPLITS:
                dataset = datasets[(rounds, seed, split)]
                views, manifests, prefix_equal = extract_window_depth_views(
                    dataset,
                    exact_two=r5_exact_two if rounds == 5 else r6_exact_two,
                    exact_three=r6_exact_three if rounds == 6 else None,
                    wrong_three=r6_wrong_three if rounds == 6 else None,
                    rounds=rounds,
                    batch_size=batch_size,
                )
                split_views[(rounds, seed, split)] = views
                if rounds == 6:
                    prefix_checks[f"seed{seed}_{split}_three_prefix_matches_two"] = (
                        prefix_equal
                    )
                dataset_sha = differential_dataset_sha256(dataset)
                for view in views_for_round:
                    feature_rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": "uknit64",
                            "rounds": rounds,
                            "seed": seed,
                            "split": split,
                            "view": view,
                            "rows": int(views[view].shape[0]),
                            "feature_dim": int(views[view].shape[1]),
                            "dataset_sha256": dataset_sha,
                            **manifests[view],
                        }
                    )

            train_dataset = datasets[(rounds, seed, "train_seen")]
            train_labels = np.asarray(train_dataset.labels, dtype=np.uint8)
            shuffled_labels, permutation_sha = deterministic_label_shuffle(
                train_labels,
                seed=seed,
            )
            for view in views_for_round:
                fit_labels = shuffled_labels if view in LABEL_SHUFFLE_VIEWS else train_labels
                scorer = fit_diagonal_fisher(
                    split_views[(rounds, seed, "train_seen")][view],
                    fit_labels,
                    variance_floor=VARIANCE_FLOOR,
                )
                scorer_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": "uknit64",
                        "rounds": rounds,
                        "seed": seed,
                        "view": view,
                        "fit_split": "train_seen",
                        "fit_rows": len(train_labels),
                        "feature_dim": int(scorer.weights.shape[0]),
                        "variance_floor": scorer.variance_floor,
                        "class0_rows": scorer.class_counts[0],
                        "class1_rows": scorer.class_counts[1],
                        "weight_l2_norm": float(np.linalg.norm(scorer.weights)),
                        "nonzero_weight_count": int(np.count_nonzero(scorer.weights)),
                        "scorer_sha256": scorer.sha256,
                        "label_permutation_sha256": (
                            permutation_sha if view in LABEL_SHUFFLE_VIEWS else None
                        ),
                        "training_performed": False,
                        "neural_parameter_count": 0,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
                for split in EXPECTED_SPLITS:
                    dataset = datasets[(rounds, seed, split)]
                    labels = np.asarray(dataset.labels, dtype=np.uint8)
                    values = split_views[(rounds, seed, split)][view]
                    scores = scorer.score(values)
                    result_rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": "uknit64",
                            "rounds": rounds,
                            "seed": seed,
                            "split": split,
                            "view": view,
                            "rows": len(labels),
                            "auc": binary_auc(labels, scores),
                            "zero_threshold_accuracy": float(
                                ((scores >= 0.0).astype(np.uint8) == labels).mean()
                            ),
                            "score_mean": float(scores.mean()),
                            "score_std": float(scores.std()),
                            "score_min": float(scores.min()),
                            "score_max": float(scores.max()),
                            "feature_dim": int(values.shape[1]),
                            "feature_sha256": numpy_array_sha256(values),
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "scorer_sha256": scorer.sha256,
                            "fit_split": "train_seen",
                            "fit_rows": len(train_labels),
                            "pairs_per_sample": EXPECTED_PAIRS,
                            "negative_mode": "encrypted_random_plaintexts",
                            "variance_floor": VARIANCE_FLOOR,
                            "training_performed": False,
                            "neural_parameter_count": 0,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return feature_rows, scorer_rows, result_rows, prefix_checks


def adjudicate_k1bo(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_auc: Mapping[tuple[int, int, str], float],
    source_checks: Mapping[str, bool],
    prefix_checks: Mapping[str, bool],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    expected_results = {
        (rounds, seed, split, view)
        for rounds, views in ((5, R5_VIEWS), (6, R6_VIEWS))
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for view in views
    }
    expected_scorers = {
        (rounds, seed, view)
        for rounds, views in ((5, R5_VIEWS), (6, R6_VIEWS))
        for seed in EXPECTED_SEEDS
        for view in views
    }
    observed_scorers = {
        (int(row["rounds"]), int(row["seed"]), str(row["view"]))
        for row in scorer_rows
    }
    tolerance = float(thresholds["source_auc_replay_tolerance"])
    replay_deltas = {
        f"r{rounds}_seed{seed}_{split}": abs(
            float(results[(rounds, seed, split, view)]["auc"])
            - float(source_auc[(rounds, seed, split)])
        )
        for rounds, view in ((5, R5_EXACT2_POSITION), (6, R6_EXACT2_POSITION))
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        if (rounds, seed, split, view) in results
        and (rounds, seed, split) in source_auc
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(prefix_checks),
        "fifty_four_results_complete": len(result_rows) == len(expected_results)
        and set(results) == expected_results,
        "fifty_four_feature_rows_complete": len(feature_rows) == len(expected_results)
        and set(features) == expected_results,
        "eighteen_scorers_complete": len(scorer_rows) == len(expected_scorers)
        and observed_scorers == expected_scorers,
        "feature_dimensions_exact": all(
            int(row.get("feature_dim", -1))
            == EXPECTED_FEATURE_DIMS.get(str(row.get("view")), -2)
            for row in (*feature_rows, *result_rows, *scorer_rows)
        ),
        "histograms_normalized_finite_nonnegative": all(
            row.get("normalized") is True
            and row.get("finite") is True
            and row.get("nonnegative") is True
            for row in feature_rows
        ),
        "source_two_round_auc_replay_exact": len(replay_deltas) == 12
        and all(delta <= tolerance for delta in replay_deltas.values()),
        "r6_same_dataset_across_views": set(results) == expected_results
        and all(
            len(
                {
                    results[(6, seed, split, view)]["dataset_sha256"]
                    for view in R6_VIEWS
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "position_shuffle_features_identical": set(features) == expected_results
        and all(
            features[(6, seed, split, R6_EXACT3_POSITION)]["feature_sha256"]
            == features[(6, seed, split, R6_SHUFFLE3_POSITION)]["feature_sha256"]
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "invariant_shuffle_features_identical": set(features) == expected_results
        and all(
            features[(6, seed, split, R6_EXACT3_INVARIANT)]["feature_sha256"]
            == features[(6, seed, split, R6_SHUFFLE3_INVARIANT)]["feature_sha256"]
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "wrong_sbox_features_distinct_equal_shape": set(features) == expected_results
        and all(
            features[(6, seed, split, exact)]["feature_sha256"]
            != features[(6, seed, split, wrong)]["feature_sha256"]
            and features[(6, seed, split, exact)]["feature_dim"]
            == features[(6, seed, split, wrong)]["feature_dim"]
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
            for exact, wrong in (
                (R6_EXACT3_POSITION, R6_WRONG3_POSITION),
                (R6_EXACT3_INVARIANT, R6_WRONG3_INVARIANT),
            )
        ),
        "closed_form_only_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("neural_parameter_count", -1)) == 0
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            for row in (*scorer_rows, *result_rows)
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
    }

    route_specs = {
        "position": (
            R6_EXACT3_POSITION,
            R6_WRONG3_POSITION,
            R6_SHUFFLE3_POSITION,
        ),
        "invariant": (
            R6_EXACT3_INVARIANT,
            R6_WRONG3_INVARIANT,
            R6_SHUFFLE3_INVARIANT,
        ),
    }
    route_results: dict[str, dict[str, dict[str, Any]]] = {
        route: {} for route in route_specs
    }
    research_checks: dict[str, bool] = {}
    for route, (exact_view, wrong_view, shuffle_view) in route_specs.items():
        for seed in EXPECTED_SEEDS:
            route_results[route][str(seed)] = {}
            for split in FRESH_SPLITS:
                summary = _route_summary(
                    results,
                    seed=seed,
                    split=split,
                    exact_view=exact_view,
                    wrong_view=wrong_view,
                    shuffle_view=shuffle_view,
                )
                route_results[route][str(seed)][split] = summary
                prefix = f"{route}_seed{seed}_{split}"
                research_checks[f"{prefix}_auc_floor"] = (
                    summary["exact3_auc"] >= float(thresholds["auc_floor"])
                )
                research_checks[f"{prefix}_beats_exact2"] = (
                    summary["exact3_minus_exact2"] >= float(thresholds["depth_margin"])
                )
                research_checks[f"{prefix}_beats_raw"] = (
                    summary["exact3_minus_raw"] >= float(thresholds["raw_margin"])
                )
                research_checks[f"{prefix}_beats_wrong_sbox"] = (
                    summary["exact3_minus_wrong_sbox"]
                    >= float(thresholds["wrong_sbox_margin"])
                )
                research_checks[f"{prefix}_beats_label_shuffle"] = (
                    summary["exact3_minus_label_shuffle"]
                    >= float(thresholds["label_shuffle_margin"])
                )

    route_pass = {
        route: all(
            passed
            for name, passed in research_checks.items()
            if name.startswith(f"{route}_")
        )
        for route in route_specs
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    any_route_pass = any(route_pass.values())
    any_depth_gain = any(
        summary["exact3_minus_exact2"] >= float(thresholds["depth_margin"])
        for route in route_results.values()
        for seed in route.values()
        for summary in seed.values()
    )
    any_semantic_failure = any(
        not passed
        for name, passed in research_checks.items()
        if name.endswith("beats_wrong_sbox")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_r6_k1bo_protocol_invalid"
        next_action = (
            "repair only the failed source binding, two-round replay, generalized "
            "composition geometry or artifact invariant and rerun K1-BO"
        )
    elif any_route_pass:
        status = "pass"
        decision = "innovation1_uknit_r6_k1bo_three_round_window_signal_supported"
        passed = ", ".join(route for route, value in route_pass.items() if value)
        next_action = (
            "run one separate local r6 neural attribution matrix using only the "
            f"passed aggregation route(s): {passed}; keep K1-U data, four pairs, "
            "seeds, keys, epochs and exact/wrong-S-box/no-structure controls fixed"
        )
    elif any_depth_gain and any_semantic_failure:
        status = "hold"
        decision = "innovation1_uknit_r6_k1bo_extra_view_signal_not_structure_attributed"
        next_action = (
            "do not train or scale r6; the deeper view did not establish correct "
            "uKNIT S-box attribution on every frozen fresh row"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_r6_k1bo_three_round_window_signal_not_supported"
        next_action = (
            "retain the K1-BN searched-family r5-to-r6 boundary, reject insufficient "
            "public-window depth as its explanation, and do not train or scale r6"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "neural_training_authorized": bool(protocol_valid and any_route_pass),
        "passed_routes": [route for route, value in route_pass.items() if value],
        "thresholds": dict(thresholds),
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "route_results": route_results,
        "source_auc_replay_deltas": replay_deltas,
        "next_action": next_action,
        "claim_scope": (
            "two-seed local zero-neural-training uKNIT r6 cell11 public-window "
            "depth attribution using reused 2048/class train and 1024/class fresh "
            "caches; not formal scale, attack, SOTA, universal r6 randomness or "
            "neural architecture evidence"
        ),
        "blocked_actions": [
            "remote r6 scale without a passed local neural attribution matrix",
            "changing difference, pairs, seeds, keys, samples or thresholds after results",
            "calling public inverse-operator views recovered keyed internal states",
        ],
    }


def _flatten(values: torch.Tensor) -> np.ndarray:
    return values.reshape(values.shape[0], -1).cpu().numpy()


def _feature_manifest(name: str, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if name == R6_RAW:
        normalized = array.reshape(len(array), 1, 16, 16).sum(axis=-1)
    elif name.endswith("invariant"):
        normalized = array.reshape(len(array), 7, 16).sum(axis=-1)
    else:
        stages = 5 if "exact2" in name else 7
        normalized = array.reshape(len(array), stages, 16, 16).sum(axis=-1)
    return {
        "feature_sha256": numpy_array_sha256(array),
        "finite": bool(np.all(np.isfinite(array))),
        "nonnegative": bool(np.all(array >= 0.0)),
        "normalized": bool(np.allclose(normalized, 1.0, atol=1e-7, rtol=0.0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    return {
        (
            int(row.get("rounds", -1)),
            int(row.get("seed", -1)),
            str(row.get("split")),
            str(row.get("view")),
        ): row
        for row in rows
    }


def _feature_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    return _result_map(rows)


def _route_summary(
    results: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    *,
    seed: int,
    split: str,
    exact_view: str,
    wrong_view: str,
    shuffle_view: str,
) -> dict[str, float]:
    exact = float(results[(6, seed, split, exact_view)]["auc"])
    exact2 = float(results[(6, seed, split, R6_EXACT2_POSITION)]["auc"])
    wrong = float(results[(6, seed, split, wrong_view)]["auc"])
    shuffled = float(results[(6, seed, split, shuffle_view)]["auc"])
    raw = float(results[(6, seed, split, R6_RAW)]["auc"])
    return {
        "exact3_auc": exact,
        "exact2_auc": exact2,
        "wrong_sbox_auc": wrong,
        "label_shuffled_auc": shuffled,
        "raw_auc": raw,
        "exact3_minus_exact2": exact - exact2,
        "exact3_minus_wrong_sbox": exact - wrong,
        "exact3_minus_label_shuffle": exact - shuffled,
        "exact3_minus_raw": exact - raw,
    }


def feature_binding_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


__all__ = [
    "EXPECTED_FEATURE_DIMS",
    "EXPECTED_SEEDS",
    "EXPECTED_SPLITS",
    "FRESH_SPLITS",
    "R5_EXACT2_POSITION",
    "R5_VIEWS",
    "R6_EXACT2_POSITION",
    "R6_EXACT3_INVARIANT",
    "R6_EXACT3_POSITION",
    "R6_RAW",
    "R6_SHUFFLE3_INVARIANT",
    "R6_SHUFFLE3_POSITION",
    "R6_VIEWS",
    "R6_WRONG3_INVARIANT",
    "R6_WRONG3_POSITION",
    "RUN_ID",
    "VIEW_NAMES",
    "adjudicate_k1bo",
    "evaluate_k1bo",
    "extract_window_depth_views",
]
