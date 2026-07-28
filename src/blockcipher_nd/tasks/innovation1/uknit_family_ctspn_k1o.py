from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    COMPOSITION_STAGE_NAMES,
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import project_features
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i1_uknit_family_ctspn_exact_partial_state_signal_audit_k1o_20260728"
SOURCE_RUN_ID = (
    "i1_uknit_family_ctspn_exact_operator_composition_k1n_"
    "2048_seed0_seed1_20260728"
)
SOURCE_DECISION = (
    "innovation1_uknit_family_ctspn_k1n_dialga_retained_uknit_signal_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate.json": "e2aed925c5d285f2856be791e1f6450b5e338f10e470572844539d86c1134a4f",
    "dataset_manifest.jsonl": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "validation.json": (
        "8743497b33f78c6e1bda7f49ca0900f78c7b669396d1b604c25d0a97a087634d"
    ),
    "preflight.json": (
        "b87335f6d36b2eb377f0deb6999af1af4546553d01c9dc1d3bc049b42e824e8d"
    ),
}
EXPECTED_SEEDS = (0, 1)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
VIEW_NAMES = (
    "raw_position_histogram",
    "exact_five_stage_position_histogram",
    "no_sbox_five_stage_position_histogram",
    "wrong_sbox_five_stage_position_histogram",
    "exact_five_stage_invariant_histogram",
    "label_shuffled_exact_position_histogram",
)
CANDIDATE_VIEW = "exact_five_stage_position_histogram"
RAW_VIEW = "raw_position_histogram"
NO_SBOX_VIEW = "no_sbox_five_stage_position_histogram"
WRONG_SBOX_VIEW = "wrong_sbox_five_stage_position_histogram"
INVARIANT_VIEW = "exact_five_stage_invariant_histogram"
LABEL_SHUFFLE_VIEW = "label_shuffled_exact_position_histogram"
EXPECTED_FEATURE_DIMS = {
    RAW_VIEW: 256,
    CANDIDATE_VIEW: 1280,
    NO_SBOX_VIEW: 1280,
    WRONG_SBOX_VIEW: 1280,
    INVARIANT_VIEW: 80,
    LABEL_SHUFFLE_VIEW: 1280,
}
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(VIEW_NAMES)
EXPECTED_FEATURE_ROWS = EXPECTED_RESULT_ROWS
EXPECTED_SCORER_ROWS = len(EXPECTED_SEEDS) * len(VIEW_NAMES)
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
EXPECTED_PAIRS = 4
EXPECTED_CELLS = 16
EXPECTED_STAGES = len(COMPOSITION_STAGE_NAMES)
VARIANCE_FLOOR = 1e-6
AUC_FLOOR = 0.550
RAW_MARGIN = 0.010
SEMANTIC_MARGIN = 0.005
LABEL_SHUFFLE_MARGIN = 0.030
POSITION_MARGIN = 0.010


@dataclass(frozen=True)
class DiagonalFisherScorer:
    midpoint: np.ndarray
    weights: np.ndarray
    variance_floor: float
    class_counts: tuple[int, int]

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("Fisher feature geometry does not match fitted scorer")
        scores = (values - self.midpoint) @ self.weights
        if not np.all(np.isfinite(scores)):
            raise ValueError("Fisher scorer produced non-finite values")
        return scores

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.midpoint, dtype=np.float64).tobytes())
        digest.update(np.asarray(self.weights, dtype=np.float64).tobytes())
        digest.update(np.asarray([self.variance_floor], dtype=np.float64).tobytes())
        digest.update(np.asarray(self.class_counts, dtype=np.int64).tobytes())
        return digest.hexdigest()


def validate_k1o_source(
    *,
    source_root: Path,
    source_gate: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    source_preflight: Mapping[str, Any],
    dataset_manifest: Sequence[Mapping[str, Any]],
    plan_path: Path,
) -> dict[str, bool]:
    uknit_rows = [
        row for row in dataset_manifest if row.get("cipher_key") == "uknit64"
    ]
    expected_uknit = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    observed_uknit = {
        (int(row.get("seed", -1)), str(row.get("split", "")))
        for row in uknit_rows
    }
    return {
        "source_artifact_digests_exact": all(
            (source_root / name).is_file()
            and file_sha256(source_root / name) == expected
            for name, expected in EXPECTED_SOURCE_DIGESTS.items()
        ),
        "source_gate_clean_hold": (
            source_gate.get("run_id") == SOURCE_RUN_ID
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == SOURCE_DECISION
            and bool(source_gate.get("protocol_checks"))
            and all(source_gate.get("protocol_checks", {}).values())
        ),
        "source_validation_passed": (
            source_validation.get("run_id") == SOURCE_RUN_ID
            and source_validation.get("status") == "pass"
            and not source_validation.get("errors")
        ),
        "source_preflight_matches_plan": (
            source_preflight.get("run_id") == SOURCE_RUN_ID
            and source_preflight.get("status") == "pass"
            and source_preflight.get("plan_sha256") == file_sha256(plan_path)
        ),
        "six_uknit_dataset_rows_exact": (
            len(uknit_rows) == 6 and observed_uknit == expected_uknit
        ),
        "uknit_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in uknit_rows
        ),
        "all_source_cache_payloads_present": all(
            all(
                (Path(str(row.get("cache_dir", ""))) / name).is_file()
                for name in ("metadata.json", "features.npy", "labels.npy")
            )
            for row in dataset_manifest
        ),
    }


def extract_k1o_feature_views(
    dataset: DifferentialDataset,
    *,
    exact_structure: RuntimeSpnStructure,
    wrong_sbox_structure: RuntimeSpnStructure,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("K1-O batch_size must be positive")
    if exact_structure.block_bits != 64 or exact_structure.cells != EXPECTED_CELLS:
        raise ValueError("K1-O requires the 64-bit, sixteen-cell uKNIT structure")
    if not torch.equal(
        exact_structure.linear_matrices,
        wrong_sbox_structure.linear_matrices,
    ):
        raise ValueError("K1-O wrong-S-box control must preserve linear matrices")
    if torch.equal(
        exact_structure.sbox_truth_bits,
        wrong_sbox_structure.sbox_truth_bits,
    ):
        raise ValueError("K1-O wrong-S-box control must change S-box semantics")
    features = np.asarray(dataset.features)
    if features.ndim != 2 or features.shape[1] != EXPECTED_PAIRS * 2 * 64:
        raise ValueError("K1-O requires four 64-bit ciphertext pairs per sample")

    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in VIEW_NAMES if name != LABEL_SHUFFLE_VIEW
    }
    for start in range(0, int(features.shape[0]), batch_size):
        stop = min(start + batch_size, int(features.shape[0]))
        batch = torch.as_tensor(
            np.asarray(features[start:stop]).copy(),
            dtype=torch.float32,
        )
        runtime = project_features(batch, exact_structure)
        exact = _position_histograms(
            exact_operator_composition_views(runtime, exact_structure),
            exact_structure,
        )
        no_sbox = _position_histograms(
            exact_operator_composition_views(
                runtime,
                exact_structure,
                apply_sboxes=False,
            ),
            exact_structure,
        )
        wrong_sbox = _position_histograms(
            exact_operator_composition_views(runtime, wrong_sbox_structure),
            exact_structure,
        )
        chunks[RAW_VIEW].append(exact[:, 0].reshape(stop - start, -1))
        chunks[CANDIDATE_VIEW].append(exact.reshape(stop - start, -1))
        chunks[NO_SBOX_VIEW].append(no_sbox.reshape(stop - start, -1))
        chunks[WRONG_SBOX_VIEW].append(wrong_sbox.reshape(stop - start, -1))
        chunks[INVARIANT_VIEW].append(
            exact.mean(axis=2).reshape(stop - start, -1)
        )

    views = {
        name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for name, values in chunks.items()
    }
    views[LABEL_SHUFFLE_VIEW] = views[CANDIDATE_VIEW]
    manifests = {
        name: _feature_manifest(name, values)
        for name, values in views.items()
    }
    if manifests[CANDIDATE_VIEW]["feature_sha256"] != manifests[LABEL_SHUFFLE_VIEW][
        "feature_sha256"
    ]:
        raise ValueError("K1-O label shuffle must reuse exact features")
    if manifests[CANDIDATE_VIEW]["feature_sha256"] == manifests[WRONG_SBOX_VIEW][
        "feature_sha256"
    ]:
        raise ValueError("K1-O wrong S-box did not change exact features")
    return views, manifests


def fit_diagonal_fisher(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    variance_floor: float = VARIANCE_FLOOR,
) -> DiagonalFisherScorer:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("Fisher training rows and labels do not align")
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ValueError("Fisher training requires both binary classes")
    if variance_floor != VARIANCE_FLOOR:
        raise ValueError("K1-O variance floor is frozen")
    class0 = values[targets == 0]
    class1 = values[targets == 1]
    if min(len(class0), len(class1)) < 2:
        raise ValueError("Fisher training requires at least two rows per class")
    mean0 = class0.mean(axis=0)
    mean1 = class1.mean(axis=0)
    pooled = (
        (len(class0) - 1) * class0.var(axis=0, ddof=1)
        + (len(class1) - 1) * class1.var(axis=0, ddof=1)
    ) / (len(class0) + len(class1) - 2)
    weights = (mean1 - mean0) / (pooled + variance_floor)
    midpoint = 0.5 * (mean0 + mean1)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(midpoint)):
        raise ValueError("Fisher fit produced non-finite parameters")
    return DiagonalFisherScorer(
        midpoint=midpoint,
        weights=weights,
        variance_floor=variance_floor,
        class_counts=(len(class0), len(class1)),
    )


def deterministic_label_shuffle(labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, str]:
    targets = np.asarray(labels, dtype=np.uint8)
    permutation = np.random.default_rng(20_260_728 + seed).permutation(len(targets))
    if np.array_equal(permutation, np.arange(len(targets))):
        raise ValueError("K1-O label permutation must be nonidentity")
    shuffled = targets[permutation]
    if np.array_equal(shuffled, targets):
        permutation = np.roll(permutation, 1)
        shuffled = targets[permutation]
    if np.array_equal(shuffled, targets):
        raise ValueError("K1-O label shuffle must change label assignment")
    if not np.array_equal(np.sort(shuffled), np.sort(targets)):
        raise ValueError("K1-O label shuffle must preserve class counts")
    digest = hashlib.sha256(np.asarray(permutation, dtype=np.int64).tobytes()).hexdigest()
    return shuffled, digest


def evaluate_k1o(
    *,
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    exact_structures: Mapping[int, RuntimeSpnStructure],
    wrong_sbox_structures: Mapping[int, RuntimeSpnStructure],
    batch_size: int = 256,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    expected_datasets = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-O requires both seeds and all three frozen splits")
    if set(exact_structures) != set(EXPECTED_SEEDS) or set(
        wrong_sbox_structures
    ) != set(EXPECTED_SEEDS):
        raise ValueError("K1-O requires exact and wrong structures for both seeds")

    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        split_views: dict[str, dict[str, np.ndarray]] = {}
        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            views, manifests = extract_k1o_feature_views(
                dataset,
                exact_structure=exact_structures[seed],
                wrong_sbox_structure=wrong_sbox_structures[seed],
                batch_size=batch_size,
            )
            split_views[split] = views
            dataset_digest = differential_dataset_sha256(dataset)
            for view in VIEW_NAMES:
                feature_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": "uknit64",
                        "seed": seed,
                        "split": split,
                        "view": view,
                        "rows": int(views[view].shape[0]),
                        "feature_dim": int(views[view].shape[1]),
                        "dataset_sha256": dataset_digest,
                        **manifests[view],
                    }
                )

        train_labels = np.asarray(datasets[(seed, "train_seen")].labels, dtype=np.uint8)
        shuffled_labels, permutation_sha = deterministic_label_shuffle(
            train_labels,
            seed=seed,
        )
        for view in VIEW_NAMES:
            fit_labels = shuffled_labels if view == LABEL_SHUFFLE_VIEW else train_labels
            scorer = fit_diagonal_fisher(
                split_views["train_seen"][view],
                fit_labels,
            )
            scorer_rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "uknit64",
                    "seed": seed,
                    "view": view,
                    "fit_split": "train_seen",
                    "fit_rows": EXPECTED_TRAIN_ROWS,
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
                dataset = datasets[(seed, split)]
                labels = np.asarray(dataset.labels, dtype=np.uint8)
                scores = scorer.score(split_views[split][view])
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "source_run_id": SOURCE_RUN_ID,
                        "cipher_key": "uknit64",
                        "rounds": 5,
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
                        "feature_dim": int(split_views[split][view].shape[1]),
                        "feature_sha256": numpy_array_sha256(
                            split_views[split][view]
                        ),
                        "dataset_sha256": differential_dataset_sha256(dataset),
                        "scorer_sha256": scorer.sha256,
                        "fit_split": "train_seen",
                        "fit_rows": EXPECTED_TRAIN_ROWS,
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


def adjudicate_k1o(
    *,
    result_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    results = _result_map(result_rows)
    features = _feature_map(feature_rows)
    scorers = _scorer_map(scorer_rows)
    expected_results = {
        (seed, split, view)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for view in VIEW_NAMES
    }
    expected_scorers = {
        (seed, view) for seed in EXPECTED_SEEDS for view in VIEW_NAMES
    }
    protocol_checks = {
        **dict(source_checks),
        "thirty_six_results_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(results) == expected_results
        ),
        "thirty_six_feature_rows_complete": (
            len(feature_rows) == EXPECTED_FEATURE_ROWS
            and set(features) == expected_results
        ),
        "twelve_scorers_complete": (
            len(scorer_rows) == EXPECTED_SCORER_ROWS
            and set(scorers) == expected_scorers
        ),
        "run_source_cipher_rounds_exact": all(
            row.get("run_id") == RUN_ID
            and row.get("source_run_id") == SOURCE_RUN_ID
            and row.get("cipher_key") == "uknit64"
            and row.get("rounds") == 5
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
            for row in result_rows
        ),
        "histograms_normalized_and_finite": all(
            row.get("finite") is True
            and row.get("nonnegative") is True
            and row.get("normalized") is True
            for row in feature_rows
        ),
        "exact_and_label_shuffle_features_identical": all(
            features[(seed, split, CANDIDATE_VIEW)].get("feature_sha256")
            == features[(seed, split, LABEL_SHUFFLE_VIEW)].get("feature_sha256")
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ) if set(features) == expected_results else False,
        "wrong_sbox_features_distinct": all(
            features[(seed, split, CANDIDATE_VIEW)].get("feature_sha256")
            != features[(seed, split, WRONG_SBOX_VIEW)].get("feature_sha256")
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ) if set(features) == expected_results else False,
        "same_dataset_per_seed_split": all(
            len(
                {
                    features[(seed, split, view)].get("dataset_sha256")
                    for view in VIEW_NAMES
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ) if set(features) == expected_results else False,
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
                bool(scorers[(seed, LABEL_SHUFFLE_VIEW)].get("label_permutation_sha256"))
                and scorers[(seed, LABEL_SHUFFLE_VIEW)].get(
                    "label_permutation_sha256"
                )
                != scorers[(1 - seed, LABEL_SHUFFLE_VIEW)].get(
                    "label_permutation_sha256"
                )
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
    }

    seed_results: dict[str, dict[str, dict[str, Any]]] = {}
    research_checks: dict[str, bool] = {}
    if set(results) == expected_results:
        for seed in EXPECTED_SEEDS:
            seed_results[str(seed)] = {}
            for split in EXPECTED_SPLITS:
                summary = _split_summary(results, seed, split)
                seed_results[str(seed)][split] = summary
                if split in FRESH_SPLITS:
                    prefix = f"seed{seed}_{split}"
                    research_checks[f"{prefix}_exact_auc_floor"] = (
                        summary["exact_auc"] >= AUC_FLOOR
                    )
                    research_checks[f"{prefix}_beats_raw"] = (
                        summary["exact_minus_raw"] >= RAW_MARGIN
                    )
                    research_checks[f"{prefix}_beats_no_sbox"] = (
                        summary["exact_minus_no_sbox"] >= SEMANTIC_MARGIN
                    )
                    research_checks[f"{prefix}_beats_wrong_sbox"] = (
                        summary["exact_minus_wrong_sbox"] >= SEMANTIC_MARGIN
                    )
                    research_checks[f"{prefix}_beats_label_shuffle"] = (
                        summary["exact_minus_label_shuffle"]
                        >= LABEL_SHUFFLE_MARGIN
                    )
                    research_checks[f"{prefix}_position_beats_invariant"] = (
                        summary["exact_minus_invariant"] >= POSITION_MARGIN
                    )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    core_names = [
        name
        for name in research_checks
        if not name.endswith("position_beats_invariant")
    ]
    position_names = [
        name for name in research_checks if name.endswith("position_beats_invariant")
    ]
    all_core = bool(core_names) and all(research_checks[name] for name in core_names)
    all_position = bool(position_names) and all(
        research_checks[name] for name in position_names
    )
    fresh_summaries = [
        seed_results[str(seed)][split]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
        if str(seed) in seed_results
    ]
    all_exact_below_floor = bool(fresh_summaries) and all(
        row["exact_auc"] < AUC_FLOOR for row in fresh_summaries
    )
    all_exact_above_floor = bool(fresh_summaries) and all(
        row["exact_auc"] >= AUC_FLOOR for row in fresh_summaries
    )
    semantic_tie = all_exact_above_floor and any(
        row["exact_minus_no_sbox"] < SEMANTIC_MARGIN
        or row["exact_minus_wrong_sbox"] < SEMANTIC_MARGIN
        for row in fresh_summaries
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1o_protocol_invalid"
        next_action = (
            "repair only the failed K1-O source, feature, scorer, or artifact "
            "invariant and rerun the frozen audit"
        )
    elif all_core and all_position:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1o_position_preserving_signal_supported"
        )
        next_action = (
            "implement K1-P with one variable: replace invariant aggregation by a "
            "position-preserving cell-stage head while retaining all K1-O controls"
        )
    elif all_core:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1o_invariant_stage_signal_supported"
        next_action = (
            "implement a minimal K1-P invariant stage-histogram branch; do not add "
            "absolute cell identity because position attribution did not pass"
        )
    elif all_exact_below_floor:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1o_current_differential_signal_not_supported"
        )
        next_action = (
            "stop modifying the network on the current uKNIT r5 differential and "
            "audit or replace the input differential/data protocol under strict negatives"
        )
    elif semantic_tie:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1o_signal_without_sbox_identifiability"
        )
        next_action = (
            "localize the deterministic signal to raw or linear stages before another "
            "network; do not prioritize S-box conditioning"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1o_partial_state_signal_unstable"
        next_action = (
            "audit the frozen differential by seed, key scope, and stage before any "
            "architecture change or scale-up"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": {
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_raw": RAW_MARGIN,
            "exact_minus_no_sbox": SEMANTIC_MARGIN,
            "exact_minus_wrong_sbox": SEMANTIC_MARGIN,
            "exact_minus_label_shuffle": LABEL_SHUFFLE_MARGIN,
            "exact_minus_invariant": POSITION_MARGIN,
        },
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "next_action": next_action,
        "claim_scope": (
            "two-seed local uKNIT r5 2048/class deterministic partial-state signal "
            "audit; not neural training, formal scale, attack, SOTA, transfer, or ceiling"
        ),
        "blocked_actions": [
            "remote scale or more samples, epochs, pairs, width, seeds, or keys",
            "MoE, experts, DDT/trail inputs, cipher identity, or another network before K1-O decision",
            "using train-seen metrics or averages to hide a failed fresh seed or split",
        ],
    }


def numpy_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _position_histograms(
    composition_views: torch.Tensor,
    structure: RuntimeSpnStructure,
) -> np.ndarray:
    batch, pairs, bits, channels = composition_views.shape
    if pairs != EXPECTED_PAIRS or bits != structure.block_bits or channels != 15:
        raise ValueError("K1-O composition view geometry is invalid")
    stages = composition_views.reshape(batch, pairs, bits, EXPECTED_STAGES, 3)[
        ..., 2
    ].permute(0, 1, 3, 2)
    indices = torch.empty(structure.cells, 4, dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bit_indices
    cell_bits = stages[..., indices].to(torch.long)
    weights = 1 << torch.arange(3, -1, -1, dtype=torch.long)
    cell_values = torch.sum(cell_bits * weights, dim=-1)
    one_hot = F.one_hot(cell_values, num_classes=16).to(torch.float32)
    histograms = one_hot.mean(dim=1).cpu().numpy()
    expected_shape = (batch, EXPECTED_STAGES, EXPECTED_CELLS, 16)
    if histograms.shape != expected_shape:
        raise ValueError("K1-O position histogram geometry is invalid")
    return histograms


def _feature_manifest(name: str, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if name == RAW_VIEW:
        normalized = array.reshape(len(array), 1, EXPECTED_CELLS, 16).sum(axis=-1)
    elif name == INVARIANT_VIEW:
        normalized = array.reshape(len(array), EXPECTED_STAGES, 16).sum(axis=-1)
    else:
        normalized = array.reshape(
            len(array), EXPECTED_STAGES, EXPECTED_CELLS, 16
        ).sum(axis=-1)
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
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["view"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-O result row: {key}")
        mapped[key] = row
    return mapped


def _feature_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["view"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-O feature row: {key}")
        mapped[key] = row
    return mapped


def _scorer_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["view"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-O scorer row: {key}")
        mapped[key] = row
    return mapped


def _split_summary(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, Any]:
    aucs = {
        view: float(rows[(seed, split, view)]["auc"])
        for view in VIEW_NAMES
    }
    exact = aucs[CANDIDATE_VIEW]
    return {
        "exact_auc": exact,
        "raw_auc": aucs[RAW_VIEW],
        "no_sbox_auc": aucs[NO_SBOX_VIEW],
        "wrong_sbox_auc": aucs[WRONG_SBOX_VIEW],
        "invariant_auc": aucs[INVARIANT_VIEW],
        "label_shuffle_auc": aucs[LABEL_SHUFFLE_VIEW],
        "exact_minus_raw": exact - aucs[RAW_VIEW],
        "exact_minus_no_sbox": exact - aucs[NO_SBOX_VIEW],
        "exact_minus_wrong_sbox": exact - aucs[WRONG_SBOX_VIEW],
        "exact_minus_invariant": exact - aucs[INVARIANT_VIEW],
        "exact_minus_label_shuffle": exact - aucs[LABEL_SHUFFLE_VIEW],
    }


__all__ = [
    "AUC_FLOOR",
    "CANDIDATE_VIEW",
    "DiagonalFisherScorer",
    "EXPECTED_FEATURE_DIMS",
    "EXPECTED_FEATURE_ROWS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SCORER_ROWS",
    "EXPECTED_SEEDS",
    "EXPECTED_SPLITS",
    "INVARIANT_VIEW",
    "LABEL_SHUFFLE_VIEW",
    "NO_SBOX_VIEW",
    "RAW_VIEW",
    "RUN_ID",
    "SOURCE_DECISION",
    "SOURCE_RUN_ID",
    "VARIANCE_FLOOR",
    "VIEW_NAMES",
    "WRONG_SBOX_VIEW",
    "adjudicate_k1o",
    "deterministic_label_shuffle",
    "evaluate_k1o",
    "extract_k1o_feature_views",
    "fit_diagonal_fisher",
    "numpy_array_sha256",
    "validate_k1o_source",
]
