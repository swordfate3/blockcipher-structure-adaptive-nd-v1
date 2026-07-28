from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    exact_operator_composition_views,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import project_features
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW as K1Q_EXACT_VIEW,
    LABEL_SHUFFLE_VIEW as K1Q_LABEL_SHUFFLE_VIEW,
    _position_histograms,
    deterministic_label_shuffle,
    fit_diagonal_fisher,
    numpy_array_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    RUN_ID as K1Q_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_PAIRS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
    K1Q_DECISION,
    RUN_ID as K1R_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i1_uknit_family_ctspn_representation_access_k1s_seed3_seed4_20260728"
K1R_DECISION = (
    "innovation1_uknit_family_ctspn_k1r_cell11_neural_signal_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1q_gate": "1af79fa865736635d40f729fe6621e677a4378e64c6779fc449756ae48609f8b",
    "k1q_dataset_manifest": (
        "16d9549df5d1a6b2d88fd95e10ceec484e6f5443bd774f11d0f7d68dc85494f2"
    ),
    "k1q_results": "faf78bc287b35f0237101869d53a89347451369d7e03d6a1253e32ab6f14bc91",
    "k1q_feature_manifest": (
        "e242aa8bd1f723954a6dcca14352b762163d1771f3e00ebdf1eb8afd7cf10868"
    ),
    "k1q_scorer_manifest": (
        "46f8e7823f11aabc03101fce3ba8ffdc20ddb35b8a1e88140305d94f5fac3261"
    ),
    "k1q_validation": (
        "25b59f9b0eeab8eb894c4b3a40513437306a2c660f0c68f4ab478260689d8059"
    ),
    "k1r_plan": "8e612988656163602db20a80241b7b4cfdf01a7c16c37e3ae1e30447f2a4ab00",
    "k1r_gate": "73371777ddef3369b58132939a0d85bf5021e8a5233e6c6c549f1d506f37e299",
    "k1r_checkpoint_manifest": (
        "a4ac9044df2dd1e0276ff88449e21c3c1d2b0a16c3113e7a059a9273adb04b2f"
    ),
    "k1r_results": "b3ad0aaf3de1a974c149f2fd546e48696f4ae9ad7dabb8e364000c536dd57cf3",
    "k1r_controls": "167fc68b40762d7a0781c78c5ed8d2ca4be427a5741cd96f4748bbb23d965acd",
    "k1r_validation": (
        "161e4c64cc692955711b61e5ec9291a2d95c45e15a310f5e0540d5af7917d7b2"
    ),
}
EXPECTED_CHECKPOINT_SHAS = {
    3: "030d280458654dcbda6a38aafe77f39c3d9f43cdee6ec350742e3d36252071e4",
    4: "a64b3f326795adf955aba6ee87ebc9b9a5b44861322aaa6a7087ea75c9c45e21",
}
TAPS = (
    "T0_exact_position_histogram",
    "T1_bit_encoder_position",
    "T2_topology_delta_position",
    "T3_invariant_cell_pool",
)
SCORER_MODES = ("interpreted", "label_shuffle")
EXPECTED_FEATURE_DIMS = {
    TAPS[0]: 1280,
    TAPS[1]: 8192,
    TAPS[2]: 2048,
    TAPS[3]: 384,
}
EXPECTED_FEATURE_ROWS = len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(TAPS)
EXPECTED_SCORER_ROWS = len(EXPECTED_SEEDS) * len(TAPS) * len(SCORER_MODES)
EXPECTED_RESULT_ROWS = EXPECTED_FEATURE_ROWS * len(SCORER_MODES)
AUC_FLOOR = 0.550
LABEL_SHUFFLE_MARGIN = 0.030
POSITION_TO_POOL_MARGIN = 0.030
REPLAY_TOLERANCE = 0.0


def source_binding_checks(
    *,
    source_digests: Mapping[str, str],
    k1q_gate: Mapping[str, Any],
    k1q_validation: Mapping[str, Any],
    k1r_gate: Mapping[str, Any],
    k1r_validation: Mapping[str, Any],
    dataset_manifest: Sequence[Mapping[str, Any]],
    checkpoint_entries: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    expected_dataset_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    observed_dataset_keys = {
        (int(row.get("seed", -1)), str(row.get("split", "")))
        for row in dataset_manifest
    }
    exact_checkpoints = [
        row for row in checkpoint_entries if row.get("condition") == "exact_composition"
    ]
    observed_checkpoints = {
        int(row.get("seed", -1)): str(row.get("sha256", ""))
        for row in exact_checkpoints
    }
    return {
        "source_artifact_digests_exact": (
            dict(source_digests) == EXPECTED_SOURCE_DIGESTS
        ),
        "k1q_confirmed_cell11_source_exact": (
            k1q_gate.get("run_id") == K1Q_RUN_ID
            and k1q_gate.get("status") == "pass"
            and k1q_gate.get("decision") == K1Q_DECISION
            and 11 in k1q_gate.get("confirmed_cells", [])
            and bool(k1q_gate.get("protocol_checks"))
            and all(k1q_gate.get("protocol_checks", {}).values())
            and k1q_validation.get("run_id") == K1Q_RUN_ID
            and k1q_validation.get("status") == "pass"
            and not k1q_validation.get("errors")
        ),
        "k1r_clean_hold_source_exact": (
            k1r_gate.get("run_id") == K1R_RUN_ID
            and k1r_gate.get("status") == "hold"
            and k1r_gate.get("decision") == K1R_DECISION
            and bool(k1r_gate.get("protocol_checks"))
            and all(k1r_gate.get("protocol_checks", {}).values())
            and k1r_validation.get("run_id") == K1R_RUN_ID
            and k1r_validation.get("status") == "pass"
            and not k1r_validation.get("errors")
        ),
        "six_cell11_confirmation_caches_exact": (
            len(dataset_manifest) == len(expected_dataset_keys)
            and observed_dataset_keys == expected_dataset_keys
            and all(
                row.get("run_id") == K1Q_RUN_ID
                and row.get("phase") == "confirmation"
                and int(row.get("cell", -1)) == 11
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rows", -1))
                == (
                    EXPECTED_TRAIN_ROWS
                    if row.get("split") == "train_seen"
                    else EXPECTED_HOLDOUT_ROWS
                )
                and row.get("cache_payloads_present") is True
                for row in dataset_manifest
            )
        ),
        "two_exact_checkpoint_bindings": (
            len(exact_checkpoints) == len(EXPECTED_SEEDS)
            and observed_checkpoints == EXPECTED_CHECKPOINT_SHAS
            and all(row.get("selected_checkpoint") == "best" for row in exact_checkpoints)
        ),
    }


def extract_k1s_feature_views(
    dataset: DifferentialDataset,
    model: torch.nn.Module,
    *,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("K1-S batch_size must be positive")
    features = np.asarray(dataset.features)
    if features.ndim != 2 or features.shape[1] != EXPECTED_PAIRS * 2 * 64:
        raise ValueError("K1-S requires four 64-bit ciphertext pairs per sample")
    model.eval()
    state_before = tensor_mapping_sha256(model.state_dict())
    chunks = {tap: [] for tap in TAPS}
    logits_bit_exact = True
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.as_tensor(
                np.asarray(features[start:stop]).copy(), dtype=torch.float32
            )
            batch_views, replayed = extract_k1s_batch_taps(batch, model)
            logits_bit_exact = logits_bit_exact and replayed
            for tap in TAPS:
                chunks[tap].append(batch_views[tap])
    views = {
        tap: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for tap, parts in chunks.items()
    }
    state_after = tensor_mapping_sha256(model.state_dict())
    for tap, values in views.items():
        expected = EXPECTED_FEATURE_DIMS[tap]
        if values.shape != (len(features), expected):
            raise ValueError(f"K1-S {tap} feature geometry is invalid")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"K1-S {tap} contains non-finite values")
    return views, {
        "ordinary_logits_bit_exact": logits_bit_exact,
        "state_dict_sha256_before": state_before,
        "state_dict_sha256_after": state_after,
        "state_dict_unchanged": state_before == state_after,
    }


def extract_k1s_batch_taps(
    features: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[dict[str, np.ndarray], bool]:
    backbone = model.backbone
    structure = model.runtime_structure
    if not bool(getattr(model, "apply_sboxes", False)):
        raise ValueError("K1-S requires the exact-composition K1-R model")
    runtime = project_features(features, structure)
    exact_views = exact_operator_composition_views(runtime, structure)
    t0 = _position_histograms(exact_views, structure).reshape(len(features), -1)
    captures: dict[str, Any] = {"updates": []}

    def capture_bit(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        captures["bit_hidden"] = output.detach().clone()

    def capture_initial(
        _module: torch.nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        captures["initial_cells"] = output.detach().clone()

    def capture_update(
        _module: torch.nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        captures["updates"].append(output.detach().clone())

    def capture_pool_input(_module: torch.nn.Module, inputs: Any) -> None:
        captures["invariant_pool"] = inputs[0].detach().clone()

    before = model(features).detach().clone()
    handles = [
        backbone.composition_bit_encoder.register_forward_hook(capture_bit),
        backbone.cell_encoder.register_forward_hook(capture_initial),
        backbone.cell_update_norm.register_forward_hook(capture_update),
        backbone.residual_pair_projection.register_forward_pre_hook(capture_pool_input),
    ]
    try:
        backbone.edge_residual_embedding(
            runtime,
            structure,
            apply_sboxes=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    after = model(features).detach()
    if not torch.equal(before, after):
        raise ValueError("K1-S introspection changed ordinary model logits")
    required = {"bit_hidden", "initial_cells", "invariant_pool"}
    if not required.issubset(captures) or len(captures["updates"]) != 2:
        raise ValueError("K1-S failed to capture the exact residual path")

    batch = len(features)
    pairs = EXPECTED_PAIRS
    bit_hidden = captures["bit_hidden"]
    initial_cells = captures["initial_cells"]
    topology_delta = captures["updates"][-1] - initial_cells
    invariant = captures["invariant_pool"]
    return {
        TAPS[0]: np.asarray(t0, dtype=np.float32),
        TAPS[1]: bit_hidden.reshape(batch, -1).cpu().numpy(),
        TAPS[2]: topology_delta.reshape(batch, pairs, -1).reshape(batch, -1).cpu().numpy(),
        TAPS[3]: invariant.reshape(batch, pairs, -1).reshape(batch, -1).cpu().numpy(),
    }, True


def evaluate_k1s(
    *,
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    models: Mapping[int, torch.nn.Module],
    checkpoint_bindings: Mapping[int, Mapping[str, Any]],
    k1q_feature_rows: Sequence[Mapping[str, Any]],
    k1q_scorer_rows: Sequence[Mapping[str, Any]],
    k1q_result_rows: Sequence[Mapping[str, Any]],
    batch_size: int = 256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_datasets = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets or set(models) != set(EXPECTED_SEEDS):
        raise ValueError("K1-S requires both seeds and all three frozen splits")
    references = k1q_reference_maps(
        k1q_feature_rows,
        k1q_scorer_rows,
        k1q_result_rows,
    )
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        split_views: dict[str, dict[str, np.ndarray]] = {}
        checkpoint = checkpoint_bindings[seed]
        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            views, introspection = extract_k1s_feature_views(
                dataset, models[seed], batch_size=batch_size
            )
            split_views[split] = views
            dataset_sha = differential_dataset_sha256(dataset)
            for tap, values in views.items():
                reference = references["features"].get((seed, split)) if tap == TAPS[0] else None
                feature_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": "uknit64",
                        "rounds": 5,
                        "seed": seed,
                        "split": split,
                        "tap": tap,
                        "rows": int(len(values)),
                        "feature_dim": int(values.shape[1]),
                        "feature_sha256": numpy_array_sha256(values),
                        "dataset_sha256": dataset_sha,
                        "checkpoint_sha256": (
                            None if tap == TAPS[0] else checkpoint["sha256"]
                        ),
                        "source_feature_sha256": (
                            None if reference is None else reference.get("feature_sha256")
                        ),
                        "finite": bool(np.all(np.isfinite(values))),
                        "ordinary_logits_bit_exact": introspection[
                            "ordinary_logits_bit_exact"
                        ],
                        "state_dict_sha256_before": introspection[
                            "state_dict_sha256_before"
                        ],
                        "state_dict_sha256_after": introspection[
                            "state_dict_sha256_after"
                        ],
                        "state_dict_unchanged": introspection["state_dict_unchanged"],
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )

        train_labels = np.asarray(
            datasets[(seed, "train_seen")].labels, dtype=np.uint8
        )
        for tap_index, tap in enumerate(TAPS):
            for mode in SCORER_MODES:
                if mode == "label_shuffle":
                    shuffle_seed = (
                        20_260_728 + seed * 100 + 11
                        if tap == TAPS[0]
                        else 30_000_000 + seed * 100 + tap_index
                    )
                    fit_labels, permutation_sha = deterministic_label_shuffle(
                        train_labels,
                        seed=shuffle_seed,
                    )
                else:
                    fit_labels = train_labels
                    permutation_sha = None
                scorer = fit_diagonal_fisher(
                    split_views["train_seen"][tap], fit_labels
                )
                reference_scorer = (
                    references["scorers"].get((seed, mode))
                    if tap == TAPS[0]
                    else None
                )
                scorer_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": "uknit64",
                        "seed": seed,
                        "tap": tap,
                        "mode": mode,
                        "fit_split": "train_seen",
                        "fit_rows": int(len(train_labels)),
                        "feature_dim": int(scorer.weights.shape[0]),
                        "variance_floor": scorer.variance_floor,
                        "class0_rows": scorer.class_counts[0],
                        "class1_rows": scorer.class_counts[1],
                        "weight_l2_norm": float(np.linalg.norm(scorer.weights)),
                        "nonzero_weight_count": int(np.count_nonzero(scorer.weights)),
                        "scorer_sha256": scorer.sha256,
                        "label_permutation_sha256": permutation_sha,
                        "source_scorer_sha256": (
                            None
                            if reference_scorer is None
                            else reference_scorer.get("scorer_sha256")
                        ),
                        "label_assignment_changed": (
                            mode == "interpreted"
                            or not np.array_equal(fit_labels, train_labels)
                        ),
                        "class_counts_preserved": bool(
                            np.array_equal(np.sort(fit_labels), np.sort(train_labels))
                        ),
                        "checkpoint_sha256": (
                            None if tap == TAPS[0] else checkpoint["sha256"]
                        ),
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
                for split in EXPECTED_SPLITS:
                    dataset = datasets[(seed, split)]
                    labels = np.asarray(dataset.labels, dtype=np.uint8)
                    values = split_views[split][tap]
                    scores = scorer.score(values)
                    source_result = (
                        references["results"].get((seed, split, mode))
                        if tap == TAPS[0]
                        else None
                    )
                    result_rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": "uknit64",
                            "rounds": 5,
                            "seed": seed,
                            "split": split,
                            "tap": tap,
                            "mode": mode,
                            "rows": int(len(labels)),
                            "auc": binary_auc(labels, scores),
                            "zero_threshold_accuracy": float(
                                ((scores >= 0.0).astype(np.uint8) == labels).mean()
                            ),
                            "score_mean": float(scores.mean()),
                            "score_std": float(scores.std()),
                            "feature_dim": int(values.shape[1]),
                            "feature_sha256": numpy_array_sha256(values),
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "scorer_sha256": scorer.sha256,
                            "checkpoint_sha256": (
                                None if tap == TAPS[0] else checkpoint["sha256"]
                            ),
                            "source_auc": (
                                None if source_result is None else source_result.get("auc")
                            ),
                            "source_feature_sha256": (
                                None
                                if source_result is None
                                else source_result.get("feature_sha256")
                            ),
                            "source_scorer_sha256": (
                                None
                                if source_result is None
                                else source_result.get("scorer_sha256")
                            ),
                            "fit_split": "train_seen",
                            "fit_rows": int(len(train_labels)),
                            "pairs_per_sample": EXPECTED_PAIRS,
                            "negative_mode": "encrypted_random_plaintexts",
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return feature_rows, scorer_rows, result_rows


def k1q_reference_maps(
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[Any, Mapping[str, Any]]]:
    view_to_mode = {
        K1Q_EXACT_VIEW: "interpreted",
        K1Q_LABEL_SHUFFLE_VIEW: "label_shuffle",
    }
    features = {
        (int(row["seed"]), str(row["split"])): row
        for row in feature_rows
        if row.get("phase") == "confirmation"
        and int(row.get("cell", -1)) == 11
        and row.get("view") == K1Q_EXACT_VIEW
    }
    scorers = {
        (int(row["seed"]), view_to_mode[str(row["view"])]): row
        for row in scorer_rows
        if row.get("phase") == "confirmation"
        and int(row.get("cell", -1)) == 11
        and row.get("view") in view_to_mode
    }
    results = {
        (
            int(row["seed"]),
            str(row["split"]),
            view_to_mode[str(row["view"])],
        ): row
        for row in result_rows
        if row.get("phase") == "confirmation"
        and int(row.get("cell", -1)) == 11
        and row.get("view") in view_to_mode
    }
    if (
        len(features) != len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS)
        or len(scorers) != len(EXPECTED_SEEDS) * len(SCORER_MODES)
        or len(results)
        != len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(SCORER_MODES)
    ):
        raise ValueError("K1-S K1-Q replay references are incomplete")
    return {"features": features, "scorers": scorers, "results": results}


def adjudicate_k1s(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
    k1r_logit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    features = _map_rows(feature_rows, ("seed", "split", "tap"))
    scorers = _map_rows(scorer_rows, ("seed", "tap", "mode"))
    results = _map_rows(result_rows, ("seed", "split", "tap", "mode"))
    expected_features = {
        (seed, split, tap)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for tap in TAPS
    }
    expected_scorers = {
        (seed, tap, mode)
        for seed in EXPECTED_SEEDS
        for tap in TAPS
        for mode in SCORER_MODES
    }
    expected_results = {
        (seed, split, tap, mode)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for tap in TAPS
        for mode in SCORER_MODES
    }
    logit_map = {
        (int(row["seed"]), str(row["split"])): row
        for row in k1r_logit_rows
        if row.get("condition") == "exact_composition"
    }
    t0_features_replay = all(
        row.get("feature_sha256") == row.get("source_feature_sha256")
        for row in feature_rows
        if row.get("tap") == TAPS[0]
    )
    t0_scorers_replay = all(
        row.get("scorer_sha256") == row.get("source_scorer_sha256")
        for row in scorer_rows
        if row.get("tap") == TAPS[0]
    )
    t0_results_replay = all(
        row.get("feature_sha256") == row.get("source_feature_sha256")
        and row.get("scorer_sha256") == row.get("source_scorer_sha256")
        and abs(float(row["auc"]) - float(row["source_auc"])) <= REPLAY_TOLERANCE
        for row in result_rows
        if row.get("tap") == TAPS[0]
    )
    protocol_checks = {
        **dict(source_checks),
        "twenty_four_feature_rows_complete": (
            len(feature_rows) == EXPECTED_FEATURE_ROWS
            and set(features) == expected_features
        ),
        "sixteen_scorer_rows_complete": (
            len(scorer_rows) == EXPECTED_SCORER_ROWS
            and set(scorers) == expected_scorers
        ),
        "forty_eight_result_rows_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(results) == expected_results
        ),
        "t0_features_exactly_replay_k1q": t0_features_replay,
        "t0_scorers_exactly_replay_k1q": t0_scorers_replay,
        "t0_aucs_exactly_replay_k1q": t0_results_replay,
        "feature_dimensions_frozen": all(
            int(row.get("feature_dim", -1)) == EXPECTED_FEATURE_DIMS[row["tap"]]
            for row in feature_rows
        ),
        "introspection_is_read_only": all(
            row.get("ordinary_logits_bit_exact") is True
            and row.get("state_dict_unchanged") is True
            and row.get("state_dict_sha256_before")
            == row.get("state_dict_sha256_after")
            for row in feature_rows
        ),
        "zero_training_and_strict_protocol": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            for row in (*feature_rows, *scorer_rows, *result_rows)
        ),
        "label_shuffle_controls_valid": all(
            row.get("class_counts_preserved") is True
            and row.get("label_assignment_changed") is True
            for row in scorer_rows
        ),
        "finite_features_and_metrics": all(
            row.get("finite") is True for row in feature_rows
        )
        and all(
            math.isfinite(float(row.get("auc", math.nan)))
            and 0.0 <= float(row.get("auc", math.nan)) <= 1.0
            and math.isfinite(float(row.get("score_mean", math.nan)))
            and math.isfinite(float(row.get("score_std", math.nan)))
            for row in result_rows
        ),
        "split_rows_and_benchmark_frozen": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("fit_split") == "train_seen"
            for row in result_rows
        ),
        "six_k1r_logit_references_exact": (
            set(logit_map)
            == {
                (seed, split)
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
            }
        ),
    }

    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        seed_results[str(seed)] = {}
        for split in EXPECTED_SPLITS:
            split_results: dict[str, Any] = {
                "k1r_exact_logit_auc": float(logit_map[(seed, split)]["auc"])
            }
            for tap in TAPS:
                interpreted = float(results[(seed, split, tap, "interpreted")]["auc"])
                shuffled = float(results[(seed, split, tap, "label_shuffle")]["auc"])
                split_results[tap] = {
                    "auc": interpreted,
                    "label_shuffle_auc": shuffled,
                    "minus_label_shuffle": interpreted - shuffled,
                }
            split_results["T2_minus_T3"] = (
                split_results[TAPS[2]]["auc"] - split_results[TAPS[3]]["auc"]
            )
            seed_results[str(seed)][split] = split_results
            if split in FRESH_SPLITS:
                for tap in TAPS:
                    research_checks[f"seed{seed}_{split}_{tap}_accessible"] = (
                        split_results[tap]["auc"] >= AUC_FLOOR
                        and split_results[tap]["minus_label_shuffle"]
                        >= LABEL_SHUFFLE_MARGIN
                    )
                research_checks[f"seed{seed}_{split}_T2_beats_T3"] = (
                    split_results["T2_minus_T3"] >= POSITION_TO_POOL_MARGIN
                )

    tap_accessible = {
        tap: all(
            research_checks[f"seed{seed}_{split}_{tap}_accessible"]
            for seed in EXPECTED_SEEDS
            for split in FRESH_SPLITS
        )
        for tap in TAPS
    }
    t2_beats_t3 = all(
        research_checks[f"seed{seed}_{split}_T2_beats_T3"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1s_protocol_invalid"
        next_action = (
            "repair only the failed K1-S source, checkpoint, tap, replay, or artifact "
            "binding and rerun the unchanged zero-training audit"
        )
    elif tap_accessible[TAPS[2]] and not tap_accessible[TAPS[3]] and t2_beats_t3:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1s_invariant_cell_pool_"
            "bottleneck_supported"
        )
        next_action = (
            "implement one active-relative runtime-topology-derived position-preserving "
            "readout while freezing K1-R data, base branch, budget and controls"
        )
    elif tap_accessible[TAPS[3]]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1s_downstream_residual_fusion_"
            "bottleneck_supported"
        )
        next_action = (
            "retain invariant pooling and isolate only residual pair projection, "
            "bounded-gate fusion and classifier scaling on the frozen evidence"
        )
    elif tap_accessible[TAPS[1]] and not tap_accessible[TAPS[2]]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1s_cell_aggregation_or_update_"
            "bottleneck_supported"
        )
        next_action = (
            "audit only bit-to-cell aggregation versus the two ordered topology update "
            "slots before changing the readout"
        )
    elif not tap_accessible[TAPS[1]] and not tap_accessible[TAPS[2]]:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1s_learned_representation_access_"
            "not_supported"
        )
        next_action = (
            "retain the exact five-stage position histogram as one bounded deterministic "
            "residual and test that single representation change against K1-R controls"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1s_first_destructive_stage_ambiguous"
        )
        next_action = (
            "hold scale and run one unchanged-budget active-cell versus all-cell linear "
            "readout audit; do not change data or architecture family"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "tap_accessible_on_all_fresh_splits": tap_accessible,
        "t2_beats_t3_on_all_fresh_splits": t2_beats_t3,
        "seed_results": seed_results,
        "thresholds": {
            "tap_auc_floor": AUC_FLOOR,
            "tap_minus_label_shuffle": LABEL_SHUFFLE_MARGIN,
            "t2_minus_t3": POSITION_TO_POOL_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local zero-training uKNIT r5 cell11 representation-access "
            "audit; not formal training, attack, SOTA, transfer, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale, more samples, pairs, positions, seeds, or epochs",
            "MoE, DDT/trails, another cipher, or another architecture family",
            "changing K1-R labels, negatives, checkpoints, or fresh splits",
        ],
    }


def _map_rows(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    mapped: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(int(row[name]) if name == "seed" else str(row[name]) for name in fields)
        if key in mapped:
            raise ValueError(f"duplicate K1-S row: {key}")
        mapped[key] = row
    return mapped


__all__ = [
    "AUC_FLOOR",
    "EXPECTED_CHECKPOINT_SHAS",
    "EXPECTED_FEATURE_DIMS",
    "EXPECTED_SOURCE_DIGESTS",
    "LABEL_SHUFFLE_MARGIN",
    "POSITION_TO_POOL_MARGIN",
    "RUN_ID",
    "SCORER_MODES",
    "TAPS",
    "adjudicate_k1s",
    "evaluate_k1s",
    "extract_k1s_batch_taps",
    "extract_k1s_feature_views",
    "k1q_reference_maps",
    "source_binding_checks",
]
