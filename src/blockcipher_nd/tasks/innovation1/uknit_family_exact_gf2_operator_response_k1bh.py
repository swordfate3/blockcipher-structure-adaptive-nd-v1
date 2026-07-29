from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.exact_gf2_operator_response import (
    extract_exact_gf2_operator_features,
    response_feature_dim,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    fit_diagonal_fisher,
    numpy_array_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_deterministic_token_edge_basis_k1bg import (
    load_and_validate_config as load_k1bg_config,
    load_authority as load_k1bg_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bc import (
    load_and_validate_config as load_k1bc_config,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_exact_gf2_operator_response_k1bh_audit_"
    "replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_exact_gf2_operator_response_k1bh_audit_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "9a5357fd5cec70b90387192c1468c17c4930d8f39deb9d32b3205dccb3c1ea37"
)
REPLICAS = (0, 1)
SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
OPERATOR_CONDITIONS = (
    "correct_operator",
    "same_summary_corrupted_operator",
    "cross_cipher_operator",
    "identity_operator",
)
RESULT_CONDITIONS = (*OPERATOR_CONDITIONS, "label_shuffled_correct_operator")
SCORER_CONDITIONS = ("correct_operator", "label_shuffled_correct_operator")
WRONG_CONDITIONS = OPERATOR_CONDITIONS[1:]
EXPECTED_FEATURE_ROWS = 72
EXPECTED_SCORER_ROWS = 12
EXPECTED_RESULT_ROWS = 60
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_FRESH_ROWS = 2048
EXPECTED_PAIRS = 4
FEATURE_BATCH_SIZE = 256
VARIANCE_FLOOR = 1e-6


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BH config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BH identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_exact_gf2_operator_response_k1bh_audit"
    ):
        raise ValueError("K1-BH experiment name drifted")
    if config.get("feature") != {
        "raw_channels_per_bit": 3,
        "response_channels_per_bit": 12,
        "views": [
            "raw",
            "inverse_linear_0",
            "inverse_linear_1",
            "composed_1_then_0",
        ],
        "pair_reduction": "mean",
        "native_bit_order_preserved": True,
        "cell_and_bit_role_coordinates_preserved": True,
        "sbox_semantics_used": False,
        "expected_feature_dims": {
            "uknit64": 768,
            "midori64": 768,
            "dialga128": 1536,
        },
    }:
        raise ValueError("K1-BH feature contract drifted")
    probe = config.get("probe", {})
    if (
        probe.get("family") != "diagonal_fisher"
        or probe.get("variance_floor") != VARIANCE_FLOOR
        or probe.get("fit_condition") != "correct_operator"
        or probe.get("counterfactuals_reuse_correct_fit") is not True
        or probe.get("fit_rows") != EXPECTED_TRAIN_ROWS
        or probe.get("fresh_rows") != EXPECTED_FRESH_ROWS
    ):
        raise ValueError("K1-BH probe contract drifted")
    if config.get("evaluation") != {
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "conditions": list(RESULT_CONDITIONS),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "pairs_per_sample": EXPECTED_PAIRS,
        "neural_training_performed": False,
        "optimizer_steps": 0,
        "data_generation": False,
        "device": "cpu",
        "execution": "local_audit",
    }:
        raise ValueError("K1-BH evaluation contract drifted")
    gates = config.get("gates", {})
    if gates != {
        "correct_operator_auc_min": 0.55,
        "correct_minus_identity_auc_min": 0.01,
        "correct_minus_each_wrong_operator_auc_min": 0.01,
        "correct_minus_label_shuffle_auc_min": 0.03,
        "label_shuffle_auc_max": 0.53,
        "minimum_nonzero_response_rms": 0.0,
        "require_every_replica_cipher_fresh_split": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BH gate contract drifted")
    expected_shuffle_keys = {
        f"replica{replica}" for replica in REPLICAS
    }
    shuffle_seeds = probe.get("label_shuffle_seeds", {})
    if set(shuffle_seeds) != expected_shuffle_keys or any(
        set(shuffle_seeds[f"replica{replica}"]) != set(EXPECTED_CIPHERS)
        for replica in REPLICAS
    ):
        raise ValueError("K1-BH label-shuffle seed contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, RuntimeSpnStructure],
    Mapping[str, RuntimeSpnStructure],
    Mapping[str, RuntimeSpnStructure],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_config_path = project_root / str(source["config"])
    source_config = load_k1bg_config(source_config_path)
    (
        _runtime_config,
        dataset_rows,
        datasets,
        structures,
        _summaries,
        _source_checkpoints,
        corrupted_structures,
        cross_operators,
        _source_panels,
        inherited_checks,
    ) = load_k1bg_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    paths = {name: source_root / name for name in source["digests"]}
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    results = _read_jsonl(paths["results.jsonl"])
    panels = _read_jsonl(paths["panel_results.jsonl"])
    summary = _read_json(paths["summary.json"])
    geometry = _read_json(paths["geometry.json"])
    expected_datasets = {
        (cipher, seed, split)
        for replica in load_k1bc_config()["replicas"]
        for cipher, seed in replica["dataset_seeds"].items()
        for split in SPLITS
    }
    checks = {
        "source_config_digest_exact": (
            file_sha256(source_config_path) == source["config_sha256"]
        ),
        "all_six_k1bg_artifact_digests_exact": len(paths) == 6
        and all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "k1bg_clean_hold_requires_different_primitive": (
            gate.get("status") == "hold"
            and gate.get("decision") == source["required_decision"]
            and gate.get("whole_path_retention_all") is True
            and gate.get("topology_share_lift_all") is False
            and not gate.get("failed_protocol_checks")
            and not gate.get("failed_compatibility_checks")
        ),
        "k1bg_validation_passed": (
            validation.get("status") == "pass" and not validation.get("errors")
        ),
        "k1bg_results_panels_summary_geometry_complete": (
            len(results) == 3
            and len(panels) == 12
            and summary.get("decision") == source["required_decision"]
            and len(geometry.get("rows", [])) == 6
        ),
        "eighteen_bound_datasets_exact": (
            len(dataset_rows) == len(datasets) == 18
            and set(datasets) == expected_datasets
        ),
        **{f"k1bg_{name}": bool(value) for name, value in inherited_checks.items()},
    }
    return (
        dataset_rows,
        datasets,
        structures,
        corrupted_structures,
        cross_operators,
        checks,
    )


def identity_operator_structure(
    structure: RuntimeSpnStructure,
) -> RuntimeSpnStructure:
    identities = torch.eye(structure.block_bits, dtype=torch.uint8).repeat(
        structure.rounds,
        1,
        1,
    )
    return RuntimeSpnStructure(
        cell_membership=structure.cell_membership,
        bit_role=structure.bit_role,
        sbox_truth_bits=structure.sbox_truth_bits,
        linear_matrices=identities,
        inverse_linear_matrices=identities,
    )


def deterministic_label_shuffle(
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    targets = np.asarray(labels, dtype=np.uint8).reshape(-1)
    permutation = np.random.default_rng(seed).permutation(len(targets))
    shuffled = targets[permutation]
    if np.array_equal(shuffled, targets):
        permutation = np.roll(permutation, 1)
        shuffled = targets[permutation]
    if np.array_equal(shuffled, targets):
        raise ValueError("K1-BH label shuffle must change label assignment")
    if not np.array_equal(np.sort(shuffled), np.sort(targets)):
        raise ValueError("K1-BH label shuffle must preserve class counts")
    digest = hashlib.sha256(
        np.asarray(permutation, dtype=np.int64).tobytes()
    ).hexdigest()
    return shuffled, digest


def evaluate_k1bh(
    *,
    config: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, RuntimeSpnStructure],
    corrupted_structures: Mapping[str, RuntimeSpnStructure],
    cross_operators: Mapping[str, RuntimeSpnStructure],
    batch_size: int = FEATURE_BATCH_SIZE,
    feature_extractor: Callable[..., np.ndarray] = extract_exact_gf2_operator_features,
    feature_dim: Callable[[RuntimeSpnStructure], int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size != FEATURE_BATCH_SIZE:
        raise ValueError("K1-BH feature batch size is frozen at 256")
    source_rows = {
        (str(row["cipher_key"]), int(row["seed"]), str(row["split"])): row
        for row in dataset_rows
    }
    replica_configs = {
        int(row["replica"]): row for row in load_k1bc_config()["replicas"]
    }
    feature_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    expected_feature_dim = feature_dim or (
        lambda structure: response_feature_dim(structure.block_bits)
    )

    for replica in REPLICAS:
        for cipher in EXPECTED_CIPHERS:
            seed = int(replica_configs[replica]["dataset_seeds"][cipher])
            correct_structure = structures[cipher]
            operator_structures = {
                "correct_operator": correct_structure,
                "same_summary_corrupted_operator": corrupted_structures[cipher],
                "cross_cipher_operator": cross_operators[cipher],
                "identity_operator": identity_operator_structure(correct_structure),
            }
            correct_scorer = None
            shuffled_scorer = None
            label_permutation_sha = None
            for split in SPLITS:
                dataset = datasets[(cipher, seed, split)]
                source_row = source_rows[(cipher, seed, split)]
                labels = np.asarray(dataset.labels, dtype=np.uint8).reshape(-1)
                correct_features = feature_extractor(
                    dataset.features,
                    correct_structure,
                    pairs_per_sample=EXPECTED_PAIRS,
                    batch_size=batch_size,
                )
                correct_sha = numpy_array_sha256(correct_features)
                feature_rows.append(
                    _feature_row(
                        replica=replica,
                        cipher=cipher,
                        seed=seed,
                        split=split,
                        condition="correct_operator",
                        features=correct_features,
                        correct_features=correct_features,
                        structure=correct_structure,
                        source_row=source_row,
                        expected_feature_dim=expected_feature_dim(correct_structure),
                    )
                )

                if split == "train_seen":
                    correct_scorer = fit_diagonal_fisher(
                        correct_features,
                        labels,
                        variance_floor=VARIANCE_FLOOR,
                    )
                    shuffle_seed = int(
                        config["probe"]["label_shuffle_seeds"][
                            f"replica{replica}"
                        ][cipher]
                    )
                    shuffled_labels, label_permutation_sha = (
                        deterministic_label_shuffle(labels, seed=shuffle_seed)
                    )
                    shuffled_scorer = fit_diagonal_fisher(
                        correct_features,
                        shuffled_labels,
                        variance_floor=VARIANCE_FLOOR,
                    )
                    scorer_rows.extend(
                        (
                            _scorer_row(
                                replica=replica,
                                cipher=cipher,
                                seed=seed,
                                condition="correct_operator",
                                scorer=correct_scorer,
                                label_permutation_sha=None,
                            ),
                            _scorer_row(
                                replica=replica,
                                cipher=cipher,
                                seed=seed,
                                condition="label_shuffled_correct_operator",
                                scorer=shuffled_scorer,
                                label_permutation_sha=label_permutation_sha,
                            ),
                        )
                    )
                elif split in FRESH_SPLITS:
                    if correct_scorer is None or shuffled_scorer is None:
                        raise ValueError("K1-BH correct scorer must be fitted first")
                    result_rows.append(
                        _result_row(
                            replica=replica,
                            cipher=cipher,
                            seed=seed,
                            rounds=int(source_row["rounds"]),
                            split=split,
                            condition="correct_operator",
                            labels=labels,
                            features=correct_features,
                            feature_sha=correct_sha,
                            dataset_sha=str(source_row["dataset_sha256"]),
                            scorer=correct_scorer,
                            fit_condition="correct_operator",
                        )
                    )
                    result_rows.append(
                        _result_row(
                            replica=replica,
                            cipher=cipher,
                            seed=seed,
                            rounds=int(source_row["rounds"]),
                            split=split,
                            condition="label_shuffled_correct_operator",
                            labels=labels,
                            features=correct_features,
                            feature_sha=correct_sha,
                            dataset_sha=str(source_row["dataset_sha256"]),
                            scorer=shuffled_scorer,
                            fit_condition="label_shuffled_correct_operator",
                        )
                    )

                for condition in WRONG_CONDITIONS:
                    wrong_features = feature_extractor(
                        dataset.features,
                        operator_structures[condition],
                        pairs_per_sample=EXPECTED_PAIRS,
                        batch_size=batch_size,
                    )
                    wrong_sha = numpy_array_sha256(wrong_features)
                    feature_rows.append(
                        _feature_row(
                            replica=replica,
                            cipher=cipher,
                            seed=seed,
                            split=split,
                            condition=condition,
                            features=wrong_features,
                            correct_features=correct_features,
                            structure=operator_structures[condition],
                            source_row=source_row,
                            expected_feature_dim=expected_feature_dim(
                                operator_structures[condition]
                            ),
                        )
                    )
                    if split in FRESH_SPLITS:
                        if correct_scorer is None:
                            raise ValueError("K1-BH correct scorer is missing")
                        result_rows.append(
                            _result_row(
                                replica=replica,
                                cipher=cipher,
                                seed=seed,
                                rounds=int(source_row["rounds"]),
                                split=split,
                                condition=condition,
                                labels=labels,
                                features=wrong_features,
                                feature_sha=wrong_sha,
                                dataset_sha=str(source_row["dataset_sha256"]),
                                scorer=correct_scorer,
                                fit_condition="correct_operator",
                            )
                        )
    return feature_rows, scorer_rows, result_rows


def adjudicate_k1bh(
    *,
    config: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    features = _map_rows(feature_rows, include_split=True)
    scorers = _map_rows(scorer_rows, include_split=False)
    results = _map_rows(result_rows, include_split=True)
    expected_features = {
        (replica, cipher, split, condition)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in SPLITS
        for condition in OPERATOR_CONDITIONS
    }
    expected_scorers = {
        (replica, cipher, condition)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for condition in SCORER_CONDITIONS
    }
    expected_results = {
        (replica, cipher, split, condition)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for condition in RESULT_CONDITIONS
    }
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "seventy_two_feature_manifests_complete": (
            len(feature_rows) == EXPECTED_FEATURE_ROWS
            and set(features) == expected_features
        ),
        "twelve_scorers_complete": (
            len(scorer_rows) == EXPECTED_SCORER_ROWS
            and set(scorers) == expected_scorers
        ),
        "sixty_fresh_results_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(results) == expected_results
        ),
        "feature_dimensions_and_row_counts_exact": all(
            int(row.get("feature_dim", -1))
            == int(config["feature"]["expected_feature_dims"][row["cipher_key"]])
            and int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_FRESH_ROWS
            )
            for row in feature_rows
        ),
        "features_finite_and_operator_responses_distinct": all(
            row.get("finite") is True
            and (
                float(row.get("response_rms_from_correct", math.nan)) == 0.0
                if row.get("condition") == "correct_operator"
                else float(row.get("response_rms_from_correct", 0.0))
                > float(config["gates"]["minimum_nonzero_response_rms"])
            )
            for row in feature_rows
        ),
        "correct_fit_scorer_reused_for_all_operator_controls": (
            set(results) == expected_results
            and all(
                results[(replica, cipher, split, condition)].get("scorer_sha256")
                == results[(replica, cipher, split, "correct_operator")].get(
                    "scorer_sha256"
                )
                and results[(replica, cipher, split, condition)].get(
                    "fit_condition"
                )
                == "correct_operator"
                for replica in REPLICAS
                for cipher in EXPECTED_CIPHERS
                for split in FRESH_SPLITS
                for condition in OPERATOR_CONDITIONS
            )
        ),
        "label_shuffle_reuses_correct_features_and_preserves_counts": (
            set(results) == expected_results
            and set(scorers) == expected_scorers
            and all(
                results[(replica, cipher, split, "label_shuffled_correct_operator")]
                .get("feature_sha256")
                == results[(replica, cipher, split, "correct_operator")].get(
                    "feature_sha256"
                )
                and scorers[(replica, cipher, "label_shuffled_correct_operator")]
                .get("class_counts")
                == scorers[(replica, cipher, "correct_operator")].get(
                    "class_counts"
                )
                and bool(
                    scorers[(replica, cipher, "label_shuffled_correct_operator")]
                    .get("label_permutation_sha256")
                )
                for replica in REPLICAS
                for cipher in EXPECTED_CIPHERS
                for split in FRESH_SPLITS
            )
        ),
        "same_dataset_held_across_operator_conditions": (
            set(results) == expected_results
            and all(
                len(
                    {
                        results[(replica, cipher, split, condition)].get(
                            "dataset_sha256"
                        )
                        for condition in RESULT_CONDITIONS
                    }
                )
                == 1
                for replica in REPLICAS
                for cipher in EXPECTED_CIPHERS
                for split in FRESH_SPLITS
            )
        ),
        "closed_form_only_zero_neural_updates": all(
            row.get("training_performed") is False
            and int(row.get("neural_parameter_count", -1)) == 0
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            and row.get("variance_floor") == VARIANCE_FLOOR
            for row in (*scorer_rows, *result_rows)
        ),
        "all_metrics_finite": all(
            all(
                math.isfinite(float(row.get(field, math.nan)))
                for field in (
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

    panels: list[dict[str, Any]] = []
    research_checks: dict[str, bool] = {}
    if set(results) == expected_results:
        for replica in REPLICAS:
            for cipher in EXPECTED_CIPHERS:
                for split in FRESH_SPLITS:
                    aucs = {
                        condition: float(
                            results[(replica, cipher, split, condition)]["auc"]
                        )
                        for condition in RESULT_CONDITIONS
                    }
                    correct = aucs["correct_operator"]
                    panel = {
                        "replica": replica,
                        "cipher_key": cipher,
                        "split": split,
                        "correct_auc": correct,
                        "same_summary_wrong_auc": aucs[
                            "same_summary_corrupted_operator"
                        ],
                        "cross_cipher_wrong_auc": aucs["cross_cipher_operator"],
                        "identity_auc": aucs["identity_operator"],
                        "label_shuffle_auc": aucs[
                            "label_shuffled_correct_operator"
                        ],
                        "correct_minus_same_summary_wrong": correct
                        - aucs["same_summary_corrupted_operator"],
                        "correct_minus_cross_cipher_wrong": correct
                        - aucs["cross_cipher_operator"],
                        "correct_minus_identity": correct - aucs["identity_operator"],
                        "correct_minus_label_shuffle": correct
                        - aucs["label_shuffled_correct_operator"],
                    }
                    panels.append(panel)
                    prefix = f"replica{replica}_{cipher}_{split}"
                    research_checks[f"{prefix}_correct_auc_floor"] = (
                        correct >= float(config["gates"]["correct_operator_auc_min"])
                    )
                    research_checks[f"{prefix}_beats_identity"] = (
                        panel["correct_minus_identity"]
                        >= float(config["gates"]["correct_minus_identity_auc_min"])
                    )
                    research_checks[f"{prefix}_beats_same_summary_wrong"] = (
                        panel["correct_minus_same_summary_wrong"]
                        >= float(
                            config["gates"][
                                "correct_minus_each_wrong_operator_auc_min"
                            ]
                        )
                    )
                    research_checks[f"{prefix}_beats_cross_cipher_wrong"] = (
                        panel["correct_minus_cross_cipher_wrong"]
                        >= float(
                            config["gates"][
                                "correct_minus_each_wrong_operator_auc_min"
                            ]
                        )
                    )
                    research_checks[f"{prefix}_beats_label_shuffle"] = (
                        panel["correct_minus_label_shuffle"]
                        >= float(
                            config["gates"]["correct_minus_label_shuffle_auc_min"]
                        )
                    )
                    research_checks[f"{prefix}_label_shuffle_near_chance"] = (
                        panel["label_shuffle_auc"]
                        <= float(config["gates"]["label_shuffle_auc_max"])
                    )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    correct_signal_all = bool(panels) and all(
        panel["correct_auc"]
        >= float(config["gates"]["correct_operator_auc_min"])
        for panel in panels
    )
    shuffle_attribution_all = bool(panels) and all(
        panel["correct_minus_label_shuffle"]
        >= float(config["gates"]["correct_minus_label_shuffle_auc_min"])
        and panel["label_shuffle_auc"]
        <= float(config["gates"]["label_shuffle_auc_max"])
        for panel in panels
    )
    shuffle_two_sided_checks = {
        f"replica{panel['replica']}|{panel['cipher_key']}|{panel['split']}": (
            abs(float(panel["label_shuffle_auc"]) - 0.5) <= 0.03
        )
        for panel in panels
    }
    diagnostic_checks = {
        "label_shuffle_auc_within_symmetric_chance_band": bool(
            shuffle_two_sided_checks
        )
        and all(shuffle_two_sided_checks.values())
    }
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bh_protocol_invalid"
        next_action = (
            "Repair only the failed source, exact-response, scorer-reuse or artifact "
            "binding and rerun the frozen audit."
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bh_exact_operator_topology_signal_supported"
        )
        next_action = (
            "Preregister K1-BI readiness with one variable: feed the ordered exact "
            "GF(2)-transported states into a shared position-preserving residual; "
            "retain the same data, four pairs, replicas and operator controls."
        )
    elif not correct_signal_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bh_exact_operator_signal_unstable"
        next_action = (
            "Preregister K1-BI with one representation variable: replace independent "
            "bit-response means by runtime-cell 4-bit categorical response histograms. "
            "Reuse the same data, four pairs, replicas, Fisher protocol and operators; "
            "replace the exposed one-sided shuffle gate by the symmetric 0.47-0.53 "
            "chance band, and do not repeat the completed uKNIT cell11 position audit."
        )
    elif not shuffle_attribution_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bh_shuffle_attribution_not_supported"
        next_action = (
            "Audit the frozen Fisher orientation and shuffle control; do not design "
            "or train another network until label attribution is valid."
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1bh_predictive_but_not_topology_identifying"
        )
        next_action = (
            "Audit representation equivalence and wrong-operator construction before "
            "architecture work; the exact response predicts but does not identify the "
            "correct topology on every frozen panel."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": dict(config["gates"]),
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "diagnostic_checks": diagnostic_checks,
        "shuffle_two_sided_checks": shuffle_two_sided_checks,
        "diagnostic_warnings": (
            []
            if all(diagnostic_checks.values())
            else [
                "The preregistered one-sided label-shuffle AUC <= 0.53 gate "
                "accepts strongly reversed AUC; future gates must use 0.47-0.53."
            ]
        ),
        "panels": panels,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Two-replica, three-cipher local deterministic exact-GF(2) response "
            "audit on 4096 train and 2048 fresh total rows per panel with four "
            "pairs; not neural training, formal scale, an attack, SOTA, arbitrary-"
            "SPN generalization or a model improvement."
        ),
    }


def run_audit(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("K1-BH is a frozen local CPU audit")
    _require_fresh_output_root(output_root)
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    (
        dataset_rows,
        datasets,
        structures,
        corrupted_structures,
        cross_operators,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BH source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "device": device,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    rebound_rows = [
        {**row, "source_run_id": row.get("run_id"), "run_id": RUN_ID}
        for row in dataset_rows
    ]
    _write_jsonl(output_root / "dataset_manifest.jsonl", rebound_rows)
    _append_progress(
        output_root / "progress.jsonl",
        "exact_operator_response_start",
        expected_feature_rows=EXPECTED_FEATURE_ROWS,
        expected_scorer_rows=EXPECTED_SCORER_ROWS,
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    feature_rows, scorer_rows, result_rows = evaluate_k1bh(
        config=config,
        dataset_rows=dataset_rows,
        datasets=datasets,
        structures=structures,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
    )
    gate = adjudicate_k1bh(
        config=config,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_checks=source_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "feature_rows": len(feature_rows),
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "scorer_rows": len(scorer_rows),
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "remote_scale": gate["remote_scale"],
        "panels": gate["panels"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
        "feature_rows": len(feature_rows),
        "scorer_rows": len(scorer_rows),
        "result_rows": len(result_rows),
        "optimizer_steps": 0,
    }
    _write_jsonl(output_root / "feature_manifest.jsonl", feature_rows)
    _write_jsonl(output_root / "scorers.jsonl", scorer_rows)
    _write_jsonl(output_root / "results.jsonl", result_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
    )
    return {
        "preflight": preflight,
        "features": feature_rows,
        "scorers": scorer_rows,
        "results": result_rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _feature_row(
    *,
    replica: int,
    cipher: str,
    seed: int,
    split: str,
    condition: str,
    features: np.ndarray,
    correct_features: np.ndarray,
    structure: RuntimeSpnStructure,
    source_row: Mapping[str, Any],
    expected_feature_dim: int,
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float32)
    correct = np.asarray(correct_features, dtype=np.float32)
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "seed": seed,
        "rounds": int(source_row["rounds"]),
        "split": split,
        "condition": condition,
        "rows": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "expected_feature_dim": expected_feature_dim,
        "feature_sha256": numpy_array_sha256(values),
        "dataset_sha256": str(source_row["dataset_sha256"]),
        "operator_sha256": _operator_sha256(structure),
        "response_rms_from_correct": float(
            np.sqrt(np.mean(np.square(values - correct), dtype=np.float64))
        ),
        "finite": bool(np.all(np.isfinite(values))),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "pairs_per_sample": EXPECTED_PAIRS,
        "data_generation_performed": False,
    }


def _scorer_row(
    *,
    replica: int,
    cipher: str,
    seed: int,
    condition: str,
    scorer: Any,
    label_permutation_sha: str | None,
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "seed": seed,
        "condition": condition,
        "fit_condition": condition,
        "fit_split": "train_seen",
        "fit_rows": EXPECTED_TRAIN_ROWS,
        "feature_dim": int(scorer.weights.shape[0]),
        "variance_floor": scorer.variance_floor,
        "class_counts": list(scorer.class_counts),
        "weight_l2_norm": float(np.linalg.norm(scorer.weights)),
        "nonzero_weight_count": int(np.count_nonzero(scorer.weights)),
        "scorer_sha256": scorer.sha256,
        "label_permutation_sha256": label_permutation_sha,
        "training_performed": False,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }


def _result_row(
    *,
    replica: int,
    cipher: str,
    seed: int,
    rounds: int,
    split: str,
    condition: str,
    labels: np.ndarray,
    features: np.ndarray,
    feature_sha: str,
    dataset_sha: str,
    scorer: Any,
    fit_condition: str,
) -> dict[str, Any]:
    scores = scorer.score(features)
    targets = np.asarray(labels, dtype=np.uint8).reshape(-1)
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "rounds": rounds,
        "seed": seed,
        "split": split,
        "condition": condition,
        "rows": int(len(targets)),
        "auc": float(binary_auc(targets, scores)),
        "zero_threshold_accuracy": float(
            ((scores >= 0.0).astype(np.uint8) == targets).mean()
        ),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "feature_dim": int(features.shape[1]),
        "feature_sha256": feature_sha,
        "dataset_sha256": dataset_sha,
        "scorer_sha256": scorer.sha256,
        "fit_condition": fit_condition,
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


def _map_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_split: bool,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    mapped: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            (int(row["replica"]), str(row["cipher_key"]), str(row["split"]), str(row["condition"]))
            if include_split
            else (int(row["replica"]), str(row["cipher_key"]), str(row["condition"]))
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-BH row: {key}")
        mapped[key] = row
    return mapped


def _operator_sha256(structure: RuntimeSpnStructure) -> str:
    values = np.ascontiguousarray(
        structure.inverse_linear_matrices.detach().cpu().numpy(),
        dtype=np.uint8,
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"K1-BH output already exists: {path}")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_FEATURE_ROWS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SCORER_ROWS",
    "OPERATOR_CONDITIONS",
    "RESULT_CONDITIONS",
    "RUN_ID",
    "SCORER_CONDITIONS",
    "WRONG_CONDITIONS",
    "adjudicate_k1bh",
    "deterministic_label_shuffle",
    "evaluate_k1bh",
    "identity_operator_structure",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
