from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.models.structure.spn.cell_joint_gf2_operator_response import (
    cell_joint_response_feature_dim,
    extract_cell_joint_gf2_operator_features,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.uknit_family_cell_joint_gf2_operator_response_k1bi import (
    load_and_validate_config as load_k1bi_config,
    load_authority as load_k1bi_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    fit_diagonal_fisher,
    numpy_array_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    EXPECTED_FRESH_ROWS,
    EXPECTED_PAIRS,
    EXPECTED_TRAIN_ROWS,
    FEATURE_BATCH_SIZE,
    FRESH_SPLITS,
    REPLICAS,
    SPLITS,
    deterministic_label_shuffle,
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
    "i1_uknit_family_multishuffle_cell_joint_null_k1bj_audit_"
    "replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multishuffle_cell_joint_null_"
    "k1bj_audit_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "9b370b461d145336a3cf262fe996823b6ba1211938ba822667fab179d4a8d102"
)
PERMUTATIONS = tuple(range(31))
EXPECTED_FEATURE_ROWS = 18
EXPECTED_SCORER_ROWS = 192
EXPECTED_RESULT_ROWS = 384
VARIANCE_FLOOR = 1e-6


def permutation_seed(replica: int, cipher: str, permutation_index: int) -> int:
    if replica not in REPLICAS:
        raise ValueError("K1-BJ replica is not frozen")
    if cipher not in EXPECTED_CIPHERS:
        raise ValueError("K1-BJ cipher is not frozen")
    if permutation_index not in PERMUTATIONS:
        raise ValueError("K1-BJ permutation index is not frozen")
    return (
        84100
        + replica * 10000
        + list(EXPECTED_CIPHERS).index(cipher) * 100
        + permutation_index
    )


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BJ config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BJ identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multishuffle_cell_joint_null_k1bj_audit"
    ):
        raise ValueError("K1-BJ experiment name drifted")
    null = config.get("null", {})
    if null != {
        "permutations_per_replica_cipher": 31,
        "permutation_indices": list(PERMUTATIONS),
        "seed_formula": (
            "84100 + replica*10000 + cipher_index*100 + permutation_index"
        ),
        "cipher_order": list(EXPECTED_CIPHERS),
        "statistic": "abs(auc_minus_0.5)",
        "empirical_p": "(1 + count(null_strength >= correct_strength)) / 32",
        "quantile_method": "higher",
        "reuse_k1bi_cell_joint_features": True,
        "feature_or_benchmark_change": False,
    }:
        raise ValueError("K1-BJ null contract drifted")
    if config.get("probe") != {
        "family": "diagonal_fisher",
        "variance_floor": VARIANCE_FLOOR,
        "fit_rows": EXPECTED_TRAIN_ROWS,
        "fresh_rows": EXPECTED_FRESH_ROWS,
        "pairs_per_sample": EXPECTED_PAIRS,
        "neural_training_performed": False,
        "optimizer_steps": 0,
        "data_generation": False,
        "device": "cpu",
    }:
        raise ValueError("K1-BJ probe contract drifted")
    if config.get("evaluation") != {
        "replicas": list(REPLICAS),
        "ciphers": list(EXPECTED_CIPHERS),
        "splits": list(FRESH_SPLITS),
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "execution": "local_audit",
    }:
        raise ValueError("K1-BJ evaluation contract drifted")
    if config.get("gates") != {
        "correct_replay_auc_tolerance": 1e-7,
        "attributed_ciphers": ["midori64", "dialga128"],
        "empirical_p_max": 0.05,
        "correct_strength_minus_null_q95_min": 0.10,
        "uknit_correct_auc_max_exclusive": 0.55,
        "require_every_replica_cipher_fresh_split": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BJ gate contract drifted")
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
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, bool],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_config_path = project_root / str(source["config"])
    source_config = load_k1bi_config(source_config_path)
    (
        dataset_rows,
        datasets,
        structures,
        _corrupted,
        _cross,
        _k1bh_gate,
        inherited_checks,
    ) = load_k1bi_authority(
        source_config,
        project_root=project_root,
        device=device,
    )
    paths = {name: source_root / name for name in source["digests"]}
    source_gate = _read_json(paths["gate.json"])
    source_validation = _read_json(paths["validation.json"])
    source_results = _read_jsonl(paths["results.jsonl"])
    source_features = _read_jsonl(paths["feature_manifest.jsonl"])
    source_scorers = _read_jsonl(paths["scorers.jsonl"])
    source_summary = _read_json(paths["summary.json"])
    source_datasets = _read_jsonl(paths["dataset_manifest.jsonl"])
    visual_report = _read_json(paths["visual_qa_render_report.json"])
    visual_marker = paths["visual_qa_passed.marker"].read_text(encoding="utf-8")
    checks = {
        "source_config_digest_exact": (
            file_sha256(source_config_path) == source["config_sha256"]
        ),
        "all_ten_k1bi_artifact_digests_exact": len(paths) == 10
        and all(
            path.is_file() and file_sha256(path) == source["digests"][name]
            for name, path in paths.items()
        ),
        "k1bi_clean_shuffle_hold_requires_multishuffle_null": (
            source_gate.get("status") == source["required_status"]
            and source_gate.get("decision") == source["required_decision"]
            and not source_gate.get("failed_protocol_checks")
        ),
        "k1bi_validation_and_visual_qa_passed": (
            source_validation.get("status") == "pass"
            and not source_validation.get("errors")
            and visual_report.get("status") == "pass"
            and "status=pass" in visual_marker
        ),
        "k1bi_artifact_rows_complete": (
            len(source_results) == 60
            and len(source_features) == 72
            and len(source_scorers) == 12
            and len(source_datasets) == 18
            and source_summary.get("decision") == source["required_decision"]
        ),
        **{f"k1bi_{name}": bool(value) for name, value in inherited_checks.items()},
    }
    return dataset_rows, datasets, structures, source_gate, source_results, checks


def evaluate_k1bj(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, RuntimeSpnStructure],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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

    for replica in REPLICAS:
        for cipher in EXPECTED_CIPHERS:
            seed = int(replica_configs[replica]["dataset_seeds"][cipher])
            structure = structures[cipher]
            features: dict[str, np.ndarray] = {}
            labels: dict[str, np.ndarray] = {}
            for split in SPLITS:
                dataset = datasets[(cipher, seed, split)]
                values = extract_cell_joint_gf2_operator_features(
                    dataset.features,
                    structure,
                    pairs_per_sample=EXPECTED_PAIRS,
                    batch_size=FEATURE_BATCH_SIZE,
                )
                targets = np.asarray(dataset.labels, dtype=np.uint8).reshape(-1)
                features[split] = values
                labels[split] = targets
                feature_rows.append(
                    _feature_row(
                        replica=replica,
                        cipher=cipher,
                        seed=seed,
                        split=split,
                        features=values,
                        dataset_sha=str(
                            source_rows[(cipher, seed, split)]["dataset_sha256"]
                        ),
                        cells=structure.cells,
                    )
                )

            correct_scorer = fit_diagonal_fisher(
                features["train_seen"],
                labels["train_seen"],
                variance_floor=VARIANCE_FLOOR,
            )
            scorer_rows.append(
                _scorer_row(
                    replica=replica,
                    cipher=cipher,
                    seed=seed,
                    condition="true_labels",
                    permutation_index=-1,
                    permutation_seed_value=None,
                    permutation_sha=None,
                    scorer=correct_scorer,
                )
            )
            for split in FRESH_SPLITS:
                source_row = source_rows[(cipher, seed, split)]
                result_rows.append(
                    _result_row(
                        replica=replica,
                        cipher=cipher,
                        seed=seed,
                        rounds=int(source_row["rounds"]),
                        split=split,
                        condition="true_labels",
                        permutation_index=-1,
                        labels=labels[split],
                        features=features[split],
                        dataset_sha=str(source_row["dataset_sha256"]),
                        scorer=correct_scorer,
                    )
                )

            for permutation_index in PERMUTATIONS:
                seed_value = permutation_seed(replica, cipher, permutation_index)
                shuffled_labels, permutation_sha = deterministic_label_shuffle(
                    labels["train_seen"],
                    seed=seed_value,
                )
                scorer = fit_diagonal_fisher(
                    features["train_seen"],
                    shuffled_labels,
                    variance_floor=VARIANCE_FLOOR,
                )
                scorer_rows.append(
                    _scorer_row(
                        replica=replica,
                        cipher=cipher,
                        seed=seed,
                        condition="shuffled_labels",
                        permutation_index=permutation_index,
                        permutation_seed_value=seed_value,
                        permutation_sha=permutation_sha,
                        scorer=scorer,
                    )
                )
                for split in FRESH_SPLITS:
                    source_row = source_rows[(cipher, seed, split)]
                    result_rows.append(
                        _result_row(
                            replica=replica,
                            cipher=cipher,
                            seed=seed,
                            rounds=int(source_row["rounds"]),
                            split=split,
                            condition="shuffled_labels",
                            permutation_index=permutation_index,
                            labels=labels[split],
                            features=features[split],
                            dataset_sha=str(source_row["dataset_sha256"]),
                            scorer=scorer,
                        )
                    )
    return feature_rows, scorer_rows, result_rows


def adjudicate_k1bj(
    *,
    config: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    source_gate: Mapping[str, Any],
    source_results: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    feature_map = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"])): row
        for row in feature_rows
    }
    scorer_map = {
        (
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["condition"]),
            int(row["permutation_index"]),
        ): row
        for row in scorer_rows
    }
    result_map = {
        (
            int(row["replica"]),
            str(row["cipher_key"]),
            str(row["split"]),
            str(row["condition"]),
            int(row["permutation_index"]),
        ): row
        for row in result_rows
    }
    source_correct = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"])): row
        for row in source_results
        if row.get("condition") == "correct_operator"
    }
    expected_features = {
        (replica, cipher, split)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in SPLITS
    }
    expected_scorers = {
        (replica, cipher, "true_labels", -1)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
    } | {
        (replica, cipher, "shuffled_labels", permutation_index)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for permutation_index in PERMUTATIONS
    }
    expected_results = {
        (replica, cipher, split, "true_labels", -1)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    } | {
        (replica, cipher, split, "shuffled_labels", permutation_index)
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for permutation_index in PERMUTATIONS
    }
    replay_deltas = {
        f"replica{replica}|{cipher}|{split}": abs(
            float(result_map[(replica, cipher, split, "true_labels", -1)]["auc"])
            - float(source_correct[(replica, cipher, split)]["auc"])
        )
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        if (replica, cipher, split, "true_labels", -1) in result_map
        and (replica, cipher, split) in source_correct
    }
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "eighteen_feature_manifests_complete": (
            len(feature_rows) == EXPECTED_FEATURE_ROWS
            and len(feature_map) == len(feature_rows)
            and set(feature_map) == expected_features
        ),
        "one_hundred_ninety_two_scorers_complete": (
            len(scorer_rows) == EXPECTED_SCORER_ROWS
            and len(scorer_map) == len(scorer_rows)
            and set(scorer_map) == expected_scorers
        ),
        "three_hundred_eighty_four_results_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and len(result_map) == len(result_rows)
            and set(result_map) == expected_results
        ),
        "feature_dimensions_rows_and_datasets_exact": all(
            int(row.get("feature_dim", -1))
            == int(row.get("expected_feature_dim", -2))
            and int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_FRESH_ROWS
            )
            and bool(row.get("dataset_sha256"))
            and row.get("finite") is True
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and row.get("data_generation_performed") is False
            and row.get("representation")
            == "runtime_cell_joint_16_value_histogram"
            for row in feature_rows
        ),
        "thirty_one_distinct_count_preserving_permutations_each": all(
            len(
                {
                    scorer_map[(replica, cipher, "shuffled_labels", index)].get(
                        "label_permutation_sha256"
                    )
                    for index in PERMUTATIONS
                }
            )
            == len(PERMUTATIONS)
            and all(
                scorer_map[(replica, cipher, "shuffled_labels", index)].get(
                    "class_counts"
                )
                == scorer_map[(replica, cipher, "true_labels", -1)].get(
                    "class_counts"
                )
                and int(
                    scorer_map[(replica, cipher, "shuffled_labels", index)].get(
                        "permutation_seed"
                    )
                )
                == permutation_seed(replica, cipher, index)
                for index in PERMUTATIONS
            )
            for replica in REPLICAS
            for cipher in EXPECTED_CIPHERS
        )
        if set(scorer_map) == expected_scorers
        else False,
        "correct_auc_replay_exact": (
            len(replay_deltas) == 12
            and max(replay_deltas.values(), default=math.inf)
            <= float(config["gates"]["correct_replay_auc_tolerance"])
        ),
        "feature_scorer_result_bindings_exact": all(
            row.get("feature_sha256")
            == feature_map[
                (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
            ].get("feature_sha256")
            and row.get("dataset_sha256")
            == feature_map[
                (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
            ].get("dataset_sha256")
            and row.get("scorer_sha256")
            == scorer_map[
                (
                    int(row["replica"]),
                    str(row["cipher_key"]),
                    str(row["condition"]),
                    int(row["permutation_index"]),
                )
            ].get("scorer_sha256")
            and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in result_rows
        )
        if set(feature_map) == expected_features
        and set(scorer_map) == expected_scorers
        and set(result_map) == expected_results
        else False,
        "closed_form_only_zero_neural_updates": all(
            row.get("training_performed") is False
            and int(row.get("neural_parameter_count", -1)) == 0
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            for row in (*scorer_rows, *result_rows)
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and math.isfinite(float(row.get("orientation_invariant_strength", math.nan)))
            for row in result_rows
        ),
    }

    panels: list[dict[str, Any]] = []
    research_checks: dict[str, bool] = {}
    if set(result_map) == expected_results and len(source_correct) == 12:
        for replica in REPLICAS:
            for cipher in EXPECTED_CIPHERS:
                for split in FRESH_SPLITS:
                    correct = result_map[
                        (replica, cipher, split, "true_labels", -1)
                    ]
                    null_strengths = np.asarray(
                        [
                            float(
                                result_map[
                                    (
                                        replica,
                                        cipher,
                                        split,
                                        "shuffled_labels",
                                        permutation_index,
                                    )
                                ]["orientation_invariant_strength"]
                            )
                            for permutation_index in PERMUTATIONS
                        ],
                        dtype=np.float64,
                    )
                    correct_auc = float(correct["auc"])
                    correct_strength = abs(correct_auc - 0.5)
                    q95 = float(np.quantile(null_strengths, 0.95, method="higher"))
                    empirical_p = float(
                        (1 + np.count_nonzero(null_strengths >= correct_strength))
                        / (len(null_strengths) + 1)
                    )
                    panel = {
                        "replica": replica,
                        "cipher_key": cipher,
                        "split": split,
                        "correct_auc": correct_auc,
                        "correct_strength": correct_strength,
                        "null_strength_median": float(np.median(null_strengths)),
                        "null_strength_q95": q95,
                        "null_strength_max": float(null_strengths.max()),
                        "null_strengths": null_strengths.tolist(),
                        "correct_strength_minus_null_q95": correct_strength - q95,
                        "empirical_p": empirical_p,
                        "null_auc_min": float(
                            min(
                                float(
                                    result_map[
                                        (
                                            replica,
                                            cipher,
                                            split,
                                            "shuffled_labels",
                                            permutation_index,
                                        )
                                    ]["auc"]
                                )
                                for permutation_index in PERMUTATIONS
                            )
                        ),
                        "null_auc_max": float(
                            max(
                                float(
                                    result_map[
                                        (
                                            replica,
                                            cipher,
                                            split,
                                            "shuffled_labels",
                                            permutation_index,
                                        )
                                    ]["auc"]
                                )
                                for permutation_index in PERMUTATIONS
                            )
                        ),
                    }
                    panels.append(panel)
                    prefix = f"replica{replica}_{cipher}_{split}"
                    if cipher in config["gates"]["attributed_ciphers"]:
                        research_checks[f"{prefix}_empirical_p"] = (
                            empirical_p <= float(config["gates"]["empirical_p_max"])
                        )
                        research_checks[f"{prefix}_null_q95_margin"] = (
                            correct_strength - q95
                            >= float(
                                config["gates"][
                                    "correct_strength_minus_null_q95_min"
                                ]
                            )
                        )
                    else:
                        research_checks[f"{prefix}_uknit_below_signal_floor"] = (
                            correct_auc
                            < float(
                                config["gates"]["uknit_correct_auc_max_exclusive"]
                            )
                        )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    attributed_checks = [
        value
        for name, value in research_checks.items()
        if name.endswith("_empirical_p") or name.endswith("_null_q95_margin")
    ]
    uknit_checks = [
        value
        for name, value in research_checks.items()
        if name.endswith("_uknit_below_signal_floor")
    ]
    attribution_all = bool(attributed_checks) and all(attributed_checks)
    uknit_boundary_all = bool(uknit_checks) and all(uknit_checks)
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bj_protocol_invalid"
        next_action = (
            "Repair only the failed K1-BI binding, feature replay, permutation, "
            "scorer or artifact invariant and rerun unchanged."
        )
    elif attribution_all and uknit_boundary_all:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bj_linear_transport_boundary_confirmed"
        )
        next_action = (
            "Stop linear-only response redesign. Preregister a runtime S-box-aware "
            "five-stage native-cell primitive using K1-Q/K1-S as the uKNIT anchor "
            "and retain Midori/Dialga exact-transport controls; do not scale yet."
        )
    elif not attribution_all:
        status = "hold"
        decision = "innovation1_uknit_family_k1bj_null_attribution_not_supported"
        next_action = (
            "Audit the fixed Fisher/null mechanism without changing K1-BI features or "
            "benchmark variables; family-wide structure attribution is not supported."
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_k1bj_uknit_boundary_not_confirmed"
        next_action = (
            "Reconcile the exact K1-BI replay before architecture work; a uKNIT panel "
            "unexpectedly crossed the frozen 0.55 signal floor."
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
        "correct_replay_auc_deltas": replay_deltas,
        "panels": panels,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Local 31-permutation orientation-invariant label-null audit on the "
            "unchanged K1-BI cell-joint features, two replicas, three ciphers, two "
            "fresh splits and four pairs; zero neural updates and no benchmark change; "
            "not formal scale, an attack, SOTA or arbitrary-SPN generalization."
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
        raise ValueError("K1-BJ is a frozen local CPU audit")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"K1-BJ output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    (
        dataset_rows,
        datasets,
        structures,
        source_gate,
        source_results,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BJ source binding failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "permutations_per_replica_cipher": len(PERMUTATIONS),
        "device": device,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    _append_progress(
        output_root / "progress.jsonl",
        "multishuffle_null_start",
        expected_feature_rows=EXPECTED_FEATURE_ROWS,
        expected_scorer_rows=EXPECTED_SCORER_ROWS,
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    feature_rows, scorer_rows, result_rows = evaluate_k1bj(
        dataset_rows=dataset_rows,
        datasets=datasets,
        structures=structures,
    )
    gate = adjudicate_k1bj(
        config=config,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_gate=source_gate,
        source_results=source_results,
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
    features: np.ndarray,
    dataset_sha: str,
    cells: int,
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float32)
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "seed": seed,
        "split": split,
        "rows": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "expected_feature_dim": cell_joint_response_feature_dim(cells),
        "feature_sha256": numpy_array_sha256(values),
        "dataset_sha256": dataset_sha,
        "representation": "runtime_cell_joint_16_value_histogram",
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
    permutation_index: int,
    permutation_seed_value: int | None,
    permutation_sha: str | None,
    scorer: Any,
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "seed": seed,
        "condition": condition,
        "permutation_index": permutation_index,
        "permutation_seed": permutation_seed_value,
        "fit_split": "train_seen",
        "fit_rows": EXPECTED_TRAIN_ROWS,
        "feature_dim": int(scorer.weights.shape[0]),
        "variance_floor": scorer.variance_floor,
        "class_counts": list(scorer.class_counts),
        "weight_l2_norm": float(np.linalg.norm(scorer.weights)),
        "nonzero_weight_count": int(np.count_nonzero(scorer.weights)),
        "scorer_sha256": scorer.sha256,
        "label_permutation_sha256": permutation_sha,
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
    permutation_index: int,
    labels: np.ndarray,
    features: np.ndarray,
    dataset_sha: str,
    scorer: Any,
) -> dict[str, Any]:
    scores = scorer.score(features)
    targets = np.asarray(labels, dtype=np.uint8).reshape(-1)
    auc = float(binary_auc(targets, scores))
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "rounds": rounds,
        "seed": seed,
        "split": split,
        "condition": condition,
        "permutation_index": permutation_index,
        "rows": int(len(targets)),
        "auc": auc,
        "orientation_invariant_strength": abs(auc - 0.5),
        "zero_threshold_accuracy": float(
            ((scores >= 0.0).astype(np.uint8) == targets).mean()
        ),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "feature_sha256": numpy_array_sha256(features),
        "dataset_sha256": dataset_sha,
        "scorer_sha256": scorer.sha256,
        "pairs_per_sample": EXPECTED_PAIRS,
        "negative_mode": "encrypted_random_plaintexts",
        "variance_floor": VARIANCE_FLOOR,
        "training_performed": False,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }


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


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_FEATURE_ROWS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SCORER_ROWS",
    "PERMUTATIONS",
    "RUN_ID",
    "adjudicate_k1bj",
    "evaluate_k1bj",
    "load_and_validate_config",
    "load_authority",
    "permutation_seed",
    "run_audit",
]
