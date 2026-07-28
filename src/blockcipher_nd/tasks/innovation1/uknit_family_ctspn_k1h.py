from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import (
    DifferentialDataset,
    DiskDifferentialDataset,
)
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    operator_support_mean,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    ANCHOR_MODEL,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    RUN_ID as K1F_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    EXPECTED_SPLITS,
    RUN_ID as K1G_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


READINESS_RUN_ID = "i1_uknit_family_ctspn_operator_tied_latent_k1h_readiness_20260728"
RUN_ID = "i1_uknit_family_ctspn_operator_tied_latent_k1h_2048_seed0_seed1_20260728"
K1F_DECISION = "innovation1_uknit_family_ctspn_k1f_hypergraph_not_supported"
K1G_DECISION = (
    "innovation1_uknit_family_ctspn_k1g_"
    "sample_specific_hypergraph_attribution_overfit_confirmed"
)
CANDIDATE_MODEL = "runtime_spn_ct_k1h_operator_tied_true"
REVERSED_MODEL = "runtime_spn_ct_k1h_operator_tied_reversed"
CORRUPTED_MODEL = "runtime_spn_ct_k1h_operator_tied_corrupted"
NO_TOPOLOGY_MODEL = "runtime_spn_ct_k1h_operator_tied_none"
CONTROL_CONDITIONS = (
    "exact_ordered",
    "operator_reversed",
    "operator_corrupted",
    "no_topology",
)
ANCHOR_CONDITION = "frozen_runtime_e4_anchor"
EXPECTED_PARAMETER_COUNT = 100450
EXPECTED_PARAMETER_CAP = 150000
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_TRAINING_ROWS = 4
EXPECTED_EVALUATION_ROWS = 60
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
MARGIN = 0.005
UKNIT_AUC_FLOOR = 0.520


def build_k1h_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    model_keys = {
        "exact_ordered": CANDIDATE_MODEL,
        "operator_reversed": REVERSED_MODEL,
        "operator_corrupted": CORRUPTED_MODEL,
        "no_topology": NO_TOPOLOGY_MODEL,
    }
    if condition not in model_keys:
        raise ValueError("unknown K1-H operator condition")
    options = deepcopy(dict(task["model_options"]))
    options["topology_corruption_seed"] = 20260728
    _, pair_bits = input_geometry(str(task["cipher_key"]))
    return build_model(
        model_keys[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=options,
    )


def build_k1h_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    k1f_gate: Mapping[str, Any],
    k1g_gate: Mapping[str, Any],
    source_checks: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = candidate_task_map(tasks, fail_closed=False)
    expected_keys = expected_task_keys()
    protocol_checks = {
        "k1f_clean_hold_exact": (
            k1f_gate.get("run_id") == K1F_RUN_ID
            and k1f_gate.get("status") == "hold"
            and k1f_gate.get("decision") == K1F_DECISION
            and bool(k1f_gate.get("protocol_checks"))
            and all(k1f_gate.get("protocol_checks", {}).values())
        ),
        "k1g_sample_overfit_exact": (
            k1g_gate.get("run_id") == K1G_RUN_ID
            and k1g_gate.get("status") == "pass"
            and k1g_gate.get("decision") == K1G_DECISION
            and bool(k1g_gate.get("protocol_checks"))
            and all(k1g_gate.get("protocol_checks", {}).values())
        ),
        "four_frozen_candidate_tasks": (
            len(tasks) == EXPECTED_TRAINING_ROWS and set(task_map) == expected_keys
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(task_map),
        **dict(source_checks),
    }
    manifests: list[dict[str, Any]] = []
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            manifests, evidence_checks, evidence_metrics = structural_readiness(
                task_map
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))

    ready = (
        all(protocol_checks.values())
        and bool(evidence_checks)
        and all(evidence_checks.values())
        and not errors
    )
    return manifests, {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if ready else "fail",
        "decision": (
            "innovation1_uknit_family_ctspn_k1h_execution_authorized"
            if ready
            else "innovation1_uknit_family_ctspn_k1h_not_ready"
        ),
        "execution_authorized": ready,
        "optimizer_step_authorized": ready,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_evidence_checks": sorted(
            name for name, passed in evidence_checks.items() if not passed
        ),
        "evidence_metrics": evidence_metrics,
        "errors": errors,
        "training_rows": 0,
        "optimizer_steps": 0,
        "claim_scope": (
            "zero-training exact-operator routing, source-cache, anchor, geometry, "
            "and invariance readiness only; not neural efficacy or formal-scale evidence"
        ),
        "next_action": (
            "run the frozen four-row local K1-H diagnostic and evaluate candidate, "
            "three same-checkpoint controls, and frozen Runtime-E4 anchors on all splits"
            if ready
            else "repair only the failed K1-H invariant or source binding and rerun readiness unchanged"
        ),
    }


def structural_readiness(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    geometries: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for cipher in EXPECTED_CIPHERS:
        task = tasks[(cipher, 0)]
        input_bits, _ = input_geometry(cipher)
        correct = build_k1h_control(
            task=task,
            condition="exact_ordered",
            input_bits=input_bits,
        )
        correct.eval()
        state = correct.state_dict()
        state_sha = tensor_mapping_sha256(state)
        states[cipher] = state
        geometries[cipher] = [
            (name, tuple(value.shape)) for name, value in state.items()
        ]
        parameter_count = model_metadata(correct)["trainable_parameter_count"]
        features = torch.randint(
            0,
            2,
            (4, input_bits),
            generator=torch.Generator().manual_seed(20260728),
        ).float()
        with torch.inference_mode():
            correct_logits = correct(features)
        structure = correct.runtime_structure
        prefix = cipher
        checks[f"{prefix}_parameter_count_exact"] = (
            parameter_count == EXPECTED_PARAMETER_COUNT
        )
        checks[f"{prefix}_parameter_count_within_cap"] = (
            parameter_count <= EXPECTED_PARAMETER_CAP
        )
        checks[f"{prefix}_binary_invertible_operator_binding"] = all(
            torch.all((matrix == 0) | (matrix == 1))
            and torch.equal(
                torch.remainder(
                    structure.linear_matrices[index].to(torch.int64)
                    @ matrix.to(torch.int64),
                    2,
                ),
                torch.eye(structure.block_bits, dtype=torch.int64),
            )
            for index, matrix in enumerate(structure.inverse_linear_matrices)
        )
        checks[f"{prefix}_unit_support_exact"] = all(
            unit_support_exact(matrix) for matrix in structure.inverse_linear_matrices
        )
        relabel_delta = cell_relabel_logit_delta(correct, features)
        checks[f"{prefix}_joint_cell_relabel_invariant"] = relabel_delta <= 1e-6
        control_metrics: dict[str, Any] = {}
        for condition in CONTROL_CONDITIONS[1:]:
            control = build_k1h_control(
                task=task,
                condition=condition,
                input_bits=input_bits,
            )
            control.load_state_dict(state, strict=True)
            control.eval()
            with torch.inference_mode():
                logits = control(features)
            logit_delta = float((logits - correct_logits).abs().max())
            control_metrics[condition] = {
                "operator_routing_sha256": control.operator_routing_sha256,
                "logit_max_abs_delta": logit_delta,
            }
            checks[f"{prefix}_{condition}_strict_same_state"] = (
                tensor_mapping_sha256(control.state_dict()) == state_sha
            )
            checks[f"{prefix}_{condition}_operator_fingerprint_distinct"] = (
                control.operator_routing_sha256 != correct.operator_routing_sha256
            )
            checks[f"{prefix}_{condition}_changes_logits"] = logit_delta > 1e-7
        reversed_model = build_k1h_control(
            task=task,
            condition="operator_reversed",
            input_bits=input_bits,
        )
        checks[f"{prefix}_earlier_operator_consumed"] = (
            reversed_model.runtime_structure_transition_sha256s
            == tuple(reversed(correct.runtime_structure_transition_sha256s))
            and control_metrics["operator_reversed"]["logit_max_abs_delta"] > 1e-7
        )
        checks[f"{prefix}_no_forbidden_identity_or_semantics"] = (
            correct.operator_routing_only is True
            and correct.uses_path_tokens is False
            and correct.uses_absolute_cell_or_bit_identity is False
            and correct.uses_cipher_identity is False
            and correct.uses_sbox_semantics is False
        )
        metrics[cipher] = {
            "parameter_count": parameter_count,
            "operator_routing_sha256": correct.operator_routing_sha256,
            "cell_relabel_max_abs_logit_delta": relabel_delta,
            "controls": control_metrics,
        }
        for seed in EXPECTED_SEEDS:
            manifests.append(
                {
                    "run_id": READINESS_RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "model": CANDIDATE_MODEL,
                    "hidden_dim": 32,
                    "pair_embedding_dim": 128,
                    "shared_residual_instances": 1,
                    "runtime_transitions": 2,
                    "trainable_parameter_count": parameter_count,
                    "operator_routing_sha256": correct.operator_routing_sha256,
                    "training_rows": 0,
                    "optimizer_steps": 0,
                }
            )
    checks["cross_width_state_geometry_identical"] = (
        geometries["uknit64"] == geometries["dialga128"]
    )
    dialga = build_k1h_control(
        task=tasks[("dialga128", 0)],
        condition="exact_ordered",
        input_bits=1024,
    )
    dialga.load_state_dict(states["uknit64"], strict=True)
    checks["cross_width_strict_state_load"] = tensor_mapping_sha256(
        dialga.state_dict()
    ) == tensor_mapping_sha256(states["uknit64"])
    checks["readiness_zero_training"] = all(
        row["training_rows"] == 0 and row["optimizer_steps"] == 0 for row in manifests
    )
    return manifests, checks, metrics


def evaluate_k1h_panel(
    *,
    candidate_tasks: Sequence[Mapping[str, Any]],
    candidate_training_rows: Sequence[Mapping[str, Any]],
    candidate_checkpoint_manifest: Mapping[str, Any],
    anchor_tasks: Sequence[Mapping[str, Any]],
    anchor_results: Sequence[Mapping[str, Any]],
    anchor_checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks = candidate_task_map(candidate_tasks)
    trained = result_map(candidate_training_rows, CANDIDATE_MODEL)
    candidate_checkpoints = checkpoint_map(
        candidate_checkpoint_manifest,
        model=CANDIDATE_MODEL,
    )
    anchors = task_map_for_model(anchor_tasks, ANCHOR_MODEL)
    anchor_sources = result_map(anchor_results, ANCHOR_MODEL)
    anchor_checkpoints = checkpoint_map(anchor_checkpoint_manifest, model=ANCHOR_MODEL)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-H requires train, fresh-same-key, and cross-key datasets")

    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = tasks[key]
            candidate_source = trained[key]
            candidate_path = Path(
                str(candidate_source["training"]["checkpoint_output"])
            )
            candidate_state, candidate_sha = load_bound_state(
                candidate_path,
                candidate_checkpoints[key],
            )
            anchor_source = anchor_sources[key]
            anchor_path = Path(str(anchor_source["training"]["checkpoint_output"]))
            anchor_state, anchor_sha = load_bound_state(
                anchor_path,
                anchor_checkpoints[key],
            )
            for split in EXPECTED_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                dataset_sha = differential_dataset_sha256(dataset)
                labels = np.asarray(dataset.labels, dtype=np.float32)
                probabilities: dict[str, np.ndarray] = {}
                models: dict[str, torch.nn.Module] = {}
                for condition in CONTROL_CONDITIONS:
                    model = build_k1h_control(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(candidate_state, strict=True)
                    if tensor_mapping_sha256(
                        model.state_dict()
                    ) != tensor_mapping_sha256(candidate_state):
                        raise ValueError("K1-H candidate strict load changed state")
                    models[condition] = model
                    probabilities[condition] = predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                anchor = build_anchor_model(
                    anchors[key],
                    input_bits=int(dataset.features.shape[1]),
                )
                anchor.load_state_dict(anchor_state, strict=True)
                probabilities[ANCHOR_CONDITION] = predict_binary_probabilities(
                    anchor,
                    dataset,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )
                aucs = {
                    condition: binary_auc(labels, values)
                    for condition, values in probabilities.items()
                }
                reference = probabilities["exact_ordered"]
                for condition in (*CONTROL_CONDITIONS, ANCHOR_CONDITION):
                    current = probabilities[condition]
                    is_anchor = condition == ANCHOR_CONDITION
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "source_role": "anchor" if is_anchor else "candidate",
                            "condition": condition,
                            "rows": int(dataset.features.shape[0]),
                            "auc": aucs[condition],
                            "exact_minus_condition_auc": (
                                aucs["exact_ordered"] - aucs[condition]
                            ),
                            "max_abs_probability_delta_from_exact": float(
                                np.max(np.abs(reference - current))
                            ),
                            "mean_abs_probability_delta_from_exact": float(
                                np.mean(np.abs(reference - current))
                            ),
                            "dataset_sha256": dataset_sha,
                            "checkpoint_path": str(
                                anchor_path if is_anchor else candidate_path
                            ),
                            "checkpoint_sha256": anchor_sha
                            if is_anchor
                            else candidate_sha,
                            "state_dict_sha256": tensor_mapping_sha256(
                                anchor_state if is_anchor else candidate_state
                            ),
                            "operator_routing_sha256": (
                                None
                                if is_anchor
                                else models[condition].operator_routing_sha256
                            ),
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def load_bound_datasets(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], DifferentialDataset]:
    expected = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    datasets: dict[tuple[str, int, str], DifferentialDataset] = {}
    for row in manifest_rows:
        key = (str(row["cipher_key"]), int(row["seed"]), str(row["split"]))
        if key in datasets:
            raise ValueError(f"duplicate K1-H cache manifest row: {key}")
        cache_dir = Path(str(row["cache_dir"]))
        metadata_path = cache_dir / "metadata.json"
        features_path = cache_dir / "features.npy"
        labels_path = cache_dir / "labels.npy"
        if not all(
            path.is_file() for path in (metadata_path, features_path, labels_path)
        ):
            raise ValueError(f"missing K1-H source cache payload: {cache_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset = DiskDifferentialDataset(
            features=np.load(features_path, mmap_mode="r"),
            labels=np.load(labels_path, mmap_mode="r"),
            metadata=metadata,
            cache_dir=cache_dir,
        )
        if int(dataset.features.shape[0]) != int(row["rows"]):
            raise ValueError(f"K1-H source cache row count mismatch: {cache_dir}")
        if differential_dataset_sha256(dataset) != row.get("dataset_sha256"):
            raise ValueError(f"K1-H source cache digest mismatch: {cache_dir}")
        datasets[key] = dataset
    if set(datasets) != expected:
        raise ValueError("K1-H requires exactly twelve K1-G source caches")
    return datasets


def validate_k1h_source_bindings(
    *,
    candidate_tasks: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    anchor_tasks: Sequence[Mapping[str, Any]],
    anchor_results: Sequence[Mapping[str, Any]],
    anchor_checkpoint_manifest: Mapping[str, Any],
) -> dict[str, bool]:
    candidates = candidate_task_map(candidate_tasks, fail_closed=False)
    anchors = task_map_for_model(anchor_tasks, ANCHOR_MODEL)
    anchor_rows = result_map(anchor_results, ANCHOR_MODEL, fail_closed=False)
    try:
        anchor_checkpoints = checkpoint_map(
            anchor_checkpoint_manifest,
            model=ANCHOR_MODEL,
        )
    except ValueError:
        anchor_checkpoints = {}
    cache_keys = {
        (str(row.get("cipher_key")), int(row.get("seed", -1)), str(row.get("split")))
        for row in dataset_manifest
    }
    expected_cache_keys = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    cache_digest_bound = False
    try:
        cache_digest_bound = (
            set(load_bound_datasets(dataset_manifest)) == expected_cache_keys
        )
    except (OSError, ValueError, TypeError, KeyError):
        cache_digest_bound = False
    checkpoint_bound = (
        set(anchor_rows) == expected_task_keys()
        and set(anchor_checkpoints) == expected_task_keys()
        and all(
            Path(str(anchor_rows[key].get("training", {}).get("checkpoint_output", "")))
            == Path(str(anchor_checkpoints[key].get("path", "")))
            and Path(str(anchor_checkpoints[key].get("path", ""))).is_file()
            and file_sha256(Path(str(anchor_checkpoints[key]["path"])))
            == anchor_checkpoints[key].get("sha256")
            for key in expected_task_keys()
        )
    )
    aligned = (
        set(candidates) == expected_task_keys()
        and set(anchors) == expected_task_keys()
        and all(
            protocol_tasks_aligned(candidates[key], anchors[key])
            for key in expected_task_keys()
        )
    )
    return {
        "twelve_cache_manifest_rows_exact": (
            len(dataset_manifest) == 12 and cache_keys == expected_cache_keys
        ),
        "twelve_caches_reused_by_exact_digest": cache_digest_bound,
        "four_runtime_e4_anchors_bound": checkpoint_bound,
        "candidate_anchor_protocol_aligned": aligned,
    }


def protocol_tasks_aligned(
    candidate: Mapping[str, Any], anchor: Mapping[str, Any]
) -> bool:
    common_fields = (
        "cipher_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "train_key",
        "validation_key",
        "sample_structure",
        "loss",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "target_epochs",
    )
    candidate_options = candidate.get("model_options", {})
    anchor_options = anchor.get("model_options", {})
    return all(
        candidate.get(field) == anchor.get(field) for field in common_fields
    ) and all(
        candidate_options.get(field) == anchor_options.get(field)
        for field in (
            "runtime_structure_path",
            "runtime_round_start",
            "runtime_rounds",
            "pair_embedding_dim",
            "dropout",
        )
    )


def adjudicate_k1h(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    readiness_gate: Mapping[str, Any],
) -> dict[str, Any]:
    grouped = evaluation_map(evaluation_rows)
    expected = {
        (cipher, seed, split, condition)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
        for condition in (*CONTROL_CONDITIONS, ANCHOR_CONDITION)
    }
    task_rows = candidate_task_map(tasks, fail_closed=False)
    trained = result_map(training_rows, CANDIDATE_MODEL, fail_closed=False)
    seed_results = {
        cipher: {
            str(seed): {
                split: split_result(grouped, cipher, seed, split)
                for split in EXPECTED_SPLITS
            }
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    protocol_checks = {
        "readiness_exact_pass": (
            readiness_gate.get("run_id") == READINESS_RUN_ID
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("optimizer_step_authorized") is True
            and all(readiness_gate.get("protocol_checks", {}).values())
            and all(readiness_gate.get("evidence_checks", {}).values())
        ),
        "four_candidate_tasks_exact": (
            len(tasks) == EXPECTED_TRAINING_ROWS
            and set(task_rows) == expected_task_keys()
        ),
        "four_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(trained) == expected_task_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "sixty_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(grouped) == expected
        ),
        "evaluation_rows_zero_training": all(
            row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            and row.get("strict_state_dict_load") is True
            for row in evaluation_rows
        ),
        "split_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in evaluation_rows
        ),
        "same_dataset_per_seed_split": same_dataset_per_split(grouped),
        "same_candidate_state_per_seed": same_candidate_state(grouped),
        "operator_controls_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, condition)].get(
                        "operator_routing_sha256"
                    )
                    for condition in CONTROL_CONDITIONS
                }
            )
            == len(CONTROL_CONDITIONS)
            for cipher, seed in expected_task_keys()
            for split in EXPECTED_SPLITS
        ),
        "finite_metrics": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and 0.0 <= float(row.get("auc", math.nan)) <= 1.0
            for row in evaluation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            prefix = f"{cipher}_seed{seed}"
            train = seed_results[cipher][str(seed)]["train_seen"]
            research_checks[f"{prefix}_train_beats_controls"] = bool(
                train["beats_all_controls"]
            )
            for split in ("same_key_fresh", "cross_key_validation"):
                result = seed_results[cipher][str(seed)][split]
                split_prefix = f"{prefix}_{split}"
                research_checks[f"{split_prefix}_beats_controls"] = bool(
                    result["beats_all_controls"]
                )
                if cipher == "uknit64":
                    research_checks[f"{split_prefix}_auc_floor"] = (
                        result["candidate_auc"] >= UKNIT_AUC_FLOOR
                    )
                    research_checks[f"{split_prefix}_beats_anchor"] = (
                        result["candidate_minus_anchor"] >= MARGIN
                    )
                else:
                    research_checks[f"{split_prefix}_retains_anchor"] = (
                        result["candidate_minus_anchor"] >= -MARGIN
                    )
    protocol_valid = all(protocol_checks.values())
    all_research = bool(research_checks) and all(research_checks.values())
    train_pass = all(
        seed_results["uknit64"][str(seed)]["train_seen"]["beats_all_controls"]
        for seed in EXPECTED_SEEDS
    )
    same_key_pass = all(
        research_checks[f"uknit64_seed{seed}_same_key_fresh_beats_controls"]
        and research_checks[f"uknit64_seed{seed}_same_key_fresh_auc_floor"]
        and research_checks[f"uknit64_seed{seed}_same_key_fresh_beats_anchor"]
        for seed in EXPECTED_SEEDS
    )
    cross_key_pass = all(
        research_checks[f"uknit64_seed{seed}_cross_key_validation_beats_controls"]
        and research_checks[f"uknit64_seed{seed}_cross_key_validation_auc_floor"]
        and research_checks[f"uknit64_seed{seed}_cross_key_validation_beats_anchor"]
        for seed in EXPECTED_SEEDS
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1h_protocol_invalid"
        next_action = "repair only the failed protocol, cache, checkpoint, or operator binding and rerun K1-H unchanged"
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1h_operator_tied_latent_supported"
        next_action = (
            "retain K1-H and preregister a separate 65536/class remote diagnostic "
            "with disk-backed caches before any broader family or formal-scale claim"
        )
    elif train_pass and not same_key_pass:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1h_sample_shortcut_confirmed"
        next_action = (
            "close this operator-tied parameterization; do not add capacity or data, "
            "and audit whether absolute ciphertext channels should be removed"
        )
    elif same_key_pass and not cross_key_pass:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1h_key_specific_signal_confirmed"
        next_action = (
            "hold scale and preregister one difference-only input constraint while "
            "keeping the exact operator transport and budget frozen"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1h_operator_tied_latent_not_supported"
        )
        next_action = (
            "close K1-H at this diagnostic scale; inspect the failed per-seed anchor "
            "and operator controls before selecting any new uKNIT-family mechanism"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "thresholds": {
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "anchor_and_control_margin": MARGIN,
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
            "two-seed local 2048/class train and 1024/class fresh-same-key/cross-key "
            "operator-attribution diagnostic; not formal scale, attack, SOTA, transfer, "
            "arbitrary-SPN, or uKNIT ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale, extra data, epochs, width, pairs, or seeds unless every K1-H gate passes",
            "MoE, K2 S-box semantics, DDT, trail, partial decryption, or cipher identity",
            "using Dialga or averages to hide a failed uKNIT seed, split, anchor, or control",
        ],
    }


def unit_support_exact(matrix: torch.Tensor) -> bool:
    operator = torch.as_tensor(matrix, dtype=torch.uint8)
    bits = operator.shape[0]
    values = torch.eye(bits).unsqueeze(-1)
    transported = operator_support_mean(values, operator)
    observed = transported.squeeze(-1).T > 0
    return torch.equal(observed, operator.bool())


def cell_relabel_logit_delta(model: torch.nn.Module, features: torch.Tensor) -> float:
    structure = model.runtime_structure
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    runtime = features.reshape(features.shape[0], -1, 2, structure.block_bits).flip(-1)
    relabeled_runtime = torch.empty_like(runtime)
    relabeled_runtime[..., bit_permutation] = runtime
    with torch.inference_mode():
        original = model.backbone(runtime, structure)
        changed = model.backbone(relabeled_runtime, relabeled)
    return float((original - changed).abs().max())


def build_anchor_model(task: Mapping[str, Any], *, input_bits: int) -> torch.nn.Module:
    _, pair_bits = input_geometry(str(task["cipher_key"]))
    return build_model(
        ANCHOR_MODEL,
        input_bits=input_bits,
        hidden_bits=64,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )


def split_result(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cipher: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    candidate = float(grouped[(cipher, seed, split, "exact_ordered")]["auc"])
    anchor = float(grouped[(cipher, seed, split, ANCHOR_CONDITION)]["auc"])
    controls = {
        condition: float(grouped[(cipher, seed, split, condition)]["auc"])
        for condition in CONTROL_CONDITIONS[1:]
    }
    margins = {condition: candidate - auc for condition, auc in controls.items()}
    return {
        "candidate_auc": candidate,
        "anchor_auc": anchor,
        "candidate_minus_anchor": candidate - anchor,
        **{f"{condition}_auc": auc for condition, auc in controls.items()},
        **{
            f"candidate_minus_{condition}": value
            for condition, value in margins.items()
        },
        "weakest_control_margin": min(margins.values()),
        "beats_all_controls": all(value >= MARGIN for value in margins.values()),
    }


def load_bound_state(
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], str]:
    digest = file_sha256(checkpoint_path)
    if digest != manifest.get("sha256") or str(checkpoint_path) != manifest.get("path"):
        raise ValueError(f"checkpoint manifest mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    return payload["state_dict"], digest


def candidate_task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped = task_map_for_model(tasks, CANDIDATE_MODEL)
    if fail_closed and set(mapped) != expected_task_keys():
        raise ValueError("K1-H candidate tasks are incomplete")
    return mapped


def task_map_for_model(
    tasks: Sequence[Mapping[str, Any]], model: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != model:
            continue
        key = (str(task["cipher_key"]), int(task["seed"]))
        if key in mapped:
            raise ValueError(f"duplicate task: {key}")
        mapped[key] = task
    return mapped


def result_map(
    rows: Sequence[Mapping[str, Any]], model: str, *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("model") != model:
            continue
        key = (str(row["cipher_key"]), int(row["seed"]))
        if key in mapped:
            raise ValueError(f"duplicate result: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_task_keys():
        raise ValueError(f"incomplete result panel for {model}")
    return mapped


def checkpoint_map(
    manifest: Mapping[str, Any], *, model: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in manifest.get("entries", []):
        if row.get("model") != model:
            continue
        key = (str(row["cipher_key"]), int(row["seed"]))
        if key in mapped:
            raise ValueError(f"duplicate checkpoint: {key}")
        mapped[key] = row
    if set(mapped) != expected_task_keys():
        raise ValueError(f"incomplete checkpoint panel for {model}")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["cipher_key"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-H evaluation row: {key}")
        mapped[key] = row
    return mapped


def candidate_protocol_frozen(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    if set(tasks) != expected_task_keys():
        return False
    return all(
        task.get("rounds") == (5 if cipher == "uknit64" else 4)
        and task.get("samples_per_class") == 2048
        and task.get("pairs_per_sample") == 4
        and task.get("negative_mode") == "encrypted_random_plaintexts"
        and task.get("loss") == "mse"
        and task.get("learning_rate") == 1e-4
        and task.get("optimizer") == "adam"
        and task.get("weight_decay") == 1e-5
        and task.get("target_epochs") == EXPECTED_EPOCHS
        and task.get("checkpoint_metric") == "val_auc"
        and task.get("restore_best_checkpoint") is True
        and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
        and int(task.get("model_options", {}).get("runtime_round_start", -1))
        == (3 if cipher == "uknit64" else 2)
        for (cipher, _), task in tasks.items()
    )


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") == CANDIDATE_MODEL
        and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
        and row.get("samples_per_class") == 2048
        and row.get("pairs_per_sample") == 4
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("training", {}).get("batch_size") == EXPECTED_BATCH_SIZE
        and row.get("training", {}).get("epochs") == EXPECTED_EPOCHS
        and row.get("training", {}).get("checkpoint_metric") == "val_auc"
        and row.get("training", {}).get("selected_checkpoint") == "best"
        for row in rows
    )


def same_dataset_per_split(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                grouped[(cipher, seed, split, condition)].get("dataset_sha256")
                for condition in (*CONTROL_CONDITIONS, ANCHOR_CONDITION)
            }
        )
        == 1
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    )


def same_candidate_state(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                grouped[(cipher, seed, split, condition)].get("state_dict_sha256")
                for split in EXPECTED_SPLITS
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher, seed in expected_task_keys()
    )


def expected_task_keys() -> set[tuple[str, int]]:
    return {(cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS}


def input_geometry(cipher: str) -> tuple[int, int]:
    if cipher == "uknit64":
        return 512, 128
    if cipher == "dialga128":
        return 1024, 256
    raise ValueError(f"unsupported K1-H cipher: {cipher}")


__all__ = [
    "ANCHOR_CONDITION",
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "READINESS_RUN_ID",
    "RUN_ID",
    "adjudicate_k1h",
    "build_anchor_model",
    "build_k1h_control",
    "build_k1h_readiness",
    "cell_relabel_logit_delta",
    "evaluate_k1h_panel",
    "expected_task_keys",
    "input_geometry",
    "load_bound_datasets",
    "protocol_tasks_aligned",
    "structural_readiness",
    "unit_support_exact",
    "validate_k1h_source_bindings",
]
