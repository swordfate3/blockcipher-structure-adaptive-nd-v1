from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.gf2_boolean_view import (
    VIEW_NAMES,
    apply_gf2_operator,
    gf2_boolean_views,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import EXPECTED_SPLITS
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    checkpoint_map,
    evaluation_map,
    expected_task_keys,
    input_geometry,
    load_bound_datasets,
    load_bound_state,
    protocol_tasks_aligned,
    result_map,
    task_map_for_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1i import (
    CANDIDATE_MODEL as K1I_MODEL,
    RUN_ID as K1I_RUN_ID,
    build_k1i_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1j import (
    RUN_ID as K1J_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


READINESS_RUN_ID = "i1_uknit_family_ctspn_topology_edge_residual_k1k_readiness_20260728"
RUN_ID = "i1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1_20260728"
K1I_DECISION = (
    "innovation1_uknit_family_ctspn_k1i_dialga_signal_recovered_"
    "operator_attribution_not_supported"
)
K1J_DECISION = "innovation1_uknit_family_ctspn_k1j_joint_pool_branch_signal_supported"
EXPECTED_SOURCE_DIGESTS = {
    "k1i_gate": "e1823155149ce6146358650ae711269b617c93f4f7d48aaaa3e231348bfd675d",
    "k1i_checkpoint_manifest": (
        "4def7bc0019d7a258d962c622cfc79db1b69e0f85dc0b491a17bf081683e465f"
    ),
    "k1i_dataset_manifest": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "k1j_gate": "e77ab3811837fca0e9e7536df4ab2c58e0ea8f82222529e5c0c7c3903bb1d9dc",
}
CANDIDATE_MODEL = "runtime_spn_ct_k1k_edge_residual_true"
REVERSED_MODEL = "runtime_spn_ct_k1k_edge_residual_reversed"
CORRUPTED_MODEL = "runtime_spn_ct_k1k_edge_residual_corrupted"
NO_TOPOLOGY_MODEL = "runtime_spn_ct_k1k_edge_residual_none"
CONTROL_CONDITIONS = (
    "exact_ordered",
    "operator_reversed",
    "operator_corrupted",
    "no_topology",
)
ANCHOR_CONDITION = "k1i_anchor"
EXPECTED_PARAMETER_COUNT = 128707
EXPECTED_PARAMETER_CAP = 200000
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_TRAINING_ROWS = 4
EXPECTED_EVALUATION_ROWS = 60
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
MARGIN = 0.005
UKNIT_AUC_FLOOR = 0.520


def build_k1k_control(
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
        raise ValueError("unknown K1-K operator condition")
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


def build_k1k_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    k1i_gate: Mapping[str, Any],
    k1j_gate: Mapping[str, Any],
    source_checks: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = candidate_task_map(tasks, fail_closed=False)
    protocol_checks = {
        "k1i_operator_attribution_hold_exact": (
            k1i_gate.get("run_id") == K1I_RUN_ID
            and k1i_gate.get("status") == "hold"
            and k1i_gate.get("decision") == K1I_DECISION
            and bool(k1i_gate.get("protocol_checks"))
            and all(k1i_gate.get("protocol_checks", {}).values())
        ),
        "k1j_joint_pool_attribution_exact": (
            k1j_gate.get("run_id") == K1J_RUN_ID
            and k1j_gate.get("status") == "pass"
            and k1j_gate.get("decision") == K1J_DECISION
            and bool(k1j_gate.get("protocol_checks"))
            and all(k1j_gate.get("protocol_checks", {}).values())
        ),
        "four_frozen_candidate_tasks": (
            len(tasks) == EXPECTED_TRAINING_ROWS
            and set(task_map) == expected_task_keys()
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
            "innovation1_uknit_family_ctspn_k1k_execution_authorized"
            if ready
            else "innovation1_uknit_family_ctspn_k1k_not_ready"
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
            "zero-training bounded topology edge-residual, exact K1-I replay, "
            "cache, geometry, control, and nonzero-gate equivariance readiness only"
        ),
        "next_action": (
            "run the frozen four-row local K1-K diagnostic and the sixty-row "
            "three-split candidate/control/K1-I panel"
            if ready
            else "repair only the failed K1-K invariant or source binding and rerun readiness unchanged"
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
        correct = build_k1k_control(
            task=task,
            condition="exact_ordered",
            input_bits=input_bits,
        )
        k1i = build_k1i_control(
            task=task,
            condition="exact_ordered",
            input_bits=input_bits,
        )
        correct.backbone.base.load_state_dict(k1i.backbone.state_dict(), strict=True)
        correct.eval()
        k1i.eval()
        parameter_count = model_metadata(correct)["trainable_parameter_count"]
        features = torch.randint(
            0,
            2,
            (8, input_bits),
            generator=torch.Generator().manual_seed(20260728),
        ).float()
        runtime = project_features(features, correct.runtime_structure)
        views = gf2_boolean_views(runtime, correct.runtime_structure)
        with torch.inference_mode():
            zero_gate_logits = correct(features)
            k1i_logits = k1i(features)
        zero_gate_delta = float((zero_gate_logits - k1i_logits).abs().max())
        initial_gate = float(correct.backbone.residual_gate.detach())
        with torch.no_grad():
            correct.backbone.residual_gate.fill_(0.75)
        state = correct.state_dict()
        state_sha = tensor_mapping_sha256(state)
        states[cipher] = state
        geometries[cipher] = [
            (name, tuple(value.shape)) for name, value in state.items()
        ]
        with torch.inference_mode():
            correct_logits = correct(features)
            full_residual = correct.backbone.edge_residual_embedding(
                runtime,
                correct.runtime_structure,
            )
            without_first = correct.backbone.edge_residual_embedding(
                runtime,
                correct.runtime_structure,
                slot_mask=(False, True),
            )
            without_second = correct.backbone.edge_residual_embedding(
                runtime,
                correct.runtime_structure,
                slot_mask=(True, False),
            )
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
        checks[f"{prefix}_twelve_binary_channels_exact"] = (
            views.shape[-1] == 12
            and tuple(correct.boolean_view_names) == VIEW_NAMES
            and bool(torch.all((views == 0) | (views == 1)))
        )
        checks[f"{prefix}_scalar_vectorized_gf2_exact"] = scalar_vectorized_exact(
            structure
        )
        checks[f"{prefix}_transformed_difference_consistent"] = (
            transformed_difference_consistent(views)
        )
        checks[f"{prefix}_composition_order_exact"] = composition_order_exact(
            runtime, structure, views
        )
        checks[f"{prefix}_composition_nondegenerate"] = not torch.equal(
            views[..., 9:12], views[..., 3:6]
        ) and not torch.equal(views[..., 9:12], views[..., 6:9])
        checks[f"{prefix}_zero_gate_exact_k1i_replay"] = zero_gate_delta <= 1e-7
        checks[f"{prefix}_bounded_zero_initialized_gate"] = (
            initial_gate == 0.0
            and float(torch.tanh(correct.backbone.residual_gate.detach()).abs()) < 1.0
            and correct.residual_gate_bounded is True
        )
        edge_counts = tuple(int(value) for value in correct.topology_edge_counts)
        checks[f"{prefix}_two_nonempty_transition_edge_sets"] = len(
            edge_counts
        ) == 2 and all(value > 0 for value in edge_counts)
        slot_deltas = (
            float((full_residual - without_first).abs().max()),
            float((full_residual - without_second).abs().max()),
        )
        checks[f"{prefix}_earlier_edge_slot_consumed"] = slot_deltas[0] > 1e-7
        checks[f"{prefix}_later_edge_slot_consumed"] = slot_deltas[1] > 1e-7
        relabel_delta = cell_relabel_logit_delta(correct, features)
        checks[f"{prefix}_joint_cell_relabel_invariant"] = relabel_delta <= 1e-6
        individual_deltas = individual_operator_logit_deltas(correct, runtime)
        checks[f"{prefix}_earlier_operator_consumed"] = individual_deltas[0] > 1e-7
        checks[f"{prefix}_later_operator_consumed"] = individual_deltas[1] > 1e-7
        control_metrics: dict[str, Any] = {}
        for condition in CONTROL_CONDITIONS[1:]:
            control = build_k1k_control(
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
                "boolean_view_sha256": control.boolean_view_sha256,
                "topology_edge_sha256": control.topology_edge_sha256,
                "logit_max_abs_delta": logit_delta,
            }
            checks[f"{prefix}_{condition}_strict_same_state"] = (
                tensor_mapping_sha256(control.state_dict()) == state_sha
            )
            checks[f"{prefix}_{condition}_view_fingerprint_distinct"] = (
                control.boolean_view_sha256 != correct.boolean_view_sha256
            )
            checks[f"{prefix}_{condition}_edge_fingerprint_distinct"] = (
                control.topology_edge_sha256 != correct.topology_edge_sha256
            )
            checks[f"{prefix}_{condition}_changes_logits"] = logit_delta > 1e-7
        checks[f"{prefix}_no_forbidden_identity_or_semantics"] = (
            correct.deterministic_gf2_views is True
            and correct.uses_raw_bypass is False
            and correct.uses_learned_message_passing is True
            and correct.uses_path_tokens is False
            and correct.uses_absolute_cell_or_bit_identity is False
            and correct.uses_cipher_identity is False
            and correct.uses_sbox_semantics is False
            and correct.uses_ordered_cell_roles is True
            and correct.uses_explicit_source_target_edges is True
            and correct.uses_transition_slot_identity is True
        )
        metrics[cipher] = {
            "parameter_count": parameter_count,
            "boolean_view_sha256": correct.boolean_view_sha256,
            "topology_edge_sha256": correct.topology_edge_sha256,
            "topology_edge_counts": edge_counts,
            "zero_gate_max_abs_logit_delta": zero_gate_delta,
            "cell_relabel_max_abs_logit_delta": relabel_delta,
            "slot_mask_max_abs_residual_deltas": slot_deltas,
            "individual_operator_logit_deltas": individual_deltas,
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
                    "boolean_views": list(VIEW_NAMES),
                    "boolean_channels_per_bit": 12,
                    "runtime_transitions": 2,
                    "trainable_parameter_count": parameter_count,
                    "boolean_view_sha256": correct.boolean_view_sha256,
                    "topology_edge_sha256": correct.topology_edge_sha256,
                    "topology_edge_counts": edge_counts,
                    "zero_gate_max_abs_logit_delta": zero_gate_delta,
                    "training_rows": 0,
                    "optimizer_steps": 0,
                }
            )
    checks["cross_width_state_geometry_identical"] = (
        geometries["uknit64"] == geometries["dialga128"]
    )
    dialga = build_k1k_control(
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


def evaluate_k1k_panel(
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
    anchors = task_map_for_model(anchor_tasks, K1I_MODEL)
    anchor_sources = result_map(anchor_results, K1I_MODEL)
    anchor_checkpoints = checkpoint_map(anchor_checkpoint_manifest, model=K1I_MODEL)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-K requires all three cached splits")
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
                    model = build_k1k_control(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(candidate_state, strict=True)
                    if tensor_mapping_sha256(
                        model.state_dict()
                    ) != tensor_mapping_sha256(candidate_state):
                        raise ValueError("K1-K candidate strict load changed state")
                    models[condition] = model
                    probabilities[condition] = predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                anchor = build_k1i_control(
                    task=anchors[key],
                    condition="exact_ordered",
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
                            "boolean_view_sha256": (
                                None
                                if is_anchor
                                else models[condition].boolean_view_sha256
                            ),
                            "topology_edge_sha256": (
                                None
                                if is_anchor
                                else models[condition].topology_edge_sha256
                            ),
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def validate_k1k_source_bindings(
    *,
    candidate_tasks: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    anchor_tasks: Sequence[Mapping[str, Any]],
    anchor_results: Sequence[Mapping[str, Any]],
    anchor_checkpoint_manifest: Mapping[str, Any],
) -> dict[str, bool]:
    candidates = candidate_task_map(candidate_tasks, fail_closed=False)
    anchors = task_map_for_model(anchor_tasks, K1I_MODEL)
    anchor_rows = result_map(anchor_results, K1I_MODEL, fail_closed=False)
    try:
        anchor_checkpoints = checkpoint_map(
            anchor_checkpoint_manifest,
            model=K1I_MODEL,
        )
    except ValueError:
        anchor_checkpoints = {}
    expected_cache_keys = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    cache_keys = {
        (str(row.get("cipher_key")), int(row.get("seed", -1)), str(row.get("split")))
        for row in dataset_manifest
    }
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
        "four_k1i_anchors_bound": checkpoint_bound,
        "candidate_anchor_protocol_aligned": aligned,
    }


def adjudicate_k1k(
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
        "boolean_view_controls_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, condition)].get("boolean_view_sha256")
                    for condition in CONTROL_CONDITIONS
                }
            )
            == len(CONTROL_CONDITIONS)
            for cipher, seed in expected_task_keys()
            for split in EXPECTED_SPLITS
        ),
        "topology_edge_controls_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, condition)].get(
                        "topology_edge_sha256"
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
    dialga_retained = all(
        research_checks[f"dialga128_seed{seed}_{split}_retains_anchor"]
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    dialga_controls_pass = all(
        research_checks[f"dialga128_seed{seed}_{split}_beats_controls"]
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    training_controls_descriptive = {
        cipher: {
            str(seed): bool(
                seed_results[cipher][str(seed)]["train_seen"]["beats_all_controls"]
            )
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1k_protocol_invalid"
        next_action = (
            "repair only the failed K1-K protocol or binding and rerun unchanged"
        )
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1k_topology_edge_residual_supported"
        next_action = (
            "retain K1-K and preregister one disk-cached remote 65536/class "
            "diagnostic before any larger or family-wide claim"
        )
    elif dialga_retained and dialga_controls_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1k_dialga_operator_attribution_"
            "supported_uknit_not_supported"
        )
        next_action = (
            "keep K1-K only as correct-operator calibration evidence, hold scale, "
            "and test one exact heterogeneous S-box/operator composition locally"
        )
    elif dialga_retained:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1k_dialga_retained_"
            "operator_attribution_not_supported"
        )
        next_action = (
            "hold scale and audit the learned residual-gate magnitude plus exact "
            "edge attribution before another architecture change"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1k_dialga_anchor_not_retained"
        next_action = (
            "discard K1-K and return to the K1-I calibrated base without adding "
            "data, width, epochs, or experts"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "thresholds": {
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "anchor_and_control_margin": MARGIN,
        },
        "descriptive_diagnostics": {
            "dialga_retained": dialga_retained,
            "dialga_controls_pass": dialga_controls_pass,
            "training_split_beats_controls": training_controls_descriptive,
            "note": "post-result descriptive classification only; does not alter the frozen research gate",
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
            "bounded topology edge-residual diagnostic against the same K1-I base; "
            "not formal scale, attack, SOTA, arbitrary-SPN transfer, or uKNIT "
            "ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale or more data, epochs, width, pairs, seeds, or experts unless every K1-K gate passes",
            "raw bypass, S-box, DDT, trail, partial decryption, key, or cipher identity inside K1-K",
            "using Dialga or averaged metrics to hide any failed uKNIT seed, split, anchor, or control",
        ],
    }


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


def candidate_task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped = task_map_for_model(tasks, CANDIDATE_MODEL)
    if fail_closed and set(mapped) != expected_task_keys():
        raise ValueError("K1-K candidate tasks are incomplete")
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


def project_features(
    features: torch.Tensor,
    structure: RuntimeSpnStructure,
) -> torch.Tensor:
    return features.reshape(
        features.shape[0],
        -1,
        2,
        structure.block_bits,
    ).flip(-1)


def scalar_vectorized_exact(structure: RuntimeSpnStructure) -> bool:
    bits = structure.block_bits
    fixtures = [
        torch.zeros(bits, 3),
        torch.eye(bits, dtype=torch.float32)[:, :3],
        torch.randint(
            0,
            2,
            (bits, 3),
            generator=torch.Generator().manual_seed(bits + 20260728),
        ).float(),
    ]
    for matrix in structure.inverse_linear_matrices:
        for values in fixtures:
            observed = apply_gf2_operator(values, matrix)
            expected = scalar_gf2_operator(values, matrix)
            if not torch.equal(observed, expected):
                return False
    return True


def scalar_gf2_operator(values: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    matrix = torch.as_tensor(operator, dtype=torch.uint8)
    for target in range(matrix.shape[0]):
        sources = torch.nonzero(matrix[target], as_tuple=False).flatten().tolist()
        for source in sources:
            result[target] = torch.remainder(result[target] + values[source], 2.0)
    return result


def transformed_difference_consistent(views: torch.Tensor) -> bool:
    grouped = views.reshape(*views.shape[:-1], 4, 3)
    expected = torch.remainder(grouped[..., 0] + grouped[..., 1], 2.0)
    return bool(torch.equal(grouped[..., 2], expected))


def composition_order_exact(
    runtime: torch.Tensor,
    structure: RuntimeSpnStructure,
    views: torch.Tensor,
) -> bool:
    left = runtime[:, :, 0]
    right = runtime[:, :, 1]
    raw = torch.stack((left, right, torch.remainder(left + right, 2.0)), dim=-1)
    first, second = structure.inverse_linear_matrices
    expected = apply_gf2_operator(apply_gf2_operator(raw, second), first)
    return bool(torch.equal(views[..., 9:12], expected))


def cell_relabel_logit_delta(model: torch.nn.Module, features: torch.Tensor) -> float:
    structure = model.runtime_structure
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    runtime = project_features(features, structure)
    relabeled_runtime = torch.empty_like(runtime)
    relabeled_runtime[..., bit_permutation] = runtime
    with torch.inference_mode():
        original = model.backbone(runtime, structure)
        changed = model.backbone(relabeled_runtime, relabeled)
    return float((original - changed).abs().max())


def individual_operator_logit_deltas(
    model: torch.nn.Module,
    runtime: torch.Tensor,
) -> tuple[float, float]:
    structure = model.runtime_structure
    identity = torch.eye(structure.block_bits, dtype=torch.uint8)
    deltas: list[float] = []
    with torch.inference_mode():
        reference = model.backbone(runtime, structure)
        for index in range(2):
            inverses = structure.inverse_linear_matrices.clone()
            inverses[index] = identity
            changed = RuntimeSpnStructure(
                cell_membership=structure.cell_membership,
                bit_role=structure.bit_role,
                sbox_truth_bits=structure.sbox_truth_bits,
                linear_matrices=torch.stack(
                    [gf2_inverse_reference(matrix) for matrix in inverses]
                ),
                inverse_linear_matrices=inverses,
            )
            logits = model.backbone(runtime, changed)
            deltas.append(float((logits - reference).abs().max()))
    return deltas[0], deltas[1]


def gf2_inverse_reference(matrix: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(matrix, dtype=torch.uint8).clone()
    size = values.shape[0]
    reduced = values.clone()
    inverse = torch.eye(size, dtype=torch.uint8)
    for column in range(size):
        pivot = int(torch.nonzero(reduced[column:, column])[0, 0]) + column
        if pivot != column:
            reduced[[column, pivot]] = reduced[[pivot, column]]
            inverse[[column, pivot]] = inverse[[pivot, column]]
        for row in range(size):
            if row != column and int(reduced[row, column]):
                reduced[row] ^= reduced[column]
                inverse[row] ^= inverse[column]
    return inverse


__all__ = [
    "ANCHOR_CONDITION",
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "READINESS_RUN_ID",
    "RUN_ID",
    "adjudicate_k1k",
    "build_k1k_control",
    "build_k1k_readiness",
    "candidate_task_map",
    "cell_relabel_logit_delta",
    "evaluate_k1k_panel",
    "scalar_vectorized_exact",
    "structural_readiness",
    "validate_k1k_source_bindings",
]
