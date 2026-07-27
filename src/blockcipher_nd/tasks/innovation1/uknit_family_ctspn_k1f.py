from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.canonical_cell_path_hypergraph import (
    routing_fingerprint,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    CONTROL_CONDITIONS as K1D_CONTROL_CONDITIONS,
    EXPECTED_CIPHERS,
    EXPECTED_PAIRS_PER_SAMPLE,
    EXPECTED_SEEDS,
    EXPECTED_SAMPLES_PER_CLASS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1d import (
    sorted_path_token_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1e import (
    RUN_ID as K1E_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


READINESS_RUN_ID = "i1_uknit_family_ctspn_cell_path_hypergraph_k1f_readiness_20260728"
RUN_ID = "i1_uknit_family_ctspn_cell_path_hypergraph_k1f_2048_seed0_seed1_20260728"
K1E_DECISION = (
    "innovation1_uknit_family_ctspn_k1e_split_specific_relative_path_overfit_confirmed"
)
CANDIDATE_MODEL = "runtime_spn_ct_k1f_hypergraph_true"
CORRUPTED_MODEL = "runtime_spn_ct_k1f_hypergraph_corrupted"
INDEPENDENT_MODEL = "runtime_spn_ct_k1f_hypergraph_independent"
INCIDENCE_SHUFFLED_MODEL = "runtime_spn_ct_k1f_hypergraph_incidence_shuffled"
CONTROL_CONDITIONS = (*K1D_CONTROL_CONDITIONS, "incidence_shuffled")
EXPECTED_PARAMETER_CAP = 442466
EXPECTED_PARAMETER_COUNT = 361154
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_TRAINING_ROWS = 4
EXPECTED_CONTROL_ROWS = 24
MARGIN = 0.005
SOURCE_AUC_REPLAY_TOLERANCE = 5e-6


def build_k1f_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if condition not in CONTROL_CONDITIONS:
        raise ValueError("unknown K1-F control condition")
    options = deepcopy(dict(task["model_options"]))
    options.pop("canonical_schedule_control", None)
    options.pop("temporal_hidden_dim", None)
    options["runtime_structure_window_control"] = "full"
    options["topology_corruption_seed"] = 20260727
    model_key = CANDIDATE_MODEL
    if condition == "repeat_last":
        options["runtime_structure_window_control"] = "repeat_last"
    elif condition == "rotated":
        options["runtime_structure_window_control"] = "rotated"
    elif condition == "corrupted":
        model_key = CORRUPTED_MODEL
    elif condition == "no_topology":
        model_key = INDEPENDENT_MODEL
    elif condition == "incidence_shuffled":
        model_key = INCIDENCE_SHUFFLED_MODEL
    _, pair_bits = _input_geometry(str(task["cipher_key"]))
    return build_model(
        model_key,
        input_bits=input_bits,
        hidden_bits=64,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=options,
    )


def build_k1f_readiness(
    *,
    source_tasks: Sequence[Mapping[str, Any]],
    k1e_gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = _task_map(source_tasks)
    expected_keys = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        "k1e_relative_path_overfit_confirmed": (
            k1e_gate.get("run_id") == K1E_RUN_ID
            and k1e_gate.get("status") == "pass"
            and k1e_gate.get("decision") == K1E_DECISION
            and bool(k1e_gate.get("protocol_checks"))
            and all(k1e_gate.get("protocol_checks", {}).values())
        ),
        "four_frozen_source_tasks": len(source_tasks) == 4
        and set(tasks) == expected_keys,
        "source_protocol_frozen": _source_protocol_frozen(tasks),
    }
    manifests: list[dict[str, Any]] = []
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    if all(protocol_checks.values()):
        manifests, evidence = _readiness_evidence(tasks)
        evidence_checks = evidence["checks"]
        evidence_metrics = evidence["metrics"]

    all_checks_pass = (
        all(protocol_checks.values())
        and bool(evidence_checks)
        and all(evidence_checks.values())
    )
    status = "pass" if all_checks_pass else "fail"
    decision = (
        "innovation1_uknit_family_ctspn_k1f_hypergraph_execution_authorized"
        if all_checks_pass
        else "innovation1_uknit_family_ctspn_k1f_hypergraph_not_ready"
    )
    next_action = (
        "run the frozen four-row 2048/class local K1-F diagnostic and six-condition same-checkpoint panel"
        if all_checks_pass
        else "repair only the failed invariance, incidence-isolation, geometry, or source-binding check"
    )
    return manifests, {
        "run_id": READINESS_RUN_ID,
        "status": status,
        "decision": decision,
        "execution_authorized": all_checks_pass,
        "optimizer_step_authorized": all_checks_pass,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_evidence_checks": sorted(
            name for name, passed in evidence_checks.items() if not passed
        ),
        "evidence_metrics": evidence_metrics,
        "training_rows": 0,
        "optimizer_steps": 0,
        "claim_scope": (
            "zero-training cell/path hypergraph implementation and invariance readiness; "
            "not trained efficacy, formal scale, attack, SOTA, transfer, or ceiling evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "training or dataset inference if any readiness check fails",
            "remote scale-up, extra data, epochs, width, MoE, or K2 nonlinear conditioning",
            "claiming uKNIT efficacy from random-weight readiness probes",
        ],
    }


def evaluate_k1f_controls(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    validation_datasets: Mapping[tuple[str, int], DifferentialDataset],
    k1d_controls: Sequence[Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    task_map = _training_task_map(tasks)
    training_map = _training_row_map(training_rows)
    prior_hashes = _k1d_dataset_hashes(k1d_controls)
    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = task_map[key]
            source = training_map[key]
            dataset = validation_datasets[key]
            dataset_sha = differential_dataset_sha256(dataset)
            checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = payload["state_dict"]
            state_sha = tensor_mapping_sha256(state_dict)
            probabilities: dict[str, np.ndarray] = {}
            for condition in CONTROL_CONDITIONS:
                model = build_k1f_control(
                    task=task,
                    condition=condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                model.load_state_dict(state_dict, strict=True)
                if tensor_mapping_sha256(model.state_dict()) != state_sha:
                    raise ValueError("K1-F strict control load changed learned state")
                probabilities[condition] = predict_binary_probabilities(
                    model,
                    dataset,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )
            labels = np.asarray(dataset.labels, dtype=np.float32)
            aucs = {
                condition: binary_auc(labels, probabilities[condition])
                for condition in CONTROL_CONDITIONS
            }
            reference = probabilities["correct_ordered"]
            source_auc = float(source["metrics"]["auc"])
            checkpoint_sha = file_sha256(checkpoint_path)
            for condition in CONTROL_CONDITIONS:
                current = probabilities[condition]
                model = build_k1f_control(
                    task=task,
                    condition=condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "auc": aucs[condition],
                        "source_auc": source_auc,
                        "correct_minus_source_auc": aucs["correct_ordered"]
                        - source_auc,
                        "correct_minus_condition_auc": (
                            0.0
                            if condition == "correct_ordered"
                            else aucs["correct_ordered"] - aucs[condition]
                        ),
                        "max_abs_probability_delta_from_correct": float(
                            np.max(np.abs(reference - current))
                        ),
                        "mean_abs_probability_delta_from_correct": float(
                            np.mean(np.abs(reference - current))
                        ),
                        "dataset_sha256": dataset_sha,
                        "prior_k1d_dataset_sha256": prior_hashes[key],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha,
                        "state_dict_sha256": state_sha,
                        "checkpoint_selected": payload["metadata"].get(
                            "selected_checkpoint"
                        ),
                        "checkpoint_metric": payload["metadata"].get(
                            "checkpoint_metric"
                        ),
                        "model_class": type(model).__name__,
                        "incidence_mode": model.incidence_mode,
                        "routing_sha256": model.cell_path_routing_sha256,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows


def adjudicate_k1f(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    readiness_gate: Mapping[str, Any],
    k1d_gate: Mapping[str, Any],
    k1e_gate: Mapping[str, Any],
) -> dict[str, Any]:
    training = _training_row_map(training_rows, fail_closed=False)
    controls = _control_map(control_rows)
    expected_keys = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    expected_control_keys = {
        (cipher, seed, condition)
        for cipher, seed in expected_keys
        for condition in CONTROL_CONDITIONS
    }
    complete = set(controls) == expected_control_keys
    protocol_checks = {
        "four_row_frozen_plan": _training_panel_valid(tasks),
        "readiness_execution_authorized": (
            readiness_gate.get("run_id") == READINESS_RUN_ID
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("optimizer_step_authorized") is True
            and bool(readiness_gate.get("protocol_checks"))
            and all(readiness_gate.get("protocol_checks", {}).values())
            and bool(readiness_gate.get("evidence_checks"))
            and all(readiness_gate.get("evidence_checks", {}).values())
        ),
        "k1d_protocol_clean_hold": (
            k1d_gate.get("status") == "hold"
            and k1d_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1d_relative_path_not_supported"
            and bool(k1d_gate.get("protocol_checks"))
            and all(k1d_gate.get("protocol_checks", {}).values())
        ),
        "k1e_split_overfit_confirmed": (
            k1e_gate.get("run_id") == K1E_RUN_ID
            and k1e_gate.get("status") == "pass"
            and k1e_gate.get("decision") == K1E_DECISION
            and bool(k1e_gate.get("protocol_checks"))
            and all(k1e_gate.get("protocol_checks", {}).values())
        ),
        "four_training_rows_complete": len(training_rows) == EXPECTED_TRAINING_ROWS
        and set(training) == expected_keys,
        "twenty_four_control_rows_complete": len(control_rows) == EXPECTED_CONTROL_ROWS
        and complete,
        "training_protocol_frozen": _training_protocol_valid(training_rows),
        "controls_reuse_same_dataset": complete
        and _same_control_field(controls, "dataset_sha256"),
        "controls_reuse_same_state": complete
        and _same_control_field(controls, "state_dict_sha256"),
        "validation_dataset_matches_k1d": complete
        and all(
            row.get("dataset_sha256") == row.get("prior_k1d_dataset_sha256")
            for row in control_rows
        ),
        "selected_best_auc_checkpoints": complete
        and all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_metric") == "val_auc"
            for row in control_rows
        ),
        "strict_control_load_and_zero_optimizer": complete
        and all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in control_rows
        ),
        "correct_auc_replays_training_row": complete
        and all(
            abs(
                float(
                    controls[(cipher, seed, "correct_ordered")].get(
                        "correct_minus_source_auc", math.inf
                    )
                )
            )
            <= SOURCE_AUC_REPLAY_TOLERANCE
            for cipher, seed in expected_keys
        ),
        "incidence_control_isolated": complete
        and all(
            controls[(cipher, seed, "incidence_shuffled")].get("incidence_mode")
            == "shuffled"
            and controls[(cipher, seed, "correct_ordered")].get("incidence_mode")
            == "true"
            and controls[(cipher, seed, "incidence_shuffled")].get("routing_sha256")
            != controls[(cipher, seed, "correct_ordered")].get("routing_sha256")
            for cipher, seed in expected_keys
        ),
        "finite_metrics": all(_control_row_finite(row) for row in control_rows),
    }
    seed_results: dict[str, dict[str, Any]] = {
        cipher: {
            str(seed): _seed_result(controls, k1d_gate, cipher, seed)
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            result = seed_results[cipher][str(seed)]
            prefix = f"{cipher}_seed{seed}"
            if cipher == "uknit64":
                research_checks[f"{prefix}_auc_floor"] = (
                    result["candidate_auc"] >= 0.520
                )
                research_checks[f"{prefix}_beats_strongest_anchor"] = (
                    result["candidate_minus_anchor"] >= MARGIN
                )
            else:
                research_checks[f"{prefix}_retains_k1d"] = (
                    result["candidate_minus_anchor"] >= -MARGIN
                )
            for condition in CONTROL_CONDITIONS[1:]:
                research_checks[f"{prefix}_beats_{condition}"] = (
                    result[f"candidate_minus_{condition}"] >= MARGIN
                )

    if not all(protocol_checks.values()):
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1f_protocol_invalid"
        next_action = "repair only the failed plan, cache, checkpoint, or control binding and rerun K1-F unchanged"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1f_hypergraph_supported"
        next_action = "retain cell/path incidence message passing and preregister one same-budget K2 nonlinear cell-semantics candidate"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1f_hypergraph_not_supported"
        next_action = "do not scale K1-F; run frozen train-versus-validation attribution only if it can distinguish another split shortcut from relation underuse"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "thresholds": {
            "uknit_auc": 0.520,
            "uknit_anchor_margin": MARGIN,
            "dialga_retention_tolerance": MARGIN,
            "control_margin": MARGIN,
            "source_auc_replay_tolerance": SOURCE_AUC_REPLAY_TOLERANCE,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "claim_scope": (
            "uKNIT-BC prefix-r5 and Dialga-128 prefix-r4 two-seed 2048/class local "
            "cell/path hypergraph mechanism diagnostic; not formal scale, attack, "
            "SOTA, arbitrary-SPN, transfer, or ceiling evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up or mechanical sample, epoch, width, pair, or seed increase",
            "K2, MoE, DDT, trail, partial decryption, or cipher identity before K1-F passes",
            "using Dialga or a macro average to hide a failed uKNIT seed",
        ],
    }


def frozen_hypergraph_stages(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    backbone = model.backbone
    views, topology = backbone.path_views_and_routing(
        runtime,
        model.runtime_structure,
        relation_mode=model.relation_mode,
        incidence_mode=model.incidence_mode,
    )
    batch, pair_count, paths, _ = views.shape
    hidden = backbone.path_encoder(views).reshape(
        batch * pair_count,
        paths,
        backbone.token_dim,
    )
    for block in backbone.message_blocks:
        hidden = block(hidden, topology, cell_count=model.runtime_structure.cells)
    hidden = backbone.sequence_norm(hidden)
    pooled = torch.cat(
        (
            hidden.mean(dim=1),
            hidden.max(dim=1).values,
            torch.sqrt(hidden.square().mean(dim=1).clamp_min(1e-8)),
        ),
        dim=-1,
    ).reshape(batch, pair_count, -1)
    return views, pooled, model(features), hidden


def _readiness_evidence(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    geometries: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for cipher in EXPECTED_CIPHERS:
        task = tasks[(cipher, 0)]
        input_bits, _ = _input_geometry(cipher)
        correct = build_k1f_control(
            task=task,
            condition="correct_ordered",
            input_bits=input_bits,
        )
        correct.eval()
        state = correct.state_dict()
        state_sha = tensor_mapping_sha256(state)
        states[cipher] = state
        metadata = model_metadata(correct)
        geometries[cipher] = [
            (name, tuple(value.shape)) for name, value in state.items()
        ]
        generator = torch.Generator().manual_seed(20260728)
        features = torch.randint(0, 2, (4, input_bits), generator=generator).float()
        with torch.inference_mode():
            correct_views, correct_pooled, correct_logits, correct_hidden = (
                frozen_hypergraph_stages(correct, features)
            )
        correct_token_sha = sorted_path_token_sha256(correct_views)
        correct_routing_sha = correct.cell_path_routing_sha256
        topology = correct.backbone.path_views_and_routing(
            features.reshape(4, 4, 2, correct.runtime_structure.block_bits).flip(-1),
            correct.runtime_structure,
            relation_mode=correct.relation_mode,
            incidence_mode=correct.incidence_mode,
        )[1]
        degrees = {
            boundary: torch.bincount(indices, minlength=correct.runtime_structure.cells)
            for boundary, indices in (
                ("source", topology.source_cells),
                ("middle", topology.middle_cells),
                ("target", topology.target_cells),
            )
        }
        prefix = cipher
        checks[f"{prefix}_parameter_count_exact"] = (
            metadata["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
        )
        checks[f"{prefix}_parameter_count_within_cap"] = (
            metadata["trainable_parameter_count"] <= EXPECTED_PARAMETER_CAP
        )
        checks[f"{prefix}_cell_indices_routing_only"] = (
            correct.cell_indices_are_routing_only is True
            and correct.cell_indices_are_numeric_features is False
            and correct.relative_path_uses_absolute_cell_identity is False
        )
        checks[f"{prefix}_shared_cell_incidence_exists"] = all(
            int(values.max()) > 1 for values in degrees.values()
        )
        relabel = _cell_relabel_evidence(correct, features, correct_views)
        checks[f"{prefix}_cell_relabel_token_invariant"] = relabel["token_set_equal"]
        checks[f"{prefix}_cell_relabel_logit_invariant"] = (
            relabel["max_abs_logit_delta"] <= 1e-6
        )
        control_metrics: dict[str, Any] = {}
        for condition in CONTROL_CONDITIONS[1:]:
            control = build_k1f_control(
                task=task,
                condition=condition,
                input_bits=input_bits,
            )
            control.load_state_dict(state, strict=True)
            control.eval()
            with torch.inference_mode():
                views, pooled, logits, hidden = frozen_hypergraph_stages(
                    control, features
                )
            token_sha = sorted_path_token_sha256(views)
            pooled_delta = float((pooled - correct_pooled).abs().max())
            logit_delta = float((logits - correct_logits).abs().max())
            hidden_delta = (
                float((hidden - correct_hidden).abs().max())
                if hidden.shape == correct_hidden.shape
                else float("inf")
            )
            routing_changed = control.cell_path_routing_sha256 != correct_routing_sha
            token_changed = token_sha != correct_token_sha
            control_metrics[condition] = {
                "path_count": int(control.relative_path_count),
                "token_sha256": token_sha,
                "token_changed": token_changed,
                "routing_sha256": control.cell_path_routing_sha256,
                "routing_changed": routing_changed,
                "post_message_hidden_max_abs_delta": hidden_delta,
                "pooled_max_abs_delta": pooled_delta,
                "logit_max_abs_delta": logit_delta,
            }
            control_prefix = f"{prefix}_{condition}"
            checks[f"{control_prefix}_same_state_strict_load"] = (
                tensor_mapping_sha256(control.state_dict()) == state_sha
            )
            if condition == "incidence_shuffled":
                checks[f"{control_prefix}_token_set_preserved"] = not token_changed
                checks[f"{control_prefix}_routing_changes"] = routing_changed
                checks[f"{control_prefix}_post_message_noncollapsed"] = (
                    hidden_delta > 1e-6 and pooled_delta > 1e-6 and logit_delta > 1e-7
                )
            else:
                checks[f"{control_prefix}_evidence_changes"] = (
                    token_changed or routing_changed
                )
                checks[f"{control_prefix}_post_message_noncollapsed"] = (
                    pooled_delta > 1e-6 and logit_delta > 1e-7
                )
        metrics[cipher] = {
            "path_count": int(correct.relative_path_count),
            "parameter_count": metadata["trainable_parameter_count"],
            "routing_sha256": routing_fingerprint(topology),
            "max_cell_degrees": {
                name: int(values.max()) for name, values in degrees.items()
            },
            "cell_relabel": relabel,
            "controls": control_metrics,
        }
        for seed in EXPECTED_SEEDS:
            manifests.append(
                {
                    "run_id": READINESS_RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "model": CANDIDATE_MODEL,
                    "path_count": int(correct.relative_path_count),
                    "path_input_values": int(correct.backbone.path_input_dim),
                    "token_dim": int(correct.backbone.token_dim),
                    "processor_steps": len(correct.backbone.message_blocks),
                    "trainable_parameter_count": metadata["trainable_parameter_count"],
                    "cell_indices_routing_only": True,
                    "training_rows": 0,
                    "optimizer_steps": 0,
                }
            )
    checks["cross_width_state_geometry_identical"] = (
        geometries["uknit64"] == geometries["dialga128"]
    )
    dialga = build_k1f_control(
        task=tasks[("dialga128", 0)],
        condition="correct_ordered",
        input_bits=1024,
    )
    dialga.load_state_dict(states["uknit64"], strict=True)
    checks["cross_width_strict_state_load"] = tensor_mapping_sha256(
        dialga.state_dict()
    ) == tensor_mapping_sha256(states["uknit64"])
    checks["schema_has_no_absolute_cell_or_cipher_identity"] = not any(
        "cell_id" in name or "cipher" in name
        for name in dialga.relative_path_feature_schema
    )
    checks["readiness_has_zero_training"] = all(
        row["training_rows"] == 0 and row["optimizer_steps"] == 0 for row in manifests
    )
    return manifests, {"checks": checks, "metrics": metrics}


def _cell_relabel_evidence(
    model: torch.nn.Module,
    features: torch.Tensor,
    original_views: torch.Tensor,
) -> dict[str, Any]:
    structure = model.runtime_structure
    permutation = tuple(reversed(range(structure.cells)))
    relabeled, bit_permutation = structure.relabel_cells(permutation)
    runtime = features.reshape(features.shape[0], -1, 2, structure.block_bits).flip(-1)
    relabeled_runtime = torch.empty_like(runtime)
    relabeled_runtime[..., bit_permutation] = runtime
    with torch.inference_mode():
        relabeled_views = model.backbone.path_views_and_routing(
            relabeled_runtime,
            relabeled,
            relation_mode=model.relation_mode,
            incidence_mode=model.incidence_mode,
        )[0]
        original_logits = model.backbone(
            runtime,
            structure,
            relation_mode=model.relation_mode,
            incidence_mode=model.incidence_mode,
        )
        relabeled_logits = model.backbone(
            relabeled_runtime,
            relabeled,
            relation_mode=model.relation_mode,
            incidence_mode=model.incidence_mode,
        )
    return {
        "permutation": list(permutation),
        "token_set_equal": (
            sorted_path_token_sha256(original_views)
            == sorted_path_token_sha256(relabeled_views)
        ),
        "max_abs_logit_delta": float((original_logits - relabeled_logits).abs().max()),
    }


def _task_map(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != "runtime_spn_ct_k1d_relative_path_true":
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-F source task: {key}")
        result[key] = task
    return result


def _training_task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != CANDIDATE_MODEL:
            if fail_closed:
                raise ValueError("K1-F plan contains a non-candidate model")
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result and fail_closed:
            raise ValueError(f"duplicate K1-F training task: {key}")
        result[key] = task
    return result


def _training_panel_valid(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _training_task_map(tasks, fail_closed=False)
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if len(tasks) != EXPECTED_TRAINING_ROWS or set(mapped) != expected:
        return False
    for (cipher, _seed), task in mapped.items():
        options = task.get("model_options", {})
        if (
            task.get("rounds") != (5 if cipher == "uknit64" else 4)
            or task.get("samples_per_class") != EXPECTED_SAMPLES_PER_CLASS
            or task.get("pairs_per_sample") != EXPECTED_PAIRS_PER_SAMPLE
            or task.get("input_difference") != 0x40
            or task.get("negative_mode") != "encrypted_random_plaintexts"
            or task.get("sample_structure") != "independent_pairs"
            or task.get("target_epochs") != EXPECTED_EPOCHS
            or task.get("loss") != "mse"
            or task.get("optimizer") != "adam"
            or task.get("learning_rate") != 0.0001
            or task.get("weight_decay") != 0.00001
            or options.get("runtime_round_start") != (3 if cipher == "uknit64" else 2)
            or options.get("runtime_rounds") != 2
            or options.get("processor_steps") != 2
            or options.get("pair_embedding_dim") != 128
            or options.get("runtime_structure_window_control") != "full"
            or "canonical_schedule_control" in options
            or "temporal_hidden_dim" in options
        ):
            return False
    return True


def _training_row_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("model") != CANDIDATE_MODEL:
            if fail_closed:
                raise ValueError("K1-F results contain a non-candidate model")
            continue
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
        if key in result and fail_closed:
            raise ValueError(f"duplicate K1-F result row: {key}")
        result[key] = row
    return result


def _training_protocol_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("samples_per_class") == EXPECTED_SAMPLES_PER_CLASS
        and row.get("pairs_per_sample") == EXPECTED_PAIRS_PER_SAMPLE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
        and row.get("training", {}).get("batch_size") == EXPECTED_BATCH_SIZE
        and row.get("training", {}).get("epochs") == EXPECTED_EPOCHS
        and row.get("training", {}).get("checkpoint_metric") == "val_auc"
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and row.get("training", {}).get("train_rows") == 4096
        and row.get("training", {}).get("validation_rows") == 2048
        for row in rows
    )


def _control_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("cipher_key")),
            int(row.get("seed", -1)),
            str(row.get("condition")),
        )
        if key in result:
            raise ValueError(f"duplicate K1-F control row: {key}")
        result[key] = row
    return result


def _same_control_field(
    rows: Mapping[tuple[str, int, str], Mapping[str, Any]],
    field: str,
) -> bool:
    return all(
        len(
            {
                rows[(cipher, seed, condition)].get(field)
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )


def _k1d_dataset_hashes(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], str]:
    result = {
        (str(row["cipher_key"]), int(row["seed"])): str(row["dataset_sha256"])
        for row in rows
        if row.get("condition") == "correct_ordered"
    }
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if set(result) != expected:
        raise ValueError("K1-F requires all four K1-D validation dataset hashes")
    return result


def _seed_result(
    controls: Mapping[tuple[str, int, str], Mapping[str, Any]],
    k1d_gate: Mapping[str, Any],
    cipher: str,
    seed: int,
) -> dict[str, float]:
    candidate = float(
        controls.get((cipher, seed, "correct_ordered"), {}).get("auc", float("nan"))
    )
    prior = k1d_gate.get("seed_results", {}).get(cipher, {}).get(str(seed), {})
    if cipher == "uknit64":
        anchor = max(
            float(prior.get("anchor_auc", float("nan"))),
            float(prior.get("candidate_auc", float("nan"))),
        )
    else:
        anchor = float(prior.get("candidate_auc", float("nan")))
    result = {
        "candidate_auc": candidate,
        "anchor_auc": anchor,
        "candidate_minus_anchor": candidate - anchor,
    }
    for condition in CONTROL_CONDITIONS[1:]:
        auc = float(
            controls.get((cipher, seed, condition), {}).get("auc", float("nan"))
        )
        result[f"{condition}_auc"] = auc
        result[f"candidate_minus_{condition}"] = candidate - auc
    return result


def _control_row_finite(row: Mapping[str, Any]) -> bool:
    return all(
        isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
        for field in (
            "auc",
            "correct_minus_source_auc",
            "correct_minus_condition_auc",
            "max_abs_probability_delta_from_correct",
            "mean_abs_probability_delta_from_correct",
        )
    )


def _source_protocol_frozen(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    for (cipher, seed), task in tasks.items():
        options = task.get("model_options", {})
        if (
            cipher not in EXPECTED_CIPHERS
            or seed not in EXPECTED_SEEDS
            or task.get("rounds") != (5 if cipher == "uknit64" else 4)
            or task.get("samples_per_class") != EXPECTED_SAMPLES_PER_CLASS
            or task.get("pairs_per_sample") != EXPECTED_PAIRS_PER_SAMPLE
            or task.get("negative_mode") != "encrypted_random_plaintexts"
            or task.get("sample_structure") != "independent_pairs"
            or task.get("target_epochs") != 10
            or options.get("runtime_round_start") != (3 if cipher == "uknit64" else 2)
            or options.get("runtime_rounds") != 2
            or options.get("processor_steps") != 2
            or options.get("pair_embedding_dim") != 128
        ):
            return False
    return True


def _input_geometry(cipher: str) -> tuple[int, int]:
    if cipher == "uknit64":
        return 512, 128
    if cipher == "dialga128":
        return 1024, 256
    raise ValueError(f"unknown K1-F cipher: {cipher}")


__all__ = [
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "EXPECTED_PARAMETER_COUNT",
    "INCIDENCE_SHUFFLED_MODEL",
    "READINESS_RUN_ID",
    "RUN_ID",
    "build_k1f_control",
    "build_k1f_readiness",
    "adjudicate_k1f",
    "evaluate_k1f_controls",
    "frozen_hypergraph_stages",
]
