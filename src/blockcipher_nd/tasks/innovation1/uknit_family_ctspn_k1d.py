from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.canonical_relative_path import (
    PATH_FEATURE_SCHEMA,
    build_relative_path_topology,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_PAIRS_PER_SAMPLE,
    EXPECTED_SEEDS,
    EXPECTED_SAMPLES_PER_CLASS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1c import (
    RUN_ID as K1C_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728"
READINESS_RUN_ID = "i1_uknit_family_ctspn_relative_path_k1d_readiness_20260728"
K1C_DECISION = (
    "innovation1_uknit_family_ctspn_k1c_split_specific_topology_overfit_confirmed"
)
CANDIDATE_MODEL = "runtime_spn_ct_k1d_relative_path_true"
CORRUPTED_MODEL = "runtime_spn_ct_k1d_relative_path_corrupted"
INDEPENDENT_MODEL = "runtime_spn_ct_k1d_relative_path_independent"
CONTROL_CONDITIONS = (
    "correct_ordered",
    "repeat_last",
    "rotated",
    "corrupted",
    "no_topology",
)
EXPECTED_PARAMETER_COUNT = 409954
ANCHOR_PARAMETER_COUNT = 442466
EXPECTED_PATH_INPUT_VALUES = 76
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_TRAINING_ROWS = 4
EXPECTED_CONTROL_ROWS = 20
MARGIN = 0.005
SOURCE_AUC_REPLAY_TOLERANCE = 5e-6


def build_k1d_readiness(
    *,
    source_tasks: Sequence[Mapping[str, Any]],
    k1c_gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = _source_task_map(source_tasks, fail_closed=False)
    protocol_checks = {
        "frozen_k1b_source_panel": _source_panel_valid(source_tasks),
        "k1c_topology_overfit_confirmed": (
            k1c_gate.get("run_id") == K1C_RUN_ID
            and k1c_gate.get("status") == "pass"
            and k1c_gate.get("decision") == K1C_DECISION
            and bool(k1c_gate.get("protocol_checks"))
            and all(k1c_gate.get("protocol_checks", {}).values())
        ),
        "zero_training_readiness": True,
    }
    models: dict[tuple[str, int], torch.nn.Module] = {}
    manifests: list[dict[str, Any]] = []
    for key, task in task_map.items():
        cipher, seed = key
        input_bits, pair_bits = _input_geometry(cipher)
        model = build_k1d_control(
            task=task,
            condition="correct_ordered",
            input_bits=input_bits,
        )
        models[key] = model
        metadata = model_metadata(model)
        manifests.append(
            {
                "run_id": READINESS_RUN_ID,
                "cipher_key": cipher,
                "seed": seed,
                "model": CANDIDATE_MODEL,
                "trainable_parameter_count": metadata["trainable_parameter_count"],
                "path_input_values": model.backbone.path_input_dim,
                "relative_path_count": model.relative_path_count,
                "relative_path_topology_sha256": (model.relative_path_topology_sha256),
                "relative_path_compositions": model.relative_path_compositions,
                "uses_absolute_cell_identity": (
                    model.relative_path_uses_absolute_cell_identity
                ),
                "training_rows": 0,
                "optimizer_steps": 0,
            }
        )

    geometry = (
        [
            (name, tuple(value.shape))
            for name, value in next(iter(models.values())).state_dict().items()
        ]
        if models
        else []
    )
    evidence_checks = {
        "cross_width_state_geometry_equal": bool(models)
        and all(
            [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
            == geometry
            for model in models.values()
        ),
        "path_input_width_is_76": bool(models)
        and all(
            model.backbone.path_input_dim == EXPECTED_PATH_INPUT_VALUES
            for model in models.values()
        ),
        "one_cross_transition_composition": bool(models)
        and all(model.relative_path_compositions == 1 for model in models.values()),
        "no_absolute_cell_or_cipher_identity": bool(models)
        and all(
            model.relative_path_uses_absolute_cell_identity is False
            and not any(
                "cipher" in name or "cell_id" in name for name in PATH_FEATURE_SCHEMA
            )
            for model in models.values()
        ),
        "directed_role_reachability_present": (
            len(PATH_FEATURE_SCHEMA) == EXPECTED_PATH_INPUT_VALUES
            and sum(
                name.startswith("reachable_source_role") for name in PATH_FEATURE_SCHEMA
            )
            == 16
        ),
        "degenerate_temporal_mixer_removed": bool(models)
        and all(
            not hasattr(model.backbone, "temporal_depthwise")
            and not hasattr(model.backbone, "temporal_channel")
            for model in models.values()
        ),
        "candidate_not_larger_than_anchor": bool(manifests)
        and all(
            row["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
            and row["trainable_parameter_count"] <= ANCHOR_PARAMETER_COUNT
            for row in manifests
        ),
    }
    structural = _readiness_structural_evidence(task_map)
    evidence_checks.update(structural["checks"])
    authorized = all(protocol_checks.values()) and all(evidence_checks.values())
    gate = {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if authorized else "fail",
        "decision": (
            "innovation1_uknit_family_ctspn_k1d_relative_path_execution_authorized"
            if authorized
            else "innovation1_uknit_family_ctspn_k1d_relative_path_not_ready"
        ),
        "implementation_ready": authorized,
        "optimizer_step_authorized": authorized,
        "training_rows": 0,
        "optimizer_steps": 0,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "structural_evidence": structural["metrics"],
        "manifest_rows": len(manifests),
        "claim_scope": (
            "zero-training deterministic feasibility audit of one relative "
            "two-transition path representation on the frozen uKNIT-BC r5 and "
            "Dialga-128 r4 windows; not neural efficacy, scale, attack, or SOTA evidence"
        ),
        "next_action": (
            "create and run the frozen four-row K1-D 2048/class local diagnostic "
            "with five same-checkpoint topology controls"
            if authorized
            else "close this exact relative-path construction and inspect only the failed readiness invariant"
        ),
        "blocked_actions": [
            "training or dataset inference when any readiness check fails",
            "remote scale-up, extra data, epochs, width, MoE, or nonlinear K2 conditioning",
            "claiming uKNIT efficacy from zero-training fingerprints or random-weight logits",
        ],
    }
    return manifests, gate


def build_k1d_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if condition not in CONTROL_CONDITIONS:
        raise ValueError("unknown K1-D control condition")
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
    _, pair_bits = _input_geometry(str(task["cipher_key"]))
    return build_model(
        model_key,
        input_bits=input_bits,
        hidden_bits=64,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=options,
    )


def evaluate_k1d_controls(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    validation_datasets: Mapping[tuple[str, int], DifferentialDataset],
    k1b_controls: Sequence[Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    task_map = _training_task_map(tasks)
    training_map = _training_row_map(training_rows)
    prior_hashes = _k1b_dataset_hashes(k1b_controls)
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
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = payload["state_dict"]
            state_sha = tensor_mapping_sha256(state_dict)
            probabilities: dict[str, np.ndarray] = {}
            metadata: dict[str, dict[str, Any]] = {}
            for condition in CONTROL_CONDITIONS:
                model = build_k1d_control(
                    task=task,
                    condition=condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                model.load_state_dict(state_dict, strict=True)
                if tensor_mapping_sha256(model.state_dict()) != state_sha:
                    raise ValueError("K1-D strict control load changed learned state")
                probabilities[condition] = predict_binary_probabilities(
                    model,
                    dataset,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )
                metadata[condition] = _control_metadata(model, condition)
            labels = np.asarray(dataset.labels, dtype=np.float32)
            aucs = {
                condition: binary_auc(labels, probabilities[condition])
                for condition in CONTROL_CONDITIONS
            }
            reference = probabilities["correct_ordered"]
            source_auc = float(source["metrics"]["auc"])
            for condition in CONTROL_CONDITIONS:
                current = probabilities[condition]
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "auc": aucs[condition],
                        "source_auc": source_auc,
                        "correct_minus_source_auc": (
                            aucs["correct_ordered"] - source_auc
                        ),
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
                        "prior_k1b_dataset_sha256": prior_hashes[key],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                        "state_dict_sha256": state_sha,
                        "checkpoint_selected": payload["metadata"].get(
                            "selected_checkpoint"
                        ),
                        "checkpoint_metric": payload["metadata"].get(
                            "checkpoint_metric"
                        ),
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        **metadata[condition],
                    }
                )
    return rows


def adjudicate_k1d(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    readiness_gate: Mapping[str, Any],
    k1b_gate: Mapping[str, Any],
    k1c_gate: Mapping[str, Any],
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
        "k1b_protocol_clean_hold": (
            k1b_gate.get("status") == "hold"
            and k1b_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1b_native_endpoint_not_supported"
            and bool(k1b_gate.get("protocol_checks"))
            and all(k1b_gate.get("protocol_checks", {}).values())
        ),
        "k1c_topology_overfit_confirmed": (
            k1c_gate.get("run_id") == K1C_RUN_ID
            and k1c_gate.get("status") == "pass"
            and k1c_gate.get("decision") == K1C_DECISION
            and bool(k1c_gate.get("protocol_checks"))
            and all(k1c_gate.get("protocol_checks", {}).values())
        ),
        "four_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(training) == expected_keys
        ),
        "twenty_control_rows_complete": (
            len(control_rows) == EXPECTED_CONTROL_ROWS and complete
        ),
        "training_protocol_frozen": _training_protocol_valid(training_rows),
        "controls_reuse_same_dataset": complete
        and _same_control_field(controls, "dataset_sha256"),
        "controls_reuse_same_state": complete
        and _same_control_field(controls, "state_dict_sha256"),
        "validation_dataset_matches_k1b": complete
        and all(
            row.get("dataset_sha256") == row.get("prior_k1b_dataset_sha256")
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
        "finite_metrics": all(_control_row_finite(row) for row in control_rows),
    }
    seed_results: dict[str, dict[str, Any]] = {
        cipher: {
            str(seed): _seed_result(controls, k1b_gate, cipher, seed)
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
                research_checks[f"{prefix}_retains_k1b"] = (
                    result["candidate_minus_anchor"] >= -MARGIN
                )
            for condition in CONTROL_CONDITIONS[1:]:
                research_checks[f"{prefix}_beats_{condition}"] = (
                    result[f"candidate_minus_{condition}"] >= MARGIN
                )

    if not all(protocol_checks.values()):
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1d_protocol_invalid"
        next_action = (
            "repair only the failed plan, dataset, checkpoint, or control binding "
            "and rerun K1-D unchanged"
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1d_relative_path_supported"
        next_action = (
            "retain the relative cross-transition path representation and "
            "preregister one same-budget K2 nonlinear canonical-S-box composition"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1d_relative_path_not_supported"
        next_action = (
            "do not scale K1-D; compare its frozen checkpoints on training versus "
            "validation paths only if training attribution can resolve representation "
            "collapse from another split-specific shortcut"
        )
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
            "uKNIT-BC prefix-r5 and Dialga-128 prefix-r4 two-seed 2048/class "
            "local relative-path mechanism diagnostic; not formal scale, attack, "
            "SOTA, arbitrary-SPN, or uKNIT-ceiling evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up or mechanical sample, epoch, width, or pair increase",
            "K2, MoE, DDT, trail, partial decryption, or cipher identity before K1-D passes",
            "using Dialga or a macro average to hide a failed uKNIT seed",
        ],
    }


def frozen_relative_path_stages(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    backbone = model.backbone
    views = backbone.relative_path_views(
        runtime,
        model.runtime_structure,
        relation_mode=model.relation_mode,
    )
    batch, pair_count, paths, _ = views.shape
    hidden = backbone.path_encoder(views).reshape(
        batch * pair_count,
        paths,
        backbone.token_dim,
    )
    for block in backbone.mixer_blocks:
        hidden = block(hidden)
    hidden = backbone.sequence_norm(hidden)
    pooled = torch.cat(
        (
            hidden.mean(dim=1),
            hidden.max(dim=1).values,
            torch.sqrt(hidden.square().mean(dim=1).clamp_min(1e-8)),
        ),
        dim=-1,
    ).reshape(batch, pair_count, -1)
    return views, pooled, model(features)


def sorted_path_token_sha256(values: torch.Tensor) -> str:
    array = values.detach().cpu().numpy().astype(np.float32, copy=False)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape[:-2])).encode("ascii"))
    digest.update(str(array.shape[-1]).encode("ascii"))
    for panel in array.reshape(-1, array.shape[-2], array.shape[-1]):
        rows = sorted(tuple(float(value) for value in row) for row in panel)
        digest.update(np.asarray(rows, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _readiness_structural_evidence(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    for cipher in EXPECTED_CIPHERS:
        task = tasks.get((cipher, 0))
        if task is None:
            continue
        input_bits, _ = _input_geometry(cipher)
        correct = build_k1d_control(
            task=task,
            condition="correct_ordered",
            input_bits=input_bits,
        )
        state = correct.state_dict()
        state_sha = tensor_mapping_sha256(state)
        generator = torch.Generator().manual_seed(20260728)
        features = torch.randint(0, 2, (4, input_bits), generator=generator).to(
            torch.float32
        )
        correct.eval()
        with torch.inference_mode():
            correct_views, correct_pooled, correct_logits = frozen_relative_path_stages(
                correct, features
            )
        correct_token_sha = sorted_path_token_sha256(correct_views)
        topology = build_relative_path_topology(
            correct.runtime_structure,
            relation_mode=correct.relation_mode,
        )
        prefix = cipher
        checks[f"{prefix}_all_paths_exactly_connected"] = _topology_paths_exact(
            correct.runtime_structure,
            topology,
        )
        relabel_metrics = _cell_relabel_evidence(correct, features, correct_views)
        checks[f"{prefix}_cell_relabel_token_invariant"] = relabel_metrics[
            "token_set_equal"
        ]
        checks[f"{prefix}_cell_relabel_logit_invariant"] = (
            relabel_metrics["max_abs_logit_delta"] <= 1e-6
        )
        control_metrics: dict[str, Any] = {}
        for condition in CONTROL_CONDITIONS[1:]:
            control = build_k1d_control(
                task=task,
                condition=condition,
                input_bits=input_bits,
            )
            control.load_state_dict(state, strict=True)
            control.eval()
            with torch.inference_mode():
                views, pooled, logits = frozen_relative_path_stages(control, features)
            token_sha = sorted_path_token_sha256(views)
            pooled_delta = float((pooled - correct_pooled).abs().max())
            logit_delta = float((logits - correct_logits).abs().max())
            control_metrics[condition] = {
                "path_count": int(control.relative_path_count),
                "topology_sha256": control.relative_path_topology_sha256,
                "token_sha256": token_sha,
                "token_fingerprint_changed": token_sha != correct_token_sha,
                "pooled_max_abs_delta": pooled_delta,
                "logit_max_abs_delta": logit_delta,
            }
            control_prefix = f"{prefix}_{condition}"
            checks[f"{control_prefix}_same_state_strict_load"] = (
                tensor_mapping_sha256(control.state_dict()) == state_sha
            )
            checks[f"{control_prefix}_path_fingerprint_changes"] = (
                token_sha != correct_token_sha
            )
            checks[f"{control_prefix}_pooled_summary_noncollapsed"] = (
                pooled_delta > 1e-6
            )
            checks[f"{control_prefix}_logit_noncollapsed"] = logit_delta > 1e-7
        metrics[cipher] = {
            "correct_path_count": topology.path_count,
            "correct_topology_sha256": topology.fingerprint_sha256,
            "correct_token_sha256": correct_token_sha,
            "cell_relabel": relabel_metrics,
            "controls": control_metrics,
        }
    return {"checks": checks, "metrics": metrics}


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
        relabeled_views = model.backbone.relative_path_views(
            relabeled_runtime,
            relabeled,
            relation_mode=model.relation_mode,
        )
        original_logits = model.backbone(
            runtime,
            structure,
            relation_mode=model.relation_mode,
        )
        relabeled_logits = model.backbone(
            relabeled_runtime,
            relabeled,
            relation_mode=model.relation_mode,
        )
    return {
        "permutation": list(permutation),
        "original_path_count": int(original_views.shape[-2]),
        "relabeled_path_count": int(relabeled_views.shape[-2]),
        "token_set_equal": (
            sorted_path_token_sha256(original_views)
            == sorted_path_token_sha256(relabeled_views)
        ),
        "max_abs_logit_delta": float((original_logits - relabeled_logits).abs().max()),
    }


def _topology_paths_exact(
    structure: Any,
    topology: Any,
) -> bool:
    lookup = torch.empty(structure.cells, 4, dtype=torch.long)
    bits = torch.arange(structure.block_bits)
    lookup[structure.cell_membership, structure.bit_role] = bits
    first, second = structure.linear_matrices
    for index in range(topology.path_count):
        source = int(topology.source_cells[index])
        middle = int(topology.middle_cells[index])
        target = int(topology.target_cells[index])
        first_relation = first[lookup[middle]][:, lookup[source]].to(torch.bool)
        second_relation = second[lookup[target]][:, lookup[middle]].to(torch.bool)
        expected = (
            first_relation.any(dim=0)[:, None] & second_relation.any(dim=1)[None, :]
        )
        if not torch.equal(topology.reachability[index].to(torch.bool), expected):
            return False
    return True


def _source_task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != "runtime_spn_ct_k1b_endpoint_true":
            if fail_closed:
                raise ValueError("K1-D readiness requires the frozen K1-B source plan")
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result and fail_closed:
            raise ValueError(f"duplicate K1-D source task: {key}")
        result[key] = task
    return result


def _source_panel_valid(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _source_task_map(tasks, fail_closed=False)
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if len(tasks) != 4 or set(mapped) != expected:
        return False
    for (cipher, _seed), task in mapped.items():
        options = task.get("model_options", {})
        if (
            task.get("rounds") != (5 if cipher == "uknit64" else 4)
            or task.get("samples_per_class") != EXPECTED_SAMPLES_PER_CLASS
            or task.get("pairs_per_sample") != EXPECTED_PAIRS_PER_SAMPLE
            or task.get("negative_mode") != "encrypted_random_plaintexts"
            or task.get("sample_structure") != "independent_pairs"
            or options.get("runtime_round_start") != (3 if cipher == "uknit64" else 2)
            or options.get("runtime_rounds") != 2
            or options.get("processor_steps") != 2
            or options.get("pair_embedding_dim") != 128
            or task.get("target_epochs") != 10
        ):
            return False
    return True


def _training_task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != CANDIDATE_MODEL:
            if fail_closed:
                raise ValueError("K1-D plan contains a non-candidate model")
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result and fail_closed:
            raise ValueError(f"duplicate K1-D training task: {key}")
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
                raise ValueError("K1-D results contain a non-candidate model")
            continue
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
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
    return {
        (
            str(row.get("cipher_key")),
            int(row.get("seed", -1)),
            str(row.get("condition")),
        ): row
        for row in rows
    }


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


def _k1b_dataset_hashes(
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
        raise ValueError("K1-D requires all four K1-B validation dataset hashes")
    return result


def _control_metadata(model: torch.nn.Module, condition: str) -> dict[str, Any]:
    metadata = model_metadata(model)
    return {
        "condition": condition,
        "model_class": type(model).__name__,
        "trainable_parameter_count": metadata["trainable_parameter_count"],
        "relation_mode": getattr(model, "relation_mode", None),
        "runtime_structure_mode": getattr(model, "runtime_structure_mode", None),
        "runtime_structure_window_control": getattr(
            model, "runtime_structure_window_control", None
        ),
        "runtime_structure_window_sha256": getattr(
            model, "runtime_structure_window_sha256", None
        ),
        "relative_path_count": getattr(model, "relative_path_count", None),
        "relative_path_topology_sha256": getattr(
            model, "relative_path_topology_sha256", None
        ),
        "relative_path_uses_absolute_cell_identity": getattr(
            model, "relative_path_uses_absolute_cell_identity", None
        ),
    }


def _seed_result(
    controls: Mapping[tuple[str, int, str], Mapping[str, Any]],
    k1b_gate: Mapping[str, Any],
    cipher: str,
    seed: int,
) -> dict[str, float]:
    candidate = float(
        controls.get((cipher, seed, "correct_ordered"), {}).get("auc", float("nan"))
    )
    prior = k1b_gate.get("seed_results", {}).get(cipher, {}).get(str(seed), {})
    if cipher == "uknit64":
        anchor = max(
            float(prior.get("prior_anchor_auc", float("nan"))),
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


def _input_geometry(cipher: str) -> tuple[int, int]:
    if cipher == "uknit64":
        return 512, 128
    if cipher == "dialga128":
        return 1024, 256
    raise ValueError(f"unsupported K1-D cipher: {cipher}")


__all__ = [
    "ANCHOR_PARAMETER_COUNT",
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "CORRUPTED_MODEL",
    "EXPECTED_PARAMETER_COUNT",
    "INDEPENDENT_MODEL",
    "READINESS_RUN_ID",
    "RUN_ID",
    "build_k1d_control",
    "build_k1d_readiness",
    "adjudicate_k1d",
    "evaluate_k1d_controls",
    "frozen_relative_path_stages",
    "sorted_path_token_sha256",
]
