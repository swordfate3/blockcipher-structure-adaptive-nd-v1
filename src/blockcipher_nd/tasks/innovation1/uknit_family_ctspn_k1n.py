from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    COMPOSITION_STAGE_NAMES,
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.runtime_structure import apply_gf2
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import EXPECTED_SPLITS
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    checkpoint_map,
    expected_task_keys,
    input_geometry,
    load_bound_state,
    result_map,
    task_map_for_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_TRAINING_ROWS,
    project_features,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1l import (
    audit_gradient_path,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


READINESS_RUN_ID = (
    "i1_uknit_family_ctspn_exact_operator_composition_k1n_readiness_20260728"
)
RUN_ID = (
    "i1_uknit_family_ctspn_exact_operator_composition_k1n_"
    "2048_seed0_seed1_20260728"
)
K1M_DECISION = (
    "innovation1_uknit_family_ctspn_k1m_gate_opened_uknit_signal_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1m_gate": "0cf3714eaf2dbc0f052b04f4255ca007b16ab00b1c8abb93c270c3831a1876c6",
    "k1m_checkpoint_manifest": (
        "848a701087a6b0cca342461af97081f2a85cf4778a979660fa9a3b32b901e139"
    ),
    "k1m_dataset_manifest": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "k1m_controls": (
        "289c611e8a41dc9d9a2e60868f6dab1313d7cf382f10e2a0052bb39cf433d2bd"
    ),
}
CANDIDATE_MODEL = "runtime_spn_ct_k1n_exact_composition_true"
CONTROL_MODELS = {
    "exact_composition": CANDIDATE_MODEL,
    "wrong_sbox_semantics": "runtime_spn_ct_k1n_exact_composition_wrong_sbox",
    "reversed_linear_schedule": (
        "runtime_spn_ct_k1n_exact_composition_reversed_linear"
    ),
    "corrupted_linear_operators": (
        "runtime_spn_ct_k1n_exact_composition_corrupted_linear"
    ),
    "no_sbox_composition": "runtime_spn_ct_k1n_exact_composition_no_sbox",
    "no_topology": "runtime_spn_ct_k1n_exact_composition_none",
}
CONTROL_CONDITIONS = tuple(CONTROL_MODELS)
ANCHOR_CONDITION = "k1m_anchor"
EVALUATION_CONDITIONS = (*CONTROL_CONDITIONS, ANCHOR_CONDITION)
EXPECTED_PARAMETER_COUNT = 131875
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_CIPHERS)
    * len(EXPECTED_SEEDS)
    * len(EXPECTED_SPLITS)
    * len(EVALUATION_CONDITIONS)
)
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
INITIAL_EFFECTIVE_GATE = 0.05
FINAL_ACTIVE_GATE = 0.010
MARGIN = 0.005
UKNIT_AUC_FLOOR = 0.520


def build_k1n_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-N semantic control")
    options = deepcopy(dict(task["model_options"]))
    options["topology_corruption_seed"] = 20260728
    _, pair_bits = input_geometry(str(task["cipher_key"]))
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=options,
    )


def candidate_task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    mapped = task_map_for_model(tasks, CANDIDATE_MODEL)
    if fail_closed and set(mapped) != expected_task_keys():
        raise ValueError("K1-N candidate tasks are incomplete")
    return mapped


def candidate_protocol_frozen(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    return set(tasks) == expected_task_keys() and all(
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
        and float(
            task.get("model_options", {}).get(
                "residual_gate_initial_effective", math.nan
            )
        )
        == INITIAL_EFFECTIVE_GATE
        for (cipher, _), task in tasks.items()
    )


def build_k1n_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    source_checks: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_rows = candidate_task_map(tasks, fail_closed=False)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    protocol_checks = {
        "four_frozen_candidate_tasks": (
            len(tasks) == EXPECTED_TRAINING_ROWS
            and set(task_rows) == expected_task_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(task_rows),
        "twelve_bound_source_caches": set(datasets) == expected_datasets,
        **dict(source_checks),
    }
    manifests: list[dict[str, Any]] = []
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            manifests, evidence_checks, evidence_metrics = structural_readiness(
                task_rows,
                datasets,
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
            "innovation1_uknit_family_ctspn_k1n_execution_authorized"
            if ready
            else "innovation1_uknit_family_ctspn_k1n_not_ready"
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
        "next_action": (
            "run the frozen four-row local K1-N diagnostic"
            if ready
            else "repair only the failed K1-N invariant or source binding and rerun readiness unchanged"
        ),
        "claim_scope": (
            "zero-training exact inverse S-box/operator composition, control, "
            "gradient, geometry, and cache readiness only"
        ),
    }


def structural_readiness(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    geometries: dict[tuple[str, int], tuple[tuple[str, tuple[int, ...]], ...]] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = tasks[key]
            dataset = datasets[(cipher, seed, "train_seen")]
            features = torch.as_tensor(
                np.asarray(dataset.features[:EXPECTED_BATCH_SIZE]).copy(),
                dtype=torch.float32,
            )
            input_bits = int(features.shape[1])
            candidate = build_k1n_control(
                task=task,
                condition="exact_composition",
                input_bits=input_bits,
            )
            runtime = project_features(features[:8], candidate.runtime_structure)
            views = exact_operator_composition_views(
                runtime,
                candidate.runtime_structure,
            )
            reference = reference_composition_views(
                runtime,
                candidate.runtime_structure,
            )
            state = candidate.state_dict()
            state_sha = tensor_mapping_sha256(state)
            geometry = tuple(
                (name, tuple(value.shape)) for name, value in state.items()
            )
            geometries[key] = geometry
            controls: dict[str, Any] = {}
            for condition in CONTROL_CONDITIONS[1:]:
                control = build_k1n_control(
                    task=task,
                    condition=condition,
                    input_bits=input_bits,
                )
                control_geometry = tuple(
                    (name, tuple(value.shape))
                    for name, value in control.state_dict().items()
                )
                control.load_state_dict(state, strict=True)
                controls[condition] = {
                    "same_geometry": control_geometry == geometry,
                    "same_state": tensor_mapping_sha256(control.state_dict())
                    == state_sha,
                    "composition_distinct": (
                        control.composition_sha256 != candidate.composition_sha256
                    ),
                    "declared_contract": control_contract_exact(
                        candidate,
                        control,
                        condition,
                    ),
                }
            gradients = audit_gradient_path(
                candidate,
                dataset,
                effective_gate=INITIAL_EFFECTIVE_GATE,
                batch_size=EXPECTED_BATCH_SIZE,
            )
            residual_groups = (
                "composition_encoder",
                "cell_encoder",
                "edge_encoder",
                "cell_update",
                "residual_projection",
            )
            stages = [views[..., offset : offset + 3] for offset in range(0, 15, 3)]
            prefix = f"{cipher}_seed{seed}"
            parameter_count = model_metadata(candidate)["trainable_parameter_count"]
            checks[f"{prefix}_parameter_count_exact"] = (
                parameter_count == EXPECTED_PARAMETER_COUNT
            )
            checks[f"{prefix}_stage_schema_exact"] = (
                tuple(candidate.composition_stage_names) == COMPOSITION_STAGE_NAMES
                and candidate.composition_channels_per_bit == 15
                and views.shape[-1] == 15
            )
            checks[f"{prefix}_binary_channels_exact"] = bool(
                torch.all((views == 0) | (views == 1))
            )
            checks[f"{prefix}_independent_reference_exact"] = torch.equal(
                views,
                reference,
            )
            checks[f"{prefix}_forward_round_trip_exact"] = forward_round_trip_exact(
                runtime,
                stages[-1],
                candidate.runtime_structure,
            )
            checks[f"{prefix}_all_four_operator_stages_nondegenerate"] = all(
                not torch.equal(stages[index], stages[index + 1])
                for index in range(4)
            )
            checks[f"{prefix}_controls_strict_distinct_and_scoped"] = all(
                all(values.values()) for values in controls.values()
            )
            checks[f"{prefix}_all_composition_residual_groups_receive_gradient"] = all(
                float(gradients["gradient_norms"][group]) > 1e-8
                for group in residual_groups
            )
            checks[f"{prefix}_gradient_probe_restores_state"] = bool(
                gradients["state_restored_exact"]
            )
            metrics[prefix] = {
                "parameter_count": parameter_count,
                "initial_effective_gate": float(
                    torch.tanh(candidate.backbone.residual_gate.detach())
                ),
                "composition_sha256": candidate.composition_sha256,
                "gradient_norms": gradients["gradient_norms"],
                "controls": controls,
            }
            manifests.append(
                {
                    "run_id": READINESS_RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "model": CANDIDATE_MODEL,
                    "trainable_parameter_count": parameter_count,
                    "composition_sha256": candidate.composition_sha256,
                    "state_dict_sha256": state_sha,
                    "training_rows": 0,
                    "optimizer_steps": 0,
                }
            )
    checks["cross_width_and_seed_state_geometry_identical"] = len(
        set(geometries.values())
    ) == 1
    return manifests, checks, metrics


def reference_composition_views(
    ciphertext_pairs: torch.Tensor,
    structure: Any,
) -> torch.Tensor:
    left = ciphertext_pairs[:, :, 0]
    right = ciphertext_pairs[:, :, 1]
    stages = [triplet(left, right)]
    for slot in (1, 0):
        left = structure.exact_inverse(left, slot)
        right = structure.exact_inverse(right, slot)
        stages.append(triplet(left, right))
        left = structure.apply_inverse_sboxes(left, slot)
        right = structure.apply_inverse_sboxes(right, slot)
        stages.append(triplet(left, right))
    return torch.cat(stages, dim=-1)


def triplet(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack((left, right, torch.remainder(left + right, 2.0)), dim=-1)


def forward_round_trip_exact(
    ciphertext_pairs: torch.Tensor,
    inverse_state: torch.Tensor,
    structure: Any,
) -> bool:
    state = inverse_state[..., :2]
    for slot in (0, 1):
        left = structure.apply_sboxes(state[..., 0], slot)
        right = structure.apply_sboxes(state[..., 1], slot)
        state = torch.stack(
            (
                apply_gf2(structure.linear_matrices[slot], left),
                apply_gf2(structure.linear_matrices[slot], right),
            ),
            dim=-1,
        )
    return torch.equal(state.movedim(-1, -2), ciphertext_pairs)


def control_contract_exact(
    candidate: Any,
    control: Any,
    condition: str,
) -> bool:
    candidate_structure = candidate.runtime_structure
    control_structure = control.runtime_structure
    same_sbox = torch.equal(
        candidate_structure.sbox_truth_bits,
        control_structure.sbox_truth_bits,
    )
    same_linear = torch.equal(
        candidate_structure.linear_matrices,
        control_structure.linear_matrices,
    )
    if condition == "wrong_sbox_semantics":
        return not same_sbox and same_linear and control.apply_sboxes is True
    if condition == "reversed_linear_schedule":
        return (
            same_sbox
            and torch.equal(
                control_structure.linear_matrices,
                candidate_structure.linear_matrices.flip(0),
            )
            and control.apply_sboxes is True
        )
    if condition == "corrupted_linear_operators":
        return same_sbox and not same_linear and control.apply_sboxes is True
    if condition == "no_sbox_composition":
        return same_sbox and same_linear and control.apply_sboxes is False
    if condition == "no_topology":
        identity = torch.eye(candidate_structure.block_bits, dtype=torch.uint8)
        return control.apply_sboxes is False and all(
            torch.equal(matrix, identity)
            for matrix in control_structure.linear_matrices
        )
    raise ValueError("unknown K1-N control contract")


def evaluate_k1n_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_controls: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    task_rows = candidate_task_map(tasks)
    trained = result_map(training_rows, CANDIDATE_MODEL)
    checkpoints = checkpoint_map(checkpoint_manifest, model=CANDIDATE_MODEL)
    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = task_rows[key]
            checkpoint_path = Path(str(trained[key]["training"]["checkpoint_output"]))
            state, checkpoint_sha = load_bound_state(
                checkpoint_path,
                checkpoints[key],
            )
            state_sha = tensor_mapping_sha256(state)
            for split in EXPECTED_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                labels = np.asarray(dataset.labels, dtype=np.float32)
                dataset_sha = differential_dataset_sha256(dataset)
                probabilities: dict[str, np.ndarray] = {}
                models: dict[str, torch.nn.Module] = {}
                for condition in CONTROL_CONDITIONS:
                    model = build_k1n_control(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(state, strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_sha:
                        raise ValueError("K1-N strict load changed candidate state")
                    models[condition] = model
                    probabilities[condition] = predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                aucs = {
                    condition: binary_auc(labels, values)
                    for condition, values in probabilities.items()
                }
                reference = probabilities["exact_composition"]
                effective_gate = float(
                    torch.tanh(
                        models[
                            "exact_composition"
                        ].backbone.residual_gate.detach()
                    )
                )
                for condition in CONTROL_CONDITIONS:
                    values = probabilities[condition]
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "source_role": "candidate",
                            "condition": condition,
                            "rows": int(dataset.features.shape[0]),
                            "auc": aucs[condition],
                            "exact_minus_condition_auc": (
                                aucs["exact_composition"] - aucs[condition]
                            ),
                            "max_abs_probability_delta_from_exact": float(
                                np.max(np.abs(reference - values))
                            ),
                            "mean_abs_probability_delta_from_exact": float(
                                np.mean(np.abs(reference - values))
                            ),
                            "effective_gate": effective_gate,
                            "dataset_sha256": dataset_sha,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha,
                            "state_dict_sha256": state_sha,
                            "composition_sha256": models[
                                condition
                            ].composition_sha256,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
                source = source_controls[(cipher, seed, split, "exact_ordered")]
                if source.get("dataset_sha256") != dataset_sha:
                    raise ValueError("K1-N K1-M anchor dataset digest mismatch")
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "source_role": "anchor",
                        "condition": ANCHOR_CONDITION,
                        "rows": int(dataset.features.shape[0]),
                        "auc": float(source["auc"]),
                        "exact_minus_condition_auc": (
                            aucs["exact_composition"] - float(source["auc"])
                        ),
                        "effective_gate": source.get("effective_gate"),
                        "dataset_sha256": dataset_sha,
                        "checkpoint_path": source.get("checkpoint_path"),
                        "checkpoint_sha256": source.get("checkpoint_sha256"),
                        "state_dict_sha256": source.get("state_dict_sha256"),
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows


def adjudicate_k1n(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    readiness_gate: Mapping[str, Any],
) -> dict[str, Any]:
    grouped = evaluation_map(evaluation_rows)
    expected = {
        (cipher, seed, split, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in EVALUATION_CONDITIONS
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
            and candidate_protocol_frozen(task_rows)
        ),
        "four_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(trained) == expected_task_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "eighty_four_evaluation_rows_complete": (
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
        "semantic_controls_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, condition)].get(
                        "composition_sha256"
                    )
                    for condition in CONTROL_CONDITIONS
                }
            )
            == len(CONTROL_CONDITIONS)
            for cipher in EXPECTED_CIPHERS
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "finite_metrics": all(
            math.isfinite(float(row.get("auc", math.nan)))
            for row in evaluation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            for split in ("same_key_fresh", "cross_key_validation"):
                result = seed_results[cipher][str(seed)][split]
                prefix = f"{cipher}_seed{seed}_{split}"
                if cipher == "uknit64":
                    research_checks[f"{prefix}_auc_floor"] = (
                        result["candidate_auc"] >= UKNIT_AUC_FLOOR
                    )
                    research_checks[f"{prefix}_beats_anchor"] = (
                        result["candidate_minus_anchor"] >= MARGIN
                    )
                    research_checks[f"{prefix}_gate_active"] = (
                        result["effective_gate_abs"] >= FINAL_ACTIVE_GATE
                    )
                else:
                    research_checks[f"{prefix}_retains_anchor"] = (
                        result["candidate_minus_anchor"] >= -MARGIN
                    )
                research_checks[f"{prefix}_beats_controls"] = result[
                    "beats_all_controls"
                ]
    protocol_valid = all(protocol_checks.values())
    all_research = all(research_checks.values())
    dialga_retained = all(
        seed_results["dialga128"][str(seed)][split]["candidate_minus_anchor"]
        >= -MARGIN
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    uknit_signal = all(
        seed_results["uknit64"][str(seed)][split]["candidate_auc"]
        >= UKNIT_AUC_FLOOR
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1n_protocol_invalid"
        next_action = (
            "repair only the failed K1-N implementation or source binding and rerun unchanged"
        )
    elif all_research:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1n_exact_composition_supported"
        )
        next_action = (
            "freeze a separate remote 65536/class disk-cached K1-N diagnostic; "
            "do not call it formal evidence"
        )
    elif dialga_retained and not uknit_signal:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1n_dialga_retained_"
            "uknit_signal_not_supported"
        )
        next_action = (
            "stop operator-view expansion and audit the frozen uKNIT r5 differential "
            "with exact partial-state statistics before another neural redesign"
        )
    elif not dialga_retained:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1n_dialga_anchor_lost"
        next_action = "discard K1-N and return to the completed K1-M calibration"
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1n_semantic_attribution_not_supported"
        )
        next_action = (
            "run a zero-step contribution audit only for the failed semantic control; no scale"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "anchor_and_control_margin": MARGIN,
            "final_active_gate": FINAL_ACTIVE_GATE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class exact-composition diagnostic against K1-M; "
            "not formal scale, attack, SOTA, arbitrary-SPN transfer, or uKNIT ceiling"
        ),
    }


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
            raise ValueError(f"duplicate K1-N evaluation row: {key}")
        mapped[key] = row
    return mapped


def split_result(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cipher: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    candidate_row = grouped[(cipher, seed, split, "exact_composition")]
    candidate = float(candidate_row["auc"])
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
        "effective_gate": float(candidate_row["effective_gate"]),
        "effective_gate_abs": abs(float(candidate_row["effective_gate"])),
    }


def same_dataset_per_split(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                grouped[(cipher, seed, split, condition)].get("dataset_sha256")
                for condition in EVALUATION_CONDITIONS
            }
        )
        == 1
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
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
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )


__all__ = [
    "ANCHOR_CONDITION",
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_DIGESTS",
    "K1M_DECISION",
    "READINESS_RUN_ID",
    "RUN_ID",
    "adjudicate_k1n",
    "build_k1n_control",
    "build_k1n_readiness",
    "candidate_protocol_frozen",
    "candidate_task_map",
    "evaluate_k1n_panel",
]
