from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    EXPECTED_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    checkpoint_map,
    expected_task_keys,
    load_bound_state,
    result_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TRAINING_ROWS,
    MARGIN,
    UKNIT_AUC_FLOOR,
    build_k1k_control,
    candidate_task_map,
    cell_relabel_logit_delta,
    project_features,
    training_protocol_frozen,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1l import (
    audit_gradient_path,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


READINESS_RUN_ID = "i1_uknit_family_ctspn_gate_opening_k1m_readiness_20260728"
RUN_ID = "i1_uknit_family_ctspn_gate_opening_k1m_2048_seed0_seed1_20260728"
K1K_DECISION = (
    "innovation1_uknit_family_ctspn_k1k_dialga_retained_"
    "operator_attribution_not_supported"
)
K1L_DECISION = (
    "innovation1_uknit_family_ctspn_k1l_uknit_zero_gate_gradient_"
    "starvation_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1k_gate": "8922bd1d03de41547f33329b869204d2d05d664514674699f6661a7eaf758055",
    "k1k_checkpoint_manifest": (
        "1c826e182c3762d389a6d575ddbc755331a6a0123fcba87dde7f856006b8473f"
    ),
    "k1k_dataset_manifest": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "k1k_controls": (
        "b08832d8f01fe0091a1a1f07e507dc830833662204c2fbc618f9702eca06d3a0"
    ),
    "k1l_gate": "8be0ba47a207e4cf9af0c51b73c787e9d5c53c02c7ab9be47e4d57271fef6d70",
}
INITIAL_EFFECTIVE_GATE = 0.05
FINAL_ACTIVE_GATE = 0.010
ANCHOR_CONDITION = "k1k_anchor"
EVALUATION_CONDITIONS = (*CONTROL_CONDITIONS, ANCHOR_CONDITION)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_CIPHERS)
    * len(EXPECTED_SEEDS)
    * len(EXPECTED_SPLITS)
    * len(EVALUATION_CONDITIONS)
)
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048


def build_k1m_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    source_checks: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = candidate_task_map(tasks, fail_closed=False)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    protocol_checks = {
        "four_frozen_candidate_tasks": (
            len(tasks) == EXPECTED_TRAINING_ROWS
            and set(task_map) == expected_task_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(task_map),
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
                task_map,
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
            "innovation1_uknit_family_ctspn_k1m_execution_authorized"
            if ready
            else "innovation1_uknit_family_ctspn_k1m_not_ready"
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
            "run the frozen four-row K1-M local diagnostic"
            if ready
            else "repair only the failed K1-M binding or implementation and rerun readiness unchanged"
        ),
        "claim_scope": (
            "zero-training effective-gate-0.05 gradient, geometry, control, and cache readiness only"
        ),
    }


def structural_readiness(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    geometries: dict[tuple[str, int], list[tuple[str, tuple[int, ...]]]] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = tasks[key]
            dataset = datasets[(cipher, seed, "train_seen")]
            input_bits = int(dataset.features.shape[1])
            model = build_k1k_control(
                task=task,
                condition="exact_ordered",
                input_bits=input_bits,
            )
            parameter_count = model_metadata(model)["trainable_parameter_count"]
            initial_gate = float(torch.tanh(model.backbone.residual_gate.detach()))
            features = torch.as_tensor(
                np.asarray(dataset.features[:8]).copy(),
                dtype=torch.float32,
            )
            runtime = project_features(features, model.runtime_structure)
            raw_gate = model.backbone.residual_gate.detach().clone()
            with torch.no_grad():
                model.backbone.residual_gate.zero_()
            with torch.inference_mode():
                zero_logits = model(features)
                base_logits = model.backbone.base.classifier(
                    model.backbone.base.encode(runtime, model.runtime_structure)
                )
            zero_replay_delta = float((zero_logits - base_logits).abs().max())
            with torch.no_grad():
                model.backbone.residual_gate.copy_(raw_gate)
            gradient = audit_gradient_path(
                model,
                dataset,
                effective_gate=INITIAL_EFFECTIVE_GATE,
                batch_size=EXPECTED_BATCH_SIZE,
            )
            residual_gradient_min = min(
                float(gradient["gradient_norms"][group])
                for group in (
                    "cell_encoder",
                    "edge_encoder",
                    "cell_update",
                    "residual_projection",
                )
            )
            default_task = deepcopy(dict(task))
            default_options = deepcopy(dict(default_task["model_options"]))
            default_options.pop("residual_gate_initial_effective", None)
            default_task["model_options"] = default_options
            default_model = build_k1k_control(
                task=default_task,
                condition="exact_ordered",
                input_bits=input_bits,
            )
            state = model.state_dict()
            state_sha = tensor_mapping_sha256(state)
            geometries[key] = [
                (name, tuple(value.shape)) for name, value in state.items()
            ]
            controls: dict[str, Any] = {}
            for condition in CONTROL_CONDITIONS[1:]:
                control = build_k1k_control(
                    task=task,
                    condition=condition,
                    input_bits=input_bits,
                )
                control.load_state_dict(state, strict=True)
                controls[condition] = {
                    "same_state": tensor_mapping_sha256(control.state_dict())
                    == state_sha,
                    "boolean_view_distinct": (
                        control.boolean_view_sha256 != model.boolean_view_sha256
                    ),
                    "topology_edge_distinct": (
                        control.topology_edge_sha256 != model.topology_edge_sha256
                    ),
                }
            prefix = f"{cipher}_seed{seed}"
            checks[f"{prefix}_parameter_count_exact"] = (
                parameter_count == EXPECTED_PARAMETER_COUNT
            )
            checks[f"{prefix}_initial_effective_gate_exact"] = (
                abs(initial_gate - INITIAL_EFFECTIVE_GATE) <= 1e-7
            )
            checks[f"{prefix}_k1k_default_gate_still_zero"] = (
                float(default_model.backbone.residual_gate.detach()) == 0.0
            )
            checks[f"{prefix}_zero_gate_base_replay"] = zero_replay_delta <= 1e-7
            checks[f"{prefix}_all_residual_groups_receive_gradient"] = (
                residual_gradient_min > 1e-8
            )
            checks[f"{prefix}_gradient_probe_restores_state"] = bool(
                gradient["state_restored_exact"]
            )
            checks[f"{prefix}_controls_strict_and_distinct"] = all(
                all(values.values()) for values in controls.values()
            )
            checks[f"{prefix}_joint_cell_relabel_invariant"] = (
                cell_relabel_logit_delta(model, features) <= 1e-6
            )
            metrics[prefix] = {
                "parameter_count": parameter_count,
                "initial_effective_gate": initial_gate,
                "zero_gate_base_replay_max_abs_delta": zero_replay_delta,
                "residual_gradient_norms": gradient["gradient_norms"],
                "controls": controls,
            }
            manifests.append(
                {
                    "run_id": READINESS_RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "model": CANDIDATE_MODEL,
                    "trainable_parameter_count": parameter_count,
                    "initial_effective_gate": initial_gate,
                    "state_dict_sha256": state_sha,
                    "training_rows": 0,
                    "optimizer_steps": 0,
                }
            )
    checks["cross_width_and_seed_state_geometry_identical"] = len(
        {tuple(value) for value in geometries.values()}
    ) == 1
    return manifests, checks, metrics


def evaluate_k1m_panel(
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
            checkpoint_path = Path(
                str(trained[key]["training"]["checkpoint_output"])
            )
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
                    model = build_k1k_control(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(state, strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_sha:
                        raise ValueError("K1-M strict load changed state")
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
                reference = probabilities["exact_ordered"]
                effective_gate = float(
                    torch.tanh(
                        models["exact_ordered"].backbone.residual_gate.detach()
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
                                aucs["exact_ordered"] - aucs[condition]
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
                            "boolean_view_sha256": models[
                                condition
                            ].boolean_view_sha256,
                            "topology_edge_sha256": models[
                                condition
                            ].topology_edge_sha256,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
                source = source_controls[(cipher, seed, split, "exact_ordered")]
                if source.get("dataset_sha256") != dataset_sha:
                    raise ValueError("K1-M anchor dataset digest mismatch")
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
                            aucs["exact_ordered"] - float(source["auc"])
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


def adjudicate_k1m(
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
        "topology_controls_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, condition)].get(
                        "topology_edge_sha256"
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
            and 0.0 <= float(row.get("auc", math.nan)) <= 1.0
            for row in evaluation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            for split in ("same_key_fresh", "cross_key_validation"):
                result = seed_results[cipher][str(seed)][split]
                prefix = f"{cipher}_seed{seed}_{split}"
                research_checks[f"{prefix}_beats_controls"] = bool(
                    result["beats_all_controls"]
                )
                research_checks[f"{prefix}_retains_anchor"] = (
                    result["candidate_minus_anchor"] >= -MARGIN
                )
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
    protocol_valid = all(protocol_checks.values())
    all_research = bool(research_checks) and all(research_checks.values())
    dialga_retained = all(
        research_checks[
            f"dialga128_seed{seed}_{split}_retains_anchor"
        ]
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    uknit_gates_active = all(
        research_checks[f"uknit64_seed{seed}_{split}_gate_active"]
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
    )
    uknit_signal_pass = all(
        research_checks[f"uknit64_seed{seed}_{split}_{check}"]
        for seed in EXPECTED_SEEDS
        for split in ("same_key_fresh", "cross_key_validation")
        for check in ("auc_floor", "beats_anchor", "beats_controls")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1m_protocol_invalid"
        next_action = (
            "repair only the failed K1-M protocol or binding and rerun unchanged"
        )
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1m_gate_opening_supported"
        next_action = (
            "retain K1-M and preregister one remote 65536/class disk-cached diagnostic"
        )
    elif not dialga_retained:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1m_dialga_anchor_lost"
        next_action = "discard K1-M and return to K1-K/K1-I calibration"
    elif uknit_gates_active and not uknit_signal_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1m_gate_opened_"
            "uknit_signal_not_supported"
        )
        next_action = (
            "stop gate scheduling and test exact heterogeneous S-box/operator "
            "composition as the next single variable"
        )
    elif not uknit_gates_active:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1m_uknit_gate_reclosed"
        next_action = (
            "hold scale and audit whether a fixed bounded 0.05 gate preserves residual gradients"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1m_controls_not_supported"
        next_action = (
            "hold scale and retain the K1-L operator-insensitivity conclusion"
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
        "descriptive_diagnostics": {
            "dialga_retained": dialga_retained,
            "uknit_gates_active": uknit_gates_active,
            "uknit_signal_pass": uknit_signal_pass,
        },
        "thresholds": {
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "anchor_and_control_margin": MARGIN,
            "final_active_gate": FINAL_ACTIVE_GATE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class gate-initialization diagnostic against K1-K; "
            "not formal scale, attack, SOTA, arbitrary-SPN transfer, or uKNIT ceiling"
        ),
    }


def candidate_protocol_frozen(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    return set(tasks) == expected_task_keys() and all(
        task.get("model_key") == CANDIDATE_MODEL
        and task.get("samples_per_class") == 2048
        and task.get("pairs_per_sample") == 4
        and task.get("negative_mode") == "encrypted_random_plaintexts"
        and task.get("target_epochs") == EXPECTED_EPOCHS
        and task.get("loss") == "mse"
        and task.get("optimizer") == "adam"
        and float(
            task.get("model_options", {}).get(
                "residual_gate_initial_effective", math.nan
            )
        )
        == INITIAL_EFFECTIVE_GATE
        for task in tasks.values()
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
            raise ValueError(f"duplicate K1-M evaluation row: {key}")
        mapped[key] = row
    return mapped


def split_result(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cipher: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    candidate_row = grouped[(cipher, seed, split, "exact_ordered")]
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
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_SOURCE_DIGESTS",
    "FINAL_ACTIVE_GATE",
    "INITIAL_EFFECTIVE_GATE",
    "K1K_DECISION",
    "K1L_DECISION",
    "READINESS_RUN_ID",
    "RUN_ID",
    "adjudicate_k1m",
    "build_k1m_readiness",
    "candidate_protocol_frozen",
    "evaluate_k1m_panel",
]
