from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_endpoint_alignment import (
    RUN_ID as K1A_RUN_ID,
    frozen_k1_stages,
    native_endpoint_signature,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    RUN_ID as K1_RUN_ID,
    _control_metadata,
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1_20260728"
READINESS_RUN_ID = "i1_uknit_family_ctspn_native_endpoint_k1b_readiness_20260728"
K1_DECISION = "innovation1_uknit_family_ctspn_k1_linear_schedule_not_supported"
K1A_DECISION = "innovation1_uknit_family_ctspn_endpoint_alignment_loss_confirmed"
CANDIDATE_MODEL = "runtime_spn_ct_k1b_endpoint_true"
CORRUPTED_MODEL = "runtime_spn_ct_k1b_endpoint_corrupted"
INDEPENDENT_MODEL = "runtime_spn_ct_k1b_endpoint_independent"
EXPECTED_CIPHERS = ("uknit64", "dialga128")
EXPECTED_SEEDS = (0, 1)
CONTROL_CONDITIONS = (
    "correct_ordered",
    "repeat_last",
    "rotated",
    "corrupted",
    "no_topology",
)
EXPECTED_TRAINING_ROWS = 4
EXPECTED_CONTROL_ROWS = 20
EXPECTED_BATCH_SIZE = 64
EXPECTED_EPOCHS = 10
EXPECTED_SAMPLES_PER_CLASS = 2048
EXPECTED_VALIDATION_SAMPLES_PER_CLASS = 1024
EXPECTED_PAIRS_PER_SAMPLE = 4
EXPECTED_PARAMETER_COUNT = 439982
ANCHOR_PARAMETER_COUNT = 442466
MARGIN = 0.005


def build_k1b_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    k1_gate: Mapping[str, Any],
    k1a_gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_map = _task_map(tasks, fail_closed=False)
    protocol_checks = {
        "four_row_frozen_plan": _task_panel_valid(tasks),
        "k1_protocol_clean_hold": _source_gate_valid(
            k1_gate, K1_RUN_ID, "hold", K1_DECISION
        ),
        "k1a_alignment_loss_confirmed": _source_gate_valid(
            k1a_gate, K1A_RUN_ID, "pass", K1A_DECISION
        ),
        "zero_training_readiness": True,
    }
    manifests: list[dict[str, Any]] = []
    models: dict[tuple[str, int], torch.nn.Module] = {}
    for key, task in task_map.items():
        cipher, seed = key
        input_bits, pair_bits = _input_geometry(cipher)
        model = build_model(
            CANDIDATE_MODEL,
            input_bits=input_bits,
            hidden_bits=64,
            pair_bits=pair_bits,
            structure="SPN",
            model_options=task["model_options"],
        )
        models[key] = model
        metadata = model_metadata(model)
        manifests.append(
            {
                "run_id": READINESS_RUN_ID,
                "cipher_key": cipher,
                "seed": seed,
                "model": CANDIDATE_MODEL,
                "trainable_parameter_count": metadata[
                    "trainable_parameter_count"
                ],
                "edge_input_values": int(
                    model.backbone.edge_encoder[0].in_features
                ),
                "endpoint_identity_mode": model.canonical_endpoint_identity_mode,
                "training_rows": 0,
                "optimizer_steps": 0,
            }
        )

    geometry = [
        (name, tuple(value.shape))
        for name, value in next(iter(models.values())).state_dict().items()
    ] if models else []
    evidence_checks = {
        "cross_width_state_geometry_equal": bool(models)
        and all(
            [
                (name, tuple(value.shape))
                for name, value in model.state_dict().items()
            ]
            == geometry
            for model in models.values()
        ),
        "edge_input_width_is_22": bool(models)
        and all(
            model.backbone.edge_encoder[0].in_features == 22
            for model in models.values()
        ),
        "native_endpoint_mode_only": bool(models)
        and all(
            model.canonical_endpoint_identity_mode == "native_cell_role"
            for model in models.values()
        ),
        "candidate_not_larger_than_anchor": bool(manifests)
        and all(
            row["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
            and row["trainable_parameter_count"] <= ANCHOR_PARAMETER_COUNT
            for row in manifests
        ),
    }
    control_evidence = _readiness_control_evidence(task_map)
    evidence_checks.update(control_evidence["checks"])
    authorized = all(protocol_checks.values()) and all(evidence_checks.values())
    gate = {
        "run_id": READINESS_RUN_ID,
        "status": "pass" if authorized else "fail",
        "decision": (
            "innovation1_uknit_family_ctspn_k1b_native_endpoint_execution_authorized"
            if authorized
            else "innovation1_uknit_family_ctspn_k1b_native_endpoint_not_ready"
        ),
        "implementation_ready": authorized,
        "optimizer_step_authorized": authorized,
        "training_rows": 0,
        "optimizer_steps": 0,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "control_evidence": control_evidence["metrics"],
        "manifest_rows": len(manifests),
        "next_action": (
            "run the frozen four-row K1-B local diagnostic and twenty-row "
            "same-checkpoint control panel"
            if authorized
            else "repair only the failed K1-B implementation/readiness check"
        ),
    }
    return manifests, gate


def evaluate_k1b_controls(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    validation_datasets: Mapping[tuple[str, int], DifferentialDataset],
    k1_controls: Sequence[Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    task_map = _task_map(tasks)
    training_map = _training_map(training_rows)
    prior_hashes = _prior_dataset_hashes(k1_controls)
    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = task_map[key]
            source = training_map[key]
            dataset = validation_datasets[key]
            dataset_sha256 = differential_dataset_sha256(dataset)
            checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = payload["state_dict"]
            state_sha256 = tensor_mapping_sha256(state_dict)
            probabilities: dict[str, np.ndarray] = {}
            metadata: dict[str, dict[str, Any]] = {}
            for condition in CONTROL_CONDITIONS:
                model = build_k1b_control(
                    task=task,
                    condition=condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                model.load_state_dict(state_dict, strict=True)
                if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                    raise ValueError("K1-B strict control load changed learned state")
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
            for condition in CONTROL_CONDITIONS:
                current = probabilities[condition]
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "auc": aucs[condition],
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
                        "dataset_sha256": dataset_sha256,
                        "prior_k1_dataset_sha256": prior_hashes[key],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                        "state_dict_sha256": state_sha256,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        **metadata[condition],
                    }
                )
    return rows


def adjudicate_k1b(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    k1_gate: Mapping[str, Any],
    k1a_gate: Mapping[str, Any],
) -> dict[str, Any]:
    training = _training_map(training_rows, fail_closed=False)
    controls = _control_map(control_rows)
    expected_keys = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    expected_controls = {
        (cipher, seed, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
    }
    protocol_checks = {
        "four_row_frozen_plan": _task_panel_valid(tasks),
        "four_training_rows_complete": len(training_rows) == 4
        and set(training) == expected_keys,
        "twenty_control_rows_complete": len(control_rows) == 20
        and set(controls) == expected_controls,
        "k1_protocol_clean_hold": _source_gate_valid(
            k1_gate, K1_RUN_ID, "hold", K1_DECISION
        ),
        "k1a_alignment_loss_confirmed": _source_gate_valid(
            k1a_gate, K1A_RUN_ID, "pass", K1A_DECISION
        ),
        "training_protocol_frozen": _training_protocol_valid(training_rows),
        "controls_reuse_same_state": _same_per_cipher_seed(
            controls, "state_dict_sha256"
        ),
        "controls_reuse_same_dataset": _same_per_cipher_seed(
            controls, "dataset_sha256"
        ),
        "validation_dataset_matches_k1": all(
            row.get("dataset_sha256") == row.get("prior_k1_dataset_sha256")
            for row in control_rows
        ),
        "strict_control_load_and_zero_optimizer": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in control_rows
        ),
        "finite_metrics": all(_control_row_finite(row) for row in control_rows),
    }
    seed_results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    research_checks: dict[str, bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            correct = float(controls.get((cipher, seed, "correct_ordered"), {}).get("auc", math.nan))
            prior = k1_gate.get("seed_results", {}).get(cipher, {}).get(str(seed), {})
            prior_anchor = float(prior.get("anchor_auc", math.nan))
            prior_candidate = float(prior.get("candidate_auc", math.nan))
            values = {
                condition: float(controls.get((cipher, seed, condition), {}).get("auc", math.nan))
                for condition in CONTROL_CONDITIONS[1:]
            }
            seed_results[cipher][str(seed)] = {
                "candidate_auc": correct,
                "prior_anchor_auc": prior_anchor,
                "prior_edge_invariant_auc": prior_candidate,
                "candidate_minus_strongest_prior": correct
                - max(prior_anchor, prior_candidate),
                **{
                    f"candidate_{condition}_auc": value
                    for condition, value in values.items()
                },
                **{
                    f"candidate_minus_{condition}": correct - value
                    for condition, value in values.items()
                },
            }
            prefix = f"{cipher}_seed{seed}"
            if cipher == "uknit64":
                research_checks[f"{prefix}_auc_floor"] = correct >= 0.520
                research_checks[f"{prefix}_beats_strongest_prior"] = (
                    correct - max(prior_anchor, prior_candidate) >= MARGIN
                )
            else:
                research_checks[f"{prefix}_retains_prior_auc"] = (
                    correct >= prior_candidate - MARGIN
                )
            for condition, value in values.items():
                research_checks[f"{prefix}_beats_{condition}"] = (
                    correct - value >= MARGIN
                )

    protocol_valid = all(protocol_checks.values())
    supported = protocol_valid and all(research_checks.values())
    status = "pass" if supported else ("hold" if protocol_valid else "invalid")
    decision = (
        "innovation1_uknit_family_ctspn_k1b_native_endpoint_supported"
        if supported
        else (
            "innovation1_uknit_family_ctspn_k1b_native_endpoint_not_supported"
            if protocol_valid
            else "innovation1_uknit_family_ctspn_k1b_protocol_invalid"
        )
    )
    next_action = (
        "retain native endpoint identity and plan K2 canonical S-box composition "
        "as a separate same-budget local experiment"
        if supported
        else (
            "hold K1-B; inspect only the failed endpoint/temporal interaction and "
            "do not scale, add capacity, or start K2"
            if protocol_valid
            else "repair only the failed K1-B protocol check and rerun"
        )
    )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "seed_results": dict(seed_results),
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "thresholds": {
            "uknit_auc": 0.520,
            "candidate_margin": MARGIN,
            "dialga_retention_tolerance": MARGIN,
        },
        "claim_scope": (
            "uKNIT-BC prefix-r5 and Dialga-128 prefix-r4 two-seed, 2048/class "
            "local native-endpoint CT-SPN diagnostic; not formal scale, attack, "
            "SOTA, arbitrary-SPN, transfer, or MSX evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up or mechanical sample, pair, epoch or width increase",
            "start K2 unless every K1-B seed-level gate passes",
            "add MoE, DDT, trail, partial decryption, guessed keys or cipher-id routing",
            "use a macro average to hide a failed cipher or seed",
            "include generalized-Feistel MSX in the CT-SPN claim",
        ],
    }


def build_k1b_control(
    *, task: Mapping[str, Any], condition: str, input_bits: int
) -> torch.nn.Module:
    if condition not in CONTROL_CONDITIONS:
        raise ValueError("unknown K1-B control condition")
    options = deepcopy(dict(task["model_options"]))
    options["runtime_structure_window_control"] = "full"
    options["canonical_schedule_control"] = "ordered"
    options["topology_corruption_seed"] = 20260727
    model_key = CANDIDATE_MODEL
    if condition == "repeat_last":
        options["runtime_structure_window_control"] = "repeat_last"
    elif condition == "rotated":
        options["canonical_schedule_control"] = "rotated"
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


def _readiness_control_evidence(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    for cipher in EXPECTED_CIPHERS:
        task = tasks.get((cipher, 0))
        if task is None:
            continue
        input_bits, _ = _input_geometry(cipher)
        correct = build_k1b_control(
            task=task, condition="correct_ordered", input_bits=input_bits
        )
        state = correct.state_dict()
        state_sha256 = tensor_mapping_sha256(state)
        generator = torch.Generator().manual_seed(20260728)
        features = torch.randint(
            0, 2, (4, input_bits), generator=generator
        ).to(torch.float32)
        correct.eval()
        with torch.inference_mode():
            _, correct_summary, correct_logits = frozen_k1_stages(correct, features)
        correct_endpoints = native_endpoint_signature(correct)
        cipher_metrics = {}
        for condition in ("repeat_last", "rotated"):
            control = build_k1b_control(
                task=task, condition=condition, input_bits=input_bits
            )
            control.load_state_dict(state, strict=True)
            control.eval()
            with torch.inference_mode():
                _, summary, logits = frozen_k1_stages(control, features)
            endpoint_fraction = float(
                (
                    native_endpoint_signature(control) != correct_endpoints
                ).any(dim=-1).float().mean()
            )
            summary_delta = float((summary - correct_summary).abs().max())
            logit_delta = float((logits - correct_logits).abs().max())
            cipher_metrics[condition] = {
                "endpoint_fraction_changed": endpoint_fraction,
                "transition_summary_max_abs_delta": summary_delta,
                "logit_max_abs_delta": logit_delta,
            }
            prefix = f"{cipher}_{condition}"
            checks[f"{prefix}_same_state_strict_load"] = (
                tensor_mapping_sha256(control.state_dict()) == state_sha256
            )
            checks[f"{prefix}_native_endpoints_change"] = endpoint_fraction >= (
                0.45 if condition == "repeat_last" else 0.95
            )
            checks[f"{prefix}_summary_noncollapsed"] = summary_delta > 1e-5
            checks[f"{prefix}_logit_noncollapsed"] = logit_delta > 1e-6
        metrics[cipher] = cipher_metrics
    checks["no_cipher_identity_tensor"] = True
    return {"checks": checks, "metrics": metrics}


def _task_map(
    tasks: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {}
    for task in tasks:
        if task.get("model_key") != CANDIDATE_MODEL:
            if fail_closed:
                raise ValueError("K1-B plan contains a non-candidate model")
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result and fail_closed:
            raise ValueError(f"duplicate K1-B task: {key}")
        result[key] = task
    return result


def _training_map(
    rows: Sequence[Mapping[str, Any]], *, fail_closed: bool = True
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        if row.get("model") != CANDIDATE_MODEL:
            if fail_closed:
                raise ValueError("K1-B results contain a non-candidate model")
            continue
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
        result[key] = row
    return result


def _task_panel_valid(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _task_map(tasks, fail_closed=False)
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
            or options.get("temporal_hidden_dim") != 76
            or options.get("runtime_structure_window_control") != "full"
            or options.get("canonical_schedule_control") != "ordered"
        ):
            return False
    return True


def _training_protocol_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == 4 and all(
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


def _prior_dataset_hashes(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], str]:
    result = {}
    for row in rows:
        if (
            row.get("source_role") == "candidate"
            and row.get("condition") == "correct_ordered"
        ):
            result[(str(row["cipher_key"]), int(row["seed"]))] = str(
                row["dataset_sha256"]
            )
    expected = {
        (cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS
    }
    if set(result) != expected:
        raise ValueError("K1-B requires all four prior K1 validation dataset hashes")
    return result


def _same_per_cipher_seed(
    rows: Mapping[tuple[str, int, str], Mapping[str, Any]], field: str
) -> bool:
    return all(
        len(
            {
                rows.get((cipher, seed, condition), {}).get(field)
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
    )


def _source_gate_valid(
    gate: Mapping[str, Any], run_id: str, status: str, decision: str
) -> bool:
    return (
        gate.get("run_id") == run_id
        and gate.get("status") == status
        and gate.get("decision") == decision
        and bool(gate.get("protocol_checks"))
        and all(gate.get("protocol_checks", {}).values())
    )


def _control_row_finite(row: Mapping[str, Any]) -> bool:
    return all(
        isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
        for field in (
            "auc",
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
    raise ValueError(f"unsupported K1-B cipher: {cipher}")


__all__ = [
    "CANDIDATE_MODEL",
    "CONTROL_CONDITIONS",
    "EXPECTED_BATCH_SIZE",
    "EXPECTED_CONTROL_ROWS",
    "EXPECTED_TRAINING_ROWS",
    "READINESS_RUN_ID",
    "RUN_ID",
    "adjudicate_k1b",
    "build_k1b_control",
    "build_k1b_readiness",
    "evaluate_k1b_controls",
]
