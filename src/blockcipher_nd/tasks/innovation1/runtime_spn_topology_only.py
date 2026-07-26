from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_holdout import (
    HOLDOUT_CIPHER,
    SOURCE_CIPHERS,
    SOURCE_DISPLAY_NAMES,
    _cache_probe,
    _file_sha256,
    _read_json,
    _validate_checkpoint_payload,
    _validate_role_checkpoint,
    build_wrong_sbox_structure,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_anf_operator import (
    build_sbox_operator_controls,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_heterogeneous_holdout import (
    _clone_state_dict,
    _state_dict_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    RelationModeRuntimeE4,
    _evaluate_target,
    _load_resumable_role,
    _load_source_tasks,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    _training_config,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.metrics import predict_binary_probabilities
from blockcipher_nd.training.runtime_spn_joint import train_runtime_spn_joint
from blockcipher_nd.training.types import ProgressCallback


EXPECTED_SEEDS = (0, 1)
TARGET_EVALUATIONS = (
    "candidate_correct",
    "candidate_corrupted_target",
    "candidate_no_topology_target",
    "candidate_wrong_sbox_target",
)
A8_TARGET_REFERENCES = (
    "candidate_correct",
    "no_topology_trained_anchor",
)


class _BoundRuntimeE4(nn.Module):
    def __init__(
        self,
        model: RelationModeRuntimeE4,
        structure: RuntimeSpnStructure,
    ) -> None:
        super().__init__()
        self.model = model
        self.structure = structure

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features, self.structure)


def load_and_validate_topology_only_config(
    path: Path,
    *,
    project_root: Path,
    require_readiness: bool,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("C1 config schema_version must be 1")
    if config.get("experiment") != "innovation1_runtime_spn_topology_only_c1":
        raise ValueError("C1 experiment name drifted")
    if tuple(config.get("source_ciphers", ())) != SOURCE_CIPHERS:
        raise ValueError("C1 source cipher panel drifted")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("C1 must hold out Dialga")

    required_candidate = {
        "backbone": "RuntimeE4EquivariantSpnDistinguisher",
        "architecture_variant": "exact_gf2_topology_only",
        "sbox_context_mode": "edge_gate",
        "sbox_context_scale": 0.0,
        "sbox_boolean_operator_mode": "none",
        "sbox_boolean_operator_scale": 0.0,
        "gradient_combination": (
            "representation_l2_equalized_pcgrad_fixed_order"
        ),
        "expected_parameter_count": 442466,
        "seeds": [0, 1],
    }
    if config.get("candidate") != required_candidate:
        raise ValueError("C1 candidate contract drifted")

    required_training = {
        "samples_per_class_per_source": 2048,
        "validation_samples_per_class_per_source": 1024,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "epochs": 10,
        "batch_size": 256,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "checkpoint_metric": "val_macro_auc",
        "restore_best_checkpoint": True,
        "device": "cpu",
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
    }
    if config.get("training") != required_training:
        raise ValueError("C1 training contract drifted")

    required_evaluations = {
        "candidate_correct": {"structure": "correct", "relation_mode": "true"},
        "candidate_corrupted_target": {
            "structure": "corrupted",
            "relation_mode": "true",
        },
        "candidate_no_topology_target": {
            "structure": "correct",
            "relation_mode": "independent",
        },
        "candidate_wrong_sbox_target": {
            "structure": "wrong_sbox",
            "relation_mode": "true",
        },
    }
    if config.get("target_evaluations") != required_evaluations:
        raise ValueError("C1 target evaluation panel drifted")

    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "trained_no_topology_margin": 0.005,
        "a8_retention_tolerance": 0.005,
        "sbox_invariance_tolerance": 0.0000001,
        "minimum_conflict_projections_per_seed": 1,
        "required_seeds": [0, 1],
        "expected_result_rows": 28,
        "expected_history_rows": 20,
    }
    if config.get("gate") != required_gate:
        raise ValueError("C1 research gate drifted")

    source = config.get("source", {})
    for path_key, hash_key in (
        ("protocol_config_path", "protocol_config_sha256"),
        ("a8_config_path", "a8_config_sha256"),
        ("a8_gate_path", "a8_gate_sha256"),
        ("a8_validation_path", "a8_validation_sha256"),
        ("a8_results_path", "a8_results_sha256"),
        ("s1_gate_path", "s1_gate_sha256"),
        ("s2_gate_path", "s2_gate_sha256"),
    ):
        if _file_sha256(project_root / source[path_key]) != source.get(hash_key):
            raise ValueError(f"C1 frozen source hash drifted: {path_key}")

    base = load_and_validate_holdout_config(
        project_root / source["protocol_config_path"]
    )
    if base["training"]["samples_per_class"] != 2048:
        raise ValueError("C1 base samples_per_class drifted")
    if base["training"]["validation_samples_per_class"] != 1024:
        raise ValueError("C1 base validation scale drifted")
    if base["training"]["pairs_per_sample"] != 4:
        raise ValueError("C1 base pair count drifted")
    if base["training"]["negative_mode"] != "encrypted_random_plaintexts":
        raise ValueError("C1 strict negative protocol drifted")

    for gate_key, decision_key in (
        ("a8_gate_path", "a8_required_decision"),
        ("s1_gate_path", "s1_required_decision"),
        ("s2_gate_path", "s2_required_decision"),
    ):
        gate = _read_json(project_root / source[gate_key])
        if gate.get("status") != "hold" or gate.get("decision") != source.get(
            decision_key
        ):
            raise ValueError(f"C1 dependency decision drifted: {gate_key}")
    a8_validation = _read_json(project_root / source["a8_validation_path"])
    if a8_validation.get("status") != "pass" or not all(
        a8_validation.get("checks", {}).values()
    ):
        raise ValueError("C1 requires valid A8 protocol evidence")

    if require_readiness:
        readiness = _read_json(project_root / _readiness_gate_path())
        if readiness.get("status") != "pass" or readiness.get("decision") != (
            "innovation1_runtime_spn_topology_only_c1_readiness_passed"
        ):
            raise ValueError("C1 readiness did not pass")
        if not all(readiness.get("checks", {}).values()):
            raise ValueError("C1 readiness contains a failed check")
    return config


def topology_only_spec(base_model: dict[str, Any]) -> RuntimeParameterizedSpnSpec:
    return replace(
        _plain_spec(base_model),
        sbox_context_scale=0.0,
        sbox_boolean_operator_mode="none",
        sbox_boolean_operator_scale=0.0,
    )


def run_topology_only_readiness(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    spec = topology_only_spec(base["model"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26_072_601)
        candidate = RuntimeE4EquivariantSpnDistinguisher(spec).eval()
        independent = RuntimeE4EquivariantSpnDistinguisher(spec).eval()
        independent.load_state_dict(candidate.state_dict(), strict=True)

    parameter_counts = {
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
        "independent": sum(
            parameter.numel() for parameter in independent.parameters()
        ),
    }
    topology_deltas: dict[str, dict[str, float]] = {}
    sbox_deltas: dict[str, float] = {}
    pair_swap_errors: dict[str, float] = {}
    outputs: list[torch.Tensor] = []
    for index, cipher in enumerate((*SOURCE_CIPHERS, HOLDOUT_CIPHER)):
        structure = structures[cipher]
        wrong_sbox = build_sbox_operator_controls(structure)["input_permuted"]
        generator = torch.Generator().manual_seed(26_072_610 + index)
        pairs = torch.randint(
            0,
            2,
            (3, 4, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            correct = candidate(pairs, structure, relation_mode="true")
            corrupted = candidate(
                pairs,
                structure.corrupted(),
                relation_mode="true",
            )
            no_topology = candidate(pairs, structure, relation_mode="independent")
            wrong = candidate(pairs, wrong_sbox, relation_mode="true")
            pair_swapped = candidate(
                pairs.flip(2),
                structure,
                relation_mode="true",
            )
        topology_deltas[cipher] = {
            "corrupted": float((correct - corrupted).abs().max()),
            "no_topology": float((correct - no_topology).abs().max()),
        }
        sbox_deltas[cipher] = float((correct - wrong).abs().max())
        pair_swap_errors[cipher] = float((correct - pair_swapped).abs().max())
        outputs.extend((correct, corrupted, no_topology, wrong))

    relabel_errors = {}
    for index, cipher in enumerate(("uknit64", "dialga128")):
        structure = structures[cipher]
        permutation = tuple(reversed(range(structure.cells)))
        relabeled_structure, bit_permutation = structure.relabel_cells(permutation)
        generator = torch.Generator().manual_seed(26_072_620 + index)
        pairs = torch.randint(
            0,
            2,
            (2, 4, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        relabeled_pairs = torch.empty_like(pairs)
        relabeled_pairs[..., bit_permutation] = pairs
        with torch.no_grad():
            original = candidate(pairs, structure, relation_mode="true")
            relabeled = candidate(
                relabeled_pairs,
                relabeled_structure,
                relation_mode="true",
            )
        relabel_errors[cipher] = float((original - relabeled).abs().max())

    gradient_structure = structures["skinny64"]
    gradient_pairs = torch.randint(
        0,
        2,
        (3, 4, 2, gradient_structure.block_bits),
        generator=torch.Generator().manual_seed(26_072_630),
        dtype=torch.float32,
    )
    candidate.zero_grad(set_to_none=True)
    candidate(gradient_pairs, gradient_structure, relation_mode="true").sum().backward()
    gradients = [
        parameter.grad
        for parameter in candidate.parameters()
        if parameter.grad is not None
    ]
    gradient_finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    gradient_nonzero = any(float(value.abs().sum()) > 0.0 for value in gradients)
    sbox_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in candidate.named_parameters()
        if "sbox_encoder" in name and parameter.grad is not None
    )

    cache_probe = _cache_probe(base, project_root)
    a8_references = load_a8_references(
        project_root / config["source"]["a8_results_path"]
    )
    checks = {
        "dependencies_frozen_and_valid": True,
        "parameter_and_state_geometry_matched": set(parameter_counts.values())
        == {config["candidate"]["expected_parameter_count"]}
        and tuple(candidate.state_dict()) == tuple(independent.state_dict()),
        "all_sbox_paths_functionally_disabled": spec.sbox_context_scale == 0.0
        and spec.sbox_boolean_operator_mode == "none"
        and spec.sbox_boolean_operator_scale == 0.0,
        "sbox_counterfactuals_exactly_invariant": all(
            value <= config["gate"]["sbox_invariance_tolerance"]
            for value in sbox_deltas.values()
        ),
        "topology_counterfactuals_distinct": all(
            value > 1e-7
            for by_cipher in topology_deltas.values()
            for value in by_cipher.values()
        ),
        "pair_swap_invariant": all(
            value <= 1e-6 for value in pair_swap_errors.values()
        ),
        "cell_relabeling_invariant": all(
            value <= 1e-6 for value in relabel_errors.values()
        ),
        "cache_manifest_ready": cache_probe["passed"],
        "target_train_cache_not_referenced": not cache_probe[
            "target_train_referenced"
        ],
        "forward_outputs_finite": all(
            bool(torch.isfinite(value).all()) for value in outputs
        ),
        "representation_gradients_finite_and_nonzero": gradient_finite
        and gradient_nonzero,
        "disabled_sbox_encoder_gradient_zero": sbox_gradient_l1 == 0.0,
        "a8_reference_panel_complete": _a8_reference_panel_complete(a8_references),
        "target_training_rows_zero": config["training"]["target_training_rows"]
        == 0,
        "target_optimizer_steps_zero": config["training"]["target_optimizer_steps"]
        == 0,
        "artifact_contract_frozen": config["gate"]["expected_result_rows"] == 28
        and config["gate"]["expected_history_rows"] == 20,
    }
    passed = all(checks.values())
    return {
        "run_id": "i1_runtime_spn_topology_only_c1_readiness_20260726",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_topology_only_c1_readiness_passed"
            if passed
            else "innovation1_runtime_spn_topology_only_c1_readiness_failed"
        ),
        "checks": checks,
        "parameter_counts": parameter_counts,
        "topology_logit_deltas": topology_deltas,
        "sbox_logit_deltas": sbox_deltas,
        "pair_swap_errors": pair_swap_errors,
        "cell_relabel_errors": relabel_errors,
        "sbox_encoder_gradient_l1": sbox_gradient_l1,
        "cache_probe": cache_probe,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "next_action": (
            "run the frozen C1 two-seed local diagnostic"
            if passed
            else "repair only the failed C1 readiness invariant"
        ),
    }


def run_topology_only(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    spec = topology_only_spec(base["model"])
    wrong_sbox = build_wrong_sbox_structure(
        structures[HOLDOUT_CIPHER],
        structures["gift64"],
    )
    checkpoint_root = output_root / "checkpoints"
    role_root = output_root / "role-results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    role_root.mkdir(parents=True, exist_ok=True)
    config_hash = config_sha256(config_path)
    roles: dict[int, dict[str, Any]] = {}
    target_loaded_after_training: dict[int, bool] = {}

    for seed in EXPECTED_SEEDS:
        tasks = _load_source_tasks(
            base,
            seed=seed,
            structures=structures,
            progress_callback=progress_callback,
            source_ciphers=SOURCE_CIPHERS,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initializer = RelationModeRuntimeE4(spec, "true")
        initial_state = _clone_state_dict(initializer.state_dict())
        initial_hash = _state_dict_sha256(initial_state)
        checkpoint_path = checkpoint_root / f"seed{seed}-topology-only.pt"
        role_path = role_root / f"seed{seed}-topology-only.json"
        resumed = _load_resumable_role(
            role_path,
            checkpoint_path,
            config_sha256=config_hash,
        )
        if resumed is not None:
            _validate_role_checkpoint(
                resumed,
                checkpoint_path=checkpoint_path,
                seed=seed,
                role="topology_only_candidate",
                relation_mode="true",
                config_hash=config_hash,
                initial_hash=initial_hash,
            )
            role_payload = resumed
            _emit(progress_callback, "source_role_reused", seed=seed)
        else:
            model = RelationModeRuntimeE4(spec, "true")
            model.load_state_dict(initial_state, strict=True)
            _emit(progress_callback, "source_role_start", seed=seed)
            result = train_runtime_spn_joint(
                model,
                tasks,
                _training_config(base["training"], seed),
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda event, payload, seed=seed: progress_callback(
                        event,
                        {"seed": seed, "role": "topology_only_candidate", **payload},
                    )
                ),
                gradient_combination=config["candidate"]["gradient_combination"],
            )
            checkpoint = {
                "state_dict": _clone_state_dict(model.state_dict()),
                "seed": seed,
                "role": "topology_only_candidate",
                "relation_mode": "true",
                "config_sha256": config_hash,
                "initial_state_sha256": initial_hash,
                "best_epoch": result.metadata["best_epoch"],
                "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
                "holdout_cipher": HOLDOUT_CIPHER,
                "sbox_context_scale": 0.0,
            }
            torch.save(checkpoint, checkpoint_path)
            role_payload = {
                "seed": seed,
                "role": "topology_only_candidate",
                "relation_mode": "true",
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "config_sha256": config_hash,
                "initial_state_sha256": initial_hash,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "history": result.history,
                "train_metrics": result.train_metrics,
                "validation_metrics": result.validation_metrics,
                "metadata": result.metadata,
                "gradient_diagnostics": result.gradient_diagnostics,
            }
            _write_json(role_path, role_payload)
            _emit(
                progress_callback,
                "source_role_done",
                seed=seed,
                best_epoch=result.metadata["best_epoch"],
            )
        roles[seed] = role_payload

        _emit(progress_callback, "target_validation_load_start", seed=seed)
        target_dataset = _load_target_validation(
            base,
            seed=seed,
            progress_callback=progress_callback,
            holdout_cipher=HOLDOUT_CIPHER,
        )
        target_loaded_after_training[seed] = True
        checkpoint_hash = _file_sha256(checkpoint_path)
        if checkpoint_hash != role_payload["checkpoint_sha256"]:
            raise ValueError("C1 checkpoint changed after source training")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint_payload(
            checkpoint,
            seed=seed,
            role="topology_only_candidate",
            relation_mode="true",
            config_hash=config_hash,
            initial_hash=initial_hash,
        )
        evaluations: dict[str, dict[str, Any]] = {}
        probability_by_name: dict[str, np.ndarray] = {}
        for name in TARGET_EVALUATIONS:
            evaluation = config["target_evaluations"][name]
            structure = structures[HOLDOUT_CIPHER]
            if evaluation["structure"] == "corrupted":
                structure = structure.corrupted()
            elif evaluation["structure"] == "wrong_sbox":
                structure = wrong_sbox
            model = RelationModeRuntimeE4(spec, evaluation["relation_mode"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            metrics = _evaluate_target(
                model,
                target_dataset,
                structure,
                base["training"],
                holdout_cipher=HOLDOUT_CIPHER,
            )
            probabilities = predict_binary_probabilities(
                _BoundRuntimeE4(model, structure),
                target_dataset,
                batch_size=int(base["training"]["batch_size"]),
                device="cpu",
            )
            probability_by_name[name] = probabilities
            evaluations[name] = {
                "structure": evaluation["structure"],
                "relation_mode": evaluation["relation_mode"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "metrics": metrics,
                "optimizer_steps": 0,
                "target_head_trained": False,
            }
            _emit(
                progress_callback,
                "target_evaluation_done",
                seed=seed,
                evaluation=name,
                auc=metrics["auc"],
            )
        role_payload["target_evaluations"] = evaluations
        role_payload["sbox_probability_delta"] = float(
            np.max(
                np.abs(
                    probability_by_name["candidate_correct"]
                    - probability_by_name["candidate_wrong_sbox_target"]
                )
            )
        )

    a8_references = load_a8_references(
        project_root / config["source"]["a8_results_path"]
    )
    return _assemble_payload(
        config=config,
        config_hash=config_hash,
        base=base,
        roles=roles,
        a8_references=a8_references,
        target_loaded_after_training=target_loaded_after_training,
        readiness=_read_json(project_root / _readiness_gate_path()),
    )


def adjudicate_topology_only(payload: dict[str, Any]) -> dict[str, Any]:
    gate_config = payload["config"]["gate"]
    per_seed = {}
    full_pass = payload["validation"]["status"] == "pass"
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        target = payload["target_auc"][key]
        source_macro = payload["source_macro_auc"][key]
        correct = target["candidate_correct"]
        checks = {
            "target_auc_floor": correct >= gate_config["target_auc_floor"],
            "corrupted_topology_margin": correct
            - target["candidate_corrupted_target"]
            >= gate_config["target_topology_margin"],
            "no_topology_margin": correct
            - target["candidate_no_topology_target"]
            >= gate_config["target_topology_margin"],
            "trained_no_topology_margin": correct
            - target["a8_trained_no_topology"]
            >= gate_config["trained_no_topology_margin"],
            "source_anchor_retained": source_macro["candidate"]
            >= source_macro["a8_correct"]
            - gate_config["a8_retention_tolerance"],
            "dialga_anchor_retained": correct
            >= target["a8_correct"] - gate_config["a8_retention_tolerance"],
            "sbox_exactly_invariant": payload["sbox_probability_delta"][key]
            <= gate_config["sbox_invariance_tolerance"],
            "conflict_projections_observed": payload[
                "conflict_projections_by_seed"
            ][key]
            >= gate_config["minimum_conflict_projections_per_seed"],
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "source_macro_auc": source_macro,
            "target_auc": target,
            "target_margins": {
                "corrupted": correct - target["candidate_corrupted_target"],
                "no_topology": correct
                - target["candidate_no_topology_target"],
                "a8_trained_no_topology": correct
                - target["a8_trained_no_topology"],
                "a8_correct": correct - target["a8_correct"],
            },
            "source_anchor_delta": source_macro["candidate"]
            - source_macro["a8_correct"],
            "sbox_probability_delta": payload["sbox_probability_delta"][key],
            "checks": checks,
            "pass": seed_pass,
        }

    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_topology_only_c1_protocol_invalid"
        next_action = "repair only the failed readiness, cache, row or checkpoint invariant"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_topology_only_dialga_supported"
        next_action = (
            "preregister a medium remote confirmation of the frozen topology-only "
            "method with the same A8 controls; do not reopen S-box conditioning"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_topology_only_dialga_not_supported"
        next_action = (
            "close topology-only whole-cipher transfer at this budget and retain "
            "only per-cipher exact-GF(2) Runtime-E4 evidence"
        )
    return {
        "run_id": payload["config"]["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": payload["validation"]["status"] == "pass",
        "full_pass": full_pass,
        "per_seed": per_seed,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "claim_scope": (
            "local 2048/class/source topology-only two-seed whole-cipher "
            "mechanism diagnostic; not formal scale, universality, attack, SOTA "
            "or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "reopen S-box truth-table, ANF, DDT, inverse-triplet or delta-U routes",
            "add Adapter, FiLM, typed residual, MoE, target head or target rows",
            "increase samples, epochs or pairs to rescue a local hold",
            "launch remote training unless this exact two-seed gate passes",
        ],
    }


def write_topology_only_readiness_artifacts(
    readiness: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cipher, deltas in readiness["topology_logit_deltas"].items():
        rows.append(
            {
                "run_id": readiness["run_id"],
                "row_kind": "counterfactual_logit",
                "cipher": cipher,
                "corrupted_delta": deltas["corrupted"],
                "no_topology_delta": deltas["no_topology"],
                "wrong_sbox_delta": readiness["sbox_logit_deltas"][cipher],
                "training_performed": False,
            }
        )
    _write_jsonl(output_root / "results.jsonl", rows)
    _write_json(
        output_root / "validation.json",
        {"status": readiness["status"], "checks": readiness["checks"]},
    )
    _write_json(output_root / "gate.json", readiness)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": readiness["run_id"],
            "status": readiness["status"],
            "decision": readiness["decision"],
            "next_action": readiness["next_action"],
        },
    )


def write_topology_only_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    _write_jsonl(output_root / "results.jsonl", payload["rows"])
    _write_csv(output_root / "history.csv", payload["history"])
    _write_json(output_root / "validation.json", payload["validation"])
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    render_topology_only_svg(payload, gate, output_root / "curves.svg")


def render_topology_only_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    source_labels = {
        "gift64": "GIFT r6",
        "skinny64": "SKINNY r7",
        "rectangle80": "RECTANGLE r6",
        "uknit64": "uKNIT prefix-r5",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
        for column, seed in enumerate(EXPECTED_SEEDS):
            key = str(seed)
            source = payload["source_auc"][key]
            y = np.arange(len(SOURCE_CIPHERS))
            for values, label, offset, color in (
                (source["candidate"], "C1纯GF(2)拓扑", -0.14, "#0072B2"),
                (source["a8_correct"], "A8含S盒门控锚点", 0.14, "#999999"),
            ):
                bars = axes[0, column].barh(
                    y + offset,
                    [values[name] for name in SOURCE_CIPHERS],
                    height=0.25,
                    color=color,
                    label=label,
                )
                axes[0, column].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            axes[0, column].axvline(0.5, color="#475569", linestyle="--")
            all_source = [
                value
                for role_values in source.values()
                for value in role_values.values()
            ]
            axes[0, column].set_xlim(
                max(0.44, min(all_source) - 0.035),
                min(1.0, max(all_source) + 0.06),
            )
            axes[0, column].set_yticks(
                y,
                [source_labels[name] for name in SOURCE_CIPHERS],
            )
            axes[0, column].set_xlabel("四个训练来源的验证 AUC")
            axes[0, column].set_title(
                f"seed{seed}：关闭S盒条件化后的来源保持",
                loc="left",
                fontweight="bold",
            )
            axes[0, column].legend(frameon=False, loc="lower right", fontsize=9)

            target = payload["target_auc"][key]
            correct = target["candidate_correct"]
            labels = (
                "来源宏：C1 - A8",
                "正确 - 损坏GF(2)",
                "正确 - 同权重无拓扑",
                "正确 - A8重训无拓扑",
                "正确 - A8正确锚点",
            )
            values = (
                payload["source_macro_auc"][key]["candidate"]
                - payload["source_macro_auc"][key]["a8_correct"],
                correct - target["candidate_corrupted_target"],
                correct - target["candidate_no_topology_target"],
                correct - target["a8_trained_no_topology"],
                correct - target["a8_correct"],
            )
            checks = gate["per_seed"][key]["checks"]
            passed = (
                checks["source_anchor_retained"],
                checks["corrupted_topology_margin"],
                checks["no_topology_margin"],
                checks["trained_no_topology_margin"],
                checks["dialga_anchor_retained"],
            )
            margin_y = np.arange(len(labels))
            bars = axes[1, column].barh(
                margin_y,
                values,
                color=["#009E73" if value else "#C0392B" for value in passed],
                height=0.58,
            )
            axes[1, column].axvline(0.0, color="#111827", linewidth=1.1)
            axes[1, column].axvline(
                0.005,
                color="#0072B2",
                linestyle="--",
                label="拓扑优势门 +0.005",
            )
            axes[1, column].axvline(
                -0.005,
                color="#8E44AD",
                linestyle=":",
                label="A8保持门 -0.005",
            )
            axes[1, column].set_yticks(margin_y, labels)
            axes[1, column].invert_yaxis()
            limit = max(0.02, max(abs(value) for value in values) + 0.02)
            axes[1, column].set_xlim(-limit, limit)
            axes[1, column].set_xlabel("AUC 差值（正数表示C1更好）")
            axes[1, column].set_title(
                f"seed{seed}：来源保持与未训练Dialga的GF(2)归因\n"
                f"C1 AUC={correct:.4f}，错误S盒概率差="
                f"{payload['sbox_probability_delta'][key]:.1e}",
                loc="left",
                fontweight="bold",
            )
            for bar, value, is_pass in zip(bars, values, passed, strict=True):
                if abs(value) < 0.08:
                    text_x = 0.025 if value >= 0.0 else -0.025
                    horizontal_alignment = "left" if value >= 0.0 else "right"
                else:
                    text_x = value - 0.004
                    horizontal_alignment = "right"
                axes[1, column].text(
                    text_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.4f}  {'通过' if is_pass else '未过'}",
                    ha=horizontal_alignment,
                    va="center",
                    fontsize=8.5,
                )
            axes[1, column].legend(frameon=False, loc="lower right", fontsize=8.5)

        verdict = (
            "通过：纯GF(2)拓扑在双seed未见Dialga上保持归因"
            if gate["status"] == "pass"
            else "暂缓：纯GF(2)拓扑未通过双seed来源保持与整密码留出门"
        )
        figure.suptitle(
            "创新1 C1：关闭不可识别S盒条件化，单独检验运行时GF(2)拓扑",
            x=0.04,
            y=0.985,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.04,
            0.944,
            "四个来源共同训练；Dialga只验证不训练。正确与错误S盒必须严格同输出，性能差只归因于GF(2)拓扑。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.04,
            0.912,
            f"裁决：{verdict}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0.03, 0.035, 0.99, 0.88), h_pad=2.8, w_pad=3.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def load_a8_references(path: Path) -> dict[str, Any]:
    source: dict[str, dict[str, float]] = {str(seed): {} for seed in EXPECTED_SEEDS}
    target: dict[str, dict[str, float]] = {str(seed): {} for seed in EXPECTED_SEEDS}
    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seed_key = str(row["seed"])
            if (
                row.get("row_kind") == "source_validation"
                and row.get("role") == "correct_candidate"
                and row.get("cipher") in SOURCE_CIPHERS
            ):
                source[seed_key][row["cipher"]] = float(
                    row["metrics"]["validation"]["auc"]
                )
                source_rows.append(row)
            if (
                row.get("row_kind") == "holdout_target"
                and row.get("evaluation") in A8_TARGET_REFERENCES
            ):
                target[seed_key][row["evaluation"]] = float(
                    row["metrics"]["validation"]["auc"]
                )
                target_rows.append(row)
    return {
        "source_auc": source,
        "target_auc": target,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }


def _a8_reference_panel_complete(references: dict[str, Any]) -> bool:
    return (
        len(references["source_rows"]) == 8
        and len(references["target_rows"]) == 4
        and all(
            set(references["source_auc"][str(seed)]) == set(SOURCE_CIPHERS)
            and set(references["target_auc"][str(seed)])
            == set(A8_TARGET_REFERENCES)
            for seed in EXPECTED_SEEDS
        )
    )


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_hash: str,
    base: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    a8_references: dict[str, Any],
    target_loaded_after_training: dict[int, bool],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    source_auc: dict[str, dict[str, dict[str, float]]] = {}
    source_macro_auc: dict[str, dict[str, float]] = {}
    target_auc: dict[str, dict[str, float]] = {}
    conflict_counts: dict[str, int] = {}
    sbox_probability_delta: dict[str, float] = {}
    protocol_by_name = {item["name"]: item for item in base["protocols"]}

    for seed in EXPECTED_SEEDS:
        key = str(seed)
        role = roles[seed]
        candidate_source = {
            cipher: float(role["validation_metrics"][cipher]["auc"])
            for cipher in SOURCE_CIPHERS
        }
        a8_source = a8_references["source_auc"][key]
        source_auc[key] = {
            "candidate": candidate_source,
            "a8_correct": a8_source,
        }
        source_macro_auc[key] = {
            name: float(np.mean(list(values.values())))
            for name, values in source_auc[key].items()
        }
        diagnostics = role["gradient_diagnostics"]
        conflict_counts[key] = int(
            sum(diagnostics["task_conflict_projection_counts"].values())
        )
        for epoch_row in role["history"]:
            history.append(
                {"seed": seed, "role": "topology_only_candidate", **epoch_row}
            )
        for cipher in SOURCE_CIPHERS:
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "source_validation",
                    "seed": seed,
                    "role": "topology_only_candidate",
                    "cipher": cipher,
                    "cipher_display_name": SOURCE_DISPLAY_NAMES[cipher],
                    "rounds": protocol_by_name[cipher]["rounds"],
                    "samples_per_class": base["training"]["samples_per_class"],
                    "validation_samples_per_class": base["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": base["training"]["pairs_per_sample"],
                    "negative_mode": base["training"]["negative_mode"],
                    "parameter_count": role["parameter_count"],
                    "checkpoint": role["checkpoint_path"],
                    "checkpoint_sha256": role["checkpoint_sha256"],
                    "metrics": {
                        "train": role["train_metrics"][cipher],
                        "validation": role["validation_metrics"][cipher],
                    },
                    "config_sha256": config_hash,
                    "sbox_context_scale": 0.0,
                }
            )
        evaluations = role["target_evaluations"]
        target_auc[key] = {
            name: float(evaluations[name]["metrics"]["auc"])
            for name in TARGET_EVALUATIONS
        }
        target_auc[key]["a8_correct"] = a8_references["target_auc"][key][
            "candidate_correct"
        ]
        target_auc[key]["a8_trained_no_topology"] = a8_references["target_auc"][
            key
        ]["no_topology_trained_anchor"]
        sbox_probability_delta[key] = float(role["sbox_probability_delta"])
        for name in TARGET_EVALUATIONS:
            evaluation = evaluations[name]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "holdout_target",
                    "seed": seed,
                    "evaluation": name,
                    "cipher": HOLDOUT_CIPHER,
                    "cipher_display_name": "Dialga-128 prefix-r4（整密码留出）",
                    "rounds": protocol_by_name[HOLDOUT_CIPHER]["rounds"],
                    "training_samples_per_class": 0,
                    "validation_samples_per_class": base["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": base["training"]["pairs_per_sample"],
                    "negative_mode": base["training"]["negative_mode"],
                    "structure": evaluation["structure"],
                    "relation_mode": evaluation["relation_mode"],
                    "checkpoint": evaluation["checkpoint_path"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "metrics": {"validation": evaluation["metrics"]},
                    "optimizer_steps": 0,
                    "target_head_trained": False,
                    "sbox_context_scale": 0.0,
                    "config_sha256": config_hash,
                }
            )

    for row in a8_references["source_rows"]:
        rows.append(
            {
                "run_id": config["run_id"],
                "row_kind": "frozen_a8_source_reference",
                "source_run_id": row["run_id"],
                "seed": row["seed"],
                "cipher": row["cipher"],
                "role": "a8_correct",
                "metrics": row["metrics"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "optimizer_steps": 0,
            }
        )
    for row in a8_references["target_rows"]:
        rows.append(
            {
                "run_id": config["run_id"],
                "row_kind": "frozen_a8_target_reference",
                "source_run_id": row["run_id"],
                "seed": row["seed"],
                "cipher": row["cipher"],
                "evaluation": row["evaluation"],
                "metrics": row["metrics"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "optimizer_steps": 0,
            }
        )

    expected_result_rows = config["gate"]["expected_result_rows"]
    expected_history_rows = config["gate"]["expected_history_rows"]
    checks = {
        "readiness_gate_matches": readiness.get("status") == "pass"
        and readiness.get("decision")
        == "innovation1_runtime_spn_topology_only_c1_readiness_passed"
        and all(readiness.get("checks", {}).values()),
        "expected_result_rows": len(rows) == expected_result_rows,
        "expected_history_rows": len(history) == expected_history_rows,
        "candidate_parameter_count": all(
            roles[seed]["parameter_count"]
            == config["candidate"]["expected_parameter_count"]
            for seed in EXPECTED_SEEDS
        ),
        "complete_source_panel": all(
            set(source_auc[str(seed)]["candidate"]) == set(SOURCE_CIPHERS)
            for seed in EXPECTED_SEEDS
        ),
        "complete_target_panel": all(
            set(roles[seed]["target_evaluations"]) == set(TARGET_EVALUATIONS)
            for seed in EXPECTED_SEEDS
        ),
        "a8_reference_panel_complete": _a8_reference_panel_complete(a8_references),
        "target_loaded_after_source_training": all(
            target_loaded_after_training.values()
        ),
        "target_never_trained": config["training"]["target_training_rows"] == 0
        and config["training"]["target_optimizer_steps"] == 0,
        "strict_negative_and_fixed_protocol": base["training"]["negative_mode"]
        == "encrypted_random_plaintexts"
        and base["training"]["pairs_per_sample"] == 4,
        "all_metrics_finite": all(
            np.isfinite(value)
            for by_seed in target_auc.values()
            for value in by_seed.values()
        )
        and all(
            np.isfinite(value)
            for by_seed in source_auc.values()
            for by_role in by_seed.values()
            for value in by_role.values()
        ),
        "sbox_probability_delta_finite": all(
            np.isfinite(value) for value in sbox_probability_delta.values()
        ),
    }
    return {
        "config": config,
        "rows": rows,
        "history": history,
        "source_auc": source_auc,
        "source_macro_auc": source_macro_auc,
        "target_auc": target_auc,
        "conflict_projections_by_seed": conflict_counts,
        "sbox_probability_delta": sbox_probability_delta,
        "validation": {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "result_rows": len(rows),
            "expected_result_rows": expected_result_rows,
            "history_rows": len(history),
            "expected_history_rows": expected_history_rows,
            "target_training_rows": 0,
            "target_optimizer_steps": 0,
        },
    }


def _readiness_gate_path() -> Path:
    return Path(
        "outputs/local_readiness/"
        "i1_runtime_spn_topology_only_c1_readiness_20260726/gate.json"
    )


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "EXPECTED_SEEDS",
    "TARGET_EVALUATIONS",
    "adjudicate_topology_only",
    "load_a8_references",
    "load_and_validate_topology_only_config",
    "render_topology_only_svg",
    "run_topology_only",
    "run_topology_only_readiness",
    "topology_only_spec",
    "write_topology_only_artifacts",
    "write_topology_only_readiness_artifacts",
]
