from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_h1_gradient_equalization import (
    config_sha256,
    run_h1_gradient_equalization,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    HOLDOUT_CIPHER,
    RelationModeRuntimeE4,
    _evaluate_target,
    _load_source_tasks,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.runtime_spn_joint import evaluate_runtime_spn_joint
from blockcipher_nd.training.types import ProgressCallback


POOLING_CONTROLS = ("uniform", "shuffled")
HETEROGENEOUS_SIGNATURE_SOURCES = ("skinny64", "uknit64")
TARGET_POOLING_CONTROLS = (
    "candidate_uniform_pooling_target",
    "candidate_shuffled_pooling_target",
)


def load_and_validate_h1_relation_activity_pooling_config(
    path: Path,
    *,
    project_root: Path,
    require_readiness: bool,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("H1-A5 config schema_version must be 1")
    source = config.get("source", {})
    candidate = config.get("candidate", {})
    gate = config.get("gate", {})
    required_source = {
        "h1_required_decision": "innovation1_runtime_spn_rectangle_holdout_not_supported",
        "a1_required_decision": (
            "innovation1_runtime_spn_h1_source_gradient_imbalance_supported"
        ),
        "a3_required_decision": ("innovation1_runtime_spn_h1_equalized_pcgrad_partial"),
        "a4_required_decision": "innovation1_runtime_spn_h1_shared_representation_weak",
        "readiness_required_decision": (
            "innovation1_runtime_spn_h1_relation_activity_pooling_readiness_passed"
        ),
    }
    for key, expected in required_source.items():
        if source.get(key) != expected:
            raise ValueError(f"H1-A5 source field drifted: {key}")
    required_candidate = {
        "gradient_combination": ("representation_l2_equalized_pcgrad_fixed_order"),
        "representation_parameters": "all_except_shared_classifier",
        "classifier_gradient_combination": "raw_arithmetic_mean",
        "task_sampling": "unchanged_equal_one_batch_per_task",
        "conflict_projection": ("fixed_source_order_pcgrad_after_l2_equalization"),
        "relation_activity_pooling_mode": "correct",
        "pooling_controls": ["uniform", "shuffled"],
        "signature": "gf2_row_fan_in_times_nonempty_source_bit_role_count",
        "heterogeneous_signature_sources": list(HETEROGENEOUS_SIGNATURE_SOURCES),
        "seeds": [0, 1],
        "expected_parameter_count": 442466,
    }
    for key, expected in required_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"H1-A5 candidate field drifted: {key}")
    expected_target = {
        "candidate_correct": {"structure": "correct", "relation_mode": "true"},
        "candidate_corrupted_target": {
            "structure": "corrupted",
            "relation_mode": "true",
        },
        "candidate_no_topology_target": {
            "structure": "correct",
            "relation_mode": "independent",
        },
    }
    if config.get("target_evaluations") != expected_target:
        raise ValueError("H1-A5 target evaluations drifted")
    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "a3_target_retention_tolerance": 0.02,
        "a3_source_macro_retention_tolerance": 0.005,
        "h1_skinny_retention_tolerance": 0.01,
        "pooling_control_margin": 0.005,
        "partial_skinny_improvement_over_a3": 0.005,
        "minimum_conflict_projections_per_seed": 1,
        "required_seeds": [0, 1],
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"H1-A5 gate field drifted: {key}")
    for path_key, hash_key in (
        ("h1_config_path", "h1_config_sha256"),
        ("a1_config_path", "a1_config_sha256"),
        ("a3_config_path", "a3_config_sha256"),
    ):
        if config_sha256(project_root / source[path_key]) != source[hash_key]:
            raise ValueError(f"H1-A5 source hash drifted: {path_key}")
    if require_readiness:
        readiness = _read_json(project_root / source["readiness_gate_path"])
        if readiness.get("decision") != source["readiness_required_decision"]:
            raise ValueError("H1-A5 readiness gate drifted")
        if readiness.get("status") != "pass":
            raise ValueError("H1-A5 readiness did not pass")
    return config


def run_h1_relation_activity_pooling_readiness(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    source = config["source"]
    h1_config = load_and_validate_holdout_config(
        project_root / source["h1_config_path"]
    )
    structures = _load_structures(h1_config)
    models = {
        mode: RelationModeRuntimeE4(
            _pooling_spec(h1_config["model"], mode),
            "true",
        ).eval()
        for mode in ("uniform", "correct", "shuffled")
    }
    state = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(state, strict=True)
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    signature_rows = []
    for name, structure in structures.items():
        correct = RuntimeE4EquivariantSpnDistinguisher.relation_activity_weights(
            structure,
            mode="correct",
            relation_mode="true",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        shuffled = RuntimeE4EquivariantSpnDistinguisher.relation_activity_weights(
            structure,
            mode="shuffled",
            relation_mode="true",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        signature_rows.append(
            {
                "cipher": name,
                "cells": structure.cells,
                "unique_signature_types": int(torch.unique(correct, dim=0).shape[0]),
                "correct_shuffled_l1": float((correct - shuffled).abs().sum()),
                "minimum_weight": float(correct.min()),
                "maximum_weight": float(correct.max()),
            }
        )

    relabel_errors = []
    one_to_one_logit_errors = []
    logits_finite = True
    for index, (name, structure) in enumerate(structures.items()):
        relabeled, bit_permutation = structure.relabel_cells(
            tuple(reversed(range(structure.cells)))
        )
        generator = torch.Generator().manual_seed(26_072_600 + index)
        pairs = torch.randint(
            0,
            2,
            (2, 4, 2, structure.block_bits),
            generator=generator,
            dtype=torch.float32,
        )
        relabeled_pairs = torch.empty_like(pairs)
        relabeled_pairs[..., bit_permutation] = pairs
        for mode in ("correct", "shuffled"):
            with torch.no_grad():
                original = models[mode](pairs, structure)
                permuted = models[mode](relabeled_pairs, relabeled)
            error = float((original - permuted).abs().max())
            relabel_errors.append({"cipher": name, "mode": mode, "max_error": error})
            logits_finite = logits_finite and bool(torch.isfinite(original).all())
        if name in {"gift64", "rectangle80"}:
            with torch.no_grad():
                correct = models["correct"](pairs, structure)
                uniform = models["uniform"](pairs, structure)
            one_to_one_logit_errors.append(
                {
                    "cipher": name,
                    "max_error": float((correct - uniform).abs().max()),
                    "bit_exact": torch.equal(correct, uniform),
                }
            )

    skinny = structures["skinny64"]
    skinny_pairs = torch.randint(
        0,
        2,
        (8, 4, 2, skinny.block_bits),
        generator=torch.Generator().manual_seed(26_072_777),
        dtype=torch.float32,
    )
    with torch.no_grad():
        skinny_logits = {mode: models[mode](skinny_pairs, skinny) for mode in models}
        independent_correct = RelationModeRuntimeE4(
            _pooling_spec(h1_config["model"], "correct"),
            "independent",
        ).eval()
        independent_uniform = RelationModeRuntimeE4(
            _pooling_spec(h1_config["model"], "uniform"),
            "independent",
        ).eval()
        independent_correct.load_state_dict(state, strict=True)
        independent_uniform.load_state_dict(state, strict=True)
        independent_correct_logits = independent_correct(skinny_pairs, skinny)
        independent_uniform_logits = independent_uniform(skinny_pairs, skinny)

    a3_loads = True
    a3_checkpoints_loaded = []
    for seed in EXPECTED_SEEDS:
        a3_checkpoint = torch.load(
            project_root
            / source["a3_output_root"]
            / f"checkpoints/seed{seed}-candidate.pt",
            map_location="cpu",
            weights_only=True,
        )
        for model in models.values():
            try:
                model.load_state_dict(a3_checkpoint["state_dict"], strict=True)
            except RuntimeError:
                a3_loads = False
        a3_checkpoints_loaded.append(seed)

    signature_by_cipher = {row["cipher"]: row for row in signature_rows}
    target_signature = signature_by_cipher[HOLDOUT_CIPHER]
    checks = {
        "parameter_counts_exact": set(parameter_counts.values())
        == {config["candidate"]["expected_parameter_count"]},
        "state_dict_keys_matched": all(
            tuple(model.state_dict()) == tuple(state) for model in models.values()
        ),
        "one_to_one_exact_uniform": all(
            signature_by_cipher[name]["unique_signature_types"] == 1
            and signature_by_cipher[name]["minimum_weight"] == 1.0
            and signature_by_cipher[name]["maximum_weight"] == 1.0
            for name in ("gift64", "rectangle80")
        ),
        "one_to_one_logits_bit_exact": all(
            row["bit_exact"] and row["max_error"] == 0.0
            for row in one_to_one_logit_errors
        ),
        "heterogeneous_sources_distinct": all(
            signature_by_cipher[name]["unique_signature_types"] > 1
            and signature_by_cipher[name]["correct_shuffled_l1"] > 0.0
            for name in HETEROGENEOUS_SIGNATURE_SOURCES
        ),
        "target_pooling_controls_identifiable": (
            target_signature["unique_signature_types"] > 1
            and target_signature["correct_shuffled_l1"] > 0.0
        ),
        "dialga_homogeneous_allowed": (
            signature_by_cipher["dialga128"]["unique_signature_types"] == 1
        ),
        "cell_relabeling_invariant": all(
            row["max_error"] <= 1e-6 for row in relabel_errors
        ),
        "skinny_controls_distinct": all(
            not torch.equal(skinny_logits["correct"], skinny_logits[mode])
            for mode in POOLING_CONTROLS
        ),
        "independent_forces_uniform": torch.equal(
            independent_correct_logits,
            independent_uniform_logits,
        ),
        "logits_finite": logits_finite,
        "both_a3_checkpoints_load_strictly": a3_loads
        and a3_checkpoints_loaded == list(EXPECTED_SEEDS),
        "target_rows_loaded_zero": True,
    }
    status = "pass" if all(checks.values()) else "fail"
    decision = (
        "innovation1_runtime_spn_h1_relation_activity_pooling_readiness_passed"
        if status == "pass"
        else "innovation1_runtime_spn_h1_relation_activity_pooling_readiness_failed"
    )
    return {
        "run_id": "i1_runtime_spn_h1_relation_activity_pooling_a5_readiness_20260726",
        "status": status,
        "decision": decision,
        "checks": checks,
        "parameter_counts": parameter_counts,
        "signature_rows": signature_rows,
        "cell_relabel_errors": relabel_errors,
        "one_to_one_logit_errors": one_to_one_logit_errors,
        "target_rows_loaded": 0,
        "next_action": (
            "run the frozen A5 local diagnostic"
            if status == "pass"
            else (
                "repair only an implementation failure, or redesign the holdout "
                "when the target pooling controls are structurally unidentifiable"
            )
        ),
    }


def run_h1_relation_activity_pooling(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    project_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    payload = run_h1_gradient_equalization(
        config=config,
        config_sha256_value=config_sha256(config_path),
        output_root=output_root,
        project_root=project_root,
        progress_callback=progress_callback,
    )
    source = config["source"]
    h1_config = load_and_validate_holdout_config(
        project_root / source["h1_config_path"]
    )
    structures = _load_structures(h1_config)
    a3_root = project_root / source["a3_output_root"]
    payload["a3_source_auc"] = _read_json(a3_root / "source-metrics.json")
    payload["a3_target_auc"] = _read_json(a3_root / "target-metrics.json")
    payload["source_pooling_control_auc"] = {}
    payload["target_pooling_control_auc"] = {}
    control_rows = []

    for seed in EXPECTED_SEEDS:
        key = str(seed)
        tasks = _load_source_tasks(
            h1_config,
            seed=seed,
            structures=structures,
            progress_callback=progress_callback,
        )
        target = _load_target_validation(
            h1_config,
            seed=seed,
            progress_callback=progress_callback,
        )
        checkpoint_path = output_root / "checkpoints" / f"seed{seed}-candidate.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        payload["source_pooling_control_auc"][key] = {}
        payload["target_pooling_control_auc"][key] = {}
        for mode in POOLING_CONTROLS:
            model = RelationModeRuntimeE4(
                _pooling_spec(h1_config["model"], mode),
                "true",
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            source_metrics = evaluate_runtime_spn_joint(
                model,
                tasks,
                split="validation",
                batch_size=int(h1_config["training"]["batch_size"]),
                device=torch.device(h1_config["training"]["device"]),
                loss=h1_config["training"]["loss"],
            )
            payload["source_pooling_control_auc"][key][mode] = {
                task: float(source_metrics[task]["auc"]) for task in EXPECTED_SOURCES
            }
            target_metrics = _evaluate_target(
                model,
                target,
                structures[HOLDOUT_CIPHER],
                h1_config["training"],
            )
            payload["target_pooling_control_auc"][key][mode] = float(
                target_metrics["auc"]
            )
            for task in EXPECTED_SOURCES:
                control_rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": "source_pooling_counterfactual",
                        "seed": seed,
                        "cipher": task,
                        "pooling_mode": mode,
                        "auc": float(source_metrics[task]["auc"]),
                        "checkpoint": str(checkpoint_path),
                        "optimizer_steps": 0,
                    }
                )
            control_rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "target_pooling_counterfactual",
                    "seed": seed,
                    "cipher": HOLDOUT_CIPHER,
                    "pooling_mode": mode,
                    "auc": float(target_metrics["auc"]),
                    "checkpoint": str(checkpoint_path),
                    "optimizer_steps": 0,
                    "target_training_rows": 0,
                }
            )

    payload["control_rows"] = control_rows
    payload["conflict_projections_by_seed"] = {
        str(seed): sum(
            int(row["conflict_projections"])
            for row in payload["gradient_scales"]
            if row["seed"] == seed
        )
        for seed in EXPECTED_SEEDS
    }
    readiness = _read_json(project_root / source["readiness_gate_path"])
    a3_gate = _read_json(a3_root / "gate.json")
    a4_gate = _read_json(project_root / source["a4_gate_path"])
    payload["validation"]["checks"].update(
        {
            "readiness_gate_matches": readiness.get("decision")
            == source["readiness_required_decision"],
            "target_pooling_controls_identifiable": readiness.get("checks", {}).get(
                "target_pooling_controls_identifiable"
            )
            is True,
            "a3_gate_matches": a3_gate.get("decision")
            == source["a3_required_decision"],
            "a4_gate_matches": a4_gate.get("decision")
            == source["a4_required_decision"],
            "pooling_counterfactual_rows": len(control_rows)
            == len(EXPECTED_SEEDS)
            * len(POOLING_CONTROLS)
            * (len(EXPECTED_SOURCES) + 1),
            "pooling_counterfactual_optimizer_steps_zero": all(
                row["optimizer_steps"] == 0 for row in control_rows
            ),
            "conflict_projections_observed": all(
                value >= config["gate"]["minimum_conflict_projections_per_seed"]
                for value in payload["conflict_projections_by_seed"].values()
            ),
        }
    )
    payload["validation"]["status"] = (
        "pass" if all(payload["validation"]["checks"].values()) else "fail"
    )
    payload["validation"]["result_rows"] = len(payload["rows"]) + len(control_rows)
    return payload


def revalidate_existing_h1_relation_activity_pooling(
    *,
    config: dict[str, Any],
    output_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    readiness = _read_json(project_root / config["source"]["readiness_gate_path"])
    identifiable = (
        readiness.get("checks", {}).get("target_pooling_controls_identifiable") is True
    )
    if identifiable:
        raise ValueError(
            "existing A5 revalidation is only required for an unidentifiable "
            "target pooling control"
        )

    validation = _read_json(output_root / "validation.json")
    checks = dict(validation.get("checks", {}))
    checks["readiness_gate_matches"] = False
    checks["target_pooling_controls_identifiable"] = False
    validation.update({"status": "fail", "checks": checks})
    _write_json(output_root / "validation.json", validation)

    previous_gate = _read_json(output_root / "gate.json")
    superseded_decision = previous_gate.get(
        "supersedes_decision",
        previous_gate.get("decision"),
    )
    gate = {
        **previous_gate,
        "status": "invalid",
        "decision": "innovation1_runtime_spn_h1_relation_activity_pooling_invalid",
        "protocol_valid": False,
        "full_pass": False,
        "partial": False,
        "supersedes_decision": superseded_decision,
        "invalid_reason": (
            "RECTANGLE has a one-to-one linear layer, so correct, uniform and "
            "shuffled relation-activity pooling are bit-exact; the frozen "
            "+0.005 target pooling margin is structurally impossible"
        ),
        "next_action": (
            "retain the raw source and target metrics as diagnostic observations, "
            "discard the A5 supported/not-supported claim, and preregister a "
            "heterogeneous-GF(2) whole-cipher holdout before further training"
        ),
    }
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "invalid_reason": gate["invalid_reason"],
            "next_action": gate["next_action"],
        },
    )
    a3_root = project_root / config["source"]["a3_output_root"]
    payload = {
        "candidate_source_auc": _read_json(output_root / "source-metrics.json"),
        "candidate_target_auc": _read_json(output_root / "target-metrics.json"),
        "source_pooling_control_auc": _read_json(
            output_root / "source-pooling-controls.json"
        ),
        "target_pooling_control_auc": _read_json(
            output_root / "target-pooling-controls.json"
        ),
        "a3_source_auc": _read_json(a3_root / "source-metrics.json"),
        "a3_target_auc": _read_json(a3_root / "target-metrics.json"),
    }
    render_h1_relation_activity_pooling_svg(
        payload,
        gate,
        output_root / "curves.svg",
    )
    return gate


def adjudicate_h1_relation_activity_pooling(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    gate_config = config["gate"]
    per_seed = {}
    full_pass = payload["validation"]["status"] == "pass"
    partial = True
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        source = payload["candidate_source_auc"][key]
        a3_source = payload["a3_source_auc"][key]
        h1_source = payload["anchor_source_auc"][key]
        controls = payload["source_pooling_control_auc"][key]
        target = payload["candidate_target_auc"][key]
        target_controls = {
            "candidate_corrupted_target": target["candidate_corrupted_target"],
            "candidate_no_topology_target": target["candidate_no_topology_target"],
            "candidate_uniform_pooling_target": payload["target_pooling_control_auc"][
                key
            ]["uniform"],
            "candidate_shuffled_pooling_target": payload["target_pooling_control_auc"][
                key
            ]["shuffled"],
        }
        correct_target = target["candidate_correct"]
        target_margins = {
            name: correct_target - value for name, value in target_controls.items()
        }
        source_macro = float(np.mean(list(source.values())))
        a3_source_macro = float(np.mean(list(a3_source.values())))
        heterogeneous_macro = float(
            np.mean([source[name] for name in HETEROGENEOUS_SIGNATURE_SOURCES])
        )
        heterogeneous_margins = {
            mode: heterogeneous_macro
            - float(
                np.mean(
                    [controls[mode][name] for name in HETEROGENEOUS_SIGNATURE_SOURCES]
                )
            )
            for mode in POOLING_CONTROLS
        }
        skinny_margins = {
            mode: source["skinny64"] - controls[mode]["skinny64"]
            for mode in POOLING_CONTROLS
        }
        checks = {
            "target_auc_floor": correct_target >= gate_config["target_auc_floor"],
            "target_margins": all(
                margin >= gate_config["target_topology_margin"]
                for margin in target_margins.values()
            ),
            "a3_target_retained": correct_target
            >= payload["a3_target_auc"][key]["candidate_correct"]
            - gate_config["a3_target_retention_tolerance"],
            "a3_source_macro_retained": source_macro
            >= a3_source_macro - gate_config["a3_source_macro_retention_tolerance"],
            "h1_skinny_retained": source["skinny64"]
            >= h1_source["skinny64"] - gate_config["h1_skinny_retention_tolerance"],
            "skinny_pooling_margins": all(
                margin >= gate_config["pooling_control_margin"]
                for margin in skinny_margins.values()
            ),
            "heterogeneous_pooling_margins": all(
                margin >= gate_config["pooling_control_margin"]
                for margin in heterogeneous_margins.values()
            ),
            "conflict_projection_observed": payload["conflict_projections_by_seed"][key]
            >= gate_config["minimum_conflict_projections_per_seed"],
        }
        seed_pass = all(checks.values())
        full_pass = full_pass and seed_pass
        partial_seed = (
            checks["target_auc_floor"]
            and checks["target_margins"]
            and source["skinny64"]
            >= a3_source["skinny64"] + gate_config["partial_skinny_improvement_over_a3"]
        )
        partial = partial and partial_seed
        per_seed[key] = {
            "candidate_target_auc": correct_target,
            "a3_target_auc": payload["a3_target_auc"][key]["candidate_correct"],
            "target_margins": target_margins,
            "candidate_source_macro_auc": source_macro,
            "a3_source_macro_auc": a3_source_macro,
            "source_macro_delta_vs_a3": source_macro - a3_source_macro,
            "candidate_skinny_auc": source["skinny64"],
            "a3_skinny_auc": a3_source["skinny64"],
            "h1_skinny_auc": h1_source["skinny64"],
            "skinny_delta_vs_a3": source["skinny64"] - a3_source["skinny64"],
            "skinny_pooling_margins": skinny_margins,
            "heterogeneous_pooling_margins": heterogeneous_margins,
            "conflict_projections": payload["conflict_projections_by_seed"][key],
            "checks": checks,
            "pass": seed_pass,
            "partial": partial_seed,
        }

    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_h1_relation_activity_pooling_invalid"
        next_action = "repair the exact readiness, checkpoint, cache or control failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_h1_relation_activity_pooling_supported"
        next_action = (
            "preregister a second independent whole-cipher holdout with the "
            "same parameter-free relation activity pool"
        )
    elif partial:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_relation_activity_pooling_partial"
        next_action = (
            "retain only the supported pooling evidence and audit the remaining "
            "representation mode before another architecture"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_h1_relation_activity_pooling_not_supported"
        next_action = (
            "close this relation-mass pooling primitive and redesign the shared "
            "representation without optimizer or scale changes"
        )
    return {
        "run_id": config["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": payload["validation"]["status"] == "pass",
        "full_pass": full_pass,
        "partial": partial,
        "per_seed": per_seed,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "claim_scope": (
            "local 2048/class/source parameter-free pooling diagnostic; not "
            "formal scale, universality, attack or SOTA evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "change optimizer, samples, epochs or remote scale",
            "train on RECTANGLE or add target heads",
            "add cipher IDs, experts, Adapter, FiLM or typed residual rescue",
            "claim universal adaptation from one holdout cipher",
        ],
    }


def write_h1_relation_activity_pooling_readiness_artifacts(
    readiness: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = (
        [{"row_kind": "signature", **row} for row in readiness["signature_rows"]]
        + [
            {"row_kind": "cell_relabel", **row}
            for row in readiness["cell_relabel_errors"]
        ]
        + [
            {"row_kind": "one_to_one_logit_equivalence", **row}
            for row in readiness["one_to_one_logit_errors"]
        ]
    )
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
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


def write_h1_relation_activity_pooling_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    rows = payload["rows"] + payload["control_rows"]
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(output_root / "history.csv", payload["history"])
    _write_csv(output_root / "gradient_scales.csv", payload["gradient_scales"])
    _write_json(output_root / "source-metrics.json", payload["candidate_source_auc"])
    _write_json(output_root / "target-metrics.json", payload["candidate_target_auc"])
    _write_json(
        output_root / "source-pooling-controls.json",
        payload["source_pooling_control_auc"],
    )
    _write_json(
        output_root / "target-pooling-controls.json",
        payload["target_pooling_control_auc"],
    )
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
    render_h1_relation_activity_pooling_svg(payload, gate, output_root / "curves.svg")


def render_h1_relation_activity_pooling_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    display = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "uknit64": "uKNIT",
        "dialga128": "Dialga",
    }
    target_labels = (
        ("correct", "A5正确关系池化", "#0072B2"),
        ("corrupted", "A5损坏结构", "#D55E00"),
        ("no_topology", "A5无拓扑", "#009E73"),
        ("uniform", "A5普通活动池化", "#CC79A7"),
        ("shuffled", "A5错误关系签名", "#E69F00"),
        ("a3", "A3历史锚点", "#7F8C8D"),
    )
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for column, seed in enumerate(EXPECTED_SEEDS):
            key = str(seed)
            target_values = {
                "correct": payload["candidate_target_auc"][key]["candidate_correct"],
                "corrupted": payload["candidate_target_auc"][key][
                    "candidate_corrupted_target"
                ],
                "no_topology": payload["candidate_target_auc"][key][
                    "candidate_no_topology_target"
                ],
                "uniform": payload["target_pooling_control_auc"][key]["uniform"],
                "shuffled": payload["target_pooling_control_auc"][key]["shuffled"],
                "a3": payload["a3_target_auc"][key]["candidate_correct"],
            }
            bars = axes[0, column].barh(
                range(len(target_labels)),
                [target_values[name] for name, _, _ in target_labels],
                color=[color for _, _, color in target_labels],
            )
            axes[0, column].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            axes[0, column].axvline(0.5, color="#34495E")
            axes[0, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[0, column].set_xlim(0.45, 0.75)
            axes[0, column].set_yticks(
                range(len(target_labels)), [label for _, label, _ in target_labels]
            )
            axes[0, column].set_xlabel("未见 RECTANGLE 验证 AUC")
            axes[0, column].set_title(f"seed{seed}：零微调目标归因")

            y = np.arange(len(EXPECTED_SOURCES))
            series = (
                (payload["candidate_source_auc"][key], "A5正确", -0.27, "#0072B2"),
                (
                    payload["source_pooling_control_auc"][key]["uniform"],
                    "普通池化",
                    -0.09,
                    "#CC79A7",
                ),
                (
                    payload["source_pooling_control_auc"][key]["shuffled"],
                    "错误签名",
                    0.09,
                    "#E69F00",
                ),
                (payload["a3_source_auc"][key], "A3锚点", 0.27, "#7F8C8D"),
            )
            for values, label, offset, color in series:
                source_bars = axes[1, column].barh(
                    y + offset,
                    [values[name] for name in EXPECTED_SOURCES],
                    height=0.16,
                    color=color,
                    label=label,
                )
                axes[1, column].bar_label(
                    source_bars, fmt="%.3f", padding=2, fontsize=6.5
                )
            axes[1, column].axvline(0.5, color="#34495E")
            axes[1, column].set_xlim(0.4, 1.0)
            axes[1, column].set_yticks(y, [display[name] for name in EXPECTED_SOURCES])
            axes[1, column].set_xlabel("四源验证 AUC")
            axes[1, column].set_title(f"seed{seed}：同权重池化反事实")
            axes[1, column].legend(frameon=False, loc="lower right", ncol=2)
        fig.suptitle(
            "创新1 H1-A5：GF(2) 多源关系质量活动池化的整密码留出\n"
            "参数量不变；RECTANGLE 不参与训练、选模或微调",
            fontsize=17,
            y=0.985,
        )
        fig.text(
            0.5,
            0.025,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="center",
            fontsize=12,
        )
        fig.subplots_adjust(
            left=0.13,
            right=0.98,
            top=0.86,
            bottom=0.1,
            wspace=0.34,
            hspace=0.42,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _pooling_spec(model_config: dict[str, Any], mode: str) -> Any:
    return _plain_spec(
        {
            **model_config,
            "relation_activity_pooling_mode": mode,
        }
    )


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_h1_relation_activity_pooling_supported": (
            "双seed源保持与目标归因全过，开放第二独立整密码留出"
        ),
        "innovation1_runtime_spn_h1_relation_activity_pooling_partial": (
            "SKINNY部分改善但未全过，保留机制证据并转表示模式审计"
        ),
        "innovation1_runtime_spn_h1_relation_activity_pooling_not_supported": (
            "关系质量池化未修复共享表示，关闭该原语并重新设计"
        ),
        "innovation1_runtime_spn_h1_relation_activity_pooling_invalid": "协议无效",
    }.get(decision, decision)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "adjudicate_h1_relation_activity_pooling",
    "load_and_validate_h1_relation_activity_pooling_config",
    "revalidate_existing_h1_relation_activity_pooling",
    "run_h1_relation_activity_pooling",
    "run_h1_relation_activity_pooling_readiness",
    "write_h1_relation_activity_pooling_artifacts",
    "write_h1_relation_activity_pooling_readiness_artifacts",
]
