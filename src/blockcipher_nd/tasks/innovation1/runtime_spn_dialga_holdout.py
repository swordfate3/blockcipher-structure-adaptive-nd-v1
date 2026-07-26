from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_holdout_qualification import (
    atomic_gf2_relation_types,
    sbox_truth_hashes,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_heterogeneous_holdout import (
    _clone_state_dict,
    _state_dict_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    RelationModeRuntimeE4,
    _evaluate_target,
    _file_sha256,
    _load_resumable_role,
    _load_source_tasks,
    _load_structures,
    _load_target_validation,
    _plain_spec,
    _synthetic_dataset,
    _training_config,
    config_sha256,
    load_and_validate_holdout_config,
)
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import ProgressCallback, TrainingConfig


SOURCE_CIPHERS = ("gift64", "skinny64", "rectangle80", "uknit64")
SOURCE_DISPLAY_NAMES = {
    "gift64": "GIFT-64 r6（训练来源）",
    "skinny64": "SKINNY-64/64 r7（训练来源）",
    "rectangle80": "RECTANGLE-80 r6（训练来源）",
    "uknit64": "uKNIT-BC prefix-r5（训练来源）",
}
HOLDOUT_CIPHER = "dialga128"
TRAINING_ROLES = ("correct_candidate", "no_topology_anchor")
EXPECTED_SEEDS = (0, 1)
TARGET_EVALUATIONS = (
    "candidate_correct",
    "candidate_corrupted_target",
    "candidate_no_topology_target",
    "candidate_wrong_sbox_target",
    "no_topology_trained_anchor",
)
CANDIDATE_COUNTERFACTUALS = TARGET_EVALUATIONS[:4]


def load_and_validate_dialga_holdout_config(
    path: Path,
    *,
    project_root: Path,
    require_readiness: bool,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("A8 config schema_version must be 1")
    if config.get("experiment") != "innovation1_runtime_spn_dialga_holdout_a8":
        raise ValueError("A8 experiment name drifted")
    if tuple(config.get("source_ciphers", ())) != SOURCE_CIPHERS:
        raise ValueError("A8 source cipher panel drifted")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("A8 must hold out Dialga")
    if HOLDOUT_CIPHER in config["source_ciphers"]:
        raise ValueError("A8 Dialga holdout leaked into source panel")

    required_roles = {
        "correct_candidate": {"relation_mode": "true"},
        "no_topology_anchor": {"relation_mode": "independent"},
    }
    if config.get("training_roles") != required_roles:
        raise ValueError("A8 training roles drifted")
    required_candidate = {
        "backbone": "RuntimeE4EquivariantSpnDistinguisher",
        "architecture_variant": "base_exact_gf2_no_extra_residual",
        "gradient_combination": (
            "representation_l2_equalized_pcgrad_fixed_order"
        ),
        "representation_parameters": "all_except_shared_classifier",
        "classifier_gradient_combination": "raw_arithmetic_mean",
        "task_sampling": "unchanged_equal_one_batch_per_task",
        "initialization": "same_state_per_seed_across_roles",
        "expected_parameter_count": 442466,
        "seeds": [0, 1],
        "forbidden_modes": {
            "typed_relation_mode": "none",
            "primitive_adapter_mode": "none",
            "primitive_film_mode": "none",
            "relation_activity_pooling_mode": "uniform",
        },
    }
    if config.get("candidate") != required_candidate:
        raise ValueError("A8 candidate contract drifted")
    required_target = {
        "candidate_correct": {
            "source_role": "correct_candidate",
            "structure": "correct",
            "relation_mode": "true",
        },
        "candidate_corrupted_target": {
            "source_role": "correct_candidate",
            "structure": "corrupted",
            "relation_mode": "true",
        },
        "candidate_no_topology_target": {
            "source_role": "correct_candidate",
            "structure": "correct",
            "relation_mode": "independent",
        },
        "candidate_wrong_sbox_target": {
            "source_role": "correct_candidate",
            "structure": "wrong_sbox",
            "relation_mode": "true",
            "wrong_sbox_source_cipher": "gift64",
        },
        "no_topology_trained_anchor": {
            "source_role": "no_topology_anchor",
            "structure": "correct",
            "relation_mode": "independent",
        },
    }
    if config.get("target_evaluations") != required_target:
        raise ValueError("A8 target evaluation panel drifted")
    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "target_wrong_sbox_margin": 0.005,
        "trained_anchor_margin": 0.005,
        "source_macro_retention_tolerance": 0.005,
        "minimum_conflict_projections_per_role_seed": 1,
        "required_seeds": [0, 1],
    }
    if config.get("gate") != required_gate:
        raise ValueError("A8 gate contract drifted")

    source = config.get("source", {})
    expected_source = {
        "a7_required_decision": (
            "innovation1_runtime_spn_holdout_qualification_dialga128_selected"
        ),
        "d1_required_decision": (
            "innovation1_dialga_runtime_e4_d1_two_seed_supported"
        ),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"A8 source field drifted: {key}")
    for path_key, hash_key in (
        ("protocol_config_path", "protocol_config_sha256"),
        ("a7_config_path", "a7_config_sha256"),
        ("a7_gate_path", "a7_gate_sha256"),
        ("a7_validation_path", "a7_validation_sha256"),
        ("d1_gate_path", "d1_gate_sha256"),
    ):
        if config_sha256(project_root / source[path_key]) != source.get(hash_key):
            raise ValueError(f"A8 source hash drifted: {path_key}")
    base = load_and_validate_holdout_config(
        project_root / source["protocol_config_path"]
    )
    spec = _plain_spec(base["model"])
    for key, expected in config["candidate"]["forbidden_modes"].items():
        if getattr(spec, key) != expected:
            raise ValueError(f"A8 base model mode drifted: {key}")
    if require_readiness:
        readiness_path = project_root / (
            "outputs/local_readiness/"
            "i1_runtime_spn_dialga_holdout_a8_readiness_20260726/gate.json"
        )
        readiness = _read_json(readiness_path)
        if readiness.get("status") != "pass":
            raise ValueError("A8 readiness did not pass")
        if readiness.get("decision") != (
            "innovation1_runtime_spn_dialga_holdout_readiness_passed"
        ):
            raise ValueError("A8 readiness decision drifted")
        if not all(readiness.get("checks", {}).values()):
            raise ValueError("A8 readiness contains a failed check")
    return config


def build_wrong_sbox_structure(
    target: RuntimeSpnStructure,
    source: RuntimeSpnStructure,
) -> RuntimeSpnStructure:
    source_truth = source.sbox_truth_bits[-1, 0]
    truth = source_truth.reshape(1, 1, 64).repeat(target.rounds, target.cells, 1)
    return runtime_spn_structure_from_truth_bits(
        target.cell_membership,
        target.bit_role,
        truth,
        target.linear_matrices,
    )


def run_dialga_holdout_readiness(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    target = structures[HOLDOUT_CIPHER]
    wrong_sbox = build_wrong_sbox_structure(target, structures["gift64"])
    models = {
        role: RelationModeRuntimeE4(
            _plain_spec(base["model"]),
            config["training_roles"][role]["relation_mode"],
        ).eval()
        for role in TRAINING_ROLES
    }
    state = _clone_state_dict(models["correct_candidate"].state_dict())
    for model in models.values():
        model.load_state_dict(state, strict=True)
    parameter_counts = {
        role: sum(parameter.numel() for parameter in model.parameters())
        for role, model in models.items()
    }
    generator = torch.Generator().manual_seed(26_072_608)
    features = torch.randint(
        0,
        2,
        (8, 4, 2, target.block_bits),
        generator=generator,
        dtype=torch.float32,
    )
    corrupted = target.corrupted()
    with torch.no_grad():
        logits = {
            "correct": models["correct_candidate"](features, target),
            "corrupted": models["correct_candidate"](features, corrupted),
            "no_topology": models["no_topology_anchor"](features, target),
            "wrong_sbox": models["correct_candidate"](features, wrong_sbox),
        }
    relabel_errors = {
        "correct": _cell_relabel_error(
            models["correct_candidate"], features, target
        ),
        "wrong_sbox": _cell_relabel_error(
            models["correct_candidate"], features, wrong_sbox
        ),
    }
    target_types = atomic_gf2_relation_types(target)
    source_types = set().union(
        *(atomic_gf2_relation_types(structures[name]) for name in SOURCE_CIPHERS)
    )
    target_sboxes = sbox_truth_hashes(target)
    source_sboxes = set().union(
        *(sbox_truth_hashes(structures[name]) for name in SOURCE_CIPHERS)
    )
    cache_probe = _cache_probe(base, project_root)
    smoke = _synthetic_source_only_smoke(base, structures)
    a7_gate = _read_json(project_root / config["source"]["a7_gate_path"])
    a7_validation = _read_json(
        project_root / config["source"]["a7_validation_path"]
    )
    d1_gate = _read_json(project_root / config["source"]["d1_gate_path"])
    checks = {
        "a7_selected_dialga": a7_gate.get("status") == "pass"
        and a7_gate.get("decision") == config["source"]["a7_required_decision"]
        and a7_gate.get("selected_holdout") == HOLDOUT_CIPHER
        and a7_validation.get("status") == "pass"
        and all(a7_validation.get("checks", {}).values()),
        "d1_oracle_valid": d1_gate.get("status") == "pass"
        and d1_gate.get("decision") == config["source"]["d1_required_decision"]
        and all(d1_gate.get("protocol_checks", {}).values()),
        "source_panel_exact_and_holdout_absent": tuple(config["source_ciphers"])
        == SOURCE_CIPHERS
        and HOLDOUT_CIPHER not in SOURCE_CIPHERS,
        "parameter_and_state_geometry_matched": set(parameter_counts.values())
        == {config["candidate"]["expected_parameter_count"]}
        and all(tuple(model.state_dict()) == tuple(state) for model in models.values()),
        "initial_states_bit_exact": all(
            all(torch.equal(state[name], model.state_dict()[name]) for name in state)
            for model in models.values()
        ),
        "base_model_has_no_closed_residuals": all(
            getattr(models["correct_candidate"].backbone.spec, key) == expected
            for key, expected in config["candidate"]["forbidden_modes"].items()
        ),
        "atomic_gf2_source_coverage_complete": target_types <= source_types
        and len(target_types) == 16,
        "target_sbox_exactly_unseen": not (target_sboxes & source_sboxes),
        "wrong_sbox_changes_only_truth_table": _wrong_sbox_contract(
            target,
            wrong_sbox,
            structures["gift64"],
        ),
        "target_counterfactual_logits_distinct": all(
            not torch.equal(logits["correct"], logits[name])
            for name in ("corrupted", "no_topology", "wrong_sbox")
        ),
        "target_cell_relabeling_invariant": all(
            error <= 1e-6 for error in relabel_errors.values()
        ),
        "cache_manifest_ready": cache_probe["passed"],
        "target_train_cache_not_referenced": not cache_probe[
            "target_train_referenced"
        ],
        "synthetic_source_panel_exact": smoke["source_task_names"]
        == list(SOURCE_CIPHERS),
        "synthetic_target_after_both_roles": smoke[
            "target_evaluated_after_both_roles"
        ],
        "synthetic_target_optimizer_steps_zero": smoke["target_optimizer_steps"]
        == 0,
        "outputs_finite": all(
            bool(torch.isfinite(value).all()) for value in logits.values()
        ),
    }
    passed = all(checks.values())
    return {
        "run_id": "i1_runtime_spn_dialga_holdout_a8_readiness_20260726",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_dialga_holdout_readiness_passed"
            if passed
            else "innovation1_runtime_spn_dialga_holdout_readiness_failed"
        ),
        "checks": checks,
        "parameter_counts": parameter_counts,
        "target_atomic_gf2_types": len(target_types),
        "covered_atomic_gf2_types": len(target_types & source_types),
        "target_unique_sboxes": len(target_sboxes),
        "exact_source_sbox_overlap": len(target_sboxes & source_sboxes),
        "logit_differences": {
            name: float((logits["correct"] - logits[name]).abs().max())
            for name in ("corrupted", "no_topology", "wrong_sbox")
        },
        "cell_relabel_errors": relabel_errors,
        "cache_probe": cache_probe,
        "smoke": smoke,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "next_action": (
            "run the frozen A8 local diagnostic"
            if passed
            else "repair only the failed A8 readiness invariant before training"
        ),
    }


def run_dialga_holdout(
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
    target_loaded_after_roles: dict[int, bool] = {}

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
            initializer = RelationModeRuntimeE4(_plain_spec(base["model"]), "true")
        initial_state = _clone_state_dict(initializer.state_dict())
        initial_hash = _state_dict_sha256(initial_state)
        roles[seed] = {}
        for role in TRAINING_ROLES:
            relation_mode = config["training_roles"][role]["relation_mode"]
            checkpoint_path = checkpoint_root / f"seed{seed}-{role}.pt"
            role_path = role_root / f"seed{seed}-{role}.json"
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
                    role=role,
                    relation_mode=relation_mode,
                    config_hash=config_hash,
                    initial_hash=initial_hash,
                )
                roles[seed][role] = resumed
                _emit(progress_callback, "source_role_reused", seed=seed, role=role)
                continue
            model = RelationModeRuntimeE4(
                _plain_spec(base["model"]),
                relation_mode,
            )
            model.load_state_dict(initial_state, strict=True)
            _emit(progress_callback, "source_role_start", seed=seed, role=role)
            result = train_runtime_spn_joint(
                model,
                tasks,
                _training_config(base["training"], seed),
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda event, payload, seed=seed, role=role: progress_callback(
                        event,
                        {"seed": seed, "role": role, **payload},
                    )
                ),
                gradient_combination=config["candidate"]["gradient_combination"],
            )
            checkpoint = {
                "state_dict": _clone_state_dict(model.state_dict()),
                "seed": seed,
                "role": role,
                "relation_mode": relation_mode,
                "config_sha256": config_hash,
                "initial_state_sha256": initial_hash,
                "best_epoch": result.metadata["best_epoch"],
                "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
                "holdout_cipher": HOLDOUT_CIPHER,
            }
            torch.save(checkpoint, checkpoint_path)
            role_payload = {
                "seed": seed,
                "role": role,
                "relation_mode": relation_mode,
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
            roles[seed][role] = role_payload
            _emit(
                progress_callback,
                "source_role_done",
                seed=seed,
                role=role,
                best_epoch=result.metadata["best_epoch"],
            )
        both_roles_done = set(roles[seed]) == set(TRAINING_ROLES)
        if not both_roles_done:
            raise RuntimeError("both A8 source roles must finish before target load")
        _emit(progress_callback, "target_validation_load_start", seed=seed)
        target_dataset = _load_target_validation(
            base,
            seed=seed,
            progress_callback=progress_callback,
            holdout_cipher=HOLDOUT_CIPHER,
        )
        target_loaded_after_roles[seed] = both_roles_done
        evaluations = {}
        for name, evaluation in config["target_evaluations"].items():
            source_role = evaluation["source_role"]
            role_payload = roles[seed][source_role]
            checkpoint_path = Path(role_payload["checkpoint_path"])
            checkpoint_hash = _file_sha256(checkpoint_path)
            if checkpoint_hash != role_payload["checkpoint_sha256"]:
                raise ValueError("A8 checkpoint changed after role completion")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            _validate_checkpoint_payload(
                checkpoint,
                seed=seed,
                role=source_role,
                relation_mode=config["training_roles"][source_role]["relation_mode"],
                config_hash=config_hash,
                initial_hash=role_payload["initial_state_sha256"],
            )
            model = RelationModeRuntimeE4(
                _plain_spec(base["model"]),
                evaluation["relation_mode"],
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            structure = structures[HOLDOUT_CIPHER]
            if evaluation["structure"] == "corrupted":
                structure = structure.corrupted()
            elif evaluation["structure"] == "wrong_sbox":
                structure = wrong_sbox
            metrics = _evaluate_target(
                model,
                target_dataset,
                structure,
                base["training"],
                holdout_cipher=HOLDOUT_CIPHER,
            )
            evaluations[name] = {
                "source_role": source_role,
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
        roles[seed]["target_evaluations"] = evaluations

    return _assemble_payload(
        config=config,
        config_sha256_value=config_hash,
        base=base,
        roles=roles,
        target_loaded_after_roles=target_loaded_after_roles,
        readiness=_read_json(
            project_root
            / "outputs/local_readiness/"
            "i1_runtime_spn_dialga_holdout_a8_readiness_20260726/gate.json"
        ),
        d1_gate=_read_json(project_root / config["source"]["d1_gate_path"]),
        wrong_sbox=wrong_sbox,
        structures=structures,
    )


def adjudicate_dialga_holdout(payload: dict[str, Any]) -> dict[str, Any]:
    gate_config = payload["config"]["gate"]
    per_seed = {}
    full_pass = payload["validation"]["status"] == "pass"
    functional_pass = payload["validation"]["status"] == "pass"
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        target = payload["target_auc"][key]
        correct = target["candidate_correct"]
        topology_margins = {
            name: correct - target[name]
            for name in (
                "candidate_corrupted_target",
                "candidate_no_topology_target",
            )
        }
        wrong_sbox_margin = correct - target["candidate_wrong_sbox_target"]
        trained_anchor_margin = correct - target["no_topology_trained_anchor"]
        candidate_macro = payload["source_macro_auc"][key]["correct_candidate"]
        anchor_macro = payload["source_macro_auc"][key]["no_topology_anchor"]
        conflict_counts = payload["conflict_projections_by_role_seed"][key]
        functional_checks = {
            "target_auc_floor": correct >= gate_config["target_auc_floor"],
            "target_topology_margins": all(
                value >= gate_config["target_topology_margin"]
                for value in topology_margins.values()
            ),
            "target_wrong_sbox_margin": wrong_sbox_margin
            >= gate_config["target_wrong_sbox_margin"],
            "trained_anchor_margin": trained_anchor_margin
            >= gate_config["trained_anchor_margin"],
        }
        checks = {
            **functional_checks,
            "source_macro_retained": candidate_macro
            >= anchor_macro - gate_config["source_macro_retention_tolerance"],
            "conflict_projections_observed": all(
                value
                >= gate_config["minimum_conflict_projections_per_role_seed"]
                for value in conflict_counts.values()
            ),
        }
        seed_functional = all(functional_checks.values())
        seed_pass = all(checks.values())
        functional_pass = functional_pass and seed_functional
        full_pass = full_pass and seed_pass
        per_seed[key] = {
            "candidate_target_auc": correct,
            "target_trained_oracle_auc": payload["oracle_auc"][key],
            "topology_margins": topology_margins,
            "wrong_sbox_margin": wrong_sbox_margin,
            "trained_anchor_margin": trained_anchor_margin,
            "candidate_source_macro_auc": candidate_macro,
            "no_topology_anchor_source_macro_auc": anchor_macro,
            "source_macro_delta": candidate_macro - anchor_macro,
            "conflict_projections": conflict_counts,
            "checks": checks,
            "functional_pass": seed_functional,
            "pass": seed_pass,
        }

    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_dialga_holdout_invalid"
        next_action = "repair the exact readiness, cache, checkpoint or zero-target-step failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_dialga_holdout_supported"
        next_action = (
            "consolidate the two independent whole-cipher holdouts and design a "
            "formal-scale confirmation without changing the base method"
        )
    elif functional_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_dialga_holdout_partial"
        next_action = (
            "retain unseen-Dialga structural attribution but audit source retention "
            "before any formal-scale confirmation"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_dialga_holdout_not_supported"
        next_action = (
            "stop second-holdout training and consolidate the supported per-cipher "
            "Runtime-E4 evidence without architecture or scale rescue"
        )
    return {
        "run_id": payload["config"]["run_id"],
        "status": status,
        "decision": decision,
        "protocol_valid": payload["validation"]["status"] == "pass",
        "full_pass": full_pass,
        "functional_pass": functional_pass,
        "per_seed": per_seed,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
        "claim_scope": (
            "local 2048/class/source second whole-cipher holdout diagnostic; not "
            "formal scale, universality, attack, SOTA or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "load Dialga training rows or select checkpoints on Dialga",
            "change optimizer, samples, epochs or launch remote scale as rescue",
            "add typed relation, relation-mass, Adapter, FiLM, MoE or target head",
            "report the D1 oracle as cross-cipher evidence",
        ],
    }


def write_dialga_holdout_readiness_artifacts(
    readiness: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": readiness["run_id"],
            "row_kind": "structure_coverage",
            "cipher": HOLDOUT_CIPHER,
            "target_atomic_gf2_types": readiness["target_atomic_gf2_types"],
            "covered_atomic_gf2_types": readiness["covered_atomic_gf2_types"],
            "target_unique_sboxes": readiness["target_unique_sboxes"],
            "exact_source_sbox_overlap": readiness["exact_source_sbox_overlap"],
            "training_performed": False,
        },
        *(
            {
                "run_id": readiness["run_id"],
                "row_kind": "counterfactual_logit",
                "cipher": HOLDOUT_CIPHER,
                "condition": name,
                "max_logit_difference": value,
                "training_performed": False,
            }
            for name, value in readiness["logit_differences"].items()
        ),
    ]
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


def write_dialga_holdout_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    _write_jsonl(output_root / "results.jsonl", payload["rows"])
    _write_csv(output_root / "history.csv", payload["history"])
    _write_csv(output_root / "gradient_scales.csv", payload["gradient_scales"])
    _write_json(output_root / "source-metrics.json", payload["source_auc"])
    _write_json(output_root / "target-metrics.json", payload["target_auc"])
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
    render_dialga_holdout_svg(payload, gate, output_root / "curves.svg")


def render_dialga_holdout_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    display = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "rectangle80": "RECTANGLE",
        "uknit64": "uKNIT",
    }
    target_labels = (
        ("candidate_correct", "正确结构", "#2563EB"),
        ("candidate_corrupted_target", "损坏拓扑", "#D97706"),
        ("candidate_no_topology_target", "同权重无拓扑", "#64748B"),
        ("candidate_wrong_sbox_target", "同权重错误S盒", "#C2417B"),
        ("no_topology_trained_anchor", "重训无拓扑锚点", "#0F9D76"),
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
        figure, axes = plt.subplots(2, 2, figsize=(16, 10))
        for column, seed in enumerate(EXPECTED_SEEDS):
            key = str(seed)
            target = payload["target_auc"][key]
            correct = target["candidate_correct"]
            bars = axes[0, column].barh(
                range(len(target_labels)),
                [target[name] for name, _, _ in target_labels],
                color=[color for _, _, color in target_labels],
            )
            axes[0, column].bar_label(
                bars,
                labels=[
                    f"{target[name]:.6f}"
                    if name == "candidate_correct"
                    else f"{target[name]:.6f}  Δ={correct - target[name]:+.6f}"
                    for name, _, _ in target_labels
                ],
                padding=3,
                fontsize=8,
            )
            axes[0, column].axvline(0.5, color="#94A3B8", linestyle="--")
            axes[0, column].axvline(0.55, color="#DC2626", linestyle=":")
            values = [target[name] for name, _, _ in target_labels]
            axes[0, column].set_xlim(max(0.45, min(values) - 0.04), min(1.0, max(values) + 0.08))
            axes[0, column].set_yticks(
                range(len(target_labels)),
                [label for _, label, _ in target_labels],
            )
            axes[0, column].set_xlabel("未见 Dialga 验证 AUC")
            axes[0, column].set_title(
                f"seed{seed}：同一候选检查点结构反事实",
                loc="left",
                fontweight="bold",
            )

            y = np.arange(len(SOURCE_CIPHERS))
            candidate = payload["source_auc"][key]["correct_candidate"]
            anchor = payload["source_auc"][key]["no_topology_anchor"]
            for values_by_cipher, label, offset, color in (
                (candidate, "正确结构训练", -0.13, "#2563EB"),
                (anchor, "无拓扑训练", 0.13, "#0F9D76"),
            ):
                source_bars = axes[1, column].barh(
                    y + offset,
                    [values_by_cipher[name] for name in SOURCE_CIPHERS],
                    height=0.24,
                    color=color,
                    label=label,
                )
                axes[1, column].bar_label(
                    source_bars,
                    fmt="%.3f",
                    padding=2,
                    fontsize=7,
                )
            axes[1, column].axvline(0.5, color="#94A3B8", linestyle="--")
            all_source = [
                value
                for values_by_cipher in (candidate, anchor)
                for value in values_by_cipher.values()
            ]
            axes[1, column].set_xlim(
                max(0.4, min(all_source) - 0.04),
                min(1.0, max(all_source) + 0.08),
            )
            axes[1, column].set_yticks(y, [display[name] for name in SOURCE_CIPHERS])
            axes[1, column].set_xlabel("四源验证 AUC")
            axes[1, column].set_title(
                f"seed{seed}：同预算源任务保持",
                loc="left",
                fontweight="bold",
            )
            axes[1, column].legend(frameon=False, loc="lower right")

        figure.suptitle(
            "创新1 A8：Dialga 整密码零训练行留出与结构归因",
            x=0.06,
            y=0.98,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.06,
            0.935,
            "共享基础 Runtime-E4 只在 GIFT、SKINNY、RECTANGLE、uKNIT 上训练；Dialga 只用于最终验证。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.06,
            0.900,
            f"裁决：{_decision_chinese(gate['decision'])}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.87), h_pad=2.5, w_pad=2.4)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches=None)
        plt.close(figure)


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_sha256_value: str,
    base: dict[str, Any],
    roles: dict[int, dict[str, Any]],
    target_loaded_after_roles: dict[int, bool],
    readiness: dict[str, Any],
    d1_gate: dict[str, Any],
    wrong_sbox: RuntimeSpnStructure,
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    source_auc: dict[str, dict[str, dict[str, float]]] = {}
    source_macro_auc: dict[str, dict[str, float]] = {}
    target_auc: dict[str, dict[str, float]] = {}
    conflict_counts: dict[str, dict[str, int]] = {}
    history: list[dict[str, Any]] = []
    gradient_scales: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    protocol_by_name = {item["name"]: item for item in base["protocols"]}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        source_auc[key] = {}
        source_macro_auc[key] = {}
        conflict_counts[key] = {}
        for role in TRAINING_ROLES:
            role_payload = roles[seed][role]
            source_auc[key][role] = {
                cipher: float(role_payload["validation_metrics"][cipher]["auc"])
                for cipher in SOURCE_CIPHERS
            }
            source_macro_auc[key][role] = float(
                np.mean(list(source_auc[key][role].values()))
            )
            diagnostics = role_payload["gradient_diagnostics"]
            conflict_counts[key][role] = int(
                sum(diagnostics["task_conflict_projection_counts"].values())
            )
            for epoch_row in role_payload["history"]:
                history.append({"seed": seed, "role": role, **epoch_row})
            for cipher in SOURCE_CIPHERS:
                gradient_scales.append(
                    {
                        "seed": seed,
                        "role": role,
                        "cipher": cipher,
                        "mean_raw_representation_gradient_l2": diagnostics[
                            "task_representation_gradient_mean_l2"
                        ][cipher],
                        "mean_applied_scale": diagnostics[
                            "task_gradient_scale_mean"
                        ][cipher],
                        "observations": diagnostics[
                            "task_gradient_scale_observations"
                        ][cipher],
                    }
                )
                rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": "source_validation",
                        "seed": seed,
                        "role": role,
                        "relation_mode": config["training_roles"][role][
                            "relation_mode"
                        ],
                        "cipher": cipher,
                        "cipher_display_name": SOURCE_DISPLAY_NAMES[cipher],
                        "rounds": protocol_by_name[cipher]["rounds"],
                        "samples_per_class": base["training"]["samples_per_class"],
                        "validation_samples_per_class": base["training"][
                            "validation_samples_per_class"
                        ],
                        "pairs_per_sample": base["training"]["pairs_per_sample"],
                        "negative_mode": base["training"]["negative_mode"],
                        "parameter_count": role_payload["parameter_count"],
                        "checkpoint": role_payload["checkpoint_path"],
                        "checkpoint_sha256": role_payload["checkpoint_sha256"],
                        "metrics": {
                            "train": role_payload["train_metrics"][cipher],
                            "validation": role_payload["validation_metrics"][cipher],
                        },
                        "config_sha256": config_sha256_value,
                    }
                )
        evaluations = roles[seed]["target_evaluations"]
        target_auc[key] = {
            name: float(evaluations[name]["metrics"]["auc"])
            for name in TARGET_EVALUATIONS
        }
        for name in TARGET_EVALUATIONS:
            evaluation = evaluations[name]
            rows.append(
                {
                    "run_id": config["run_id"],
                    "row_kind": "holdout_target",
                    "seed": seed,
                    "evaluation": name,
                    "source_role": evaluation["source_role"],
                    "structure": evaluation["structure"],
                    "relation_mode": evaluation["relation_mode"],
                    "cipher": HOLDOUT_CIPHER,
                    "cipher_display_name": "Dialga-128 prefix-r4（整密码留出）",
                    "rounds": protocol_by_name[HOLDOUT_CIPHER]["rounds"],
                    "training_samples_per_class": 0,
                    "validation_samples_per_class": base["training"][
                        "validation_samples_per_class"
                    ],
                    "pairs_per_sample": base["training"]["pairs_per_sample"],
                    "negative_mode": base["training"]["negative_mode"],
                    "checkpoint": evaluation["checkpoint_path"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "metrics": {"validation": evaluation["metrics"]},
                    "optimizer_steps": 0,
                    "target_head_trained": False,
                    "config_sha256": config_sha256_value,
                }
            )
    candidate_shared = all(
        len(
            {
                roles[seed]["target_evaluations"][name]["checkpoint_sha256"]
                for name in CANDIDATE_COUNTERFACTUALS
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
    )
    checkpoints = [
        Path(roles[seed][role]["checkpoint_path"])
        for seed in EXPECTED_SEEDS
        for role in TRAINING_ROLES
    ]
    checks = {
        "readiness_gate_matches": readiness.get("status") == "pass"
        and readiness.get("decision")
        == "innovation1_runtime_spn_dialga_holdout_readiness_passed"
        and all(readiness.get("checks", {}).values()),
        "d1_oracle_gate_matches": d1_gate.get("status") == "pass"
        and d1_gate.get("decision") == config["source"]["d1_required_decision"],
        "four_checkpoints_exist": len(checkpoints) == 4
        and all(path.is_file() for path in checkpoints),
        "parameter_count_matches": {
            roles[seed][role]["parameter_count"]
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
        }
        == {config["candidate"]["expected_parameter_count"]},
        "same_initial_state_per_seed": all(
            len(
                {
                    roles[seed][role]["initial_state_sha256"]
                    for role in TRAINING_ROLES
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
        ),
        "source_panel_exact_and_holdout_absent": all(
            tuple(roles[seed][role]["metadata"]["task_names"]) == SOURCE_CIPHERS
            and HOLDOUT_CIPHER not in roles[seed][role]["metadata"]["task_names"]
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
        ),
        "source_only_checkpoint_selection": all(
            roles[seed][role]["metadata"]["selected_checkpoint"] == "best"
            and tuple(roles[seed][role]["metadata"]["task_names"]) == SOURCE_CIPHERS
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
        ),
        "target_loaded_after_both_roles": all(target_loaded_after_roles.values()),
        "candidate_same_checkpoint_counterfactuals": candidate_shared,
        "target_optimizer_steps_zero": all(
            roles[seed]["target_evaluations"][name]["optimizer_steps"] == 0
            for seed in EXPECTED_SEEDS
            for name in TARGET_EVALUATIONS
        ),
        "target_head_never_trained": all(
            not roles[seed]["target_evaluations"][name]["target_head_trained"]
            for seed in EXPECTED_SEEDS
            for name in TARGET_EVALUATIONS
        ),
        "gradient_combination_exact": all(
            roles[seed][role]["metadata"]["gradient_combination"]
            == config["candidate"]["gradient_combination"]
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
        ),
        "gradient_scales_observed": all(
            row["observations"] > 0
            and np.isfinite(row["mean_applied_scale"])
            and row["mean_applied_scale"] > 0.0
            for row in gradient_scales
        ),
        "strict_negative_mode": base["training"]["negative_mode"]
        == "encrypted_random_plaintexts",
        "wrong_sbox_changes_only_truth_table": _wrong_sbox_contract(
            structures[HOLDOUT_CIPHER],
            wrong_sbox,
            structures["gift64"],
        ),
        "all_metrics_finite": all(
            np.isfinite(metric)
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
            for cipher in SOURCE_CIPHERS
            for metric in roles[seed][role]["validation_metrics"][cipher].values()
        )
        and all(
            np.isfinite(metric)
            for seed in EXPECTED_SEEDS
            for name in TARGET_EVALUATIONS
            for metric in roles[seed]["target_evaluations"][name]["metrics"].values()
        ),
        "result_rows": len(rows) == 26,
    }
    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "result_rows": len(rows),
        "checkpoint_count": len(checkpoints),
        "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
        "holdout_cipher": HOLDOUT_CIPHER,
        "target_training_rows": 0,
        "target_optimizer_steps": 0,
    }
    oracle_auc = {
        str(seed): float(d1_gate["aucs"][f"seed{seed}"]["correct"])
        for seed in EXPECTED_SEEDS
    }
    return {
        "config": config,
        "source_auc": source_auc,
        "source_macro_auc": source_macro_auc,
        "target_auc": target_auc,
        "oracle_auc": oracle_auc,
        "conflict_projections_by_role_seed": conflict_counts,
        "rows": rows,
        "history": history,
        "gradient_scales": gradient_scales,
        "validation": validation,
    }


def _synthetic_source_only_smoke(
    base: dict[str, Any],
    structures: dict[str, RuntimeSpnStructure],
) -> dict[str, Any]:
    tasks = [
        RuntimeSpnJointTask(
            name=name,
            group="source",
            structure=structures[name],
            train_dataset=_synthetic_dataset(structures[name].block_bits, seed=index),
            validation_dataset=_synthetic_dataset(
                structures[name].block_bits,
                seed=100 + index,
            ),
        )
        for index, name in enumerate(SOURCE_CIPHERS)
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        initializer = RelationModeRuntimeE4(_plain_spec(base["model"]), "true")
    initial_state = _clone_state_dict(initializer.state_dict())
    role_task_names = []
    trained_models = []
    for relation_mode in ("true", "independent"):
        model = RelationModeRuntimeE4(_plain_spec(base["model"]), relation_mode)
        model.load_state_dict(initial_state, strict=True)
        result = train_runtime_spn_joint(
            model,
            tasks,
            TrainingConfig(
                epochs=1,
                batch_size=16,
                learning_rate=1e-4,
                seed=0,
                device="cpu",
                optimizer="adam",
                weight_decay=1e-5,
                lr_scheduler="none",
                checkpoint_metric="val_macro_auc",
                restore_best_checkpoint=True,
                loss="mse",
            ),
            gradient_combination=(
                "representation_l2_equalized_pcgrad_fixed_order"
            ),
        )
        role_task_names.append(result.metadata["task_names"])
        trained_models.append((model, relation_mode))
    target_dataset = _synthetic_dataset(
        structures[HOLDOUT_CIPHER].block_bits,
        seed=999,
    )
    target_finite = True
    for model, _ in trained_models:
        metrics = _evaluate_target(
            model,
            target_dataset,
            structures[HOLDOUT_CIPHER],
            {"batch_size": 16, "loss": "mse"},
            holdout_cipher=HOLDOUT_CIPHER,
        )
        target_finite = target_finite and bool(np.isfinite(metrics["auc"]))
    return {
        "source_task_names": role_task_names[-1],
        "source_task_names_by_role": role_task_names,
        "target_evaluated_after_both_roles": target_finite,
        "target_optimizer_steps": 0,
    }


def _cache_probe(base: dict[str, Any], project_root: Path) -> dict[str, Any]:
    root = project_root / base["training"]["cache_source_root"]
    required = []
    for seed in EXPECTED_SEEDS:
        for cipher in SOURCE_CIPHERS:
            for split in ("train", "validation"):
                required.extend(
                    root / f"seed{seed}" / cipher / split / name
                    for name in ("features.npy", "labels.npy", "metadata.json")
                )
        required.extend(
            root / f"seed{seed}" / HOLDOUT_CIPHER / "validation" / name
            for name in ("features.npy", "labels.npy", "metadata.json")
        )
    target_train_paths = [
        root / f"seed{seed}" / HOLDOUT_CIPHER / "train" for seed in EXPECTED_SEEDS
    ]
    return {
        "passed": all(path.is_file() for path in required),
        "required_file_count": len(required),
        "required_files_present": sum(path.is_file() for path in required),
        "target_train_referenced": any(
            target_path in path.parents
            for path in required
            for target_path in target_train_paths
        ),
        "historical_target_train_cache_exists": any(
            path.exists() for path in target_train_paths
        ),
    }


def _wrong_sbox_contract(
    correct: RuntimeSpnStructure,
    wrong: RuntimeSpnStructure,
    source: RuntimeSpnStructure,
) -> bool:
    expected_truth = source.sbox_truth_bits[-1, 0].reshape(1, 1, 64).repeat(
        correct.rounds,
        correct.cells,
        1,
    )
    return bool(
        not torch.equal(correct.sbox_truth_bits, wrong.sbox_truth_bits)
        and torch.equal(wrong.sbox_truth_bits, expected_truth)
        and torch.equal(correct.cell_membership, wrong.cell_membership)
        and torch.equal(correct.bit_role, wrong.bit_role)
        and torch.equal(correct.linear_matrices, wrong.linear_matrices)
        and torch.equal(
            correct.inverse_linear_matrices,
            wrong.inverse_linear_matrices,
        )
    )


def _cell_relabel_error(
    model: RelationModeRuntimeE4,
    features: torch.Tensor,
    structure: RuntimeSpnStructure,
) -> float:
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    relabeled_features = torch.empty_like(features)
    relabeled_features[..., bit_permutation] = features
    with torch.no_grad():
        original = model(features, structure)
        transformed = model(relabeled_features, relabeled)
    return float((original - transformed).abs().max())


def _validate_role_checkpoint(
    role_payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    seed: int,
    role: str,
    relation_mode: str,
    config_hash: str,
    initial_hash: str,
) -> None:
    if role_payload.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("A8 resumed role checkpoint hash drifted")
    if role_payload.get("seed") != seed or role_payload.get("role") != role:
        raise ValueError("A8 resumed role identity drifted")
    if role_payload.get("relation_mode") != relation_mode:
        raise ValueError("A8 resumed role relation mode drifted")
    if role_payload.get("parameter_count") != 442466:
        raise ValueError("A8 resumed role parameter count drifted")
    if role_payload.get("initial_state_sha256") != initial_hash:
        raise ValueError("A8 resumed role initial state drifted")
    if tuple(role_payload.get("metadata", {}).get("task_names", ())) != SOURCE_CIPHERS:
        raise ValueError("A8 resumed role source panel drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_payload(
        checkpoint,
        seed=seed,
        role=role,
        relation_mode=relation_mode,
        config_hash=config_hash,
        initial_hash=initial_hash,
    )


def _validate_checkpoint_payload(
    checkpoint: dict[str, Any],
    *,
    seed: int,
    role: str,
    relation_mode: str,
    config_hash: str,
    initial_hash: str,
) -> None:
    expected = {
        "seed": seed,
        "role": role,
        "relation_mode": relation_mode,
        "config_sha256": config_hash,
        "initial_state_sha256": initial_hash,
        "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
        "holdout_cipher": HOLDOUT_CIPHER,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"A8 checkpoint metadata drifted: {key}")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("A8 checkpoint state_dict is missing")


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_dialga_holdout_supported": (
            "双seed未见Dialga信号、拓扑、S盒与重训锚点门全过"
        ),
        "innovation1_runtime_spn_dialga_holdout_partial": (
            "未见Dialga结构归因成立但源保持未全过"
        ),
        "innovation1_runtime_spn_dialga_holdout_not_supported": (
            "GF(2)拓扑门通过，但S盒原语门失败；停止救援性扩样"
        ),
        "innovation1_runtime_spn_dialga_holdout_invalid": "协议无效",
    }.get(decision, decision)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)
