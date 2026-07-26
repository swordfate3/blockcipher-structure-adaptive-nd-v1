from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
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


SOURCE_CIPHERS = ("gift64", "skinny64", "rectangle80", "dialga128")
HOLDOUT_CIPHER = "uknit64"
TRAINING_ROLES = ("correct_pooling", "uniform_pooling_anchor")
EXPECTED_SEEDS = (0, 1)
TARGET_EVALUATIONS = (
    "candidate_correct",
    "candidate_corrupted_target",
    "candidate_no_topology_target",
    "candidate_uniform_same_checkpoint",
    "candidate_shuffled_same_checkpoint",
    "uniform_trained_anchor",
)
CANDIDATE_COUNTERFACTUALS = TARGET_EVALUATIONS[:5]


def load_and_validate_uknit_heterogeneous_holdout_config(
    path: Path,
    *,
    project_root: Path,
    require_readiness: bool,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("H1-A6 config schema_version must be 1")
    source = config.get("source", {})
    required_source = {
        "a5_required_decision": (
            "innovation1_runtime_spn_h1_relation_activity_pooling_invalid"
        ),
        "readiness_required_decision": (
            "innovation1_runtime_spn_uknit_heterogeneous_holdout_readiness_passed"
        ),
    }
    for key, expected in required_source.items():
        if source.get(key) != expected:
            raise ValueError(f"H1-A6 source field drifted: {key}")
    if tuple(config.get("source_ciphers", ())) != SOURCE_CIPHERS:
        raise ValueError("H1-A6 source cipher panel drifted")
    if config.get("holdout_cipher") != HOLDOUT_CIPHER:
        raise ValueError("H1-A6 must hold out uKNIT")
    if HOLDOUT_CIPHER in config["source_ciphers"]:
        raise ValueError("H1-A6 holdout leaked into the source panel")
    required_roles = {
        "correct_pooling": {
            "relation_mode": "true",
            "pooling_mode": "correct",
        },
        "uniform_pooling_anchor": {
            "relation_mode": "true",
            "pooling_mode": "uniform",
        },
    }
    if config.get("training_roles") != required_roles:
        raise ValueError("H1-A6 training role panel drifted")
    required_candidate = {
        "gradient_combination": (
            "representation_l2_equalized_pcgrad_fixed_order"
        ),
        "representation_parameters": "all_except_shared_classifier",
        "classifier_gradient_combination": "raw_arithmetic_mean",
        "task_sampling": "unchanged_equal_one_batch_per_task",
        "initialization": "same_state_per_seed_across_roles",
        "expected_parameter_count": 442466,
        "seeds": [0, 1],
    }
    for key, expected in required_candidate.items():
        if config.get("candidate", {}).get(key) != expected:
            raise ValueError(f"H1-A6 candidate field drifted: {key}")
    required_target = {
        "candidate_correct": {
            "source_role": "correct_pooling",
            "structure": "correct",
            "relation_mode": "true",
            "pooling_mode": "correct",
        },
        "candidate_corrupted_target": {
            "source_role": "correct_pooling",
            "structure": "corrupted",
            "relation_mode": "true",
            "pooling_mode": "correct",
        },
        "candidate_no_topology_target": {
            "source_role": "correct_pooling",
            "structure": "correct",
            "relation_mode": "independent",
            "pooling_mode": "correct",
        },
        "candidate_uniform_same_checkpoint": {
            "source_role": "correct_pooling",
            "structure": "correct",
            "relation_mode": "true",
            "pooling_mode": "uniform",
        },
        "candidate_shuffled_same_checkpoint": {
            "source_role": "correct_pooling",
            "structure": "correct",
            "relation_mode": "true",
            "pooling_mode": "shuffled",
        },
        "uniform_trained_anchor": {
            "source_role": "uniform_pooling_anchor",
            "structure": "correct",
            "relation_mode": "true",
            "pooling_mode": "uniform",
        },
    }
    if config.get("target_evaluations") != required_target:
        raise ValueError("H1-A6 target evaluation panel drifted")
    required_gate = {
        "target_auc_floor": 0.55,
        "target_topology_margin": 0.005,
        "target_pooling_margin": 0.005,
        "trained_anchor_retention_tolerance": 0.01,
        "source_macro_retention_tolerance": 0.005,
        "minimum_conflict_projections_per_role_seed": 1,
        "required_seeds": [0, 1],
    }
    for key, expected in required_gate.items():
        if config.get("gate", {}).get(key) != expected:
            raise ValueError(f"H1-A6 gate field drifted: {key}")
    for path_key, hash_key in (
        ("protocol_config_path", "protocol_config_sha256"),
        ("a5_config_path", "a5_config_sha256"),
    ):
        if config_sha256(project_root / source[path_key]) != source.get(hash_key):
            raise ValueError(f"H1-A6 source hash drifted: {path_key}")
    load_and_validate_holdout_config(project_root / source["protocol_config_path"])
    if require_readiness:
        readiness = _read_json(project_root / source["readiness_gate_path"])
        if readiness.get("status") != "pass":
            raise ValueError("H1-A6 readiness did not pass")
        if readiness.get("decision") != source["readiness_required_decision"]:
            raise ValueError("H1-A6 readiness decision drifted")
        if not all(readiness.get("checks", {}).values()):
            raise ValueError("H1-A6 readiness contains a failed check")
    return config


def run_uknit_heterogeneous_holdout_readiness(
    *,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    base = load_and_validate_holdout_config(
        project_root / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    target = structures[HOLDOUT_CIPHER]
    models = {
        mode: RelationModeRuntimeE4(_pooling_spec(base["model"], mode), "true").eval()
        for mode in ("correct", "uniform", "shuffled")
    }
    state = _clone_state_dict(models["correct"].state_dict())
    for model in models.values():
        model.load_state_dict(state, strict=True)
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    target_weights = {
        mode: RuntimeE4EquivariantSpnDistinguisher.relation_activity_weights(
            target,
            mode=mode,
            relation_mode="true",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for mode in ("correct", "uniform", "shuffled")
    }
    features = torch.randint(
        0,
        2,
        (8, 4, 2, target.block_bits),
        generator=torch.Generator().manual_seed(26_072_806),
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = {mode: model(features, target) for mode, model in models.items()}
        independent_correct = RelationModeRuntimeE4(
            _pooling_spec(base["model"], "correct"),
            "independent",
        ).eval()
        independent_uniform = RelationModeRuntimeE4(
            _pooling_spec(base["model"], "uniform"),
            "independent",
        ).eval()
        independent_correct.load_state_dict(state, strict=True)
        independent_uniform.load_state_dict(state, strict=True)
        independent_equal = torch.equal(
            independent_correct(features, target),
            independent_uniform(features, target),
        )
    relabeled, bit_permutation = target.relabel_cells(
        tuple(reversed(range(target.cells)))
    )
    relabeled_features = torch.empty_like(features)
    relabeled_features[..., bit_permutation] = features
    relabel_errors = {}
    for mode in ("correct", "shuffled"):
        with torch.no_grad():
            relabeled_logits = models[mode](relabeled_features, relabeled)
        relabel_errors[mode] = float((logits[mode] - relabeled_logits).abs().max())
    cache_probe = _cache_probe(base, project_root)
    smoke = _synthetic_source_only_smoke(base, structures)
    a5_gate = _read_json(project_root / config["source"]["a5_gate_path"])
    signature_types = int(torch.unique(target_weights["correct"], dim=0).shape[0])
    checks = {
        "source_panel_exact_and_holdout_absent": tuple(config["source_ciphers"])
        == SOURCE_CIPHERS
        and HOLDOUT_CIPHER not in SOURCE_CIPHERS,
        "a5_invalid_for_unidentifiable_target_control": (
            a5_gate.get("decision") == config["source"]["a5_required_decision"]
            and a5_gate.get("protocol_valid") is False
            and "one-to-one" in a5_gate.get("invalid_reason", "")
        ),
        "target_signature_heterogeneous": signature_types > 1,
        "target_correct_differs_from_uniform": not torch.equal(
            target_weights["correct"], target_weights["uniform"]
        )
        and not torch.equal(logits["correct"], logits["uniform"]),
        "target_correct_differs_from_shuffled": not torch.equal(
            target_weights["correct"], target_weights["shuffled"]
        )
        and not torch.equal(logits["correct"], logits["shuffled"]),
        "parameter_and_state_keys_matched": set(parameter_counts.values())
        == {config["candidate"]["expected_parameter_count"]}
        and all(tuple(model.state_dict()) == tuple(state) for model in models.values()),
        "initial_states_bit_exact": all(
            all(torch.equal(state[name], model.state_dict()[name]) for name in state)
            for model in models.values()
        ),
        "independent_forces_uniform": independent_equal,
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
        "outputs_finite": all(bool(torch.isfinite(value).all()) for value in logits.values()),
    }
    passed = all(checks.values())
    return {
        "run_id": "i1_runtime_spn_uknit_heterogeneous_holdout_a6_readiness_20260726",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_uknit_heterogeneous_holdout_readiness_passed"
            if passed
            else "innovation1_runtime_spn_uknit_heterogeneous_holdout_readiness_failed"
        ),
        "checks": checks,
        "parameter_counts": parameter_counts,
        "target_signature_types": signature_types,
        "target_correct_uniform_l1": float(
            (target_weights["correct"] - target_weights["uniform"]).abs().sum()
        ),
        "target_correct_shuffled_l1": float(
            (target_weights["correct"] - target_weights["shuffled"]).abs().sum()
        ),
        "target_logit_differences": {
            mode: float((logits["correct"] - logits[mode]).abs().max())
            for mode in ("uniform", "shuffled")
        },
        "cell_relabel_errors": relabel_errors,
        "cache_probe": cache_probe,
        "smoke": smoke,
        "target_training_rows": 0,
        "next_action": (
            "run the frozen A6 local diagnostic"
            if passed
            else "repair only the failed readiness invariant before training"
        ),
    }


def run_uknit_heterogeneous_holdout(
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
    checkpoint_root = output_root / "checkpoints"
    role_root = output_root / "role-results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    role_root.mkdir(parents=True, exist_ok=True)
    config_hash = config_sha256(config_path)
    roles: dict[int, dict[str, dict[str, Any]]] = {}
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
            initializer = RelationModeRuntimeE4(
                _pooling_spec(base["model"], "correct"),
                "true",
            )
        initial_state = _clone_state_dict(initializer.state_dict())
        initial_hash = _state_dict_sha256(initial_state)
        roles[seed] = {}
        for role in TRAINING_ROLES:
            checkpoint_path = checkpoint_root / f"seed{seed}-{role}.pt"
            role_path = role_root / f"seed{seed}-{role}.json"
            resumed = _load_resumable_role(
                role_path,
                checkpoint_path,
                config_sha256=config_hash,
            )
            if resumed is not None:
                if resumed.get("initial_state_sha256") != initial_hash:
                    raise ValueError("H1-A6 resumed role initial state drifted")
                _validate_role_checkpoint(
                    resumed,
                    checkpoint_path=checkpoint_path,
                    seed=seed,
                    role=role,
                    pooling_mode=config["training_roles"][role]["pooling_mode"],
                    config_hash=config_hash,
                    initial_hash=initial_hash,
                )
                roles[seed][role] = resumed
                _emit(progress_callback, "source_role_reused", seed=seed, role=role)
                continue
            role_config = config["training_roles"][role]
            model = RelationModeRuntimeE4(
                _pooling_spec(base["model"], role_config["pooling_mode"]),
                role_config["relation_mode"],
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
                "pooling_mode": role_config["pooling_mode"],
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
                "pooling_mode": role_config["pooling_mode"],
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
            raise RuntimeError("both H1-A6 source roles must finish before target load")
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
            checkpoint_path = Path(roles[seed][source_role]["checkpoint_path"])
            checkpoint_hash = _file_sha256(checkpoint_path)
            if checkpoint_hash != roles[seed][source_role]["checkpoint_sha256"]:
                raise ValueError("H1-A6 checkpoint changed after role completion")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            _validate_checkpoint_payload(
                checkpoint,
                seed=seed,
                role=source_role,
                pooling_mode=config["training_roles"][source_role]["pooling_mode"],
                config_hash=config_hash,
                initial_hash=roles[seed][source_role]["initial_state_sha256"],
            )
            model = RelationModeRuntimeE4(
                _pooling_spec(base["model"], evaluation["pooling_mode"]),
                evaluation["relation_mode"],
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            structure = structures[HOLDOUT_CIPHER]
            if evaluation["structure"] == "corrupted":
                structure = structure.corrupted()
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
                "pooling_mode": evaluation["pooling_mode"],
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
            project_root / config["source"]["readiness_gate_path"]
        ),
        a5_gate=_read_json(project_root / config["source"]["a5_gate_path"]),
    )


def adjudicate_uknit_heterogeneous_holdout(
    payload: dict[str, Any],
) -> dict[str, Any]:
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
        pooling_margins = {
            name: correct - target[name]
            for name in (
                "candidate_uniform_same_checkpoint",
                "candidate_shuffled_same_checkpoint",
            )
        }
        candidate_macro = payload["source_macro_auc"][key]["correct_pooling"]
        anchor_macro = payload["source_macro_auc"][key][
            "uniform_pooling_anchor"
        ]
        conflict_counts = payload["conflict_projections_by_role_seed"][key]
        functional_checks = {
            "target_auc_floor": correct >= gate_config["target_auc_floor"],
            "target_topology_margins": all(
                value >= gate_config["target_topology_margin"]
                for value in topology_margins.values()
            ),
            "target_pooling_margins": all(
                value >= gate_config["target_pooling_margin"]
                for value in pooling_margins.values()
            ),
        }
        checks = {
            **functional_checks,
            "trained_anchor_retained": correct
            >= target["uniform_trained_anchor"]
            - gate_config["trained_anchor_retention_tolerance"],
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
            "uniform_trained_anchor_auc": target["uniform_trained_anchor"],
            "candidate_minus_trained_anchor": correct
            - target["uniform_trained_anchor"],
            "topology_margins": topology_margins,
            "pooling_margins": pooling_margins,
            "candidate_source_macro_auc": candidate_macro,
            "uniform_anchor_source_macro_auc": anchor_macro,
            "source_macro_delta": candidate_macro - anchor_macro,
            "conflict_projections": conflict_counts,
            "checks": checks,
            "functional_pass": seed_functional,
            "pass": seed_pass,
        }

    if payload["validation"]["status"] != "pass":
        status = "invalid"
        decision = "innovation1_runtime_spn_uknit_heterogeneous_holdout_invalid"
        next_action = "repair the exact readiness, initialization, cache or checkpoint failure"
    elif full_pass:
        status = "pass"
        decision = "innovation1_runtime_spn_uknit_heterogeneous_holdout_supported"
        next_action = (
            "preregister a second independent heterogeneous whole-cipher holdout "
            "with the same relation-activity primitive"
        )
    elif functional_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_uknit_heterogeneous_holdout_partial"
        next_action = (
            "retain unseen-uKNIT structural attribution and audit source calibration "
            "before any architecture change or scale-up"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_uknit_heterogeneous_holdout_not_supported"
        next_action = (
            "close relation-mass activity pooling and redesign the shared structure "
            "representation without optimizer or scale changes"
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
            "local 2048/class/source whole-cipher holdout diagnostic; not formal "
            "scale, universality, attack, SOTA or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "load uKNIT training rows or select checkpoints on uKNIT",
            "change optimizer, samples, epochs or launch remote scale",
            "add a target head, cipher ID, expert, Adapter, FiLM or typed residual",
            "claim universal SPN adaptation from one heterogeneous holdout",
        ],
    }


def write_uknit_heterogeneous_holdout_readiness_artifacts(
    readiness: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": readiness["run_id"],
            "row_kind": "target_signature",
            "cipher": HOLDOUT_CIPHER,
            "signature_types": readiness["target_signature_types"],
            "correct_uniform_l1": readiness["target_correct_uniform_l1"],
            "correct_shuffled_l1": readiness["target_correct_shuffled_l1"],
            **readiness["target_logit_differences"],
        },
        *(
            {
                "run_id": readiness["run_id"],
                "row_kind": "cell_relabel",
                "cipher": HOLDOUT_CIPHER,
                "pooling_mode": mode,
                "max_error": error,
            }
            for mode, error in readiness["cell_relabel_errors"].items()
        ),
    ]
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


def write_uknit_heterogeneous_holdout_artifacts(
    *,
    payload: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> None:
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["rows"]),
        encoding="utf-8",
    )
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
    render_uknit_heterogeneous_holdout_svg(
        payload,
        gate,
        output_root / "curves.svg",
    )


def render_uknit_heterogeneous_holdout_svg(
    payload: dict[str, Any],
    gate: dict[str, Any],
    output: Path,
) -> None:
    display = {
        "gift64": "GIFT",
        "skinny64": "SKINNY",
        "rectangle80": "RECTANGLE",
        "dialga128": "Dialga",
    }
    target_labels = (
        ("candidate_correct", "正确关系池化", "#0072B2"),
        ("candidate_corrupted_target", "损坏结构", "#D55E00"),
        ("candidate_no_topology_target", "无拓扑", "#009E73"),
        ("candidate_uniform_same_checkpoint", "同权重普通池化", "#CC79A7"),
        ("candidate_shuffled_same_checkpoint", "同权重错误签名", "#E69F00"),
        ("uniform_trained_anchor", "重训普通池化锚点", "#7F8C8D"),
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
            target = payload["target_auc"][key]
            bars = axes[0, column].barh(
                range(len(target_labels)),
                [target[name] for name, _, _ in target_labels],
                color=[color for _, _, color in target_labels],
            )
            correct_auc = target["candidate_correct"]
            value_labels = []
            for name, _, _ in target_labels:
                value = target[name]
                value_labels.append(
                    f"{value:.6f}"
                    if name == "candidate_correct"
                    else f"{value:.6f}  Δ={correct_auc - value:+.6f}"
                )
            axes[0, column].bar_label(
                bars,
                labels=value_labels,
                padding=3,
                fontsize=7.5,
            )
            axes[0, column].axvline(0.5, color="#34495E")
            axes[0, column].axvline(0.55, color="#7B2CBF", linestyle="--")
            axes[0, column].set_xlim(0.46, 0.56)
            axes[0, column].set_yticks(
                range(len(target_labels)),
                [label for _, label, _ in target_labels],
            )
            axes[0, column].set_xlabel("未见 uKNIT 验证 AUC")
            axes[0, column].set_title(
                f"seed{seed}：零训练结构归因（Δ=正确关系减该项）"
            )

            y = np.arange(len(SOURCE_CIPHERS))
            candidate = payload["source_auc"][key]["correct_pooling"]
            anchor = payload["source_auc"][key]["uniform_pooling_anchor"]
            for values_by_cipher, label, offset, color in (
                (candidate, "正确关系池化", -0.13, "#0072B2"),
                (anchor, "普通池化锚点", 0.13, "#7F8C8D"),
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
            axes[1, column].axvline(0.5, color="#34495E")
            axes[1, column].set_xlim(0.4, 1.0)
            axes[1, column].set_yticks(
                y,
                [display[name] for name in SOURCE_CIPHERS],
            )
            axes[1, column].set_xlabel("四源验证 AUC")
            axes[1, column].set_title(f"seed{seed}：同预算源任务保持")
            axes[1, column].legend(frameon=False, loc="lower right")
        fig.suptitle(
            "创新1 H1-A6：uKNIT 异构 GF(2) 整密码零训练留出\n"
            "训练仅含 GIFT / SKINNY / RECTANGLE / Dialga；两角色同初始化同预算",
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
            left=0.14,
            right=0.98,
            top=0.86,
            bottom=0.1,
            wspace=0.36,
            hspace=0.42,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg", bbox_inches="tight")
        plt.close(fig)


def _assemble_payload(
    *,
    config: dict[str, Any],
    config_sha256_value: str,
    base: dict[str, Any],
    roles: dict[int, dict[str, dict[str, Any]]],
    target_loaded_after_roles: dict[int, bool],
    readiness: dict[str, Any],
    a5_gate: dict[str, Any],
) -> dict[str, Any]:
    source_auc = {}
    source_macro_auc = {}
    target_auc = {}
    rows = []
    history = []
    gradient_scales = []
    protocol_by_name = {item["name"]: item for item in base["protocols"]}
    conflict_projections_by_role_seed = {}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        source_auc[key] = {}
        source_macro_auc[key] = {}
        conflict_projections_by_role_seed[key] = {}
        for role in TRAINING_ROLES:
            role_payload = roles[seed][role]
            source_auc[key][role] = {
                name: float(role_payload["validation_metrics"][name]["auc"])
                for name in SOURCE_CIPHERS
            }
            source_macro_auc[key][role] = float(
                np.mean(list(source_auc[key][role].values()))
            )
            for name in SOURCE_CIPHERS:
                rows.append(
                    {
                        "run_id": config["run_id"],
                        "row_kind": "source_validation",
                        "seed": seed,
                        "role": role,
                        "pooling_mode": role_payload["pooling_mode"],
                        "cipher": name,
                        "cipher_display_name": protocol_by_name[name]["display_name"],
                        "rounds": protocol_by_name[name]["rounds"],
                        "parameter_count": role_payload["parameter_count"],
                        "samples_per_class": base["training"]["samples_per_class"],
                        "validation_samples_per_class": base["training"][
                            "validation_samples_per_class"
                        ],
                        "pairs_per_sample": base["training"]["pairs_per_sample"],
                        "negative_mode": base["training"]["negative_mode"],
                        "checkpoint": role_payload["checkpoint_path"],
                        "metrics": {
                            "train": role_payload["train_metrics"][name],
                            "validation": role_payload["validation_metrics"][name],
                        },
                        "config_sha256": config_sha256_value,
                    }
                )
            history.extend(
                {"seed": seed, "role": role, **row}
                for row in role_payload["history"]
            )
            diagnostics = role_payload["gradient_diagnostics"]
            conflict_total = 0
            for task in SOURCE_CIPHERS:
                conflicts = int(
                    diagnostics["task_conflict_projection_counts"][task]
                )
                conflict_total += conflicts
                gradient_scales.append(
                    {
                        "seed": seed,
                        "role": role,
                        "task": task,
                        "mean_raw_representation_gradient_l2": diagnostics[
                            "task_representation_gradient_mean_l2"
                        ][task],
                        "mean_applied_scale": diagnostics[
                            "task_gradient_scale_mean"
                        ][task],
                        "observations": diagnostics[
                            "task_gradient_scale_observations"
                        ][task],
                        "conflict_projections": conflicts,
                    }
                )
            conflict_projections_by_role_seed[key][role] = conflict_total
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
                    "cipher": HOLDOUT_CIPHER,
                    "cipher_display_name": "uKNIT-BC prefix-r5（整密码留出）",
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
        == config["source"]["readiness_required_decision"]
        and all(readiness.get("checks", {}).values()),
        "a5_invalid_gate_matches": a5_gate.get("decision")
        == config["source"]["a5_required_decision"]
        and a5_gate.get("protocol_valid") is False,
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
            tuple(roles[seed][role]["metadata"]["task_names"])
            == SOURCE_CIPHERS
            and HOLDOUT_CIPHER not in roles[seed][role]["metadata"]["task_names"]
            for seed in EXPECTED_SEEDS
            for role in TRAINING_ROLES
        ),
        "source_only_checkpoint_selection": all(
            roles[seed][role]["metadata"]["selected_checkpoint"] == "best"
            and tuple(
                roles[seed][role]["metadata"]["task_names"]
            )
            == SOURCE_CIPHERS
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
        "result_rows": len(rows) == 28,
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
    return {
        "config": config,
        "source_auc": source_auc,
        "source_macro_auc": source_macro_auc,
        "target_auc": target_auc,
        "conflict_projections_by_role_seed": conflict_projections_by_role_seed,
        "rows": rows,
        "history": history,
        "gradient_scales": gradient_scales,
        "validation": validation,
    }


def _synthetic_source_only_smoke(
    base: dict[str, Any],
    structures: dict[str, Any],
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
        initializer = RelationModeRuntimeE4(
            _pooling_spec(base["model"], "correct"),
            "true",
        )
    initial_state = _clone_state_dict(initializer.state_dict())
    trained_models = []
    role_task_names = []
    for mode in ("correct", "uniform"):
        model = RelationModeRuntimeE4(_pooling_spec(base["model"], mode), "true")
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
        trained_models.append(model)

    target_dataset = _synthetic_dataset(
        structures[HOLDOUT_CIPHER].block_bits,
        seed=999,
    )
    target_finite = True
    for model in trained_models:
        target_metrics = _evaluate_target(
            model,
            target_dataset,
            structures[HOLDOUT_CIPHER],
            {"batch_size": 16, "loss": "mse"},
            holdout_cipher=HOLDOUT_CIPHER,
        )
        target_finite = target_finite and bool(np.isfinite(target_metrics["auc"]))
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
            target_path in path.parents for path in required for target_path in target_train_paths
        ),
        "historical_target_train_cache_exists": any(
            path.exists() for path in target_train_paths
        ),
    }


def _pooling_spec(model: dict[str, Any], mode: str) -> Any:
    return _plain_spec({**model, "relation_activity_pooling_mode": mode})


def _validate_role_checkpoint(
    role_payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    seed: int,
    role: str,
    pooling_mode: str,
    config_hash: str,
    initial_hash: str,
) -> None:
    if role_payload.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("H1-A6 resumed role checkpoint hash drifted")
    if role_payload.get("seed") != seed or role_payload.get("role") != role:
        raise ValueError("H1-A6 resumed role identity drifted")
    if role_payload.get("pooling_mode") != pooling_mode:
        raise ValueError("H1-A6 resumed role pooling mode drifted")
    if role_payload.get("parameter_count") != 442466:
        raise ValueError("H1-A6 resumed role parameter count drifted")
    metadata = role_payload.get("metadata", {})
    if tuple(metadata.get("task_names", ())) != SOURCE_CIPHERS:
        raise ValueError("H1-A6 resumed role source panel drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_payload(
        checkpoint,
        seed=seed,
        role=role,
        pooling_mode=pooling_mode,
        config_hash=config_hash,
        initial_hash=initial_hash,
    )


def _validate_checkpoint_payload(
    checkpoint: dict[str, Any],
    *,
    seed: int,
    role: str,
    pooling_mode: str,
    config_hash: str,
    initial_hash: str,
) -> None:
    expected = {
        "seed": seed,
        "role": role,
        "pooling_mode": pooling_mode,
        "config_sha256": config_hash,
        "initial_state_sha256": initial_hash,
        "checkpoint_selection_tasks": list(SOURCE_CIPHERS),
        "holdout_cipher": HOLDOUT_CIPHER,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"H1-A6 checkpoint metadata drifted: {key}")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("H1-A6 checkpoint state_dict is missing")


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _decision_chinese(decision: str) -> str:
    return {
        "innovation1_runtime_spn_uknit_heterogeneous_holdout_supported": (
            "双seed未见uKNIT结构归因与同预算保持全过，开放第二异构整密码留出"
        ),
        "innovation1_runtime_spn_uknit_heterogeneous_holdout_partial": (
            "未见uKNIT结构归因成立但同预算保持未全过，转源校准审计"
        ),
        "innovation1_runtime_spn_uknit_heterogeneous_holdout_not_supported": (
            "关系活动池化未通过未见uKNIT结构控制，关闭该原语"
        ),
        "innovation1_runtime_spn_uknit_heterogeneous_holdout_invalid": "协议无效",
    }.get(decision, decision)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "adjudicate_uknit_heterogeneous_holdout",
    "load_and_validate_uknit_heterogeneous_holdout_config",
    "render_uknit_heterogeneous_holdout_svg",
    "run_uknit_heterogeneous_holdout",
    "run_uknit_heterogeneous_holdout_readiness",
    "write_uknit_heterogeneous_holdout_artifacts",
    "write_uknit_heterogeneous_holdout_readiness_artifacts",
]
