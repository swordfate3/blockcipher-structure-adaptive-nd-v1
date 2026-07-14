from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.four_role_attribution_gate import (
    _checkpoint_history_errors,
)


TRANSFER_MODEL_ROLES = {
    "gift_anchor": "gift_cross_spn_aligned_token_mixer_raw_anchor",
    "gift_typed_scratch": "gift_cross_spn_typed_cell_true",
    "true_to_true": "gift_cross_spn_typed_cell_true_from_present_true",
    "shuffled_to_true": "gift_cross_spn_typed_cell_true_from_present_shuffled",
    "true_to_shuffled": "gift_cross_spn_typed_cell_shuffled_from_present_true",
}

_TYPED_OPTIONS = {
    "mixer_depth": 2,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "attention_mean_max",
    "dropout": 0.0,
}
_ANCHOR_OPTIONS = {
    "mixer_depth": 1,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "topk_logsumexp",
    "dropout": 0.0,
    "top_k": 2,
    "lse_temperature": 1.0,
}
_TRUE_SOURCE_SHA = "eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1"
_SHUFFLED_SOURCE_SHA = "fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22"
_EXPERIMENT_BUDGETS = {
    "e4_r2": (8192, 10),
    "e4_r3": (65536, 10),
}


def gate_cross_spn_typed_transfer(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    readiness_only: bool = False,
    experiment_stage: str = "e4_r2",
) -> dict[str, Any]:
    argument_errors = _argument_errors(
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        experiment_stage=experiment_stage,
    )
    if argument_errors:
        return _invalid(argument_errors, experiment_stage=experiment_stage)
    if len(results_paths) != 1 or len(progress_paths) != 1:
        return _invalid(
            [
                f"{experiment_stage.upper()} requires exactly one result/progress run "
                f"results={len(results_paths)} progress={len(progress_paths)}"
            ],
            experiment_stage=experiment_stage,
        )
    rows, result_errors = _read_jsonl(results_paths[0], "results")
    progress, progress_errors = _read_jsonl(progress_paths[0], "progress")
    errors = [
        *result_errors,
        *progress_errors,
        *_result_errors(
            rows,
            expected_seed=expected_seeds[0],
            samples_per_class=samples_per_class,
            epochs=epochs,
        ),
        *_progress_errors(
            progress,
            rows=rows,
            result_path=results_paths[0],
            expected_seed=expected_seeds[0],
            samples_per_class=samples_per_class,
        ),
    ]
    if errors:
        return _invalid(errors, experiment_stage=experiment_stage)

    by_role = {
        role: next(
            row
            for row in rows
            if row["selected_model"] == model
        )
        for role, model in TRANSFER_MODEL_ROLES.items()
    }
    aucs = {role: float(row["metrics"]["auc"]) for role, row in by_role.items()}
    margins = {
        "anchor_margin": aucs["true_to_true"] - aucs["gift_anchor"],
        "scratch_margin": aucs["true_to_true"] - aucs["gift_typed_scratch"],
        "source_topology_margin": aucs["true_to_true"] - aucs["shuffled_to_true"],
        "target_topology_margin": aucs["true_to_true"] - aucs["true_to_shuffled"],
    }
    evidence = {
        "models": TRANSFER_MODEL_ROLES,
        "aucs": aucs,
        "margins": margins,
        "source_checkpoint_sha256": {
            "true": _TRUE_SOURCE_SHA,
            "shuffled": _SHUFFLED_SOURCE_SHA,
        },
        "cache_root": rows[0]["training"]["dataset_cache_root"],
        "typed_parameter_count": by_role["gift_typed_scratch"]["parameter_count"],
    }
    if readiness_only:
        return {
            "status": "pass",
            "decision": "implementation_ready",
            "errors": [],
            "expected_seeds": list(expected_seeds),
            "samples_per_class": samples_per_class,
            "epochs": epochs,
            "experiment_stage": experiment_stage,
            **evidence,
            "research_decision_applied": False,
            "claim_scope": (
                f"{experiment_stage.upper()} readiness only; metrics not interpreted"
            ),
            "next_action": (
                f"run_frozen_{experiment_stage}_seed"
                f"{expected_seeds[0]}_local_diagnostic"
            ),
            "stopped_actions": _stopped_actions("implementation_ready"),
        }

    decision, next_action = _decision(
        aucs,
        margins,
        expected_seed=expected_seeds[0],
        experiment_stage=experiment_stage,
    )
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "experiment_stage": experiment_stage,
        **evidence,
        "research_decision_applied": True,
        "claim_scope": (
            f"{samples_per_class}/class {experiment_stage.upper()} local transfer diagnostic; "
            "not formal, paper-scale, remote, or breakthrough evidence"
        ),
        "next_action": next_action,
        "stopped_actions": _stopped_actions(decision),
    }


def _argument_errors(
    *,
    expected_seeds: Any,
    samples_per_class: Any,
    epochs: Any,
    readiness_only: bool,
    experiment_stage: Any,
) -> list[str]:
    errors: list[str] = []
    if experiment_stage not in _EXPERIMENT_BUDGETS:
        errors.append(
            "experiment_stage must be one of "
            f"{sorted(_EXPERIMENT_BUDGETS)} actual={experiment_stage!r}"
        )
    if expected_seeds not in {(0,), (1,)}:
        errors.append(
            "expected_seeds must equal one frozen target seed "
            f"actual={expected_seeds!r}"
        )
    if type(samples_per_class) is not int or samples_per_class <= 0:
        errors.append(
            f"samples_per_class must be a positive integer actual={samples_per_class!r}"
        )
    if type(epochs) is not int or epochs <= 0:
        errors.append(f"epochs must be a positive integer actual={epochs!r}")
    if readiness_only and (samples_per_class != 64 or epochs != 1):
        errors.append(
            "readiness_only requires 64/class and 1 epoch "
            f"actual={samples_per_class}/class epochs={epochs}"
        )
    expected_budget = _EXPERIMENT_BUDGETS.get(experiment_stage)
    if (
        not readiness_only
        and expected_budget is not None
        and (samples_per_class, epochs) != expected_budget
    ):
        errors.append(
            f"{experiment_stage.upper()} requires "
            f"{expected_budget[0]}/class and {expected_budget[1]} epochs "
            f"actual={samples_per_class}/class epochs={epochs}"
        )
    return errors


def _result_errors(
    rows: list[dict[str, Any]],
    *,
    expected_seed: int,
    samples_per_class: int,
    epochs: int,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != 5:
        errors.append(f"result rows={len(rows)} expected=5")
    expected_models = set(TRANSFER_MODEL_ROLES.values())
    for model in expected_models:
        count = sum(row.get("selected_model") == model for row in rows)
        if count != 1:
            errors.append(f"model={model} rows={count} expected=1")

    cache_roots: list[Any] = []
    typed_counts: set[int] = set()
    typed_trainable_counts: set[int] = set()
    by_model = {
        row.get("selected_model"): row
        for row in rows
        if row.get("selected_model") in expected_models
    }
    for index, row in enumerate(rows):
        model = row.get("selected_model")
        label = f"row={index} model={model}"
        exact_row = {
            "cipher": "GIFT-64",
            "cipher_key": "gift64",
            "structure": "SPN",
            "rounds": 6,
            "seed": expected_seed,
            "samples_per_class": samples_per_class,
            "pairs_per_sample": 4,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "difference_profile": "gift64_shen2024_spn_screen",
            "difference_member": 0,
        }
        _check_fields(row, exact_row, label, errors)
        if row.get("model") != model:
            errors.append(f"{label} model must equal selected_model")
        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label} training must be an object")
            training = {}
        runtime = {
            "batch_size": 32 if samples_per_class == 64 else 256,
            "dataset_cache_chunk_size": 64 if samples_per_class == 64 else 512,
            "dataset_cache_workers": 1 if samples_per_class == 64 else 4,
        }
        exact_training = {
            **runtime,
            "device": "cpu",
            "key_schedule": "fixed",
            "input_bits": 512,
            "pair_bits": 128,
            "pairs_per_sample": 4,
            "sample_structure": "independent_pairs",
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
            "optimizer_state_transition": "reset_each_stage",
            "optimizer_state_reused": False,
            "train_eval_interval": 1,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "lr_scheduler": "none",
            "max_learning_rate": None,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
            "epochs": epochs,
            "samples_total": 2 * samples_per_class,
            "positive_rows": samples_per_class,
            "negative_rows": samples_per_class,
            "validation_rows": samples_per_class,
            "validation_positive_rows": samples_per_class // 2,
            "validation_negative_rows": samples_per_class // 2,
        }
        _check_fields(training, exact_training, f"{label} training", errors)
        expected_options = (
            _ANCHOR_OPTIONS
            if model == TRANSFER_MODEL_ROLES["gift_anchor"]
            else _TYPED_OPTIONS
        )
        if training.get("model_options") != expected_options:
            errors.append(
                f"{label} training model_options mismatch "
                f"expected={expected_options!r} actual={training.get('model_options')!r}"
            )
        cache_roots.append(training.get("dataset_cache_root"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or any(
            not _finite(metrics.get(metric))
            for metric in ("auc", "accuracy", "calibrated_accuracy", "loss")
        ):
            errors.append(f"{label} metrics must contain finite values")
            metrics = metrics if isinstance(metrics, dict) else {}
        errors.extend(
            _checkpoint_history_errors(
                row,
                training=training,
                metrics=metrics,
                configured_epochs=epochs,
                label=label,
            )
        )
        if model != TRANSFER_MODEL_ROLES["gift_anchor"]:
            if type(row.get("parameter_count")) is int:
                typed_counts.add(row["parameter_count"])
            if type(row.get("trainable_parameter_count")) is int:
                typed_trainable_counts.add(row["trainable_parameter_count"])

    if (
        len(cache_roots) != 5
        or any(not isinstance(root, str) or not root for root in cache_roots)
        or len(set(cache_roots)) != 1
    ):
        errors.append(f"dataset_cache_root must be one shared non-empty path: {cache_roots!r}")
    if len(typed_counts) != 1 or len(typed_trainable_counts) != 1:
        errors.append(
            "typed transfer capacity mismatch "
            f"total={sorted(typed_counts)} trainable={sorted(typed_trainable_counts)}"
        )
    if expected_models.issubset(by_model):
        errors.extend(_initialization_errors(by_model))
    return errors


def _initialization_errors(by_model: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected = {
        "gift_anchor": ("scratch", "aligned", None, None),
        "gift_typed_scratch": ("scratch", "true", None, None),
        "true_to_true": ("checkpoint", "true", "true", _TRUE_SOURCE_SHA),
        "shuffled_to_true": (
            "checkpoint",
            "true",
            "shuffled",
            _SHUFFLED_SOURCE_SHA,
        ),
        "true_to_shuffled": (
            "checkpoint",
            "shuffled",
            "true",
            _TRUE_SOURCE_SHA,
        ),
    }
    initial_hashes: dict[str, Any] = {}
    for role, model in TRANSFER_MODEL_ROLES.items():
        initialization = by_model[model].get("initialization")
        label = f"role={role} initialization"
        if not isinstance(initialization, dict):
            errors.append(f"{label} must be an object")
            continue
        kind, target_mapping, source_mapping, source_sha = expected[role]
        _check_fields(
            initialization,
            {
                "kind": kind,
                "target_model": model,
                "target_mapping": target_mapping,
                "strict_state_dict_load": kind == "checkpoint",
            },
            label,
            errors,
        )
        initial_hash = initialization.get("initial_state_sha256")
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            errors.append(f"{label} initial_state_sha256 must be 64 hex characters")
        initial_hashes[role] = initial_hash
        if kind == "checkpoint":
            _check_fields(
                initialization,
                {
                    "source_cipher": "PRESENT-80",
                    "source_rounds": 7,
                    "source_seed": 0,
                    "source_samples_per_class": 8192,
                    "source_epochs": 10,
                    "source_mapping": source_mapping,
                    "source_checkpoint_sha256": source_sha,
                },
                label,
                errors,
            )
    if initial_hashes.get("true_to_true") != initial_hashes.get("true_to_shuffled"):
        errors.append("true source initial_state_sha256 must match across target mappings")
    return errors


def _progress_errors(
    progress: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    result_path: Path,
    expected_seed: int,
    samples_per_class: int,
) -> list[str]:
    errors: list[str] = []
    expected_models = set(TRANSFER_MODEL_ROLES.values())
    init_events = [row for row in progress if row.get("event") == "initialization_ready"]
    terminals = [
        row for row in progress if row.get("event") in {"cache_done", "cache_reuse"}
    ]
    for model in expected_models:
        model_init = [row for row in init_events if row.get("model") == model]
        if len(model_init) != 1:
            errors.append(
                f"progress initialization_ready model={model} count={len(model_init)} expected=1"
            )
        else:
            _check_fields(
                model_init[0],
                {"seed": expected_seed},
                f"progress initialization_ready model={model}",
                errors,
            )
        for split in ("train", "validation"):
            events = [
                row
                for row in terminals
                if row.get("model") == model and row.get("split") == split
            ]
            if len(events) != 1:
                errors.append(
                    f"progress cache model={model} split={split} count={len(events)} expected=1"
                )
                continue
            event = events[0]
            expected_rows = 2 * samples_per_class if split == "train" else samples_per_class
            _check_fields(
                event,
                {
                    "cipher_key": "gift64",
                    "rounds": 6,
                    "pairs_per_sample": 4,
                    "sample_structure": "independent_pairs",
                    "difference_profile": "gift64_shen2024_spn_screen",
                    "difference_member": 0,
                    "input_bits": 512,
                    "seed": expected_seed,
                    "samples_per_class": samples_per_class,
                    "total_rows": expected_rows,
                },
                f"progress cache model={model} split={split}",
                errors,
            )
    run_done = [row for row in progress if row.get("event") == "run_done"]
    if len(run_done) != 1 or run_done[0].get("total") != 5:
        errors.append(f"progress run_done must appear once with total=5 actual={run_done!r}")
    elif Path(str(run_done[0].get("output"))).resolve() != result_path.resolve():
        errors.append("progress run_done output must match results path")
    if len(rows) == 5:
        result_initializations = {
            row["selected_model"]: row.get("initialization") for row in rows
        }
        for event in init_events:
            model = event.get("model")
            initialization = result_initializations.get(model)
            if isinstance(initialization, dict):
                for field in (
                    "kind",
                    "source_model",
                    "source_checkpoint_sha256",
                    "strict_state_dict_load",
                    "target_model",
                    "target_mapping",
                ):
                    if event.get(field) != initialization.get(field):
                        errors.append(
                            f"progress initialization model={model} field={field} "
                            f"result={initialization.get(field)!r} event={event.get(field)!r}"
                        )
    return errors


def _decision(
    aucs: dict[str, float],
    margins: dict[str, float],
    *,
    expected_seed: int,
    experiment_stage: str,
) -> tuple[str, str]:
    if experiment_stage == "e4_r3":
        if margins["source_topology_margin"] <= 0.0:
            return (
                "e4_r3_source_topology_not_attributed",
                "stop_mechanical_scale_and_audit_source_topology",
            )
        if margins["target_topology_margin"] <= 0.0:
            return (
                "e4_r3_target_topology_not_attributed",
                "stop_mechanical_scale_and_audit_target_topology",
            )
        if margins["anchor_margin"] <= 0.0 or margins["scratch_margin"] <= 0.0:
            return (
                "e4_r3_transfer_rejected",
                "stop_mechanical_scale_keep_within_cipher_evidence",
            )
        if _passes_transfer_thresholds(aucs, margins):
            return "e4_r3_seed_signal_preserved", "run_e4_r3_joint_gate"
        return (
            "e4_r3_seed_margin_miss",
            "stop_mechanical_scale_and_audit_seed_variance",
        )
    if margins["source_topology_margin"] <= 0.0:
        return (
            "generic_pretraining_not_typed_transfer",
            "stop_e4_transfer_source_topology_not_attributed",
        )
    if margins["target_topology_margin"] <= 0.0:
        return (
            "target_topology_not_attributed",
            "stop_e4_transfer_target_topology_not_attributed",
        )
    if margins["anchor_margin"] <= 0.0 or margins["scratch_margin"] <= 0.0:
        return "reject_e4_transfer", "stop_e4_transfer_keep_within_cipher_evidence"
    if _passes_transfer_thresholds(aucs, margins):
        if expected_seed == 0:
            return "promote_e4_transfer_seed1", "freeze_identical_e4_r2_seed1_repeat"
        return "promote_e4_transfer_joint_gate", "run_frozen_e4_r2_joint_gate"
    return "weak_transfer_no_scale", "stop_e4_transfer_scale_after_seed0_margin_miss"


def _passes_transfer_thresholds(
    aucs: dict[str, float],
    margins: dict[str, float],
) -> bool:
    return (
        aucs["true_to_true"] >= 0.52
        and margins["anchor_margin"] >= 0.003
        and margins["scratch_margin"] >= 0.005
        and margins["source_topology_margin"] >= 0.003
        and margins["target_topology_margin"] >= 0.003
    )


def gate_cross_spn_typed_transfer_joint(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    samples_per_class: int = 8192,
    epochs: int = 10,
    experiment_stage: str = "e4_r2",
) -> dict[str, Any]:
    if len(results_paths) != 2 or len(progress_paths) != 2:
        return _invalid(
            [
                f"{experiment_stage.upper()} joint gate requires seed0 and seed1 "
                "result/progress paths "
                f"results={len(results_paths)} progress={len(progress_paths)}"
            ],
            experiment_stage=experiment_stage,
        )
    per_seed = {
        str(seed): gate_cross_spn_typed_transfer(
            [results_paths[seed]],
            progress_paths=[progress_paths[seed]],
            expected_seeds=(seed,),
            samples_per_class=samples_per_class,
            epochs=epochs,
            experiment_stage=experiment_stage,
        )
        for seed in (0, 1)
    }
    invalid = [
        f"seed={seed} invalid: {error}"
        for seed, report in per_seed.items()
        if report["status"] != "pass"
        for error in report["errors"]
    ]
    if invalid:
        report = _invalid(invalid, experiment_stage=experiment_stage)
        report["expected_seeds"] = [0, 1]
        report["per_seed"] = per_seed
        return report
    if experiment_stage == "e4_r3":
        if all(
            report["decision"] == "e4_r3_seed_signal_preserved"
            for report in per_seed.values()
        ):
            decision = "e4_r3_two_seed_medium_signal_confirmed"
            return {
                "status": "pass",
                "decision": decision,
                "errors": [],
                "expected_seeds": [0, 1],
                "experiment_stage": experiment_stage,
                "per_seed": per_seed,
                "research_decision_applied": True,
                "claim_scope": (
                    "two-seed E4-R3 65536/class local medium diagnostic; "
                    "not formal, paper-scale, remote, or breakthrough evidence"
                ),
                "next_action": (
                    "design_e4_r4_262144_class_diagnostic_with_remote_readiness"
                ),
                "stopped_actions": _stopped_actions(decision),
            }
        decision = "e4_r3_two_seed_medium_signal_unstable"
        return {
            "status": "pass",
            "decision": decision,
            "errors": [],
            "expected_seeds": [0, 1],
            "experiment_stage": experiment_stage,
            "per_seed": per_seed,
            "research_decision_applied": True,
            "claim_scope": (
                "two-seed E4-R3 65536/class local medium diagnostic; "
                "controls did not replicate across both seeds"
            ),
            "next_action": "stop_mechanical_scale_and_audit_seed_variance",
            "stopped_actions": _stopped_actions(decision),
        }
    if (
        per_seed["0"]["decision"] == "promote_e4_transfer_seed1"
        and per_seed["1"]["decision"] == "promote_e4_transfer_joint_gate"
    ):
        return {
            "status": "pass",
            "decision": "two_seed_transfer_signal_confirmed",
            "errors": [],
            "expected_seeds": [0, 1],
            "experiment_stage": experiment_stage,
            "per_seed": per_seed,
            "research_decision_applied": True,
            "claim_scope": (
                "two-seed E4-R2 local transfer diagnostic; not formal, "
                "paper-scale, remote, or breakthrough evidence"
            ),
            "next_action": "design_e4_r3_same_protocol_medium_diagnostic",
            "stopped_actions": _stopped_actions(
                "two_seed_transfer_signal_confirmed"
            ),
        }
    return {
        "status": "pass",
        "decision": "two_seed_transfer_unstable_no_scale",
        "errors": [],
        "expected_seeds": [0, 1],
        "experiment_stage": experiment_stage,
        "per_seed": per_seed,
        "research_decision_applied": True,
        "claim_scope": (
            "two-seed E4-R2 local transfer diagnostic; controls did not "
            "replicate across both seeds"
        ),
        "next_action": "stop_e4_transfer_scale_after_two_seed_variance",
        "stopped_actions": _stopped_actions("two_seed_transfer_unstable_no_scale"),
    }


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision in {
        "promote_e4_transfer_seed1",
        "promote_e4_transfer_joint_gate",
    }:
        actions = ("remote_scale", "sample_scale", "formal_claim")
    elif decision == "two_seed_transfer_signal_confirmed":
        actions = ("remote_scale", "formal_claim")
    elif decision == "e4_r3_two_seed_medium_signal_confirmed":
        actions = ("remote_launch", "sample_scale", "formal_claim")
    elif decision == "e4_r3_seed_signal_preserved":
        actions = ("remote_launch", "sample_scale", "formal_claim")
    else:
        actions = ("seed1", "remote_scale", "sample_scale", "formal_claim")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


def _read_jsonl(
    path: Path,
    label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"{label} path={path} read_error={exc}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label} line={line_number} invalid_json={exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{label} line={line_number} must be an object")
            continue
        rows.append(row)
    return rows, errors


def _check_fields(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    for field, expected_value in expected.items():
        if actual.get(field) != expected_value:
            errors.append(
                f"{label} {field} expected={expected_value!r} actual={actual.get(field)!r}"
            )


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _invalid(
    errors: list[str],
    *,
    experiment_stage: str = "e4_r2",
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "invalid_e4_protocol",
        "errors": errors,
        "research_decision_applied": False,
        "experiment_stage": experiment_stage,
        "claim_scope": f"invalid {experiment_stage.upper()} transfer evidence",
        "next_action": (
            "repair_e4_r3_evidence_before_interpretation"
            if experiment_stage == "e4_r3"
            else "repair_e4_r2_protocol_and_replay_same_matrix"
        ),
        "stopped_actions": _stopped_actions("invalid_e4_protocol"),
    }


__all__ = [
    "TRANSFER_MODEL_ROLES",
    "gate_cross_spn_typed_transfer",
    "gate_cross_spn_typed_transfer_joint",
]
