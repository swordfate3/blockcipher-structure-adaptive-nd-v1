from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    load_score_artifact,
)
from blockcipher_nd.planning.four_role_attribution_gate import (
    _checkpoint_history_errors,
)
from blockcipher_nd.planning.result_alignment import (
    validate_result_plan_alignment,
)
from blockcipher_nd.training.metrics import binary_auc


ADAPTATION_MODEL_ROLES = {
    "typed_scratch": "gift_cross_spn_typed_cell_true",
    "true_to_true": "gift_cross_spn_typed_cell_true_from_present_true",
    "shuffled_to_true": "gift_cross_spn_typed_cell_true_from_present_shuffled",
    "true_to_shuffled": "gift_cross_spn_typed_cell_shuffled_from_present_true",
}

_TRUE_SOURCE_SHA = "eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1"
_SHUFFLED_SOURCE_SHA = "fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22"
_TYPED_OPTIONS = {
    "mixer_depth": 2,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "attention_mean_max",
    "dropout": 0.0,
}
_MARGIN_THRESHOLDS = {
    "scratch_margin": 0.004,
    "source_topology_margin": 0.005,
    "target_topology_margin": 0.003,
}
_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 20260715


def gate_cross_spn_target_adaptation(
    *,
    plan_path: Path,
    results_path: Path,
    progress_path: Path,
    score_artifact_paths: dict[str, Path],
    expected_seed: int,
    samples_per_class: int = 65536,
    epochs: int = 1,
    readiness_only: bool = False,
    bootstrap_replicates: int = _BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
    paired_scores_output: Path | None = None,
) -> dict[str, Any]:
    argument_errors = _argument_errors(
        expected_seed=expected_seed,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        score_artifact_paths=score_artifact_paths,
    )
    if argument_errors:
        return _invalid(argument_errors)

    rows, result_errors = _read_jsonl(results_path, "results")
    progress, progress_errors = _read_jsonl(progress_path, "progress")
    alignment = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=4,
    )
    errors = [*result_errors, *progress_errors, *alignment["errors"]]
    errors.extend(
        _result_errors(
            rows,
            expected_seed=expected_seed,
            samples_per_class=samples_per_class,
            epochs=epochs,
            expected_device="cpu" if readiness_only else "cuda",
        )
    )
    errors.extend(
        _progress_errors(
            progress,
            rows=rows,
            result_path=results_path,
            expected_seed=expected_seed,
            samples_per_class=samples_per_class,
        )
    )
    artifacts, score_errors = _load_and_validate_score_artifacts(
        score_artifact_paths,
        rows=rows,
        expected_seed=expected_seed,
        samples_per_class=samples_per_class,
    )
    errors.extend(score_errors)
    if errors:
        return _invalid(errors, alignment=alignment)

    if paired_scores_output is not None:
        write_paired_score_csv_gz(
            paired_scores_output,
            artifacts,
            target_seed=expected_seed,
        )

    aucs = {
        role: binary_auc(artifact.labels, artifact.probabilities)
        for role, artifact in artifacts.items()
    }
    margins = {
        "scratch_margin": aucs["true_to_true"] - aucs["typed_scratch"],
        "source_topology_margin": (
            aucs["true_to_true"] - aucs["shuffled_to_true"]
        ),
        "target_topology_margin": (
            aucs["true_to_true"] - aucs["true_to_shuffled"]
        ),
    }
    evidence = {
        "aucs": aucs,
        "margins": margins,
        "models": ADAPTATION_MODEL_ROLES,
        "source_checkpoint_sha256": {
            "true": _TRUE_SOURCE_SHA,
            "shuffled": _SHUFFLED_SOURCE_SHA,
        },
        "target_checkpoint_sha256": {
            role: str(artifact.metadata["checkpoint_sha256"])
            for role, artifact in artifacts.items()
        },
        "score_rows": len(next(iter(artifacts.values())).labels),
        "score_pairing": "same validation cache, identical sample_ids and labels",
        "paired_scores_output": (
            str(paired_scores_output) if paired_scores_output is not None else None
        ),
        "alignment": alignment,
        "source_pretraining_cost": {
            "cipher": "PRESENT-80",
            "rounds": 7,
            "seed": 0,
            "samples_per_class": 8192,
            "epochs": 10,
            "accounting": "reported separately; excluded from target epoch budget",
        },
    }
    if readiness_only:
        return {
            "status": "pass",
            "decision": "implementation_ready",
            "errors": [],
            "expected_seed": expected_seed,
            "samples_per_class": samples_per_class,
            "epochs": epochs,
            "experiment_stage": "e4_r4",
            **evidence,
            "bootstrap": None,
            "research_decision_applied": False,
            "claim_scope": "E4-R4 readiness only; AUC values are not interpreted",
            "next_action": f"run_frozen_e4_r4_seed{expected_seed}_remote_medium",
            "stopped_actions": _stopped_actions("implementation_ready"),
        }

    bootstrap = paired_stratified_bootstrap_auc_differences(
        next(iter(artifacts.values())).labels,
        {role: artifact.probabilities for role, artifact in artifacts.items()},
        candidate_role="true_to_true",
        control_roles=("typed_scratch", "shuffled_to_true", "true_to_shuffled"),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    core_ci_lower = bootstrap["comparisons"]["typed_scratch"]["ci_lower"]
    margins_pass = all(
        margins[name] >= threshold
        for name, threshold in _MARGIN_THRESHOLDS.items()
    )
    controls_positive = all(value > 0.0 for value in margins.values())
    if margins_pass and core_ci_lower > 0.0:
        decision = "e4_r4_target_adaptation_efficiency_confirmed"
        next_action = "design_formal_multiseed_adaptation_protocol"
    elif controls_positive:
        decision = "e4_r4_target_adaptation_signal_unstable"
        next_action = "stop_transfer_branch_keep_typed_representation_result"
    else:
        decision = "e4_r4_target_adaptation_rejected"
        next_action = "stop_transfer_branch_keep_typed_representation_result"
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seed": expected_seed,
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "experiment_stage": "e4_r4",
        **evidence,
        "bootstrap": bootstrap,
        "thresholds": {
            **_MARGIN_THRESHOLDS,
            "core_scratch_ci_lower_strictly_greater_than": 0.0,
        },
        "research_decision_applied": True,
        "claim_scope": (
            "single-seed E4-R4 65536/class remote medium target-adaptation "
            "diagnostic; conditional target-training efficiency only; source "
            "pretraining cost reported separately; not formal, paper-scale, SOTA, "
            "breakthrough, or end-to-end compute evidence"
        ),
        "next_action": next_action,
        "stopped_actions": _stopped_actions(decision),
    }


def gate_cross_spn_target_adaptation_joint(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(seed_reports) != 2:
        errors.append(f"joint gate requires two reports actual={len(seed_reports)}")
    by_seed: dict[int, dict[str, Any]] = {}
    for report in seed_reports:
        seed = report.get("expected_seed")
        if type(seed) is not int or seed in by_seed:
            errors.append(f"joint gate invalid or duplicate seed={seed!r}")
            continue
        by_seed[seed] = report
        if report.get("status") != "pass" or report.get("errors") != []:
            errors.append(f"seed={seed} report is not valid pass evidence")
        if report.get("experiment_stage") != "e4_r4":
            errors.append(f"seed={seed} experiment_stage must equal e4_r4")
        if report.get("samples_per_class") != 65536 or report.get("epochs") != 1:
            errors.append(f"seed={seed} budget must equal 65536/class and 1 epoch")
        if report.get("research_decision_applied") is not True:
            errors.append(f"seed={seed} research decision was not applied")
    if set(by_seed) != {2, 3}:
        errors.append(f"joint gate requires target seeds [2, 3] actual={sorted(by_seed)}")
    if errors:
        return _invalid(errors, claim_scope="invalid E4-R4 joint evidence")

    decisions = {report["decision"] for report in seed_reports}
    if decisions == {"e4_r4_target_adaptation_efficiency_confirmed"}:
        decision = "e4_r4_two_seed_target_adaptation_efficiency_confirmed"
        next_action = "design_formal_multiseed_adaptation_protocol"
    elif "e4_r4_target_adaptation_rejected" in decisions:
        decision = "e4_r4_two_seed_target_adaptation_rejected"
        next_action = "stop_transfer_branch_keep_typed_representation_result"
    else:
        decision = "e4_r4_two_seed_target_adaptation_signal_unstable"
        next_action = "stop_transfer_branch_keep_typed_representation_result"
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": [2, 3],
        "samples_per_class": 65536,
        "epochs": 1,
        "experiment_stage": "e4_r4",
        "per_seed": {str(seed): by_seed[seed] for seed in (2, 3)},
        "research_decision_applied": True,
        "claim_scope": (
            "two-seed E4-R4 65536/class remote medium target-adaptation "
            "diagnostic; conditional target-training efficiency only; not formal, "
            "paper-scale, SOTA, breakthrough, or end-to-end compute evidence"
        ),
        "next_action": next_action,
        "stopped_actions": _stopped_actions(decision),
    }


def paired_stratified_bootstrap_auc_differences(
    labels: np.ndarray,
    scores_by_role: dict[str, np.ndarray],
    *,
    candidate_role: str,
    control_roles: tuple[str, ...],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 32,
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.float32)
    if label_array.ndim != 1 or set(np.unique(label_array)) != {0.0, 1.0}:
        raise ValueError("paired bootstrap labels must be a one-dimensional binary array")
    if candidate_role not in scores_by_role:
        raise ValueError(f"missing candidate role: {candidate_role}")
    if any(role not in scores_by_role for role in control_roles):
        raise ValueError("paired bootstrap is missing a control role")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")

    scores = {
        role: np.asarray(values, dtype=np.float64)
        for role, values in scores_by_role.items()
    }
    if any(values.shape != label_array.shape for values in scores.values()):
        raise ValueError("paired bootstrap score arrays must align with labels")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("paired bootstrap scores must be finite")

    positive_indices = np.flatnonzero(label_array == 1.0)
    negative_indices = np.flatnonzero(label_array == 0.0)
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    positive_lookup = np.full(len(label_array), -1, dtype=np.int64)
    negative_lookup = np.full(len(label_array), -1, dtype=np.int64)
    positive_lookup[positive_indices] = np.arange(positive_count)
    negative_lookup[negative_indices] = np.arange(negative_count)
    prepared = {
        role: _prepare_weighted_auc_order(
            label_array,
            values,
            positive_lookup=positive_lookup,
            negative_lookup=negative_lookup,
        )
        for role, values in scores.items()
    }
    bootstrap_differences = {
        role: np.empty(replicates, dtype=np.float64) for role in control_roles
    }
    rng = np.random.default_rng(seed)
    positive_probabilities = np.full(positive_count, 1.0 / positive_count)
    negative_probabilities = np.full(negative_count, 1.0 / negative_count)
    denominator = float(positive_count * negative_count)

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        current = stop - start
        positive_weights = rng.multinomial(
            positive_count,
            positive_probabilities,
            size=current,
        )
        negative_weights = rng.multinomial(
            negative_count,
            negative_probabilities,
            size=current,
        )
        aucs = {
            role: _weighted_auc_chunk(
                spec,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                denominator=denominator,
            )
            for role, spec in prepared.items()
        }
        for role in control_roles:
            bootstrap_differences[role][start:stop] = (
                aucs[candidate_role] - aucs[role]
            )

    alpha = (1.0 - confidence) / 2.0
    point_aucs = {
        role: binary_auc(label_array, values) for role, values in scores.items()
    }
    comparisons: dict[str, Any] = {}
    for role, differences in bootstrap_differences.items():
        comparisons[role] = {
            "candidate_role": candidate_role,
            "control_role": role,
            "candidate_auc": point_aucs[candidate_role],
            "control_auc": point_aucs[role],
            "point_difference": point_aucs[candidate_role] - point_aucs[role],
            "bootstrap_mean_difference": float(differences.mean()),
            "bootstrap_std_difference": float(differences.std(ddof=1)),
            "ci_lower": float(np.quantile(differences, alpha)),
            "ci_upper": float(np.quantile(differences, 1.0 - alpha)),
        }
    return {
        "method": "paired label-stratified fixed-size nonparametric bootstrap",
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "positive_rows": positive_count,
        "negative_rows": negative_count,
        "comparisons": comparisons,
    }


def write_paired_score_csv_gz(
    output_path: Path,
    artifacts: dict[str, EnsembleScoreArtifact],
    *,
    target_seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(artifacts.values()))
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_index",
                "label",
                "score",
                "target_seed",
                "model_role",
                "checkpoint_sha256",
            ),
        )
        writer.writeheader()
        for index, sample_id in enumerate(first.sample_ids):
            for role, artifact in artifacts.items():
                writer.writerow(
                    {
                        "sample_index": str(sample_id),
                        "label": int(first.labels[index]),
                        "score": format(float(artifact.probabilities[index]), ".10g"),
                        "target_seed": target_seed,
                        "model_role": role,
                        "checkpoint_sha256": artifact.metadata["checkpoint_sha256"],
                    }
                )


def _argument_errors(
    *,
    expected_seed: Any,
    samples_per_class: Any,
    epochs: Any,
    readiness_only: bool,
    bootstrap_replicates: Any,
    bootstrap_seed: Any,
    score_artifact_paths: dict[str, Path],
) -> list[str]:
    errors: list[str] = []
    if expected_seed not in {2, 3}:
        errors.append(f"E4-R4 expected_seed must be 2 or 3 actual={expected_seed!r}")
    expected_scale = 64 if readiness_only else 65536
    if samples_per_class != expected_scale or epochs != 1:
        errors.append(
            f"E4-R4 budget must be {expected_scale}/class and 1 epoch "
            f"actual={samples_per_class}/class epochs={epochs}"
        )
    if set(score_artifact_paths) != set(ADAPTATION_MODEL_ROLES):
        errors.append(
            "score artifact roles must exactly match "
            f"{sorted(ADAPTATION_MODEL_ROLES)} actual={sorted(score_artifact_paths)}"
        )
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        errors.append("bootstrap_replicates must be a positive integer")
    elif not readiness_only and bootstrap_replicates != _BOOTSTRAP_REPLICATES:
        errors.append(
            f"E4-R4 confirmation requires {_BOOTSTRAP_REPLICATES} bootstrap replicates"
        )
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        errors.append("bootstrap_seed must be a nonnegative integer")
    elif not readiness_only and bootstrap_seed != _BOOTSTRAP_SEED:
        errors.append(f"E4-R4 confirmation requires bootstrap seed {_BOOTSTRAP_SEED}")
    return errors


def _result_errors(
    rows: list[dict[str, Any]],
    *,
    expected_seed: int,
    samples_per_class: int,
    epochs: int,
    expected_device: str,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != 4:
        errors.append(f"result rows={len(rows)} expected=4")
    expected_models = set(ADAPTATION_MODEL_ROLES.values())
    by_model = {row.get("selected_model"): row for row in rows}
    if set(by_model) != expected_models:
        errors.append(
            f"result models must exactly match {sorted(expected_models)} "
            f"actual={sorted(str(item) for item in by_model)}"
        )
        return errors

    cache_roots: list[Any] = []
    parameter_counts: set[Any] = set()
    trainable_counts: set[Any] = set()
    for role, model in ADAPTATION_MODEL_ROLES.items():
        row = by_model[model]
        label = f"role={role} model={model}"
        _check_fields(
            row,
            {
                "cipher": "GIFT-64",
                "cipher_key": "gift64",
                "structure": "SPN",
                "model": model,
                "selected_model": model,
                "rounds": 6,
                "seed": expected_seed,
                "samples_per_class": samples_per_class,
                "pairs_per_sample": 4,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "difference_profile": "gift64_shen2024_spn_screen",
                "difference_member": 0,
            },
            label,
            errors,
        )
        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label} training must be an object")
            continue
        _check_fields(
            training,
            {
                "device": expected_device,
                "epochs": epochs,
                "batch_size": 32 if samples_per_class == 64 else 256,
                "dataset_cache_chunk_size": 64 if samples_per_class == 64 else 512,
                "dataset_cache_workers": 1 if samples_per_class == 64 else 4,
                "samples_total": 2 * samples_per_class,
                "positive_rows": samples_per_class,
                "negative_rows": samples_per_class,
                "validation_rows": samples_per_class,
                "validation_positive_rows": samples_per_class // 2,
                "validation_negative_rows": samples_per_class // 2,
                "train_dataset_storage": "disk",
                "validation_dataset_storage": "disk",
                "checkpoint_metric": "val_auc",
                "restore_best_checkpoint": True,
                "selected_checkpoint": "best",
                "model_options": _TYPED_OPTIONS,
            },
            f"{label} training",
            errors,
        )
        cache_roots.append(training.get("dataset_cache_root"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or any(
            not _finite(metrics.get(field))
            for field in ("auc", "accuracy", "calibrated_accuracy", "loss")
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
        parameter_counts.add(row.get("parameter_count"))
        trainable_counts.add(row.get("trainable_parameter_count"))
    if (
        len(cache_roots) != 4
        or any(not isinstance(root, str) or not root for root in cache_roots)
        or len(set(cache_roots)) != 1
    ):
        errors.append(f"all four rows must share one non-empty cache root: {cache_roots!r}")
    if parameter_counts != {187426} or trainable_counts != {187426}:
        errors.append(
            "typed capacity must equal 187426 for all roles "
            f"total={parameter_counts} trainable={trainable_counts}"
        )
    errors.extend(_initialization_errors(by_model))
    return errors


def _initialization_errors(by_model: dict[str, dict[str, Any]]) -> list[str]:
    expected = {
        "typed_scratch": ("scratch", "true", None, None),
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
    errors: list[str] = []
    initial_hashes: dict[str, Any] = {}
    for role, model in ADAPTATION_MODEL_ROLES.items():
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
        initial_hashes[role] = initialization.get("initial_state_sha256")
        if not isinstance(initial_hashes[role], str) or len(initial_hashes[role]) != 64:
            errors.append(f"{label} initial_state_sha256 must be 64 characters")
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
        errors.append("true source initial hash must match across target mappings")
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
    expected_models = set(ADAPTATION_MODEL_ROLES.values())
    init_events = [row for row in progress if row.get("event") == "initialization_ready"]
    terminals = [
        row for row in progress if row.get("event") in {"cache_done", "cache_reuse"}
    ]
    for model in expected_models:
        model_init = [row for row in init_events if row.get("model") == model]
        if len(model_init) != 1 or model_init[0].get("seed") != expected_seed:
            errors.append(f"progress initialization model={model} must appear once")
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
            _check_fields(
                events[0],
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
                    "total_rows": (
                        2 * samples_per_class if split == "train" else samples_per_class
                    ),
                },
                f"progress cache model={model} split={split}",
                errors,
            )
    for split in ("train", "validation"):
        split_events = [row for row in terminals if row.get("split") == split]
        if sum(row.get("event") == "cache_done" for row in split_events) != 1:
            errors.append(f"progress {split} cache must be generated exactly once")
        if sum(row.get("event") == "cache_reuse" for row in split_events) != 3:
            errors.append(f"progress {split} cache must be reused exactly three times")
    run_done = [row for row in progress if row.get("event") == "run_done"]
    if len(run_done) != 1 or run_done[0].get("total") != 4:
        errors.append("progress run_done must appear once with total=4")
    elif Path(str(run_done[0].get("output"))).resolve() != result_path.resolve():
        errors.append("progress run_done output must match results path")

    result_initializations = {
        row.get("selected_model"): row.get("initialization") for row in rows
    }
    for event in init_events:
        initialization = result_initializations.get(event.get("model"))
        if not isinstance(initialization, dict):
            continue
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
                    f"progress initialization model={event.get('model')} field={field} mismatch"
                )
    return errors


def _load_and_validate_score_artifacts(
    score_artifact_paths: dict[str, Path],
    *,
    rows: list[dict[str, Any]],
    expected_seed: int,
    samples_per_class: int,
) -> tuple[dict[str, EnsembleScoreArtifact], list[str]]:
    artifacts: dict[str, EnsembleScoreArtifact] = {}
    errors: list[str] = []
    for role, path in score_artifact_paths.items():
        try:
            artifacts[role] = load_score_artifact(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"score artifact role={role} load failed: {exc}")
    if len(artifacts) != len(ADAPTATION_MODEL_ROLES):
        return artifacts, errors

    first = next(iter(artifacts.values()))
    by_model = {row.get("selected_model"): row for row in rows}
    expected_rows = samples_per_class
    if len(first.labels) != expected_rows:
        errors.append(f"score rows={len(first.labels)} expected={expected_rows}")
    if int((first.labels == 1.0).sum()) != samples_per_class // 2:
        errors.append("score labels must contain the expected positive rows")
    if int((first.labels == 0.0).sum()) != samples_per_class // 2:
        errors.append("score labels must contain the expected negative rows")
    for role, artifact in artifacts.items():
        model = ADAPTATION_MODEL_ROLES[role]
        if not np.array_equal(first.labels, artifact.labels):
            errors.append(f"score labels differ for role={role}")
        if not np.array_equal(first.sample_ids, artifact.sample_ids):
            errors.append(f"score sample_ids differ for role={role}")
        if artifact.probabilities.shape != artifact.labels.shape:
            errors.append(f"score probability shape differs for role={role}")
        if not np.isfinite(artifact.probabilities).all():
            errors.append(f"score probabilities are not finite for role={role}")
        metadata = artifact.metadata
        _check_fields(
            metadata,
            {
                "cipher": "GIFT-64",
                "cipher_key": "gift64",
                "rounds": 6,
                "seed": expected_seed,
                "samples_per_class": samples_per_class,
                "pairs_per_sample": 4,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "difference_profile": "gift64_shen2024_spn_screen",
                "difference_member": 0,
                "model_key": model,
                "score_split": "validation",
                "score_samples_per_class": samples_per_class // 2,
                "validation_samples_per_class": samples_per_class // 2,
                "dataset_cache_enabled": True,
            },
            f"score metadata role={role}",
            errors,
        )
        checkpoint_sha = metadata.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            errors.append(f"score role={role} checkpoint_sha256 must be 64 characters")
        row = by_model.get(model)
        if row is not None:
            training = row.get("training", {})
            _check_fields(
                metadata,
                {
                    "dataset_cache_root": training.get("dataset_cache_root"),
                    "train_key": row.get("train_key"),
                    "validation_key": row.get("validation_key"),
                    "model_options": training.get("model_options"),
                },
                f"score metadata role={role}",
                errors,
            )
            checkpoint_output = row.get("training", {}).get("checkpoint_output")
            if str(metadata.get("checkpoint_path")) != str(checkpoint_output):
                errors.append(f"score role={role} checkpoint path does not match result row")
            checkpoint_metadata = metadata.get("checkpoint_metadata")
            if not isinstance(checkpoint_metadata, dict):
                errors.append(f"score role={role} checkpoint_metadata must be an object")
            else:
                _check_fields(
                    checkpoint_metadata,
                    {
                        "checkpoint_output": checkpoint_output,
                        "seed": expected_seed,
                        "epochs": 1,
                        "selected_checkpoint": "best",
                        "restore_best_checkpoint": True,
                    },
                    f"score checkpoint metadata role={role}",
                    errors,
                )
            score_auc = binary_auc(artifact.labels, artifact.probabilities)
            result_auc = row.get("metrics", {}).get("auc")
            if not _finite(result_auc) or abs(score_auc - float(result_auc)) > 1e-6:
                errors.append(
                    f"score role={role} AUC does not match restored-best result "
                    f"score={score_auc} result={result_auc!r}"
                )
    return artifacts, errors


def _prepare_weighted_auc_order(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    positive_lookup: np.ndarray,
    negative_lookup: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]).astype(np.int64)
            + 1,
        )
    )
    positive_columns = np.flatnonzero(labels[order] == 1.0)
    negative_columns = np.flatnonzero(labels[order] == 0.0)
    return {
        "starts": starts,
        "positive_columns": positive_columns,
        "negative_columns": negative_columns,
        "positive_sources": positive_lookup[order[positive_columns]],
        "negative_sources": negative_lookup[order[negative_columns]],
        "rows": np.array([len(labels)], dtype=np.int64),
    }


def _weighted_auc_chunk(
    spec: dict[str, np.ndarray],
    *,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
    denominator: float,
) -> np.ndarray:
    current = positive_weights.shape[0]
    rows = int(spec["rows"][0])
    sorted_positive = np.zeros((current, rows), dtype=np.float64)
    sorted_negative = np.zeros((current, rows), dtype=np.float64)
    sorted_positive[:, spec["positive_columns"]] = positive_weights[
        :, spec["positive_sources"]
    ]
    sorted_negative[:, spec["negative_columns"]] = negative_weights[
        :, spec["negative_sources"]
    ]
    group_positive = np.add.reduceat(sorted_positive, spec["starts"], axis=1)
    group_negative = np.add.reduceat(sorted_negative, spec["starts"], axis=1)
    negative_before = np.cumsum(group_negative, axis=1) - group_negative
    return np.sum(
        group_positive * (negative_before + 0.5 * group_negative),
        axis=1,
    ) / denominator


def _check_fields(
    mapping: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    for field, expected_value in expected.items():
        actual = mapping.get(field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            errors.append(
                f"{label} field={field} expected={expected_value!r} actual={actual!r}"
            )


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{label} read failed: {exc}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label} line={line_number} invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{label} line={line_number} must be an object")
            continue
        rows.append(payload)
    return rows, errors


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _invalid(
    errors: list[str],
    *,
    alignment: dict[str, Any] | None = None,
    claim_scope: str = "invalid E4-R4 target-adaptation evidence",
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "e4_r4_target_adaptation_invalid",
        "errors": errors,
        "experiment_stage": "e4_r4",
        "alignment": alignment,
        "research_decision_applied": False,
        "claim_scope": claim_scope,
        "next_action": "repair_e4_r4_evidence_before_interpretation",
        "stopped_actions": _stopped_actions("e4_r4_target_adaptation_invalid"),
    }


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in ("sample_scale", "formal_claim", "sota_claim", "breakthrough_claim")
    ]
