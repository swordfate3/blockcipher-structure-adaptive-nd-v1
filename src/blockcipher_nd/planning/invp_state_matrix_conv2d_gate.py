from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_state_matrix_conv2d_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_state_matrix_conv2d_spn_only",
    "delta_only": "present_nibble_delta_state_matrix_conv2d_spn_only",
}


def gate_invp_state_matrix_conv2d(
    results_paths: list[Path],
    *,
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    seed0_topology_margin: float = 0.003,
    seed0_representation_margin: float = 0.003,
    joint_architecture_margin: float = 0.001,
    joint_control_margin: float = 0.002,
) -> dict[str, Any]:
    rows, load_errors = _load_rows(results_paths)
    errors = [
        *load_errors,
        *_protocol_errors(
            rows,
            expected_seeds=expected_seeds,
            samples_per_class=samples_per_class,
            epochs=epochs,
        ),
    ]
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_protocol",
            "errors": errors,
            "next_action": "repair_protocol_and_rerun_same_matrix",
            "claim_scope": "invalid strict-protocol architecture evidence",
        }

    by_seed = _rows_by_seed_and_role(rows)
    seed_reports = {
        str(seed): _seed_report(by_seed[seed])
        for seed in expected_seeds
    }
    decision, next_action = _decision(
        seed_reports,
        expected_seeds=expected_seeds,
        seed0_topology_margin=seed0_topology_margin,
        seed0_representation_margin=seed0_representation_margin,
        joint_architecture_margin=joint_architecture_margin,
        joint_control_margin=joint_control_margin,
    )
    first_seed_rows = by_seed[expected_seeds[0]]
    counts = {
        role: int(first_seed_rows[role]["parameter_count"])
        for role in MODEL_ROLES
    }
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "models": MODEL_ROLES,
        "seeds": seed_reports,
        "parameter_counts": {
            **counts,
            "candidate_to_anchor_ratio": counts["candidate"] / counts["anchor"],
        },
        "next_action": next_action,
        "claim_scope": (
            f"{samples_per_class}/class strict PRESENT r7 architecture-attribution diagnostic; "
            "not formal, paper-scale, or breakthrough evidence"
        ),
    }


def _load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"results_path={path} read_error={exc}")
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"results_path={path} line={line_number} invalid_json={exc.msg}")
                continue
            if not isinstance(row, dict):
                errors.append(f"results_path={path} line={line_number} row_not_object")
                continue
            rows.append(row)
    return rows, errors


def _protocol_errors(
    rows: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    epochs: int,
) -> list[str]:
    errors: list[str] = []
    if len(expected_seeds) not in {1, 2} or len(set(expected_seeds)) != len(expected_seeds):
        errors.append(f"expected_seeds={expected_seeds} must_contain_one_or_two_unique_seeds")

    expected_models = set(MODEL_ROLES.values())
    model_to_role = {model: role for role, model in MODEL_ROLES.items()}
    expected_seed_set = set(expected_seeds)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        seed = row.get("seed")
        model = row.get("selected_model")
        seed_is_expected = (
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in expected_seed_set
        )
        if not seed_is_expected:
            errors.append(f"row={index} unexpected_seed={seed}")
        if model not in expected_models:
            errors.append(f"row={index} unexpected_selected_model={model}")
        if seed_is_expected and model in expected_models:
            grouped.setdefault((int(seed), model_to_role[str(model)]), []).append(row)

    for seed in expected_seeds:
        for role in MODEL_ROLES:
            role_rows = grouped.get((seed, role), [])
            if len(role_rows) != 1:
                errors.append(f"seed={seed} role={role} rows={len(role_rows)} expected_rows=1")

    exact_row_fields = {
        "rounds": 7,
        "samples_per_class": samples_per_class,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11" * 10, 16),
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
    }
    exact_training_fields = {
        "input_bits": 2048,
        "pair_bits": 128,
        "train_rows": 2 * samples_per_class,
        "validation_rows": samples_per_class,
        "epochs": epochs,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "lr_scheduler": "official_cyclic",
        "max_learning_rate": 0.002,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.0001,
    }
    cache_roots: dict[int, list[Any]] = {seed: [] for seed in expected_seeds}
    conv_counts: dict[str, set[int]] = {
        "parameter_count": set(),
        "trainable_parameter_count": set(),
    }

    for index, row in enumerate(rows):
        model = row.get("selected_model")
        seed = row.get("seed")
        label = f"row={index} seed={seed} model={model}"
        for field, expected in exact_row_fields.items():
            _check_exact(row, field, expected, label, errors)

        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label} training expected_mapping actual={training!r}")
            training = {}
        for field, expected in exact_training_fields.items():
            _check_exact(training, field, expected, f"{label} training", errors)

        validation = row.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"{label} validation expected_mapping actual={validation!r}")
            validation = {}
        _check_exact(
            validation,
            "samples_per_class",
            samples_per_class // 2,
            f"{label} validation",
            errors,
        )

        if seed in expected_seed_set:
            cache_roots[int(seed)].append(training.get("dataset_cache_root"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{label} metrics expected_mapping actual={metrics!r}")
        else:
            for metric in ("auc", "accuracy", "calibrated_accuracy", "loss"):
                if not _is_finite_number(metrics.get(metric)):
                    errors.append(f"{label} metrics.{metric} must_be_finite actual={metrics.get(metric)!r}")

        for field in ("parameter_count", "trainable_parameter_count"):
            value = row.get(field)
            if not _is_positive_integer(value):
                errors.append(f"{label} {field} must_be_positive_integer actual={value!r}")
            elif model in expected_models - {MODEL_ROLES["anchor"]}:
                conv_counts[field].add(value)

    for seed, roots in cache_roots.items():
        if len(roots) != len(MODEL_ROLES) or any(
            not isinstance(root, str) or not root.strip() for root in roots
        ) or len(set(roots)) != 1:
            errors.append(f"seed={seed} dataset_cache_root must_be_identical_and_non_empty actual={roots!r}")

    if len(conv_counts["parameter_count"]) != 1:
        errors.append(
            "conv2d_parameter_count_mismatch "
            f"values={sorted(conv_counts['parameter_count'])}"
        )
    if len(conv_counts["trainable_parameter_count"]) != 1:
        errors.append(
            "conv2d_trainable_parameter_count_mismatch "
            f"values={sorted(conv_counts['trainable_parameter_count'])}"
        )
    return errors


def _check_exact(
    mapping: dict[str, Any],
    field: str,
    expected: Any,
    label: str,
    errors: list[str],
) -> None:
    actual = mapping.get(field)
    if type(actual) is not type(expected) or actual != expected:
        errors.append(f"{label} {field} expected={expected!r} actual={actual!r}")


def _is_finite_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    return math.isfinite(value)


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _rows_by_seed_and_role(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    model_to_role = {model: role for role, model in MODEL_ROLES.items()}
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["seed"])
        role = model_to_role[str(row["selected_model"])]
        by_seed.setdefault(seed, {})[role] = row
    return by_seed


def _seed_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aucs = {
        role: float(row["metrics"]["auc"])
        for role, row in rows.items()
    }
    return {
        "aucs": aucs,
        "architecture_margin": aucs["candidate"] - aucs["anchor"],
        "topology_margin": aucs["candidate"] - aucs["shuffled_p"],
        "representation_margin": aucs["candidate"] - aucs["delta_only"],
        "candidate_above_all": aucs["candidate"] > max(
            aucs["anchor"], aucs["shuffled_p"], aucs["delta_only"]
        ),
    }


def _decision(
    seed_reports: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    seed0_topology_margin: float,
    seed0_representation_margin: float,
    joint_architecture_margin: float,
    joint_control_margin: float,
) -> tuple[str, str]:
    reports = [seed_reports[str(seed)] for seed in expected_seeds]
    if any(report["architecture_margin"] <= 0.0 for report in reports):
        return "stop_conv2d_route", "keep_token_mixer_anchor_and_do_not_scale_conv2d"
    if any(report["topology_margin"] <= 0.0 for report in reports):
        return "stop_generic_locality", "do_not_scale_generic_locality_without_true_p_topology"
    if any(report["representation_margin"] <= 0.0 for report in reports):
        return "stop_invp_attribution", "do_not_scale_without_invp_representation_attribution"

    if len(reports) == 1:
        report = reports[0]
        if (
            report["topology_margin"] >= seed0_topology_margin
            and report["representation_margin"] >= seed0_representation_margin
        ):
            return "promote_seed1", "run_identical_seed1_local_gate"
        return "weak_or_fragile_no_scale", "do_not_scale_run_bounded_local_variance_check"

    mean_architecture_margin = sum(
        report["architecture_margin"] for report in reports
    ) / len(reports)
    minimum_topology_margin = min(report["topology_margin"] for report in reports)
    minimum_representation_margin = min(
        report["representation_margin"] for report in reports
    )
    if (
        all(report["candidate_above_all"] for report in reports)
        and mean_architecture_margin >= joint_architecture_margin
        and minimum_topology_margin >= joint_control_margin
        and minimum_representation_margin >= joint_control_margin
    ):
        return "promote_medium_65536", "run_65536_per_class_two_seed_medium_confirmation"
    return "unstable_no_remote_scale", "do_not_launch_remote_scale_inspect_two_seed_variance"
