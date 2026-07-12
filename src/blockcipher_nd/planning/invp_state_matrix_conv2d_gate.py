from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from blockcipher_nd.training.trainer import is_checkpoint_improved


MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_state_matrix_conv2d_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_state_matrix_conv2d_spn_only",
    "delta_only": "present_nibble_delta_state_matrix_conv2d_spn_only",
}


@dataclass(frozen=True)
class FourRoleGateSpec:
    model_roles: Mapping[str, str]
    anchor_options: Mapping[str, Any]
    hybrid_options: Mapping[str, Any]
    capacity_label: str
    semantic_checks: Mapping[str, Any]
    readiness_next_action: str
    claim_label: str
    decide: Callable[..., tuple[str, str]]
    stopped_actions: Callable[[str], list[dict[str, str]]]


def gate_invp_state_matrix_conv2d(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    readiness_only: bool = False,
    seed0_topology_margin: float = 0.003,
    seed0_representation_margin: float = 0.003,
    joint_architecture_margin: float = 0.001,
    joint_control_margin: float = 0.002,
) -> dict[str, Any]:
    return _gate_four_role_attribution(
        results_paths,
        progress_paths=progress_paths,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        seed0_architecture_margin=None,
        seed0_topology_margin=seed0_topology_margin,
        seed0_representation_margin=seed0_representation_margin,
        joint_architecture_margin=joint_architecture_margin,
        joint_control_margin=joint_control_margin,
        spec=_CONV2D_GATE_SPEC,
    )


def _gate_four_role_attribution(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    epochs: int,
    readiness_only: bool,
    seed0_architecture_margin: float | None,
    seed0_topology_margin: float,
    seed0_representation_margin: float,
    joint_architecture_margin: float,
    joint_control_margin: float,
    spec: FourRoleGateSpec,
) -> dict[str, Any]:
    expected_seed_errors = _expected_seed_errors(expected_seeds)
    if expected_seed_errors:
        return _invalid_protocol(expected_seed_errors)

    result_runs, load_errors = _load_result_runs(results_paths)
    rows = [row for _, run_rows in result_runs for row in run_rows]
    progress_runs, progress_load_errors = _load_progress_runs(progress_paths)
    readiness_errors = []
    if readiness_only and (
        expected_seeds != (0,) or samples_per_class != 64 or epochs != 1
    ):
        readiness_errors.append(
            "readiness_only requires frozen R0 identity "
            f"expected_seeds=(0,) samples_per_class=64 epochs=1 "
            f"actual_expected_seeds={expected_seeds!r} "
            f"actual_samples_per_class={samples_per_class!r} actual_epochs={epochs!r}"
        )
    errors = [
        *load_errors,
        *progress_load_errors,
        *readiness_errors,
        *_protocol_errors(
            rows,
            expected_seeds=expected_seeds,
            samples_per_class=samples_per_class,
            epochs=epochs,
            spec=spec,
        ),
    ]
    cache_evidence, cache_errors = _cache_evidence(
        result_runs,
        progress_runs,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        spec=spec,
    )
    errors.extend(cache_errors)
    if errors:
        return _invalid_protocol(errors)

    by_seed = _rows_by_seed_and_role(rows, spec=spec)
    seed_reports = {str(seed): _seed_report(by_seed[seed]) for seed in expected_seeds}
    first_seed_rows = by_seed[expected_seeds[0]]
    counts = {
        role: int(first_seed_rows[role]["parameter_count"]) for role in spec.model_roles
    }
    parameter_counts = {
        **counts,
        "candidate_to_anchor_ratio": counts["candidate"] / counts["anchor"],
    }
    common_evidence = {
        "protocol_evidence": _protocol_evidence(by_seed, expected_seeds, spec=spec),
        "semantic_checks": dict(spec.semantic_checks),
        "cache_evidence": cache_evidence,
        "promotion_conditions": _promotion_conditions(
            expected_seeds=expected_seeds,
            samples_per_class=samples_per_class,
            seed0_architecture_margin=seed0_architecture_margin,
            seed0_topology_margin=seed0_topology_margin,
            seed0_representation_margin=seed0_representation_margin,
            joint_architecture_margin=joint_architecture_margin,
            joint_control_margin=joint_control_margin,
        ),
    }
    if readiness_only:
        return {
            "status": "pass",
            "decision": "implementation_ready",
            "errors": [],
            "expected_seeds": list(expected_seeds),
            "samples_per_class": samples_per_class,
            "models": dict(spec.model_roles),
            "seeds": seed_reports,
            "parameter_counts": parameter_counts,
            **common_evidence,
            "stopped_actions": spec.stopped_actions("implementation_ready"),
            "next_action": spec.readiness_next_action,
            "claim_scope": "implementation readiness only; metrics not interpreted",
            "research_decision_applied": False,
        }

    decision, next_action = spec.decide(
        seed_reports,
        expected_seeds=expected_seeds,
        seed0_architecture_margin=seed0_architecture_margin,
        seed0_topology_margin=seed0_topology_margin,
        seed0_representation_margin=seed0_representation_margin,
        joint_architecture_margin=joint_architecture_margin,
        joint_control_margin=joint_control_margin,
    )
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "models": dict(spec.model_roles),
        "seeds": seed_reports,
        "parameter_counts": parameter_counts,
        **common_evidence,
        "stopped_actions": spec.stopped_actions(decision),
        "next_action": next_action,
        "claim_scope": (
            f"{samples_per_class}/class strict PRESENT r7 {spec.claim_label}; "
            "not formal, paper-scale, or breakthrough evidence"
        ),
        "research_decision_applied": True,
    }


def _expected_seed_errors(expected_seeds: Any) -> list[str]:
    if type(expected_seeds) is not tuple:
        return [f"expected_seeds must_be_tuple actual={expected_seeds!r}"]
    if not expected_seeds:
        return ["expected_seeds must_be_nonempty_tuple actual=()"]

    errors = [
        f"expected_seeds index={index} must_be_exact_int actual={value!r}"
        for index, value in enumerate(expected_seeds)
        if type(value) is not int
    ]
    if errors:
        return errors
    if len(set(expected_seeds)) != len(expected_seeds):
        return [f"expected_seeds must_be_unique actual={expected_seeds!r}"]
    return []


def _invalid_protocol(errors: list[str]) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "invalid_protocol",
        "errors": errors,
        "next_action": "repair_protocol_and_rerun_same_matrix",
        "claim_scope": "invalid strict-protocol architecture evidence",
        "research_decision_applied": False,
    }


def _load_result_runs(
    paths: list[Path],
) -> tuple[list[tuple[Path, list[dict[str, Any]]]], list[str]]:
    runs: list[tuple[Path, list[dict[str, Any]]]] = []
    errors: list[str] = []
    for path in paths:
        rows, path_errors = _load_jsonl_rows([path], path_label="results_path")
        runs.append((path, rows))
        errors.extend(path_errors)
    return runs, errors


def _load_jsonl_rows(
    paths: list[Path],
    *,
    path_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{path_label}={path} read_error={exc}")
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{path_label}={path} line={line_number} invalid_json={exc.msg}"
                )
                continue
            if not isinstance(row, dict):
                errors.append(f"{path_label}={path} line={line_number} row_not_object")
                continue
            rows.append(row)
    return rows, errors


def _load_progress_runs(
    paths: list[Path],
) -> tuple[list[tuple[Path, list[dict[str, Any]]]], list[str]]:
    runs: list[tuple[Path, list[dict[str, Any]]]] = []
    errors: list[str] = []
    for path in paths:
        rows, path_errors = _load_jsonl_rows([path], path_label="progress_path")
        runs.append((path, rows))
        errors.extend(path_errors)
    return runs, errors


def _cache_evidence(
    result_runs: list[tuple[Path, list[dict[str, Any]]]],
    progress_runs: list[tuple[Path, list[dict[str, Any]]]],
    *,
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    spec: FourRoleGateSpec,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    if len(result_runs) != len(progress_runs) or not result_runs:
        return evidence, [
            "result_progress_path_count must_be_equal_and_nonzero "
            f"results={len(result_runs)} progress={len(progress_runs)}"
        ]

    expected_seed_set = set(expected_seeds)
    expected_models = set(spec.model_roles.values())
    model_to_role = {model: role for role, model in spec.model_roles.items()}
    for pair_index, ((result_path, result_rows), (progress_path, rows)) in enumerate(
        zip(result_runs, progress_runs, strict=True)
    ):
        pair_label = f"pair={pair_index} results_path={result_path} progress_path={progress_path}"
        result_seed_values = [row.get("seed") for row in result_rows]
        result_seeds = {seed for seed in result_seed_values if type(seed) is int}
        result_models = [row.get("selected_model") for row in result_rows]
        exact_string_models = [model for model in result_models if type(model) is str]
        result_seed_valid = (
            len(result_rows) == len(spec.model_roles)
            and all(type(seed) is int for seed in result_seed_values)
            and len(result_seeds) == 1
            and result_seeds <= expected_seed_set
            and len(result_models) == len(spec.model_roles)
            and len(exact_string_models) == len(spec.model_roles)
            and set(exact_string_models) == expected_models
            and all(result_models.count(model) == 1 for model in expected_models)
        )
        if not result_seed_valid:
            errors.append(
                f"{pair_label} expected_exact_four_model_rows_with_one_seed "
                f"rows={len(result_rows)} seeds={sorted(result_seeds)} models={result_models!r}"
            )
            continue
        seed = next(iter(result_seeds))

        terminal = [
            row for row in rows if row.get("event") in {"cache_done", "cache_reuse"}
        ]
        seeds = {
            row["seed"]
            for row in terminal
            if _is_expected_seed(row.get("seed"), expected_seed_set)
        }
        if seeds != {seed}:
            errors.append(
                f"{pair_label} paired_terminal_seed expected={seed} actual={sorted(seeds)}"
            )

        run_done = [row for row in rows if row.get("event") == "run_done"]
        if len(run_done) != 1:
            errors.append(f"seed={seed} run_done count={len(run_done)} expected=1")

        normalized_result_path = _normalized_path(result_path)
        normalized_cache_root: Path | None = None
        cache_roots = [
            row.get("training", {}).get("dataset_cache_root")
            if isinstance(row.get("training"), dict)
            else None
            for row in result_rows
        ]
        if (
            len(cache_roots) != len(spec.model_roles)
            or any(
                not isinstance(root, str) or not root.strip() for root in cache_roots
            )
            or len(set(cache_roots)) != 1
        ):
            errors.append(
                f"{pair_label} dataset_cache_root must_be_identical_and_non_empty "
                f"actual={cache_roots!r}"
            )
        else:
            normalized_cache_root = _normalized_path(Path(cache_roots[0]))

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        events: list[dict[str, Any]] = []
        terminal = [
            row for row in rows if row.get("event") in {"cache_done", "cache_reuse"}
        ]
        exact_fields = {
            "cipher_key": "present80",
            "rounds": 7,
            "dataset_label_mode": "balanced_per_class",
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "pairs_per_sample": 16,
            "key_rotation_interval": 0,
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "difference_profile": "present_zhang_wang2022_mcnd",
            "difference_member": 0,
            "input_bits": 2048,
            "optimizer_state_transition": "reset_each_stage",
            "loss": "mse",
            "samples_per_class": samples_per_class,
        }
        for index, row in enumerate(terminal):
            label = f"seed={seed} cache_evidence[{index}]"
            model = row.get("model")
            split = row.get("split")
            model_valid = _is_expected_model(model, expected_models)
            seed_valid = type(row.get("seed")) is int and row.get("seed") == seed
            split_valid = type(split) is str and split in {"train", "validation"}
            if not model_valid:
                errors.append(f"{label} model unexpected={model!r}")
            if not seed_valid:
                errors.append(
                    f"{label} seed expected={seed} actual={row.get('seed')!r}"
                )
            if not split_valid:
                errors.append(f"{label} split unexpected={split!r}")
            cache_path = row.get("cache_path")
            if not isinstance(cache_path, str) or not cache_path.strip():
                errors.append(
                    f"{label} cache_path must_be_nonempty actual={cache_path!r}"
                )
                normalized_cache_path = None
            else:
                normalized_cache_path = _normalized_path(Path(cache_path))
                if normalized_cache_root is not None and not _is_under(
                    normalized_cache_path, normalized_cache_root
                ):
                    errors.append(
                        f"{label} cache_path outside_cache_root "
                        f"path={normalized_cache_path} cache_root={normalized_cache_root}"
                    )
            for field, expected in exact_fields.items():
                _check_exact(row, field, expected, label, errors)
            expected_rows = (
                2 * samples_per_class if split == "train" else samples_per_class
            )
            _check_exact(row, "total_rows", expected_rows, label, errors)
            if model_valid and split_valid and seed_valid:
                role = model_to_role[model]
                grouped.setdefault((role, split), []).append(row)
                events.append(
                    {
                        "role": role,
                        "model": model,
                        "split": split,
                        "event": row.get("event"),
                        "cache_path": str(normalized_cache_path)
                        if normalized_cache_path is not None
                        else None,
                        "original_cache_path": cache_path,
                    }
                )

        for role in spec.model_roles:
            for split in ("train", "validation"):
                role_events = grouped.get((role, split), [])
                if len(role_events) != 1:
                    errors.append(
                        f"seed={seed} cache_evidence role={role} split={split} "
                        f"terminal_count={len(role_events)} expected=1"
                    )
                    continue
                event = role_events[0]["event"]
                if role != "anchor" and event != "cache_reuse":
                    errors.append(
                        f"seed={seed} cache_evidence role={role} split={split} "
                        f"event={event!r} expected='cache_reuse'"
                    )

        split_paths = {
            split: {
                str(_normalized_path(Path(row["cache_path"])))
                for role in spec.model_roles
                for row in grouped.get((role, split), [])
                if isinstance(row.get("cache_path"), str)
                and row.get("cache_path", "").strip()
            }
            for split in ("train", "validation")
        }
        for split, paths in split_paths.items():
            if len(paths) != 1:
                errors.append(
                    f"seed={seed} cache_evidence split={split} unique_paths={sorted(paths)} expected=1"
                )
        control_reuse_count = sum(
            1
            for event in events
            if event["role"] != "anchor" and event["event"] == "cache_reuse"
        )
        create_count = sum(event["event"] == "cache_done" for event in events)
        reuse_count = sum(event["event"] == "cache_reuse" for event in events)
        if create_count != 2:
            errors.append(
                f"seed={seed} cache_evidence create_count={create_count} expected=2"
            )
        if reuse_count != 6:
            errors.append(
                f"seed={seed} cache_evidence reuse_count={reuse_count} expected=6"
            )
        if control_reuse_count != 6:
            errors.append(
                f"seed={seed} cache_evidence control_reuse_count={control_reuse_count} expected=6"
            )
        train_path = next(iter(split_paths["train"]), None)
        validation_path = next(iter(split_paths["validation"]), None)
        if train_path is not None and train_path == validation_path:
            errors.append(
                f"seed={seed} cache_evidence train_and_validation_paths_must_differ "
                f"actual={train_path!r}"
            )
        result_output = run_done[0].get("output") if len(run_done) == 1 else None
        if not isinstance(result_output, str) or not result_output.strip():
            errors.append(
                f"seed={seed} run_done output must_be_nonempty actual={result_output!r}"
            )
        elif _normalized_path(Path(result_output)) != normalized_result_path:
            errors.append(
                f"seed={seed} run_done output must_match_paired_result "
                f"expected={normalized_result_path} "
                f"actual={_normalized_path(Path(result_output))}"
            )
        evidence[str(seed)] = {
            "progress_path": str(progress_path),
            "normalized_progress_path": str(_normalized_path(progress_path)),
            "result_path": str(normalized_result_path),
            "result_root": str(normalized_result_path.parent),
            "cache_root": str(normalized_cache_root)
            if normalized_cache_root is not None
            else None,
            "train_path": train_path,
            "validation_path": validation_path,
            "create_count": create_count,
            "reuse_count": reuse_count,
            "control_reuse_count": control_reuse_count,
            "events": events,
            "run_done_count": len(run_done),
            "verified": True,
        }
    observed_seeds = {
        int(seed) for seed in evidence if seed.isdigit() or seed.startswith("-")
    }
    for seed in expected_seeds:
        if seed not in observed_seeds:
            errors.append(f"seed={seed} cache_evidence paired_runs=0 expected=1")
    if errors:
        for item in evidence.values():
            item["verified"] = False
    return evidence, errors


def _normalized_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _protocol_errors(
    rows: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    epochs: int,
    spec: FourRoleGateSpec,
) -> list[str]:
    errors: list[str] = []
    if len(expected_seeds) not in {1, 2} or len(set(expected_seeds)) != len(
        expected_seeds
    ):
        errors.append(
            f"expected_seeds={expected_seeds} must_contain_one_or_two_unique_seeds"
        )

    expected_models = set(spec.model_roles.values())
    model_to_role = {model: role for role, model in spec.model_roles.items()}
    expected_seed_set = set(expected_seeds)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        seed = row.get("seed")
        model = row.get("selected_model")
        seed_is_expected = _is_expected_seed(seed, expected_seed_set)
        model_is_expected = _is_expected_model(model, expected_models)
        if not seed_is_expected:
            errors.append(f"row={index} unexpected_seed={seed}")
        if not model_is_expected:
            errors.append(f"row={index} unexpected_selected_model={model}")
        if seed_is_expected and model_is_expected:
            grouped.setdefault((int(seed), model_to_role[str(model)]), []).append(row)

    for seed in expected_seeds:
        for role in spec.model_roles:
            role_rows = grouped.get((seed, role), [])
            if len(role_rows) != 1:
                errors.append(
                    f"seed={seed} role={role} rows={len(role_rows)} expected_rows=1"
                )

    exact_row_fields = {
        "cipher": "PRESENT-80",
        "cipher_key": "present80",
        "structure": "SPN",
        "rounds": 7,
        "dataset_label_mode": "balanced_per_class",
        "input_difference": 9,
        "samples_per_class": samples_per_class,
        "train_samples_total": None,
        "validation_samples_total": None,
        "final_test_samples_total": None,
        "final_test_repeats": 0,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11" * 10, 16),
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
        "validation_integral_active_nibbles": [],
    }
    is_r0 = samples_per_class == 64 and epochs == 1
    exact_training_fields = {
        "amsgrad": False,
        "batch_size": 32 if is_r0 else 256,
        "dataset_cache_chunk_size": 64 if is_r0 else 512,
        "dataset_cache_workers": 1 if is_r0 else 4,
        "dataset_label_mode": "balanced_per_class",
        "device": "cpu",
        "feature_encoding": "ciphertext_pair_bits",
        "key_schedule": "per_pair_random",
        "input_bits": 2048,
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
        "validation_integral_active_nibbles": [],
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "selected_bit_indices": [],
        "samples_total": 2 * samples_per_class,
        "positive_rows": samples_per_class,
        "negative_rows": samples_per_class,
        "train_rows": 2 * samples_per_class,
        "train_positive_rows": samples_per_class,
        "train_negative_rows": samples_per_class,
        "validation_rows": samples_per_class,
        "validation_positive_rows": samples_per_class // 2,
        "validation_negative_rows": samples_per_class // 2,
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
        "optimizer_state_transition": "reset_each_stage",
        "optimizer_state_reused": False,
        "optimizer_state_step_before": 0,
        "optimizer_session_call": 1,
        "train_eval_interval": 1,
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
    exact_validation_fields = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": 7,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "pairs_per_sample": 16,
        "samples_per_class": samples_per_class // 2,
        "samples_total": samples_per_class,
        "positive_rows": samples_per_class // 2,
        "negative_rows": samples_per_class // 2,
        "dataset_label_mode": "balanced_per_class",
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "key_schedule": "per_pair_random",
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
    }
    cache_roots: dict[int, list[Any]] = {seed: [] for seed in expected_seeds}
    conv_counts: dict[str, set[int]] = {
        "parameter_count": set(),
        "trainable_parameter_count": set(),
    }

    for index, row in enumerate(rows):
        model = row.get("selected_model")
        seed = row.get("seed")
        seed_is_expected = _is_expected_seed(seed, expected_seed_set)
        model_is_expected = _is_expected_model(model, expected_models)
        label = f"row={index} seed={seed} model={model}"
        for field, expected in exact_row_fields.items():
            _check_exact(row, field, expected, label, errors)
        _check_exact(row, "model", model, label, errors)

        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label} training expected_mapping actual={training!r}")
            training = {}
        for field, expected in exact_training_fields.items():
            _check_exact(training, field, expected, f"{label} training", errors)
        expected_options = (
            spec.anchor_options
            if model == spec.model_roles["anchor"]
            else spec.hybrid_options
        )
        _check_exact(
            training,
            "model_options",
            expected_options,
            f"{label} training",
            errors,
        )

        validation = row.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"{label} validation expected_mapping actual={validation!r}")
            validation = {}
        for field, expected in exact_validation_fields.items():
            _check_exact(validation, field, expected, f"{label} validation", errors)

        if seed_is_expected:
            cache_roots[int(seed)].append(training.get("dataset_cache_root"))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{label} metrics expected_mapping actual={metrics!r}")
            metrics = {}
        else:
            for metric in ("auc", "accuracy", "calibrated_accuracy", "loss"):
                if not _is_finite_number(metrics.get(metric)):
                    errors.append(
                        f"{label} metrics.{metric} must_be_finite actual={metrics.get(metric)!r}"
                    )
        errors.extend(
            _checkpoint_history_errors(
                row,
                training=training,
                metrics=metrics,
                configured_epochs=epochs,
                label=label,
            )
        )

        for field in ("parameter_count", "trainable_parameter_count"):
            value = row.get(field)
            if not _is_positive_integer(value):
                errors.append(
                    f"{label} {field} must_be_positive_integer actual={value!r}"
                )
            elif model_is_expected and model != spec.model_roles["anchor"]:
                conv_counts[field].add(value)

    for seed, roots in cache_roots.items():
        if (
            len(roots) != len(spec.model_roles)
            or any(not isinstance(root, str) or not root.strip() for root in roots)
            or len(set(roots)) != 1
        ):
            errors.append(
                f"seed={seed} dataset_cache_root must_be_identical_and_non_empty actual={roots!r}"
            )

    if len(conv_counts["parameter_count"]) != 1:
        errors.append(
            f"{spec.capacity_label}_parameter_count_mismatch "
            f"values={sorted(conv_counts['parameter_count'])}"
        )
    if len(conv_counts["trainable_parameter_count"]) != 1:
        errors.append(
            f"{spec.capacity_label}_trainable_parameter_count_mismatch "
            f"values={sorted(conv_counts['trainable_parameter_count'])}"
        )
    return errors


def _checkpoint_history_errors(
    row: dict[str, Any],
    *,
    training: dict[str, Any],
    metrics: dict[str, Any],
    configured_epochs: int,
    label: str,
) -> list[str]:
    errors: list[str] = []
    history = row.get("history")
    if not isinstance(history, list) or not history:
        return [f"{label} history expected_nonempty_list actual={history!r}"]

    epochs_ran = training.get("epochs_ran")
    if type(epochs_ran) is not int or not 1 <= epochs_ran <= configured_epochs:
        errors.append(
            f"{label} training epochs_ran must_be_exact_positive_int_at_most="
            f"{configured_epochs} actual={epochs_ran!r}"
        )
    if type(epochs_ran) is int and len(history) != epochs_ran:
        errors.append(
            f"{label} history length={len(history)} must_equal_epochs_ran={epochs_ran}"
        )

    best_epoch = training.get("best_epoch")
    if (
        type(best_epoch) is not int
        or type(epochs_ran) is not int
        or not 1 <= best_epoch <= epochs_ran
    ):
        errors.append(
            f"{label} training best_epoch must_be_exact_positive_int_at_most_epochs_ran "
            f"actual={best_epoch!r} epochs_ran={epochs_ran!r}"
        )

    required_history_metrics = (
        "learning_rate",
        "train_auc",
        "train_accuracy",
        "train_loss",
        "train_eval_loss",
        "val_auc",
        "val_accuracy",
        "val_calibrated_accuracy",
        "val_loss",
    )
    valid_history_items = True
    for expected_epoch, item in enumerate(history, start=1):
        item_label = f"{label} history[{expected_epoch - 1}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label} expected_mapping actual={item!r}")
            valid_history_items = False
            continue
        epoch = item.get("epoch")
        if not _is_finite_number(epoch) or epoch != expected_epoch:
            errors.append(
                f"{item_label} epoch expected_sequential={expected_epoch} actual={epoch!r}"
            )
            valid_history_items = False
        for metric in required_history_metrics:
            if not _is_finite_number(item.get(metric)):
                errors.append(
                    f"{item_label} {metric} must_be_finite actual={item.get(metric)!r}"
                )
                valid_history_items = False

    best_checkpoint_metric = training.get("best_checkpoint_metric")
    if not _is_finite_number(best_checkpoint_metric):
        errors.append(
            f"{label} training best_checkpoint_metric must_be_finite "
            f"actual={best_checkpoint_metric!r}"
        )

    stopped_epoch = training.get("stopped_epoch")
    if type(stopped_epoch) is not int or stopped_epoch < 0:
        errors.append(
            f"{label} training stopped_epoch must_be_exact_nonnegative_int "
            f"actual={stopped_epoch!r}"
        )

    patience = training.get("early_stopping_patience")
    min_delta = training.get("early_stopping_min_delta")
    if (
        valid_history_items
        and type(patience) is int
        and patience >= 0
        and _is_finite_number(min_delta)
    ):
        replayed_best: float | None = None
        replayed_best_epoch = 0
        epochs_without_improvement = 0
        replayed_stopped_epoch = 0
        for epoch, item in enumerate(history, start=1):
            current = item["val_auc"]
            if is_checkpoint_improved(
                current=current,
                best=replayed_best,
                metric="val_auc",
                min_delta=min_delta,
            ):
                replayed_best = current
                replayed_best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                replayed_stopped_epoch = epoch
                break

        if replayed_stopped_epoch:
            if len(history) != replayed_stopped_epoch:
                errors.append(
                    f"{label} history contains_epochs_after_early_stop "
                    f"expected_length={replayed_stopped_epoch} actual={len(history)}"
                )
            if epochs_ran != replayed_stopped_epoch:
                errors.append(
                    f"{label} training epochs_ran must_match_replayed_early_stop "
                    f"expected={replayed_stopped_epoch} actual={epochs_ran!r}"
                )
            if stopped_epoch != replayed_stopped_epoch:
                errors.append(
                    f"{label} training stopped_epoch must_match_replay "
                    f"expected={replayed_stopped_epoch} actual={stopped_epoch!r}"
                )
        else:
            if len(history) != configured_epochs or epochs_ran != configured_epochs:
                errors.append(
                    f"{label} partial_history_without_early_stop "
                    f"configured_epochs={configured_epochs} history_length={len(history)} "
                    f"epochs_ran={epochs_ran!r}"
                )
            if stopped_epoch != 0:
                errors.append(
                    f"{label} training stopped_epoch expected=0_without_replayed_stop "
                    f"actual={stopped_epoch!r}"
                )

        if best_epoch != replayed_best_epoch:
            errors.append(
                f"{label} training best_epoch must_match_checkpoint_replay "
                f"expected={replayed_best_epoch} actual={best_epoch!r}"
            )
        if not _tightly_equal_finite(best_checkpoint_metric, replayed_best):
            errors.append(
                f"{label} training best_checkpoint_metric must_match_checkpoint_replay "
                f"expected={replayed_best!r} actual={best_checkpoint_metric!r}"
            )

        best_history = history[replayed_best_epoch - 1]
        comparisons = (
            ("metrics.auc", metrics.get("auc"), best_history["val_auc"]),
            ("metrics.accuracy", metrics.get("accuracy"), best_history["val_accuracy"]),
            (
                "metrics.calibrated_accuracy",
                metrics.get("calibrated_accuracy"),
                best_history["val_calibrated_accuracy"],
            ),
            ("metrics.loss", metrics.get("loss"), best_history["val_loss"]),
        )
        for field, actual, expected in comparisons:
            if not _tightly_equal_finite(actual, expected):
                errors.append(
                    f"{label} {field} must_match_best_epoch_history "
                    f"expected={expected!r} actual={actual!r}"
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
    if not _exact_equal(actual, expected):
        errors.append(f"{label} {field} expected={expected!r} actual={actual!r}")


def _exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _is_finite_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _tightly_equal_finite(left: Any, right: Any) -> bool:
    return (
        _is_finite_number(left)
        and _is_finite_number(right)
        and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    )


def _is_expected_seed(value: Any, expected: set[int]) -> bool:
    return type(value) is int and value in expected


def _is_expected_model(value: Any, expected: set[str]) -> bool:
    return isinstance(value, str) and value in expected


def _is_positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0 and _is_finite_number(value)


def _rows_by_seed_and_role(
    rows: list[dict[str, Any]],
    *,
    spec: FourRoleGateSpec,
) -> dict[int, dict[str, dict[str, Any]]]:
    model_to_role = {model: role for role, model in spec.model_roles.items()}
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["seed"])
        role = model_to_role[str(row["selected_model"])]
        by_seed.setdefault(seed, {})[role] = row
    return by_seed


def _seed_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aucs = {role: float(row["metrics"]["auc"]) for role, row in rows.items()}
    return {
        "aucs": aucs,
        "architecture_margin": aucs["candidate"] - aucs["anchor"],
        "topology_margin": aucs["candidate"] - aucs["shuffled_p"],
        "representation_margin": aucs["candidate"] - aucs["delta_only"],
        "candidate_above_all": aucs["candidate"]
        > max(aucs["anchor"], aucs["shuffled_p"], aucs["delta_only"]),
    }


def _protocol_evidence(
    by_seed: dict[int, dict[str, dict[str, Any]]],
    expected_seeds: tuple[int, ...],
    *,
    spec: FourRoleGateSpec,
) -> dict[str, Any]:
    top_fields = (
        "cipher",
        "cipher_key",
        "structure",
        "model",
        "selected_model",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "input_difference",
        "train_samples_total",
        "validation_samples_total",
        "final_test_samples_total",
        "final_test_repeats",
        "negative_mode",
        "pairs_per_sample",
        "feature_encoding",
        "train_key",
        "validation_key",
        "key_rotation_interval",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "integral_active_nibble",
        "integral_active_nibbles",
        "validation_integral_active_nibbles",
    )
    training_fields = (
        "amsgrad",
        "batch_size",
        "dataset_cache_chunk_size",
        "dataset_cache_workers",
        "dataset_label_mode",
        "device",
        "feature_encoding",
        "key_schedule",
        "input_bits",
        "integral_active_nibble",
        "integral_active_nibbles",
        "validation_integral_active_nibbles",
        "pair_bits",
        "pairs_per_sample",
        "key_rotation_interval",
        "sample_structure",
        "selected_bit_indices",
        "samples_total",
        "positive_rows",
        "negative_rows",
        "train_rows",
        "train_positive_rows",
        "train_negative_rows",
        "validation_rows",
        "validation_positive_rows",
        "validation_negative_rows",
        "train_dataset_storage",
        "validation_dataset_storage",
        "epochs",
        "dataset_cache_root",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "lr_scheduler",
        "max_learning_rate",
        "checkpoint_metric",
        "selected_checkpoint",
        "restore_best_checkpoint",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "optimizer_state_transition",
        "optimizer_state_reused",
        "optimizer_state_step_before",
        "optimizer_session_call",
        "train_eval_interval",
    )
    validation_fields = (
        "cipher",
        "structure",
        "rounds",
        "feature_encoding",
        "negative_mode",
        "pairs_per_sample",
        "samples_per_class",
        "samples_total",
        "positive_rows",
        "negative_rows",
        "dataset_label_mode",
        "key_rotation_interval",
        "sample_structure",
        "key_schedule",
        "integral_active_nibble",
        "integral_active_nibbles",
    )
    rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for role in spec.model_roles:
            row = by_seed[seed][role]
            training = row["training"]
            validation = row["validation"]
            rows.append(
                {
                    "seed": seed,
                    "role": role,
                    "model": row["selected_model"],
                    "model_options": training["model_options"],
                    "top_level": {field: row[field] for field in top_fields},
                    "training": {field: training[field] for field in training_fields},
                    "validation": {
                        field: validation[field] for field in validation_fields
                    },
                    "parameter_count": row["parameter_count"],
                    "trainable_parameter_count": row["trainable_parameter_count"],
                    "checkpoint_history_status": "pass",
                    "epochs_ran": training["epochs_ran"],
                    "best_epoch": training["best_epoch"],
                    "history_rows": len(row["history"]),
                }
            )
    return {"rows": rows, "status": "pass"}


def _semantic_checks() -> dict[str, Any]:
    return {
        "raw_input_bits": 2048,
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "state_matrix_axes": ["bit_plane", "cell"],
        "state_matrix_shape": ["batch", "pair", 4, 16],
        "role_mapping_identities": {
            "candidate": "true_inv_p",
            "shuffled_p": "deterministic_shuffled_p",
            "delta_only": "raw_delta",
        },
        "evidence_kind": "frozen/tested semantic contract; not runtime tensor equality",
        "status": "pass",
    }


def _promotion_conditions(
    *,
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    seed0_architecture_margin: float | None,
    seed0_topology_margin: float,
    seed0_representation_margin: float,
    joint_architecture_margin: float,
    joint_control_margin: float,
) -> dict[str, Any]:
    conditions = {
        "candidate_above_anchor_required": True,
        "candidate_above_all_controls_required": True,
        "seed0_topology_margin": seed0_topology_margin,
        "seed0_representation_margin": seed0_representation_margin,
        "joint_architecture_margin": joint_architecture_margin,
        "joint_control_margin": joint_control_margin,
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "scale_identity": f"{samples_per_class}/class",
    }
    if seed0_architecture_margin is not None:
        conditions["seed0_architecture_margin"] = seed0_architecture_margin
    return conditions


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_smoke_metrics", "remote_scale")
    elif decision == "promote_seed1":
        actions = ("65536_per_class", "262144_per_class", "remote_scale")
    elif decision == "promote_medium_65536":
        actions = ("262144_per_class", "remote_scale")
    else:
        actions = ("seed1", "65536_per_class", "262144_per_class", "remote_scale")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


def _decision(
    seed_reports: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    seed0_architecture_margin: float | None,
    seed0_topology_margin: float,
    seed0_representation_margin: float,
    joint_architecture_margin: float,
    joint_control_margin: float,
) -> tuple[str, str]:
    reports = [seed_reports[str(seed)] for seed in expected_seeds]
    if any(report["architecture_margin"] <= 0.0 for report in reports):
        return "stop_conv2d_route", "keep_token_mixer_anchor_and_do_not_scale_conv2d"
    if any(report["topology_margin"] <= 0.0 for report in reports):
        return (
            "stop_generic_locality",
            "do_not_scale_generic_locality_without_true_p_topology",
        )
    if any(report["representation_margin"] <= 0.0 for report in reports):
        return (
            "stop_invp_attribution",
            "do_not_scale_without_invp_representation_attribution",
        )

    if len(reports) == 1:
        report = reports[0]
        if (
            report["topology_margin"] >= seed0_topology_margin
            and report["representation_margin"] >= seed0_representation_margin
        ):
            return "promote_seed1", "run_identical_seed1_local_gate"
        return (
            "weak_or_fragile_no_scale",
            "do_not_scale_run_bounded_local_variance_check",
        )

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
        return (
            "promote_medium_65536",
            "run_65536_per_class_two_seed_medium_confirmation",
        )
    return (
        "unstable_no_remote_scale",
        "do_not_launch_remote_scale_inspect_two_seed_variance",
    )


_CONV2D_GATE_SPEC = FourRoleGateSpec(
    model_roles=MODEL_ROLES,
    anchor_options={
        "spn_mixer_depth": 2,
        "activation": "relu",
        "norm": "layernorm",
    },
    hybrid_options={
        "conv_depth": 3,
        "kernel_size": 3,
        "activation": "relu",
        "norm": "batchnorm2d",
        "dropout": 0.0,
    },
    capacity_label="conv2d",
    semantic_checks=_semantic_checks(),
    readiness_next_action="run_frozen_r1_seed0_local_diagnostic",
    claim_label="architecture-attribution diagnostic",
    decide=_decision,
    stopped_actions=_stopped_actions,
)
