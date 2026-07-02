from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import build_tasks


REMOTE_ROOT = "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a remote experiment config before launch without touching the remote host."
    )
    parser.add_argument("--config", required=True, type=Path, help="Remote config JSON path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def remote_readiness_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(config, dict):
        return {
            "status": "fail",
            "config": str(config_path),
            "errors": ["remote config must be a JSON object"],
            "warnings": [],
        }

    if config.get("launch_enabled") is False:
        reason = _str_value(config.get("disabled_reason")) or "remote config is disabled"
        return {
            "status": "fail",
            "config": str(config_path),
            "run_id": _str_value(config.get("run_id")),
            "plan": _str_value(config.get("plan")) or None,
            "expected_rows": _int_value(config.get("expected_rows")),
            "plan_rows": 0,
            "max_samples_per_class": 0,
            "errors": [f"launch_enabled=false: {reason}"],
            "warnings": [],
            "checked_invariants": ["launch_enabled"],
        }

    plan_path = _local_plan_path(config.get("plan"))
    tasks: list[dict[str, Any]] = []
    plan_is_json_matrix = False
    if plan_path is None:
        errors.append("missing plan")
    elif not plan_path.exists():
        errors.append(f"plan does not exist: {plan_path}")
    else:
        try:
            plan_is_json_matrix = _plan_is_json_matrix(plan_path)
            tasks = _load_plan_tasks(plan_path)
        except ValueError as exc:
            errors.append(str(exc))

    expected_rows = _int_value(config.get("expected_rows"))
    if expected_rows is None:
        errors.append("missing or invalid expected_rows")
    elif tasks and expected_rows != len(tasks):
        errors.append(f"expected_rows={expected_rows} plan_rows={len(tasks)}")

    run_id = _str_value(config.get("run_id"))
    if not run_id:
        errors.append("missing run_id")
    if run_id and config.get("task_name") not in {run_id, None}:
        errors.append(f"task_name={config.get('task_name')} does not match run_id={run_id}")
    if run_id and config.get("archive_work_id") not in {run_id, None}:
        errors.append(f"archive_work_id={config.get('archive_work_id')} does not match run_id={run_id}")

    _require_equal(config, "branch", "main", errors)
    _require_prefix(config, "repo_url", "git@github.com:", errors)
    if "clone_url" in config:
        _require_prefix(config, "clone_url", "git@github.com:", errors)
    _require_contains(config, "launch_policy", "cmd.exe /c", errors)
    _forbid_contains(config, "launch_policy", "cmd.exe /k", errors)
    _require_contains(config, "launch_policy", "G:\\lxy", errors)

    monitor_script = _str_value(config.get("monitor_script_name"))
    if not monitor_script:
        errors.append("missing monitor_script_name")
    elif not monitor_script.endswith(".sh"):
        errors.append(f"monitor_script_name must end with .sh: {monitor_script}")

    max_samples_per_class = _max_samples_per_class(tasks)
    if max_samples_per_class >= 65_536:
        if config.get("dataset_cache") is not True:
            errors.append("dataset_cache must be true for medium or larger remote runs")
        cache_root = _str_value(config.get("dataset_cache_root"))
        if not cache_root:
            errors.append("missing dataset_cache_root")
        elif not cache_root.startswith(REMOTE_ROOT):
            errors.append(f"dataset_cache_root must stay under {REMOTE_ROOT}: {cache_root}")
        chunk_size = _int_value(config.get("dataset_cache_chunk_size"))
        if chunk_size is None or chunk_size <= 0:
            errors.append("dataset_cache_chunk_size must be positive")
        workers = _int_value(config.get("dataset_cache_workers"))
        if workers is None or workers < 1:
            errors.append("dataset_cache_workers must be >= 1")

    if config.get("result_sync") != "local_tmux_monitor_scp_fallback":
        warnings.append("result_sync is not local_tmux_monitor_scp_fallback")
    if config.get("source_commit") in {None, "", "dirty_overlay"}:
        errors.append("source_commit must be recorded or delegated to remote run script git revision")

    train_consistency = _training_consistency(config, tasks)
    errors.extend(train_consistency["errors"])
    warnings.extend(train_consistency["warnings"])
    candidate_consistency = _candidate_trail_consistency(config, tasks, plan_is_json_matrix=plan_is_json_matrix)
    errors.extend(candidate_consistency["errors"])
    warnings.extend(candidate_consistency["warnings"])
    transition_consistency = _transition_spectrum_consistency(
        config, tasks, plan_is_json_matrix=plan_is_json_matrix
    )
    errors.extend(transition_consistency["errors"])
    warnings.extend(transition_consistency["warnings"])
    pairset_consistency = _pairset_aggregation_consistency(config)
    errors.extend(pairset_consistency["errors"])
    warnings.extend(pairset_consistency["warnings"])

    checked_invariants = [
        "plan_exists",
        "expected_rows_matches_plan",
        "run_id_task_archive_alignment",
        "github_ssh_repo",
        "cmd_exe_c_only_policy",
        "g_lxy_artifact_policy",
        "training_protocol_matches_plan",
    ]
    if max_samples_per_class >= 65_536:
        checked_invariants.append("medium_scale_dataset_cache")
    if _is_candidate_trail_config(config):
        checked_invariants.append("candidate_trail_protocol_lock")
    if _is_transition_spectrum_config(config):
        checked_invariants.append("transition_spectrum_protocol_lock")
    if _is_pairset_aggregation_config(config):
        checked_invariants.append("pairset_aggregation_stage_lock")

    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "run_id": run_id,
        "plan": str(plan_path) if plan_path is not None else None,
        "expected_rows": expected_rows,
        "plan_rows": len(tasks),
        "max_samples_per_class": max_samples_per_class,
        "errors": errors,
        "warnings": warnings,
        "checked_invariants": checked_invariants,
    }


def _training_consistency(config: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not tasks:
        return {"errors": errors, "warnings": warnings}

    comparable_fields = [
        "loss",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "lr_scheduler",
        "max_learning_rate",
        "key_rotation_interval",
        "sample_structure",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "early_stopping_patience",
        "early_stopping_min_delta",
    ]
    for field in comparable_fields:
        if field not in config:
            warnings.append(f"remote config does not override {field}; runner default or plan value will apply")
            continue
        planned_values = {task.get(field) for task in tasks}
        if len(planned_values) == 1 and config[field] != next(iter(planned_values)):
            errors.append(f"{field}={config[field]} plan_value={next(iter(planned_values))}")

    if "device" not in config:
        errors.append("missing device")
    elif not str(config["device"]).startswith(("cuda", "cpu")):
        errors.append(f"unsupported device value: {config['device']}")
    if "epochs" not in config or _int_value(config["epochs"]) is None:
        errors.append("missing or invalid epochs")
    if "batch_size" not in config or _int_value(config["batch_size"]) is None:
        errors.append("missing or invalid batch_size")
    return {"errors": errors, "warnings": warnings}


def _load_plan_tasks(plan_path: Path) -> list[dict[str, Any]]:
    if plan_path.suffix.lower() == ".json":
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError(f"JSON plan must be an object: {plan_path}")
        if "rows" in plan:
            return _candidate_json_matrix_tasks(plan)
        return [_candidate_json_plan_task(plan)]
    return build_tasks(parse_train_args(["--plan", str(plan_path)]))


def _candidate_json_matrix_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    common = plan.get("common", {})
    rows = plan.get("rows")
    if not isinstance(common, dict):
        raise ValueError("JSON matrix plan common must be an object")
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON matrix plan rows must be a non-empty list")
    tasks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("JSON matrix plan rows must be objects")
        tasks.append(_candidate_json_plan_task({**common, **row}))
    return tasks


def _candidate_json_plan_task(plan: dict[str, Any]) -> dict[str, Any]:
    model = plan.get("model")
    feature_mode = plan.get("feature_mode")
    if feature_mode is None:
        feature_mode = {
            "linear": "cell_structured",
            "mlp": "cell_structured",
            "shuffled_cells": "cell_structured_shuffled",
        }.get(str(model))
    return {
        "samples_per_class": _required_int(plan, "samples_per_class"),
        "negative_mode": plan.get("negative_mode"),
        "sample_structure": plan.get("sample_structure"),
        "key_rotation_interval": _required_int(plan, "key_rotation_interval"),
        "learning_rate": _optional_float(plan.get("learning_rate")),
        "rounds": _required_int(plan, "rounds"),
        "seed": _required_int(plan, "seed"),
        "pairs_per_sample": _required_int(plan, "pairs_per_sample"),
        "model": model,
        "validation_key": plan.get("validation_key"),
        "feature_mode": feature_mode,
    }


def _candidate_trail_consistency(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    plan_is_json_matrix: bool,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _is_candidate_trail_config(config):
        return {"errors": errors, "warnings": warnings}

    expected_values = {
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
    }
    for field, expected in expected_values.items():
        observed = config.get(field)
        if observed != expected:
            errors.append(f"candidate_trail {field}={observed} expected={expected}")

    allowed_feature_modes = {"cell_structured", "cell_structured_shuffled"}
    feature_modes = _candidate_feature_modes(config, tasks)
    if not feature_modes or any(mode not in allowed_feature_modes for mode in feature_modes):
        errors.append(
            "candidate_trail feature_mode must be explicit cell-structured control "
            f"one of {sorted(allowed_feature_modes)}: {sorted(feature_modes)}"
        )

    runner_script = _str_value(config.get("runner_script"))
    if plan_is_json_matrix and runner_script != "scripts/spn-candidate-evidence-matrix":
        errors.append(
            "candidate_trail JSON matrix remote config must set "
            "runner_script=scripts/spn-candidate-evidence-matrix"
        )
    if not plan_is_json_matrix and runner_script and runner_script != "scripts/spn-candidate-evidence":
        errors.append(f"candidate_trail single JSON plan runner_script unsupported: {runner_script}")

    cache_root = _str_value(config.get("feature_cache_root")) or _str_value(config.get("dataset_cache_root"))
    if not cache_root:
        errors.append("candidate_trail missing feature_cache_root or dataset_cache_root")
    elif not cache_root.startswith(REMOTE_ROOT):
        errors.append(f"candidate_trail cache root must stay under {REMOTE_ROOT}: {cache_root}")

    feature_cache_workers = _int_value(config.get("feature_cache_workers", config.get("dataset_cache_workers", 1)))
    if feature_cache_workers is None or feature_cache_workers < 1:
        errors.append("candidate_trail feature_cache_workers must be >= 1")
    elif _max_samples_per_class(tasks) >= 65_536 and feature_cache_workers == 1:
        warnings.append(
            "candidate_trail medium-scale feature_cache_workers=1; future medium/large runs should set >1 "
            "to avoid slow feature-cache generation"
        )

    return {"errors": errors, "warnings": warnings}


def _transition_spectrum_consistency(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    plan_is_json_matrix: bool,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _is_transition_spectrum_config(config):
        return {"errors": errors, "warnings": warnings}

    expected_values = {
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
    }
    for field, expected in expected_values.items():
        observed = config.get(field)
        if observed != expected:
            errors.append(f"transition_spectrum {field}={observed} expected={expected}")

    runner_script = _str_value(config.get("runner_script"))
    if plan_is_json_matrix and runner_script != "scripts/spn-transition-spectrum-matrix":
        errors.append(
            "transition_spectrum JSON matrix remote config must set "
            "runner_script=scripts/spn-transition-spectrum-matrix"
        )
    if not plan_is_json_matrix and runner_script and runner_script != "scripts/spn-transition-spectrum":
        errors.append(f"transition_spectrum single JSON plan runner_script unsupported: {runner_script}")

    planned_models = {_str_value(task.get("model")) for task in tasks}
    required_models = {"linear", "mlp", "shuffled_p"}
    if plan_is_json_matrix and tasks and not required_models.issubset(planned_models):
        errors.append(
            "transition_spectrum JSON matrix must include linear, mlp, and shuffled_p rows: "
            f"{sorted(planned_models)}"
        )

    cache_root = _str_value(config.get("feature_cache_root")) or _str_value(config.get("dataset_cache_root"))
    if not cache_root:
        errors.append("transition_spectrum missing feature_cache_root or dataset_cache_root")
    elif not cache_root.startswith(REMOTE_ROOT):
        errors.append(f"transition_spectrum cache root must stay under {REMOTE_ROOT}: {cache_root}")

    feature_cache_workers = _int_value(config.get("feature_cache_workers", config.get("dataset_cache_workers", 1)))
    if feature_cache_workers is None or feature_cache_workers < 1:
        errors.append("transition_spectrum feature_cache_workers must be >= 1")
    elif _max_samples_per_class(tasks) >= 65_536 and feature_cache_workers == 1:
        warnings.append(
            "transition_spectrum medium-scale feature_cache_workers=1; future medium/large runs should set >1 "
            "to avoid slow feature-cache generation"
        )

    return {"errors": errors, "warnings": warnings}


def _plan_is_json_matrix(plan_path: Path) -> bool:
    if plan_path.suffix.lower() != ".json" or not plan_path.exists():
        return False
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return isinstance(plan, dict) and isinstance(plan.get("rows"), list)


def _candidate_feature_modes(config: dict[str, Any], tasks: list[dict[str, Any]]) -> set[str]:
    mode = _str_value(config.get("feature_mode"))
    if mode:
        return {mode}
    if len(tasks) <= 1:
        return set()
    return {_str_value(task.get("feature_mode")) for task in tasks if _str_value(task.get("feature_mode"))}


def _pairset_aggregation_consistency(config: dict[str, Any]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not _is_pairset_aggregation_config(config):
        return {"errors": errors, "warnings": warnings}

    stage = config.get("pairset_stage")
    if stage == "single_pair_scorer_checkpoint":
        _require_remote_path(config, "checkpoint_output", errors)
    elif stage == "learned_pairset_plus_frozen_aggregation_gate":
        _require_remote_path(config, "requires_checkpoint", errors)
        _require_remote_path(config, "frozen_aggregation_output", errors)
    else:
        errors.append(f"pairset_aggregation unsupported or missing pairset_stage: {stage}")

    return {"errors": errors, "warnings": warnings}


def _require_remote_path(config: dict[str, Any], field: str, errors: list[str]) -> None:
    value = _str_value(config.get(field))
    if not value:
        errors.append(f"pairset_aggregation missing {field}")
    elif not value.startswith(REMOTE_ROOT):
        errors.append(f"pairset_aggregation {field} must stay under {REMOTE_ROOT}: {value}")


def _is_candidate_trail_config(config: dict[str, Any]) -> bool:
    haystack = " ".join(
        _str_value(config.get(field))
        for field in [
            "run_id",
            "task_name",
            "plan",
            "claim_scope",
            "launch_policy",
            "route",
            "experiment_route",
        ]
    ).lower()
    markers = [
        "candidate_trail",
        "candidate-trail",
        "spn_candidate_evidence",
        "spn-candidate-evidence",
        "candidate_evidence",
        "candidate-evidence",
    ]
    return any(marker in haystack for marker in markers)


def _is_transition_spectrum_config(config: dict[str, Any]) -> bool:
    haystack = " ".join(
        _str_value(config.get(field))
        for field in [
            "run_id",
            "task_name",
            "plan",
            "claim_scope",
            "launch_policy",
            "route",
            "experiment_route",
            "runner_script",
        ]
    ).lower()
    markers = [
        "transition_spectrum",
        "transition-spectrum",
        "bit_transition_spectrum",
        "bit-transition-spectrum",
        "spn-transition-spectrum",
    ]
    return any(marker in haystack for marker in markers)


def _is_pairset_aggregation_config(config: dict[str, Any]) -> bool:
    haystack = " ".join(
        _str_value(config.get(field))
        for field in [
            "run_id",
            "task_name",
            "plan",
            "claim_scope",
            "launch_policy",
            "pairset_stage",
        ]
    ).lower()
    markers = [
        "pairset_aggregation",
        "pair-set aggregation",
        "pairset aggregation",
        "frozen_aggregation",
    ]
    return any(marker in haystack for marker in markers)


def _local_plan_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value.replace("\\", "/"))


def _max_samples_per_class(tasks: list[dict[str, Any]]) -> int:
    if not tasks:
        return 0
    return max(int(task["samples_per_class"]) for task in tasks)


def _str_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"JSON plan missing integer field: {key}")
    return int(value, 0) if isinstance(value, str) else int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_equal(config: dict[str, Any], key: str, expected: Any, errors: list[str]) -> None:
    if config.get(key) != expected:
        errors.append(f"{key}={config.get(key)} expected={expected}")


def _require_prefix(config: dict[str, Any], key: str, prefix: str, errors: list[str]) -> None:
    value = _str_value(config.get(key))
    if not value.startswith(prefix):
        errors.append(f"{key} must start with {prefix}: {value}")


def _require_contains(config: dict[str, Any], key: str, needle: str, errors: list[str]) -> None:
    value = _str_value(config.get(key))
    if needle not in value:
        errors.append(f"{key} must contain {needle}")


def _forbid_contains(config: dict[str, Any], key: str, needle: str, errors: list[str]) -> None:
    value = _str_value(config.get(key))
    if needle in value:
        errors.append(f"{key} must not contain {needle}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = remote_readiness_report(args.config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
