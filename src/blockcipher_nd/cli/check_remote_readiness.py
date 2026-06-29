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

    plan_path = _local_plan_path(config.get("plan"))
    tasks: list[dict[str, Any]] = []
    if plan_path is None:
        errors.append("missing plan")
    elif not plan_path.exists():
        errors.append(f"plan does not exist: {plan_path}")
    else:
        tasks = build_tasks(parse_train_args(["--plan", str(plan_path)]))

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

    if _max_samples_per_class(tasks) >= 65_536:
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

    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "run_id": run_id,
        "plan": str(plan_path) if plan_path is not None else None,
        "expected_rows": expected_rows,
        "plan_rows": len(tasks),
        "max_samples_per_class": _max_samples_per_class(tasks),
        "errors": errors,
        "warnings": warnings,
        "checked_invariants": [
            "plan_exists",
            "expected_rows_matches_plan",
            "run_id_task_archive_alignment",
            "github_ssh_repo",
            "cmd_exe_c_only_policy",
            "g_lxy_artifact_policy",
            "medium_scale_dataset_cache",
            "training_protocol_matches_plan",
        ],
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
