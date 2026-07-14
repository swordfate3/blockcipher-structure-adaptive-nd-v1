from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.engine.checkpoint_initialization import file_sha256
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


E6_SOURCE_MODELS = {
    "off": "present_cross_spn_typed_cell_e6_off",
    "candidate": "present_cross_spn_typed_cell_e6_functional_margin",
    "placebo": "present_cross_spn_typed_cell_e6_shuffled_placebo",
}
E6_TARGET_MODELS = {
    "scratch": "gift_cross_spn_typed_cell_e6_scratch",
    "off": "gift_cross_spn_typed_cell_e6_from_present_off",
    "candidate": "gift_cross_spn_typed_cell_e6_from_present_functional_margin",
    "placebo": "gift_cross_spn_typed_cell_e6_from_present_shuffled_placebo",
}


def gate_e6_source_readiness(
    plan_path: Path,
    results_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    rows, errors = _read_jsonl(results_path)
    alignment = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=3,
    )
    errors.extend(alignment["errors"])
    by_model = {row.get("selected_model"): row for row in rows}
    if set(by_model) != set(E6_SOURCE_MODELS.values()):
        errors.append("source readiness models do not match E6 roles")

    state_keys: dict[str, set[str]] = {}
    for role, model in E6_SOURCE_MODELS.items():
        row = by_model.get(model)
        if row is None:
            continue
        errors.extend(_source_row_errors(role, row))
        checkpoint = (row.get("training") or {}).get("checkpoint_output")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            errors.append(f"source role={role} checkpoint is missing")
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            errors.append(f"source role={role} checkpoint state_dict is missing")
            continue
        state_keys[role] = set(state_dict)
    if len(state_keys) == 3 and len({tuple(sorted(keys)) for keys in state_keys.values()}) != 1:
        errors.append("source readiness checkpoint state keys differ")

    if errors:
        return _invalid(errors, alignment), None
    manifest = build_e6_readiness_manifest(by_model, results_path)
    key_count = len(next(iter(state_keys.values())))
    report = {
        "status": "pass",
        "decision": "e6_source_functional_margin_readiness_pass",
        "errors": [],
        "rows": 3,
        "parameter_count": 196003,
        "checkpoint_state_keys": key_count,
        "state_dict_keys_equal": True,
        "alignment": alignment,
        "metrics_interpreted": False,
        "roles": {
            role: _source_role_summary(by_model[model])
            for role, model in E6_SOURCE_MODELS.items()
        },
        "next_action": "run_e6_target_readiness_from_verified_manifest",
    }
    return report, manifest


def gate_e6_target_readiness(
    plan_path: Path,
    results_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    rows, errors = _read_jsonl(results_path)
    progress, progress_errors = _read_jsonl(progress_path)
    errors.extend(progress_errors)
    alignment = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=4,
    )
    errors.extend(alignment["errors"])
    by_model = {row.get("selected_model"): row for row in rows}
    if set(by_model) != set(E6_TARGET_MODELS.values()):
        errors.append("target readiness models do not match E6 roles")
    for role, model in E6_TARGET_MODELS.items():
        row = by_model.get(model)
        if row is None:
            continue
        errors.extend(_target_row_errors(role, row))

    cache_start = sum(event.get("event") == "cache_start" for event in progress)
    cache_reuse = sum(event.get("event") == "cache_reuse" for event in progress)
    if cache_start != 2 or cache_reuse != 6:
        errors.append(
            f"target readiness cache events expected start=2 reuse=6 actual={cache_start}/{cache_reuse}"
        )
    if errors:
        return _invalid(errors, alignment)
    return {
        "status": "pass",
        "decision": "e6_target_readiness_pass",
        "errors": [],
        "rows": 4,
        "parameter_count": 196003,
        "alignment": alignment,
        "metrics_interpreted": False,
        "target_auxiliary_loss_max": max(
            float(item.get("train_auxiliary_loss", 0.0))
            for row in rows
            for item in row.get("history", [])
        ),
        "cache": {
            "create_events": cache_start,
            "reuse_events": cache_reuse,
            "shared_train_and_validation_cache": True,
        },
        "transfer_initializations": {
            "checkpoint_rows": 3,
            "strict_state_dict_load_rows": 3,
            "state_dict_key_count": 59,
        },
        "next_action": "run_e6_local_8192_source_and_target_gate",
    }


def build_e6_readiness_manifest(
    source_rows: dict[Any, dict[str, Any]],
    results_path: Path,
) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {
        E6_TARGET_MODELS["scratch"]: {
            "kind": "scratch",
            "target_mapping": "true",
        }
    }
    for role in ("off", "candidate", "placebo"):
        source_model = E6_SOURCE_MODELS[role]
        source_row = source_rows[source_model]
        checkpoint = Path(source_row["training"]["checkpoint_output"])
        targets[E6_TARGET_MODELS[role]] = {
            "kind": "checkpoint",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha256(checkpoint),
            "source_results": str(results_path),
            "source_model": source_model,
            "source_cipher": "PRESENT-80",
            "source_rounds": 7,
            "source_seed": 0,
            "source_samples_per_class": 64,
            "source_epochs": 3,
            "source_mapping": "true",
            "target_mapping": "true",
        }
    return {"version": 1, "targets": targets}


def _source_row_errors(role: str, row: dict[str, Any]) -> list[str]:
    errors = _common_row_errors(role, row, cipher="PRESENT-80", rounds=7, seed=0)
    history = row.get("history", [])
    auxiliary = [float(item.get("train_auxiliary_loss", 0.0)) for item in history]
    functional_keys = (
        "train_functional_preferred_loss",
        "train_functional_comparison_loss",
        "train_functional_loss_gap",
        "train_functional_margin_loss",
        "train_functional_violation_rate",
    )
    if role == "off":
        if any(value != 0.0 for value in auxiliary):
            errors.append("source off auxiliary loss must be zero")
        if any(any(key in item for key in functional_keys) for item in history):
            errors.append("source off must not emit functional metrics")
    else:
        if not auxiliary or not all(math.isfinite(value) and value > 0.0 for value in auxiliary):
            errors.append(f"source role={role} auxiliary losses must be finite positive")
        for item in history:
            if not all(key in item and math.isfinite(float(item[key])) for key in functional_keys):
                errors.append(f"source role={role} functional metrics are missing or non-finite")
                break
    return errors


def _target_row_errors(role: str, row: dict[str, Any]) -> list[str]:
    errors = _common_row_errors(role, row, cipher="GIFT-64", rounds=6, seed=2)
    history = row.get("history", [])
    if any(float(item.get("train_auxiliary_loss", 0.0)) != 0.0 for item in history):
        errors.append(f"target role={role} auxiliary loss must be zero")
    if any(any("functional" in key for key in item) for item in history):
        errors.append(f"target role={role} must not emit functional metrics")
    initialization = row.get("initialization")
    if not isinstance(initialization, dict):
        errors.append(f"target role={role} initialization is missing")
        return errors
    if role == "scratch":
        if initialization.get("kind") != "scratch":
            errors.append("target scratch initialization must be scratch")
    else:
        if initialization.get("kind") != "checkpoint":
            errors.append(f"target role={role} must load a checkpoint")
        if initialization.get("strict_state_dict_load") is not True:
            errors.append(f"target role={role} strict state load must pass")
        if initialization.get("source_model") != E6_SOURCE_MODELS[role]:
            errors.append(f"target role={role} source model mismatch")
        if initialization.get("state_dict_key_count") != 59:
            errors.append(f"target role={role} state key count mismatch")
    return errors


def _common_row_errors(
    role: str,
    row: dict[str, Any],
    *,
    cipher: str,
    rounds: int,
    seed: int,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "cipher": cipher,
        "rounds": rounds,
        "seed": seed,
        "samples_per_class": 64,
        "parameter_count": 196003,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(
                f"role={role} {field} expected={value!r} actual={row.get(field)!r}"
            )
    training = row.get("training")
    if not isinstance(training, dict):
        errors.append(f"role={role} training metadata is missing")
    elif training.get("epochs") != (3 if cipher == "PRESENT-80" else 1):
        errors.append(f"role={role} readiness epoch count mismatch")
    return errors


def _source_role_summary(row: dict[str, Any]) -> dict[str, Any]:
    history = row["history"]
    return {
        "auxiliary_loss_min": min(
            float(item.get("train_auxiliary_loss", 0.0)) for item in history
        ),
        "auxiliary_loss_max": max(
            float(item.get("train_auxiliary_loss", 0.0)) for item in history
        ),
        "functional_loss_gap_final": history[-1].get(
            "train_functional_loss_gap"
        ),
        "functional_margin_loss_final": history[-1].get(
            "train_functional_margin_loss"
        ),
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"cannot read {path}: {exc}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON line={index} path={path}: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"JSONL row line={index} path={path} must be an object")
            continue
        rows.append(payload)
    return rows, errors


def _invalid(errors: list[str], alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "invalid_e6_readiness",
        "errors": errors,
        "alignment": alignment,
        "metrics_interpreted": False,
        "next_action": "repair_e6_readiness_before_diagnostic",
    }


__all__ = [
    "E6_SOURCE_MODELS",
    "E6_TARGET_MODELS",
    "build_e6_readiness_manifest",
    "gate_e6_source_readiness",
    "gate_e6_target_readiness",
]
