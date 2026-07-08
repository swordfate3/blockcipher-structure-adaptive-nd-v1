from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PLAN_RESULT_FIELD_PAIRS = [
    ("cipher", "cipher"),
    ("structure", "structure"),
    ("rounds", "rounds"),
    ("seed", "seed"),
    ("samples_per_class", "samples_per_class"),
    ("feature_encoding", "feature_encoding"),
    ("negative_mode", "negative_mode"),
    ("train_key", "train_key"),
    ("validation_key", "validation_key"),
    ("pairs_per_sample", "pairs_per_sample"),
    ("sample_structure", "sample_structure"),
    ("integral_active_nibble", "integral_active_nibble"),
    ("key_rotation_interval", "key_rotation_interval"),
    ("difference_profile", "difference_profile"),
    ("difference_member", "difference_member"),
    ("selected_bit_indices", ("training", "selected_bit_indices")),
    ("loss", ("training", "loss")),
    ("learning_rate", ("training", "learning_rate")),
    ("optimizer", ("training", "optimizer")),
    ("weight_decay", ("training", "weight_decay")),
    ("checkpoint_metric", ("training", "checkpoint_metric")),
    ("restore_best_checkpoint", ("training", "restore_best_checkpoint")),
    ("early_stopping_patience", ("training", "early_stopping_patience")),
    ("early_stopping_min_delta", ("training", "early_stopping_min_delta")),
    ("pretrain_epochs", ("training", "pretraining", "epochs_ran")),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a result JSONL file exactly matches its experiment plan rows."
    )
    parser.add_argument("--plan", required=True, help="Plan CSV path.")
    parser.add_argument("--results", required=True, help="Result JSONL path.")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON validation report path.")
    return parser.parse_args(argv)


def validate_result_plan_alignment(
    plan_path: Path,
    results_path: Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    plan_rows = _load_plan_rows(plan_path)
    result_rows = _load_jsonl_rows(results_path)

    expected_rows = len(plan_rows) if expected_rows is None else expected_rows
    errors: list[str] = []

    if len(plan_rows) != expected_rows:
        errors.append(f"plan_rows={len(plan_rows)} expected_rows={expected_rows}")
    if len(result_rows) != expected_rows:
        errors.append(f"result_rows={len(result_rows)} expected_rows={expected_rows}")

    optional_key_fields = _optional_alignment_key_fields(plan_rows, result_rows)
    plan_keys = [_alignment_key(row, optional_key_fields=optional_key_fields) for row in plan_rows]
    result_keys = [_alignment_key(row, optional_key_fields=optional_key_fields) for row in result_rows]
    plan_counter = Counter(plan_keys)
    result_counter = Counter(result_keys)

    duplicate_plan_keys = sorted(key for key, count in plan_counter.items() if count > 1)
    duplicate_result_keys = sorted(key for key, count in result_counter.items() if count > 1)
    missing_result_keys = sorted((plan_counter - result_counter).elements())
    unexpected_result_keys = sorted((result_counter - plan_counter).elements())

    if duplicate_plan_keys:
        errors.append(f"duplicate_plan_key={duplicate_plan_keys}")
    if duplicate_result_keys:
        errors.append(f"duplicate_result_key={duplicate_result_keys}")
    if missing_result_keys:
        errors.append(f"missing_result_key={missing_result_keys}")
    if unexpected_result_keys:
        errors.append(f"unexpected_result_key={unexpected_result_keys}")

    plan_by_key = {
        _alignment_key(row, optional_key_fields=optional_key_fields): row
        for row in plan_rows
        if plan_counter[_alignment_key(row, optional_key_fields=optional_key_fields)] == 1
    }
    field_mismatches = _field_mismatches(
        plan_by_key,
        result_rows,
        optional_key_fields=optional_key_fields,
    )
    if field_mismatches:
        errors.append(f"field_mismatches={field_mismatches[:10]}")

    return {
        "status": "pass" if not errors else "fail",
        "expected_rows": expected_rows,
        "plan_rows": len(plan_rows),
        "result_rows": len(result_rows),
        "optional_key_fields": list(optional_key_fields),
        "plan_keys": plan_keys,
        "result_keys": result_keys,
        "duplicate_plan_keys": duplicate_plan_keys,
        "duplicate_result_keys": duplicate_result_keys,
        "missing_result_keys": missing_result_keys,
        "unexpected_result_keys": unexpected_result_keys,
        "field_mismatches": field_mismatches,
        "errors": errors,
    }


def _load_plan_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError(f"JSON plan must be an object: {path}")
        if "rows" not in plan:
            return [_json_plan_row(plan)]
        common = plan.get("common", {})
        rows = plan.get("rows")
        if not isinstance(common, dict):
            raise ValueError(f"JSON matrix plan common must be an object: {path}")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"JSON matrix plan rows must be a non-empty list: {path}")
        return [_json_plan_row({**common, **row}) for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_plan_row(row: dict[str, Any]) -> dict[str, str]:
    model = row.get("model_key") or row.get("model") or row.get("route")
    feature_route = str(row.get("feature_route") or row.get("route") or "")
    route_models = _json_plan_route_models(feature_route)
    if model in route_models:
        model = route_models[str(model)]
    return {
        "rounds": str(row.get("rounds", "")),
        "seed": str(row.get("seed", "")),
        "model_key": str(model or ""),
        "samples_per_class": str(row.get("samples_per_class", "")),
        "pairs_per_sample": str(row.get("pairs_per_sample", "")),
        "negative_mode": str(row.get("negative_mode", "")),
        "sample_structure": str(row.get("sample_structure", "")),
        "key_rotation_interval": str(row.get("key_rotation_interval", "")),
        "difference_profile": str(row.get("difference_profile", "")),
        "difference_member": str(row.get("difference_member", "")),
    }


def _json_plan_route_models(feature_route: str) -> dict[str, str]:
    if feature_route == "bit_transition_spectrum":
        return {
            "linear": "bit_transition_spectrum_linear",
            "mlp": "bit_transition_spectrum_mlp",
            "shuffled_p": "bit_transition_spectrum_shuffled_p",
        }
    if feature_route == "trail_family_consistency":
        return {
            "linear": "trail_family_consistency_linear",
            "mlp": "trail_family_consistency_mlp",
            "false_family": "trail_family_consistency_false_family",
        }
    return {
        "linear": "candidate_trail_consistency_linear",
        "mlp": "candidate_trail_consistency_mlp",
        "shuffled_cells": "candidate_trail_consistency_shuffled_cells",
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _optional_alignment_key_fields(
    plan_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    fields = []
    for field in ("difference_profile", "difference_member"):
        if _any_nonempty(plan_rows, field) and _any_nonempty(result_rows, field):
            fields.append(field)
    return tuple(fields)


def _any_nonempty(rows: list[dict[str, Any]], field: str) -> bool:
    return any(_normalize_value(row.get(field, "")) != "" for row in rows)


def _alignment_key(
    row: dict[str, Any],
    *,
    optional_key_fields: tuple[str, ...] = (),
) -> tuple[Any, ...]:
    key: list[Any] = [
        int(row["rounds"]),
        int(row["seed"]),
        _model_key(row),
        int(row["samples_per_class"]),
        _normalize_value(row.get("feature_encoding", "")),
        _selected_indices_key(row),
    ]
    key.extend(_normalize_value(row.get(field, "")) for field in optional_key_fields)
    return tuple(key)


def _model_key(row: dict[str, Any]) -> str:
    for field in ("model_key", "model", "selected_model"):
        if field in row and row[field] not in {None, ""}:
            return _normalize_value(row[field])
    return ""


def _selected_indices_key(row: dict[str, Any]) -> str:
    value = row.get("selected_bit_indices")
    if value is None and isinstance(row.get("training"), dict):
        value = row["training"].get("selected_bit_indices")
    if value is None or value == "":
        return "[]"
    if isinstance(value, (list, tuple)):
        return json.dumps([int(item) for item in value], separators=(",", ":"))
    text = str(value).strip()
    if not text:
        return "[]"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return json.dumps([int(item) for item in parsed], separators=(",", ":"))
    return text


def _field_mismatches(
    plan_by_key: dict[tuple[Any, ...], dict[str, str]],
    result_rows: list[dict[str, Any]],
    *,
    optional_key_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for result_row in result_rows:
        key = _alignment_key(result_row, optional_key_fields=optional_key_fields)
        plan_row = plan_by_key.get(key)
        if not plan_row:
            continue
        for plan_field, result_field in PLAN_RESULT_FIELD_PAIRS:
            if plan_field not in plan_row:
                continue
            result_value_raw = _nested_get(result_row, result_field)
            if result_value_raw is None:
                continue
            if plan_field == "selected_bit_indices":
                plan_value = _selected_indices_key(plan_row)
                result_value = _selected_indices_key({"selected_bit_indices": result_value_raw})
            else:
                plan_value = _normalize_value(plan_row[plan_field])
                result_value = _normalize_value(result_value_raw)
            if plan_value != result_value:
                mismatches.append(
                    {
                        "key": key,
                        "plan_field": plan_field,
                        "result_field": _field_label(result_field),
                        "plan_value": plan_value,
                        "result_value": result_value,
                    }
                )
        if "model_key" in plan_row:
            model_values = {
                _normalize_value(result_row[field])
                for field in ("model", "selected_model")
                if field in result_row
            }
            if model_values and _normalize_value(plan_row["model_key"]) not in model_values:
                mismatches.append(
                    {
                        "key": key,
                        "plan_field": "model_key",
                        "result_field": "model/selected_model",
                        "plan_value": _normalize_value(plan_row["model_key"]),
                        "result_value": sorted(model_values),
                    }
                )
    return mismatches


def _normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value).strip()
    if text.lower().startswith("0x"):
        return str(int(text, 16))
    if text.lower() in {"true", "false"}:
        return text.lower()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _nested_get(row: dict[str, Any], field: str | tuple[str, ...]) -> Any:
    if isinstance(field, str):
        return row.get(field)
    current: Any = row
    for part in field:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _field_label(field: str | tuple[str, ...]) -> str:
    if isinstance(field, str):
        return field
    return ".".join(field)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_result_plan_alignment(
        Path(args.plan),
        Path(args.results),
        expected_rows=args.expected_rows,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
