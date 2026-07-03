from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.spn_active_auxiliary import args_from_config, run_active_auxiliary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an active-pattern auxiliary-head JSON matrix and write one JSONL row per matrix row."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = _load_matrix_plan(args.config)
    output = args.output or Path(str(plan["output"]))
    rows = run_active_auxiliary_matrix(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_active_auxiliary_matrix(plan: dict[str, Any]) -> list[dict[str, Any]]:
    common = _common_plan(plan)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_matrix_rows(plan)):
        row_config = {**common, **row}
        row_type = str(row_config.get("row_type", "candidate"))
        if row_type == "external_anchor":
            rows.append(_external_anchor_row(row_config, index))
            continue
        if row_type != "candidate":
            raise SystemExit(f"unsupported active-auxiliary matrix row_type: {row_type}")
        result = run_active_auxiliary(args_from_config(row_config))
        result["matrix_row"] = index
        result["row_type"] = row_type
        rows.append(result)
    return rows


def _load_matrix_plan(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"active-auxiliary matrix config must be a JSON object: {path}")
    if "rows" not in data:
        raise SystemExit("active-auxiliary matrix config must contain rows")
    if "output" not in data:
        raise SystemExit("active-auxiliary matrix config must contain output")
    return data


def _common_plan(plan: dict[str, Any]) -> dict[str, Any]:
    common = plan.get("common", {})
    if not isinstance(common, dict):
        raise SystemExit("active-auxiliary matrix common must be a JSON object")
    return dict(common)


def _matrix_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("active-auxiliary matrix rows must be a non-empty list")
    if not all(isinstance(row, dict) for row in rows):
        raise SystemExit("active-auxiliary matrix rows must be JSON objects")
    return rows


def _external_anchor_row(config: dict[str, Any], index: int) -> dict[str, Any]:
    model = _required_str(config, "model")
    auc = _required_float(config, "anchor_auc")
    calibrated = config.get("anchor_calibrated_accuracy")
    accuracy = config.get("anchor_accuracy", calibrated)
    loss = config.get("anchor_loss")
    row = _base_result_fields(config)
    row.update(
        {
            "matrix_row": index,
            "row_type": "external_anchor",
            "route": model,
            "model": model,
            "selected_model": model,
            "metrics": {
                "auc": auc,
                "accuracy": _optional_float(accuracy),
                "calibrated_accuracy": _optional_float(calibrated),
                "loss": _optional_float(loss),
            },
        }
    )
    return row


def _base_result_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "rounds": int(config.get("rounds", 7)),
        "seed": int(config.get("seed", 0)),
        "samples_per_class": int(config.get("samples_per_class", 0)),
        "pairs_per_sample": int(config.get("pairs_per_sample", 16)),
        "negative_mode": config.get("negative_mode"),
        "sample_structure": config.get("sample_structure"),
        "difference_profile": config.get("difference_profile"),
        "difference_member": int(config.get("difference_member", 0)),
        "key_rotation_interval": int(config.get("key_rotation_interval", 0)),
        "feature_route": config.get("feature_route", "active_pattern_auxiliary_head"),
    }


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value in {None, ""}:
        raise SystemExit(f"active-auxiliary matrix row missing {key}")
    return str(value)


def _required_float(config: dict[str, Any], key: str) -> float:
    value = _optional_float(config.get(key))
    if value is None:
        raise SystemExit(f"active-auxiliary matrix row missing numeric {key}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
