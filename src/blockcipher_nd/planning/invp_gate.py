from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ZHANG_WANG_1M_AUC = 0.793897025948
DEFAULT_STRONG_DELTA = 0.003
DEFAULT_WEAK_DELTA = 0.001


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate an InvP-only 1M PRESENT/SPN result against a fixed reference AUC."
    )
    parser.add_argument("--results", required=True, help="Result JSONL path.")
    parser.add_argument("--output", default=None, help="Optional JSON gate report path.")
    parser.add_argument(
        "--reference-auc",
        type=float,
        default=DEFAULT_ZHANG_WANG_1M_AUC,
        help="Reference Zhang/Wang 1M AUC.",
    )
    parser.add_argument(
        "--strong-delta",
        type=float,
        default=DEFAULT_STRONG_DELTA,
        help="AUC delta required to launch seed1 confirmation.",
    )
    parser.add_argument(
        "--weak-delta",
        type=float,
        default=DEFAULT_WEAK_DELTA,
        help="AUC delta treated as weak positive survival.",
    )
    parser.add_argument(
        "--expected-model",
        default="present_nibble_invp_only_spn_only",
        help="Expected single result model key.",
    )
    parser.add_argument("--expected-rows", type=int, default=1)
    return parser.parse_args(argv)


def gate_invp_only_result(
    results_path: Path,
    *,
    reference_auc: float = DEFAULT_ZHANG_WANG_1M_AUC,
    strong_delta: float = DEFAULT_STRONG_DELTA,
    weak_delta: float = DEFAULT_WEAK_DELTA,
    expected_model: str = "present_nibble_invp_only_spn_only",
    expected_rows: int = 1,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    row = rows[0] if rows else {}
    model = str(row.get("model", row.get("selected_model", "")))
    if row and model != expected_model:
        errors.append(f"model={model} expected_model={expected_model}")

    metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
    auc = _float_metric(metrics, "auc", errors)
    calibrated_accuracy = _optional_float_metric(metrics, "calibrated_accuracy")
    accuracy = _optional_float_metric(metrics, "accuracy")
    loss = _optional_float_metric(metrics, "loss")
    delta = None if auc is None else auc - reference_auc

    decision = "invalid"
    action = "fix_result_or_alignment_before_branching"
    interpretation = "gate cannot be evaluated"
    if not errors and delta is not None:
        if delta >= strong_delta:
            decision = "launch_invp_seed1_confirmation"
            action = "launch_prepared_seed1_1m_config"
            interpretation = "strong single-seed paper-scale improvement over reference"
        elif delta >= weak_delta:
            decision = "run_seed1_before_claiming"
            action = "launch_prepared_seed1_1m_config"
            interpretation = "weak positive survival; stability evidence required"
        elif delta >= -weak_delta:
            decision = "enter_ddt_graph_route"
            action = "implement_ddt_graph_conditional_plan"
            interpretation = "medium-scale InvP signal is tied at paper scale"
        else:
            decision = "discard_invp_only_as_main_1m_candidate"
            action = "implement_ddt_graph_conditional_plan_or_return_to_baseline"
            interpretation = "InvP-only underperforms the reference at paper scale"

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "expected_rows": expected_rows,
        "result_rows": len(rows),
        "expected_model": expected_model,
        "model": model,
        "reference_auc": reference_auc,
        "auc": auc,
        "auc_delta": delta,
        "strong_delta": strong_delta,
        "weak_delta": weak_delta,
        "accuracy": accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "loss": loss,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "1000000/class single-seed gate only; not formal multi-seed evidence "
            "and not a breakthrough claim"
        ),
        "errors": errors,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _float_metric(metrics: dict[str, Any], key: str, errors: list[str]) -> float | None:
    value = _optional_float_metric(metrics, key)
    if value is None:
        errors.append(f"missing_or_invalid_metric={key}")
    return value


def _optional_float_metric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_invp_only_result(
        Path(args.results),
        reference_auc=args.reference_auc,
        strong_delta=args.strong_delta,
        weak_delta=args.weak_delta,
        expected_model=args.expected_model,
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
