from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PAIRSET_MARGIN = 0.001
FLOAT_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate learned pair-set consistency against frozen single-pair aggregation."
    )
    parser.add_argument("--learned-results", required=True, type=Path)
    parser.add_argument("--frozen-summary", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--learned-model", default="present_nibble_invp_pair_consistency_spn_only")
    parser.add_argument("--anchor-model", default="present_nibble_invp_only_spn_only")
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--expected-learned-rows", type=int, default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_PAIRSET_MARGIN)
    return parser.parse_args(argv)


def gate_pairset_aggregation_control(
    learned_results_path: Path,
    frozen_summary_path: Path,
    *,
    learned_model: str = "present_nibble_invp_pair_consistency_spn_only",
    anchor_model: str = "present_nibble_invp_only_spn_only",
    anchor_auc: float | None = None,
    expected_learned_rows: int | None = None,
    margin: float = DEFAULT_PAIRSET_MARGIN,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    learned_rows = _load_jsonl_rows(learned_results_path)
    if expected_learned_rows is not None and len(learned_rows) != expected_learned_rows:
        errors.append(f"learned_result_rows={len(learned_rows)} expected_rows={expected_learned_rows}")

    learned_by_model = {_model_key(row): row for row in learned_rows}
    learned_row = learned_by_model.get(learned_model)
    if learned_row is None:
        errors.append(f"missing_learned_model={learned_model}")
    anchor_row = learned_by_model.get(anchor_model)
    if anchor_row is None and anchor_auc is None:
        errors.append(f"missing_anchor_model={anchor_model} and anchor_auc not provided")

    learned_auc = _row_auc(learned_row, errors, learned_model)
    learned_accuracy = _row_metric(learned_row, "accuracy")
    learned_calibrated_accuracy = _row_metric(learned_row, "calibrated_accuracy")
    resolved_anchor_auc = anchor_auc if anchor_auc is not None else _row_auc(anchor_row, errors, anchor_model)

    frozen_summary = _load_json(frozen_summary_path)
    frozen_status = str(frozen_summary.get("status", ""))
    if frozen_status != "pass":
        errors.append(f"frozen_summary_status={frozen_status!r}")
    frozen_metrics = frozen_summary.get("metrics", {})
    frozen_auc = _metric_auc(frozen_metrics, errors, "frozen_summary")
    frozen_accuracy = _optional_float(frozen_metrics.get("accuracy"))
    frozen_calibrated_accuracy = _optional_float(frozen_metrics.get("calibrated_accuracy"))
    frozen_scope = str(frozen_summary.get("claim_scope", ""))
    if "not a learned pair-set model" not in frozen_scope:
        warnings.append("frozen_summary claim_scope does not explicitly say it is not learned")

    margin_vs_frozen = _delta(learned_auc, frozen_auc)
    margin_vs_anchor = _delta(learned_auc, resolved_anchor_auc)
    calibrated_delta_vs_frozen = _delta(learned_calibrated_accuracy, frozen_calibrated_accuracy)
    calibrated_delta_vs_anchor = _delta(learned_calibrated_accuracy, _row_metric(anchor_row, "calibrated_accuracy"))

    decision = "invalid"
    action = "fix_pairset_gate_inputs_before_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and margin_vs_frozen is not None and margin_vs_anchor is not None:
        if _at_least(margin_vs_frozen, margin) and _at_least(margin_vs_anchor, margin):
            decision = "support_learned_pairset_consistency"
            action = "repeat_262k_seed1_before_1m_pairset_scale"
            interpretation = (
                "learned pair-set consistency beats frozen single-pair aggregation "
                "and the InvP anchor by the required diagnostic margin"
            )
        elif _at_least(margin_vs_frozen, 0.0) and _at_least(margin_vs_anchor, 0.0):
            decision = "weak_pairset_consistency_signal"
            action = "repeat_262k_or_run_variance_check_before_scaling"
            interpretation = (
                "learned pair-set consistency is best but below the required margin"
            )
        else:
            decision = "stop_pairset_consistency_route"
            action = "treat_as_aggregation_or_diagnostic_context"
            interpretation = (
                "learned pair-set consistency does not beat frozen aggregation and the InvP anchor; "
                "do not scale this route as a main contribution"
            )

    return {
        "status": "pass" if not errors else "fail",
        "learned_results_path": str(learned_results_path),
        "frozen_summary_path": str(frozen_summary_path),
        "learned_result_rows": len(learned_rows),
        "expected_learned_rows": expected_learned_rows,
        "learned_model": learned_model,
        "anchor_model": anchor_model,
        "learned_auc": learned_auc,
        "frozen_auc": frozen_auc,
        "anchor_auc": resolved_anchor_auc,
        "margin_vs_frozen_auc": margin_vs_frozen,
        "margin_vs_anchor_auc": margin_vs_anchor,
        "learned_accuracy": learned_accuracy,
        "frozen_accuracy": frozen_accuracy,
        "learned_calibrated_accuracy": learned_calibrated_accuracy,
        "frozen_calibrated_accuracy": frozen_calibrated_accuracy,
        "calibrated_delta_vs_frozen": calibrated_delta_vs_frozen,
        "calibrated_delta_vs_anchor": calibrated_delta_vs_anchor,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "pair-set aggregation diagnostic gate; compares learned joint pair-set "
            "model to frozen single-pair aggregation, not formal route evidence"
        ),
        "errors": errors,
        "warnings": warnings,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object in {path}")
    return parsed


def _model_key(row: dict[str, Any]) -> str:
    for key in ("model", "selected_model", "model_key"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _row_auc(row: dict[str, Any] | None, errors: list[str], label: str) -> float | None:
    if row is None:
        return None
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
    return _metric_auc(metrics, errors, label)


def _metric_auc(metrics: Any, errors: list[str], label: str) -> float | None:
    if not isinstance(metrics, dict):
        errors.append(f"{label} missing_metrics")
        return None
    value = _optional_float(metrics.get("auc"))
    if value is None:
        errors.append(f"{label} missing_or_invalid_auc")
    return value


def _row_metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
    return _optional_float(metrics.get(key))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - FLOAT_TOLERANCE


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_pairset_aggregation_control(
        args.learned_results,
        args.frozen_summary,
        learned_model=args.learned_model,
        anchor_model=args.anchor_model,
        anchor_auc=args.anchor_auc,
        expected_learned_rows=args.expected_learned_rows,
        margin=args.margin,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
