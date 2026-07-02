from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_DDT_GRAPH_MODELS = {
    "present_nibble_invp_only_spn_only",
    "present_nibble_paligned_transition_residual",
    "present_nibble_no_ddt_graph",
    "present_nibble_ddt_graph",
    "present_nibble_shuffled_ddt_graph",
}
DEFAULT_DDT_GRAPH_MARGIN = 0.001
FLOAT_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate PRESENT DDT graph results against InvP/no-DDT/shuffled controls."
    )
    parser.add_argument("--results", required=True, type=Path, help="DDT graph result JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON gate report path.")
    parser.add_argument("--expected-rows", type=int, default=5)
    parser.add_argument("--margin", type=float, default=DEFAULT_DDT_GRAPH_MARGIN)
    return parser.parse_args(argv)


def gate_ddt_graph_result(
    results_path: Path,
    *,
    expected_rows: int = 5,
    margin: float = DEFAULT_DDT_GRAPH_MARGIN,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    models: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = _model_key(row)
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
        auc = _float_metric(metrics, "auc", errors, model)
        models[model] = {
            "auc": auc,
            "accuracy": _optional_float_metric(metrics, "accuracy"),
            "calibrated_accuracy": _optional_float_metric(metrics, "calibrated_accuracy"),
            "loss": _optional_float_metric(metrics, "loss"),
        }

    missing = sorted(EXPECTED_DDT_GRAPH_MODELS - set(models))
    unexpected = sorted(set(models) - EXPECTED_DDT_GRAPH_MODELS)
    if missing:
        errors.append(f"missing_models={missing}")
    if unexpected:
        warnings.append(f"unexpected_models={unexpected}")

    ddt = models.get("present_nibble_ddt_graph", {})
    invp = models.get("present_nibble_invp_only_spn_only", {})
    transition_no_ddt = models.get("present_nibble_paligned_transition_residual", {})
    graph_no_ddt = models.get("present_nibble_no_ddt_graph", {})
    shuffled = models.get("present_nibble_shuffled_ddt_graph", {})
    ddt_auc = ddt.get("auc")
    control_aucs = [
        value
        for value in [
            invp.get("auc"),
            transition_no_ddt.get("auc"),
            graph_no_ddt.get("auc"),
            shuffled.get("auc"),
        ]
        if value is not None
    ]
    max_control_auc = max(control_aucs) if control_aucs else None
    margin_vs_best_control = (
        None if ddt_auc is None or max_control_auc is None else ddt_auc - max_control_auc
    )
    margin_vs_invp = _metric_delta(ddt.get("auc"), invp.get("auc"))
    margin_vs_transition_no_ddt = _metric_delta(ddt.get("auc"), transition_no_ddt.get("auc"))
    margin_vs_no_ddt_graph = _metric_delta(ddt.get("auc"), graph_no_ddt.get("auc"))
    margin_vs_shuffled = _metric_delta(ddt.get("auc"), shuffled.get("auc"))
    calibrated_delta_vs_invp = _metric_delta(
        ddt.get("calibrated_accuracy"),
        invp.get("calibrated_accuracy"),
    )

    decision = "invalid"
    action = "fix_result_or_alignment_before_ddt_graph_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and margin_vs_best_control is not None:
        calibrated_not_worse = calibrated_delta_vs_invp is None or _at_least(calibrated_delta_vs_invp, 0.0)
        beats_best_control = margin_vs_best_control > FLOAT_TOLERANCE
        if _at_least(margin_vs_best_control, margin) and calibrated_not_worse:
            decision = "support_ddt_graph_route"
            action = "run_262k_seed1_confirmation_before_1m_scale"
            interpretation = (
                "DDT graph is above InvP/transition/no-DDT-graph/shuffled controls by the required margin; "
                "this is medium-scale diagnostic support for the DDT/topology route"
            )
        elif beats_best_control and calibrated_not_worse:
            decision = "weak_ddt_graph_signal"
            action = "run_prepared_262k_seed1_variance_check_before_scaling"
            interpretation = (
                "DDT graph is best but below the required margin; treat as weak diagnostic signal"
            )
        else:
            decision = "stop_ddt_graph_route"
            action = "record_negative_or_tied_evidence_and_switch_hypothesis"
            interpretation = (
                "DDT graph does not beat the strongest same-budget control, or calibrated accuracy regresses; "
                "do not scale this route to 1M"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "expected_rows": expected_rows,
        "result_rows": len(rows),
        "expected_models": sorted(EXPECTED_DDT_GRAPH_MODELS),
        "models": models,
        "max_control_auc": max_control_auc,
        "margin_vs_best_control_auc": margin_vs_best_control,
        "margin_vs_invp_auc": margin_vs_invp,
        "margin_vs_transition_no_ddt_auc": margin_vs_transition_no_ddt,
        "margin_vs_no_ddt_graph_auc": margin_vs_no_ddt_graph,
        "margin_vs_no_ddt_auc": margin_vs_no_ddt_graph,
        "margin_vs_shuffled_auc": margin_vs_shuffled,
        "calibrated_delta_vs_invp": calibrated_delta_vs_invp,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "262144/class medium diagnostic DDT graph gate; "
            "not paper-scale, formal, or breakthrough evidence"
        ),
        "errors": errors,
        "warnings": warnings,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _model_key(row: dict[str, Any]) -> str:
    for key in ("model", "selected_model", "model_key"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _float_metric(metrics: dict[str, Any], key: str, errors: list[str], model: str) -> float | None:
    value = _optional_float_metric(metrics, key)
    if value is None:
        errors.append(f"model={model} missing_or_invalid_metric={key}")
    return value


def _optional_float_metric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def _metric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - FLOAT_TOLERANCE


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_ddt_graph_result(
        args.results,
        expected_rows=args.expected_rows,
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
