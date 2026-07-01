from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TOPOLOGY_MODELS = {
    "present_nibble_invp_only_spn_only",
    "present_nibble_invp_p_layer_graph_spn_only",
    "present_nibble_invp_shuffled_p_layer_graph_spn_only",
}
DEFAULT_TOPOLOGY_MARGIN = 0.001
FLOAT_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate PRESENT topology-aware network results against InvP and shuffled controls."
    )
    parser.add_argument("--results", required=True, type=Path, help="Topology-aware result JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON gate report path.")
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument("--margin", type=float, default=DEFAULT_TOPOLOGY_MARGIN)
    return parser.parse_args(argv)


def gate_topology_aware_result(
    results_path: Path,
    *,
    expected_rows: int = 3,
    margin: float = DEFAULT_TOPOLOGY_MARGIN,
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

    missing = sorted(EXPECTED_TOPOLOGY_MODELS - set(models))
    unexpected = sorted(set(models) - EXPECTED_TOPOLOGY_MODELS)
    if missing:
        errors.append(f"missing_models={missing}")
    if unexpected:
        warnings.append(f"unexpected_models={unexpected}")

    true_graph = models.get("present_nibble_invp_p_layer_graph_spn_only", {})
    invp = models.get("present_nibble_invp_only_spn_only", {})
    shuffled = models.get("present_nibble_invp_shuffled_p_layer_graph_spn_only", {})
    margin_vs_invp = _metric_delta(true_graph.get("auc"), invp.get("auc"))
    margin_vs_shuffled = _metric_delta(true_graph.get("auc"), shuffled.get("auc"))
    calibrated_delta_vs_invp = _metric_delta(
        true_graph.get("calibrated_accuracy"),
        invp.get("calibrated_accuracy"),
    )

    decision = "invalid"
    action = "fix_result_or_alignment_before_topology_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and margin_vs_invp is not None and margin_vs_shuffled is not None:
        calibrated_not_worse = calibrated_delta_vs_invp is None or _at_least(calibrated_delta_vs_invp, 0.0)
        if _at_least(margin_vs_invp, margin) and _at_least(margin_vs_shuffled, margin) and calibrated_not_worse:
            decision = "support_topology_aware_network_route"
            action = "run_262k_seed1_confirmation_before_1m_scale"
            interpretation = (
                "true-P graph beats InvP-only and shuffled-P controls by the required margin; "
                "this is medium-scale diagnostic support for topology-aware network structure"
            )
        elif _at_least(margin_vs_invp, 0.0) and _at_least(margin_vs_shuffled, 0.0) and calibrated_not_worse:
            decision = "weak_topology_aware_network_signal"
            action = "run_262k_seed1_variance_check_before_scaling"
            interpretation = (
                "true-P graph is best but below at least one required margin; treat as weak diagnostic signal"
            )
        else:
            decision = "stop_topology_aware_network_route"
            action = "record_negative_or_tied_evidence_and_switch_hypothesis"
            interpretation = (
                "true-P graph does not beat InvP-only and shuffled-P controls, or calibrated accuracy regresses; "
                "do not scale this architecture"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "expected_rows": expected_rows,
        "result_rows": len(rows),
        "expected_models": sorted(EXPECTED_TOPOLOGY_MODELS),
        "models": models,
        "margin_vs_invp_auc": margin_vs_invp,
        "margin_vs_shuffled_auc": margin_vs_shuffled,
        "calibrated_delta_vs_invp": calibrated_delta_vs_invp,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "262144/class medium diagnostic topology-aware network gate; "
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
    report = gate_topology_aware_result(
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
