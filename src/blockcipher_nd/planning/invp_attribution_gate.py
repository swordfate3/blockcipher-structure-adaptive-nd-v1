from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INVP_SEED0_AUC = 0.797470988906
DEFAULT_INVP_SEED1_AUC = 0.797347588554
DEFAULT_ZHANG_WANG_1M_AUC = 0.793897025948
DEFAULT_ATTRIBUTION_MARGIN = 0.001
EXPECTED_CONTROL_MODELS = {
    "present_nibble_delta_only_spn_only",
    "present_nibble_shuffled_paligned_spn_only",
}
FLOAT_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate InvP-only attribution controls against completed InvP-only 1M evidence."
    )
    parser.add_argument("--results", required=True, type=Path, help="Attribution-control result JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON gate report path.")
    parser.add_argument("--expected-rows", type=int, default=2)
    parser.add_argument("--invp-seed0-auc", type=float, default=DEFAULT_INVP_SEED0_AUC)
    parser.add_argument("--invp-seed1-auc", type=float, default=DEFAULT_INVP_SEED1_AUC)
    parser.add_argument("--reference-auc", type=float, default=DEFAULT_ZHANG_WANG_1M_AUC)
    parser.add_argument("--margin", type=float, default=DEFAULT_ATTRIBUTION_MARGIN)
    return parser.parse_args(argv)


def gate_invp_attribution_controls(
    results_path: Path,
    *,
    expected_rows: int = 2,
    invp_seed0_auc: float = DEFAULT_INVP_SEED0_AUC,
    invp_seed1_auc: float = DEFAULT_INVP_SEED1_AUC,
    reference_auc: float = DEFAULT_ZHANG_WANG_1M_AUC,
    margin: float = DEFAULT_ATTRIBUTION_MARGIN,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    controls: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = _model_key(row)
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
        auc = _float_metric(metrics, "auc", errors, model)
        controls[model] = {
            "auc": auc,
            "accuracy": _optional_float_metric(metrics, "accuracy"),
            "calibrated_accuracy": _optional_float_metric(metrics, "calibrated_accuracy"),
            "loss": _optional_float_metric(metrics, "loss"),
            "delta_vs_reference_auc": None if auc is None else auc - reference_auc,
        }

    missing = sorted(EXPECTED_CONTROL_MODELS - set(controls))
    unexpected = sorted(set(controls) - EXPECTED_CONTROL_MODELS)
    if missing:
        errors.append(f"missing_control_models={missing}")
    if unexpected:
        warnings.append(f"unexpected_control_models={unexpected}")

    valid_control_aucs = [control["auc"] for control in controls.values() if control["auc"] is not None]
    invp_min_auc = min(invp_seed0_auc, invp_seed1_auc)
    invp_mean_auc = (invp_seed0_auc + invp_seed1_auc) / 2.0
    max_control_auc = max(valid_control_aucs) if valid_control_aucs else None
    attribution_margin = None if max_control_auc is None else invp_min_auc - max_control_auc

    decision = "invalid"
    action = "fix_result_or_alignment_before_attribution_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and attribution_margin is not None:
        if _at_least(attribution_margin, margin):
            decision = "support_invp_structural_attribution"
            action = "write_route_level_attribution_summary"
            interpretation = (
                "InvP-only remains above DeltaC-only and shuffled-P controls at paper scale; "
                "true InvP/P-layer alignment is supported as the useful SPN structure signal"
            )
        elif _at_least(attribution_margin, 0.0):
            decision = "weak_attribution_support"
            action = "add_variance_or_additional_controls_before_formal_claim"
            interpretation = (
                "InvP-only remains above controls but by less than the attribution margin; "
                "support is positive but too small for strong route attribution"
            )
        else:
            decision = "weaken_invp_structural_attribution"
            action = "switch_to_new_spn_structure_hypothesis_or_variance_audit"
            interpretation = (
                "At least one control reaches or exceeds the InvP-only confirmation band; "
                "the current InvP-only structural attribution claim is not supported"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "expected_rows": expected_rows,
        "result_rows": len(rows),
        "expected_control_models": sorted(EXPECTED_CONTROL_MODELS),
        "controls": controls,
        "reference_auc": reference_auc,
        "invp_seed0_auc": invp_seed0_auc,
        "invp_seed1_auc": invp_seed1_auc,
        "invp_min_auc": invp_min_auc,
        "invp_mean_auc": invp_mean_auc,
        "max_control_auc": max_control_auc,
        "attribution_margin": attribution_margin,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "1000000/class attribution-control gate against completed InvP-only seed0/seed1; "
            "not formal route evidence by itself"
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


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - FLOAT_TOLERANCE


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_invp_attribution_controls(
        args.results,
        expected_rows=args.expected_rows,
        invp_seed0_auc=args.invp_seed0_auc,
        invp_seed1_auc=args.invp_seed1_auc,
        reference_auc=args.reference_auc,
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
