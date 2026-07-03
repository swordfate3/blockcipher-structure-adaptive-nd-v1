from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SBOX_PRIOR_MARGIN = 0.001
FLOAT_TOLERANCE = 1e-12
DEFAULT_ANCHOR_MODEL = "present_nibble_invp_only_spn_only"
DEFAULT_CANDIDATE_MODEL = "present_nibble_invp_sbox_prior_gate"
DEFAULT_CONTROL_MODELS = (
    "present_nibble_invp_no_ddt_gate",
    "present_nibble_invp_shuffled_sbox_prior_gate",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate S-box transition-prior routes against InvP and controls.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--control-model", action="append", default=[])
    parser.add_argument("--require-controls", action="store_true")
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--anchor-calibrated-accuracy", type=float, default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_SBOX_PRIOR_MARGIN)
    return parser.parse_args(argv)


def gate_sbox_prior_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    control_models: tuple[str, ...] = DEFAULT_CONTROL_MODELS,
    require_controls: bool = False,
    anchor_auc: float | None = None,
    anchor_calibrated_accuracy: float | None = None,
    margin: float = DEFAULT_SBOX_PRIOR_MARGIN,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    warnings: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    models = {_model_key(row): _metrics(row) for row in rows}
    candidate_metrics = models.get(candidate_model)
    if candidate_metrics is None:
        errors.append(f"missing_candidate_model={candidate_model}")

    anchor_metrics = models.get(anchor_model)
    resolved_anchor_auc = anchor_auc if anchor_auc is not None else _metric(anchor_metrics, "auc")
    if resolved_anchor_auc is None:
        errors.append(f"missing_anchor_auc for {anchor_model}")
    resolved_anchor_calibrated = (
        anchor_calibrated_accuracy
        if anchor_calibrated_accuracy is not None
        else _metric(anchor_metrics, "calibrated_accuracy")
    )

    candidate_auc = _metric(candidate_metrics, "auc")
    candidate_calibrated = _metric(candidate_metrics, "calibrated_accuracy")
    if candidate_metrics is not None and candidate_auc is None:
        errors.append(f"missing_candidate_auc={candidate_model}")
    if resolved_anchor_calibrated is None:
        errors.append(f"missing_anchor_calibrated_accuracy for {anchor_model}")
    if candidate_metrics is not None and candidate_calibrated is None:
        errors.append(f"missing_candidate_calibrated_accuracy for {candidate_model}")

    controls: dict[str, dict[str, float | None]] = {}
    for model in control_models:
        metrics = models.get(model)
        if metrics is None:
            missing = f"missing_control_model={model}"
            if require_controls:
                errors.append(missing)
            else:
                warnings.append(missing)
        else:
            controls[model] = metrics

    best_control_model, best_control_metrics = _best_auc_model(controls)
    best_control_auc = _metric(best_control_metrics, "auc")
    margin_vs_anchor = _delta(candidate_auc, resolved_anchor_auc)
    margin_vs_best_control = _delta(candidate_auc, best_control_auc)
    calibrated_delta_vs_anchor = _delta(candidate_calibrated, resolved_anchor_calibrated)

    decision = "invalid"
    action = "fix_sbox_prior_gate_inputs_before_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and margin_vs_anchor is not None:
        calibration_not_worse = calibrated_delta_vs_anchor is None or _at_least(calibrated_delta_vs_anchor, 0.0)
        controls_strong = margin_vs_best_control is not None and _at_least(margin_vs_best_control, margin)
        controls_missing = margin_vs_best_control is None
        if _at_least(margin_vs_anchor, margin) and calibration_not_worse and controls_strong:
            decision = "support_sbox_prior_route"
            action = "prepare_262k_seed1_confirmation_before_1m_scale"
            interpretation = (
                "true S-box transition prior gate beats the InvP anchor by the required diagnostic margin "
                "and separates from no-DDT and shuffled-prior controls"
            )
        elif (
            _at_least(margin_vs_anchor, 0.0)
            and calibration_not_worse
            and not _at_least(margin_vs_anchor, margin)
            and margin_vs_best_control is not None
            and _at_least(margin_vs_best_control, 0.0)
        ):
            decision = "weak_sbox_prior_signal"
            action = "prepare_262k_variance_check_only_if_branch_selected"
            interpretation = (
                "true S-box transition prior gate is best but below the required diagnostic margin; "
                "treat as weak medium-scale evidence only"
            )
        elif _at_least(margin_vs_anchor, 0.0) and calibration_not_worse and controls_missing:
            decision = "weak_sbox_prior_signal"
            action = "prepare_262k_variance_check_only_if_branch_selected"
            interpretation = (
                "true S-box transition prior gate is at or above the InvP anchor but below the required margin, "
                "but missing controls prevent a route decision"
            )
        else:
            decision = "stop_sbox_prior_route"
            action = "record_tied_or_negative_evidence_and_switch_hypothesis"
            interpretation = (
                "true S-box transition prior gate does not beat the InvP anchor, calibration regresses, "
                "or the no-DDT or shuffled prior control matches/exceeds the true route; do not scale this route"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "result_rows": len(rows),
        "expected_rows": expected_rows,
        "anchor_model": anchor_model,
        "candidate_model": candidate_model,
        "control_models": list(control_models),
        "require_controls": require_controls,
        "models": models,
        "candidate_auc": candidate_auc,
        "anchor_auc": resolved_anchor_auc,
        "best_control_model": best_control_model,
        "best_control_auc": best_control_auc,
        "margin_vs_anchor_auc": margin_vs_anchor,
        "margin_vs_best_control_auc": margin_vs_best_control,
        "candidate_calibrated_accuracy": candidate_calibrated,
        "anchor_calibrated_accuracy": resolved_anchor_calibrated,
        "calibrated_delta_vs_anchor": calibrated_delta_vs_anchor,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": "S-box transition prior diagnostic gate; not paper-scale, formal, or breakthrough evidence",
        "errors": errors,
        "warnings": warnings,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _model_key(row: dict[str, Any]) -> str:
    for key in ("model", "selected_model", "model_key", "route"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    metrics = row["metrics"] if isinstance(row.get("metrics"), dict) else row
    return {
        "auc": _optional_float(metrics.get("auc") if "auc" in metrics else metrics.get("val_auc")),
        "accuracy": _optional_float(
            metrics.get("accuracy") if "accuracy" in metrics else metrics.get("val_accuracy")
        ),
        "calibrated_accuracy": _optional_float(metrics.get("calibrated_accuracy")),
        "loss": _optional_float(metrics.get("loss") if "loss" in metrics else metrics.get("val_loss")),
    }


def _best_auc_model(models: dict[str, dict[str, float | None]]) -> tuple[str | None, dict[str, float | None] | None]:
    best_name: str | None = None
    best_metrics: dict[str, float | None] | None = None
    best_auc: float | None = None
    for name, metrics in models.items():
        auc = _metric(metrics, "auc")
        if auc is not None and (best_auc is None or auc > best_auc):
            best_name = name
            best_metrics = metrics
            best_auc = auc
    return best_name, best_metrics


def _metric(metrics: dict[str, float | None] | None, key: str) -> float | None:
    if metrics is None:
        return None
    return metrics.get(key)


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
    control_models = tuple(args.control_model) if args.control_model else DEFAULT_CONTROL_MODELS
    report = gate_sbox_prior_result(
        args.results,
        expected_rows=args.expected_rows,
        anchor_model=args.anchor_model,
        candidate_model=args.candidate_model,
        control_models=control_models,
        require_controls=args.require_controls,
        anchor_auc=args.anchor_auc,
        anchor_calibrated_accuracy=args.anchor_calibrated_accuracy,
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
