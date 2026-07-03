from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TRAIL_FAMILY_MARGIN = 0.001
FLOAT_TOLERANCE = 1e-12
DEFAULT_ANCHOR_MODEL = "present_nibble_invp_only_spn_only"
DEFAULT_CANDIDATE_MODELS = (
    "trail_family_consistency_linear",
    "trail_family_consistency_mlp",
)
DEFAULT_FALSE_FAMILY_MODEL = "trail_family_consistency_false_family"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate trail-family-consistency routes against an InvP anchor.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--candidate-model", action="append", default=[])
    parser.add_argument("--false-family-model", default=DEFAULT_FALSE_FAMILY_MODEL)
    parser.add_argument("--require-false-family-control", action="store_true")
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--anchor-calibrated-accuracy", type=float, default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_TRAIL_FAMILY_MARGIN)
    return parser.parse_args(argv)


def gate_trail_family_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_models: tuple[str, ...] = DEFAULT_CANDIDATE_MODELS,
    false_family_model: str = DEFAULT_FALSE_FAMILY_MODEL,
    require_false_family_control: bool = False,
    anchor_auc: float | None = None,
    anchor_calibrated_accuracy: float | None = None,
    margin: float = DEFAULT_TRAIL_FAMILY_MARGIN,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    warnings: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    models = {_model_key(row): _metrics(row) for row in rows}
    candidate_metrics = {model: models[model] for model in candidate_models if model in models}
    if not candidate_metrics:
        errors.append(f"missing_candidate_models={list(candidate_models)}")

    anchor_metrics = models.get(anchor_model)
    resolved_anchor_auc = anchor_auc if anchor_auc is not None else _metric(anchor_metrics, "auc")
    if resolved_anchor_auc is None:
        errors.append(f"missing_anchor_auc for {anchor_model}")
    resolved_anchor_calibrated = (
        anchor_calibrated_accuracy
        if anchor_calibrated_accuracy is not None
        else _metric(anchor_metrics, "calibrated_accuracy")
    )

    best_model, best_metrics = _best_auc_model(candidate_metrics)
    best_auc = _metric(best_metrics, "auc")
    best_calibrated = _metric(best_metrics, "calibrated_accuracy")
    if candidate_metrics and best_auc is None:
        errors.append(f"missing_candidate_auc={list(candidate_metrics)}")
    if resolved_anchor_calibrated is None:
        errors.append(f"missing_anchor_calibrated_accuracy for {anchor_model}")
    if best_model is not None and best_calibrated is None:
        errors.append(f"missing_candidate_calibrated_accuracy for {best_model}")
    false_family_metrics = models.get(false_family_model)
    false_family_auc = _metric(false_family_metrics, "auc")
    if false_family_metrics is None:
        missing_control = f"missing_false_family_control={false_family_model}"
        if require_false_family_control:
            errors.append(missing_control)
        else:
            warnings.append(missing_control)

    margin_vs_anchor = _delta(best_auc, resolved_anchor_auc)
    margin_vs_false_family = _delta(best_auc, false_family_auc)
    calibrated_delta_vs_anchor = _delta(best_calibrated, resolved_anchor_calibrated)

    decision = "invalid"
    action = "fix_trail_family_gate_inputs_before_claim"
    interpretation = "gate cannot be evaluated"
    if not errors and margin_vs_anchor is not None:
        calibration_not_worse = calibrated_delta_vs_anchor is None or _at_least(calibrated_delta_vs_anchor, 0.0)
        false_family_not_matching = margin_vs_false_family is not None and _at_least(margin_vs_false_family, margin)
        candidate_beats_false_family = margin_vs_false_family is None or margin_vs_false_family > FLOAT_TOLERANCE
        if _at_least(margin_vs_anchor, margin) and calibration_not_worse and false_family_not_matching:
            decision = "support_trail_family_route"
            action = "run_262k_seed1_confirmation_before_1m_scale"
            interpretation = (
                "trail-family features beat the InvP anchor by the required diagnostic margin "
                "and separate from the false-family control"
            )
        elif _at_least(margin_vs_anchor, 0.0) and calibration_not_worse and candidate_beats_false_family:
            decision = "weak_trail_family_signal"
            action = "run_262k_variance_check_only_if_branch_selected"
            interpretation = (
                "trail-family features are at or above the InvP anchor but below the required margin, "
                "or lack a strong false-family-control margin"
            )
        else:
            decision = "stop_trail_family_route"
            action = "record_tied_or_negative_evidence_and_switch_hypothesis"
            interpretation = (
                "trail-family features do not beat the InvP anchor, calibration regresses, "
                "or the false-family control matches/exceeds the true route; do not scale this route"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "result_rows": len(rows),
        "expected_rows": expected_rows,
        "anchor_model": anchor_model,
        "candidate_models": list(candidate_models),
        "false_family_model": false_family_model,
        "require_false_family_control": require_false_family_control,
        "models": models,
        "best_candidate_model": best_model,
        "best_candidate_auc": best_auc,
        "anchor_auc": resolved_anchor_auc,
        "false_family_auc": false_family_auc,
        "margin_vs_anchor_auc": margin_vs_anchor,
        "margin_vs_false_family_auc": margin_vs_false_family,
        "best_candidate_calibrated_accuracy": best_calibrated,
        "anchor_calibrated_accuracy": resolved_anchor_calibrated,
        "calibrated_delta_vs_anchor": calibrated_delta_vs_anchor,
        "required_margin": margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": "trail-family diagnostic gate; not paper-scale, formal, or breakthrough evidence",
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
    candidate_models = tuple(args.candidate_model) if args.candidate_model else DEFAULT_CANDIDATE_MODELS
    report = gate_trail_family_result(
        args.results,
        expected_rows=args.expected_rows,
        anchor_model=args.anchor_model,
        candidate_models=candidate_models,
        false_family_model=args.false_family_model,
        require_false_family_control=args.require_false_family_control,
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
