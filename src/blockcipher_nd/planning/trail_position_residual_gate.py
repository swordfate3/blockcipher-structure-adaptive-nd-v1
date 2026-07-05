from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_MODEL = "present_trail_position_stats_pairset"
DEFAULT_GLOBAL_CONTROL_MODEL = "present_pairset_global_stats"
DEFAULT_TRAIL_POSITION_RESIDUAL_MARGIN = 0.01
FLOAT_TOLERANCE = 1e-12
MISMATCH_CONTROL_KINDS = {"active_nibble", "input_difference"}
ORDER_CONTROL_KINDS = {"pair_order"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate trail-position neural residual evidence against deterministic and mismatch controls."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--baseline-audit",
        action="append",
        required=True,
        type=Path,
        dest="baseline_audits",
        help="Trail-position split/control baseline JSON path. Repeat for multiple seeds.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--global-control-model", default=DEFAULT_GLOBAL_CONTROL_MODEL)
    parser.add_argument("--margin", type=float, default=DEFAULT_TRAIL_POSITION_RESIDUAL_MARGIN)
    return parser.parse_args(argv)


def gate_trail_position_residual(
    results_path: Path,
    *,
    baseline_audit_paths: list[Path],
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    global_control_model: str = DEFAULT_GLOBAL_CONTROL_MODEL,
    margin: float = DEFAULT_TRAIL_POSITION_RESIDUAL_MARGIN,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    warnings: list[str] = []
    neural_by_seed = _neural_metrics_by_seed(rows)
    audits_by_seed = _audit_metrics_by_seed(baseline_audit_paths, errors)

    seeds = sorted(set(neural_by_seed) | set(audits_by_seed))
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        neural = neural_by_seed.get(seed, {})
        audit = audits_by_seed.get(seed, {})
        candidate_auc = _metric(neural.get(candidate_model), "auc")
        global_control_auc = _metric(neural.get(global_control_model), "auc")
        deterministic_baseline_auc = audit.get("deterministic_baseline_auc")
        max_mismatch_control_auc = audit.get("max_mismatch_control_auc")
        max_order_control_auc = audit.get("max_order_control_auc")
        if candidate_auc is None:
            errors.append(f"seed={seed} missing_candidate_model={candidate_model}")
        if global_control_auc is None:
            errors.append(f"seed={seed} missing_global_control_model={global_control_model}")
        if deterministic_baseline_auc is None:
            errors.append(f"seed={seed} missing_deterministic_baseline_auc")
        if max_mismatch_control_auc is None:
            errors.append(f"seed={seed} missing_mismatch_control_auc")

        candidate_margin_vs_deterministic = _delta(candidate_auc, deterministic_baseline_auc)
        candidate_margin_vs_global = _delta(candidate_auc, global_control_auc)
        deterministic_margin_vs_mismatch = _delta(deterministic_baseline_auc, max_mismatch_control_auc)
        per_seed.append(
            {
                "seed": seed,
                "candidate_auc": candidate_auc,
                "global_control_auc": global_control_auc,
                "deterministic_baseline_auc": deterministic_baseline_auc,
                "max_mismatch_control_auc": max_mismatch_control_auc,
                "max_order_control_auc": max_order_control_auc,
                "candidate_margin_vs_deterministic_auc": candidate_margin_vs_deterministic,
                "candidate_margin_vs_global_auc": candidate_margin_vs_global,
                "deterministic_margin_vs_mismatch_auc": deterministic_margin_vs_mismatch,
                "mismatch_controls": audit.get("mismatch_controls", []),
                "order_controls": audit.get("order_controls", []),
            }
        )

    min_candidate_margin_vs_deterministic = _min_present(
        row["candidate_margin_vs_deterministic_auc"] for row in per_seed
    )
    min_candidate_margin_vs_global = _min_present(row["candidate_margin_vs_global_auc"] for row in per_seed)
    min_deterministic_margin_vs_mismatch = _min_present(
        row["deterministic_margin_vs_mismatch_auc"] for row in per_seed
    )
    pair_order_assessment = _pair_order_assessment(per_seed)

    decision = "invalid"
    action = "fix_trail_position_residual_gate_inputs_before_claim"
    interpretation = "gate cannot be evaluated"
    if not errors:
        candidate_clears_deterministic = _at_least(min_candidate_margin_vs_deterministic, margin)
        candidate_clears_global = _at_least(min_candidate_margin_vs_global, margin)
        deterministic_clears_mismatch = _at_least(min_deterministic_margin_vs_mismatch, margin)
        if candidate_clears_deterministic and candidate_clears_global and deterministic_clears_mismatch:
            decision = "support_trail_position_neural_residual_local"
            action = "run_controlled_local_medium_diagnostic_before_remote_launch"
            interpretation = (
                "trail-position neural candidate clears the deterministic split baseline, same-input "
                "global-stat control, and active/difference mismatch controls at the local diagnostic gate"
            )
        else:
            decision = "hold_trail_position_neural_residual_local"
            action = "record_local_hold_or_add_targeted_controls_before_scaling"
            interpretation = (
                "trail-position neural residual does not clear the deterministic, global-control, "
                "or mismatch-control margin on every checked seed"
            )

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "baseline_audit_paths": [str(path) for path in baseline_audit_paths],
        "candidate_model": candidate_model,
        "global_control_model": global_control_model,
        "required_margin": margin,
        "result_rows": len(rows),
        "seeds": seeds,
        "per_seed": per_seed,
        "min_candidate_margin_vs_deterministic_auc": min_candidate_margin_vs_deterministic,
        "min_candidate_margin_vs_global_auc": min_candidate_margin_vs_global,
        "min_deterministic_margin_vs_mismatch_auc": min_deterministic_margin_vs_mismatch,
        "pair_order_assessment": pair_order_assessment,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "PRESENT r8 trail-position local diagnostic gate only; not a Zhang/Wang r7 Case2 result, "
            "not paper-scale evidence, not remote-launch evidence, and not a breakthrough claim"
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


def _neural_metrics_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, float | None]]]:
    by_seed: dict[int, dict[str, dict[str, float | None]]] = {}
    for row in rows:
        seed = _int_or_none(row.get("seed"))
        if seed is None:
            continue
        by_seed.setdefault(seed, {})[_model_key(row)] = _metrics(row)
    return by_seed


def _audit_metrics_by_seed(paths: list[Path], errors: list[str]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline_report = payload.get("baseline", {}).get("report", payload.get("report", {}))
        seed = _int_or_none(baseline_report.get("seed"))
        if seed is None:
            errors.append(f"baseline_audit={path} missing_seed")
            continue
        baseline_auc = _composite_auc(baseline_report)
        mismatch_controls: list[dict[str, Any]] = []
        order_controls: list[dict[str, Any]] = []
        for control in payload.get("controls", []):
            control_auc = _composite_auc(control.get("report", {}))
            control_row = {
                "variant_kind": control.get("variant_kind"),
                "variant_label": control.get("variant_label"),
                "auc": control_auc,
            }
            if control.get("variant_kind") in MISMATCH_CONTROL_KINDS:
                mismatch_controls.append(control_row)
            if control.get("variant_kind") in ORDER_CONTROL_KINDS:
                order_controls.append(control_row)
        mismatch_aucs = [row["auc"] for row in mismatch_controls if row["auc"] is not None]
        order_aucs = [row["auc"] for row in order_controls if row["auc"] is not None]
        by_seed[seed] = {
            "deterministic_baseline_auc": baseline_auc,
            "max_mismatch_control_auc": max(mismatch_aucs) if mismatch_aucs else None,
            "max_order_control_auc": max(order_aucs) if order_aucs else None,
            "mismatch_controls": mismatch_controls,
            "order_controls": order_controls,
        }
    return by_seed


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


def _metric(metrics: dict[str, float | None] | None, key: str) -> float | None:
    if metrics is None:
        return None
    return metrics.get(key)


def _composite_auc(report: dict[str, Any]) -> float | None:
    return _optional_float(report.get("evaluation", {}).get("composite", {}).get("auc"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _min_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return min(present)


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold - FLOAT_TOLERANCE


def _pair_order_assessment(per_seed: list[dict[str, Any]]) -> str:
    order_deltas = [
        _delta(row.get("max_order_control_auc"), row.get("deterministic_baseline_auc"))
        for row in per_seed
        if row.get("max_order_control_auc") is not None
    ]
    if not order_deltas:
        return "pair_order_control_missing"
    if all(abs(delta or 0.0) <= FLOAT_TOLERANCE for delta in order_deltas):
        return "pair_order_not_bottleneck"
    if all((delta or 0.0) < 0.0 for delta in order_deltas):
        return "pair_order_control_below_baseline"
    return "pair_order_control_differs_from_baseline"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_trail_position_residual(
        args.results,
        baseline_audit_paths=args.baseline_audits,
        candidate_model=args.candidate_model,
        global_control_model=args.global_control_model,
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
