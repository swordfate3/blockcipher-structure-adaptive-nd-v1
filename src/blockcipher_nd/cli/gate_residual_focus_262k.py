from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
CANDIDATES = ("focus05", "focus10")
TRAIN_DERIVED_MODE = "train_derived_base_residual_loss_threshold"
REQUIRED_SLICE_KEYS = (
    "uniform_slice_eval",
    "focus10_shuffle_slice_eval",
    "focus05_slice_eval",
    "focus10_slice_eval",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate the PRESENT r8 262144/class residual-focused correction route "
            "after focus, uniform, and label-shuffle slice reports are available."
        )
    )
    parser.add_argument("--action-plan", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-focus-rows", type=int, default=1)
    parser.add_argument("--min-focus-auc-delta", type=float, default=0.0)
    parser.add_argument("--min-focus-loss-drop", type=float, default=0.0)
    parser.add_argument("--min-candidate-vs-uniform-loss-margin", type=float, default=0.0)
    parser.add_argument("--min-label-shuffle-loss-increase", type=float, default=0.0)
    return parser.parse_args(argv)


def gate_residual_focus_262k(
    *,
    action_plan: Path,
    min_focus_rows: int = 1,
    min_focus_auc_delta: float = 0.0,
    min_focus_loss_drop: float = 0.0,
    min_candidate_vs_uniform_loss_margin: float = 0.0,
    min_label_shuffle_loss_increase: float = 0.0,
) -> dict[str, Any]:
    plan = _load_json(action_plan)
    if plan.get("status") != "pass":
        return _report(
            status="pending",
            errors=["action_plan_not_ready"],
            runs=[],
            passing_candidates=[],
            action_plan=action_plan,
            source_decision=str(plan.get("source_decision", "")),
            source_gate_assessment=str(plan.get("source_gate_assessment", "")),
            missing_outputs=[],
        )

    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    candidate_pass_by_name = {candidate: True for candidate in CANDIDATES}
    control_errors: list[str] = []
    missing_outputs = _missing_outputs(plan)
    if missing_outputs:
        return _report(
            status="pending",
            errors=[],
            runs=[],
            passing_candidates=[],
            action_plan=action_plan,
            source_decision=str(plan.get("source_decision", "")),
            source_gate_assessment=str(plan.get("source_gate_assessment", "")),
            missing_outputs=missing_outputs,
        )
    for index, seed_plan in enumerate(plan.get("seeds", [])):
        if not isinstance(seed_plan, dict):
            errors.append(f"seed{index}: seed_plan_not_object")
            continue
        seed = int(seed_plan.get("seed", index))
        planned_outputs = seed_plan.get("planned_outputs", {})
        if not isinstance(planned_outputs, dict):
            errors.append(f"seed{seed}: planned_outputs_missing")
            continue
        run = _evaluate_seed(
            seed=seed,
            planned_outputs=planned_outputs,
            min_focus_rows=min_focus_rows,
            min_focus_auc_delta=min_focus_auc_delta,
            min_focus_loss_drop=min_focus_loss_drop,
            min_candidate_vs_uniform_loss_margin=min_candidate_vs_uniform_loss_margin,
            min_label_shuffle_loss_increase=min_label_shuffle_loss_increase,
        )
        runs.append(run)
        errors.extend(run["errors"])
        control_errors.extend(run["control_errors"])
        for candidate in CANDIDATES:
            if candidate in run["failed_candidates"]:
                candidate_pass_by_name[candidate] = False

    passing_candidates = [candidate for candidate in CANDIDATES if candidate_pass_by_name[candidate]]
    status = "pass" if passing_candidates and not control_errors else "fail"
    return _report(
        status=status,
        errors=errors,
        runs=runs,
        passing_candidates=passing_candidates,
        action_plan=action_plan,
        source_decision=str(plan.get("source_decision", "")),
        source_gate_assessment=str(plan.get("source_gate_assessment", "")),
        missing_outputs=[],
    )


def _missing_outputs(plan: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        planned_outputs = seed_plan.get("planned_outputs", {})
        if not isinstance(planned_outputs, dict):
            continue
        for key in REQUIRED_SLICE_KEYS:
            path = Path(str(planned_outputs.get(key, "")))
            if not path.exists():
                missing.append(str(path))
    return missing


def _evaluate_seed(
    *,
    seed: int,
    planned_outputs: dict[str, Any],
    min_focus_rows: int,
    min_focus_auc_delta: float,
    min_focus_loss_drop: float,
    min_candidate_vs_uniform_loss_margin: float,
    min_label_shuffle_loss_increase: float,
) -> dict[str, Any]:
    errors: list[str] = []
    control_errors: list[str] = []
    failed_candidates: list[str] = []
    uniform = _load_slice(planned_outputs, "uniform_slice_eval")
    shuffle = _load_slice(planned_outputs, "focus10_shuffle_slice_eval")
    uniform_loss_delta = _focus_loss_delta(uniform)
    shuffle_loss_delta = _focus_loss_delta(shuffle)
    if shuffle_loss_delta <= min_label_shuffle_loss_increase:
        message = f"seed{seed}: label_shuffle_did_not_worsen_focus_loss"
        errors.append(message)
        control_errors.append(message)

    candidate_reports: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        slice_report = _load_slice(planned_outputs, f"{candidate}_slice_eval")
        candidate_errors: list[str] = []
        rows = _focus_rows(slice_report)
        auc_delta = _focus_auc_delta(slice_report)
        loss_delta = _focus_loss_delta(slice_report)
        vs_uniform_loss_margin = loss_delta - uniform_loss_delta
        if _focus_mode(slice_report) != TRAIN_DERIVED_MODE:
            candidate_errors.append(f"seed{seed}: {candidate}_slice_not_train_derived")
        if rows < min_focus_rows:
            candidate_errors.append(f"seed{seed}: {candidate}_focus_rows_too_low")
        if auc_delta < min_focus_auc_delta:
            candidate_errors.append(f"seed{seed}: {candidate}_focus_auc_delta_too_low")
        if loss_delta >= -min_focus_loss_drop:
            candidate_errors.append(f"seed{seed}: {candidate}_focus_loss_not_reduced")
        if vs_uniform_loss_margin >= -min_candidate_vs_uniform_loss_margin:
            candidate_errors.append(f"seed{seed}: {candidate}_not_better_than_uniform_focus_loss")
        if candidate_errors:
            failed_candidates.append(candidate)
            errors.extend(candidate_errors)
        candidate_reports[candidate] = {
            "focus_rows": float(rows),
            "focus_auc_delta": auc_delta,
            "focus_loss_delta": loss_delta,
            "vs_uniform_focus_loss_margin": vs_uniform_loss_margin,
        }
    return {
        "seed": seed,
        "status": "fail" if errors else "pass",
        "errors": errors,
        "control_errors": control_errors,
        "failed_candidates": failed_candidates,
        "uniform_focus_loss_delta": uniform_loss_delta,
        "shuffle_focus_loss_delta": shuffle_loss_delta,
        "candidates": candidate_reports,
    }


def _load_slice(planned_outputs: dict[str, Any], key: str) -> dict[str, Any]:
    path = Path(str(planned_outputs[key]))
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _focus_mode(report: dict[str, Any]) -> str:
    focus = report.get("focus", {})
    return str(focus.get("mode", "")) if isinstance(focus, dict) else ""


def _focus_rows(report: dict[str, Any]) -> int:
    metrics = report.get("validation_focus_metrics", {})
    return int(metrics.get("rows", 0)) if isinstance(metrics, dict) else 0


def _focus_auc_delta(report: dict[str, Any]) -> float:
    delta = report.get("validation_focus_delta", {})
    return float(delta.get("auc", 0.0)) if isinstance(delta, dict) else 0.0


def _focus_loss_delta(report: dict[str, Any]) -> float:
    delta = report.get("validation_focus_delta", {})
    return float(delta.get("residual_loss_mean", 0.0)) if isinstance(delta, dict) else 0.0


def _report(
    *,
    status: str,
    errors: list[str],
    runs: list[dict[str, Any]],
    passing_candidates: list[str],
    action_plan: Path,
    source_decision: str,
    source_gate_assessment: str,
    missing_outputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": (
            "keep_residual_focus_262k_hard_slice_candidate"
            if status == "pass"
            else "hold_residual_focus_262k_controls_failed"
            if status == "fail"
            else "wait_for_residual_focus_262k_outputs"
        ),
        "action_plan": str(action_plan),
        "source_decision": source_decision,
        "source_gate_assessment": source_gate_assessment,
        "seed_count": len(runs),
        "passing_candidates": passing_candidates,
        "errors": errors,
        "missing_outputs": missing_outputs or [],
        "min_focus05_vs_uniform_loss_margin": _min_candidate_value(
            runs,
            "focus05",
            "vs_uniform_focus_loss_margin",
        ),
        "min_focus10_vs_uniform_loss_margin": _min_candidate_value(
            runs,
            "focus10",
            "vs_uniform_focus_loss_margin",
        ),
        "min_shuffle_focus_loss_delta": _min_run_value(runs, "shuffle_focus_loss_delta"),
        "runs": runs,
        "next_action": {
            "branch": (
                "residual_focus_1m_candidate_after_medium_confirmation"
                if status == "pass"
                else "repair_residual_focus_controls_before_scaleup"
                if status == "fail"
                else "finish_residual_focus_262k_outputs"
            ),
            "should_launch_remote": False,
        },
        "claim_scope": (
            "262144/class residual-focused hard-slice gate only; medium diagnostic "
            "SPN/PRESENT evidence, not formal evidence, not a breakthrough claim, "
            "and not a raw single-sample SOTA claim"
        ),
    }


def _min_run_value(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if key in run]
    return min(values) if values else None


def _min_candidate_value(runs: list[dict[str, Any]], candidate: str, key: str) -> float | None:
    values = [
        float(run["candidates"][candidate][key])
        for run in runs
        if candidate in run.get("candidates", {}) and key in run["candidates"][candidate]
    ]
    return min(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_residual_focus_262k(
        action_plan=args.action_plan,
        min_focus_rows=args.min_focus_rows,
        min_focus_auc_delta=args.min_focus_auc_delta,
        min_focus_loss_drop=args.min_focus_loss_drop,
        min_candidate_vs_uniform_loss_margin=args.min_candidate_vs_uniform_loss_margin,
        min_label_shuffle_loss_increase=args.min_label_shuffle_loss_increase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"pass", "pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
