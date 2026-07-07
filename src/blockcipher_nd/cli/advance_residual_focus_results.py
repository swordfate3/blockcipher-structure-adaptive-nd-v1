from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.evaluate_residual_guided_diverse_pool import (
    evaluate_residual_guided_diverse_pool,
)
from blockcipher_nd.cli.gate_residual_focus_262k import gate_residual_focus_262k
from blockcipher_nd.cli.plan_residual_guided_diverse_pool import plan_residual_guided_diverse_pool
from blockcipher_nd.cli.plan_residual_focus_repair import plan_residual_focus_repair
from blockcipher_nd.cli.residual_focus_status import residual_focus_status
from blockcipher_nd.cli.summarize_residual_axis_spectrum import (
    summarize_residual_axis_spectrum,
)


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_GATE_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_POOL_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
DEFAULT_POOL_EVAL_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json")
DEFAULT_REPAIR_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_repair_plan.json")
DEFAULT_STATUS_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_status.json")
DEFAULT_MONITOR_DIR = Path("outputs/remote_results/i1_present_r8_residual_focus_262k_retry1/monitor")
DEFAULT_ARTIFACT_ROOT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_advance_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local residual-focus postprocessing when retrieved outputs are ready. "
            "This command does not SSH, sync, launch, or modify remote state."
        )
    )
    parser.add_argument("--action-plan", type=Path, default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--gate-output", type=Path, default=DEFAULT_GATE_OUTPUT)
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL_OUTPUT)
    parser.add_argument("--pool-eval-output", type=Path, default=DEFAULT_POOL_EVAL_OUTPUT)
    parser.add_argument("--repair-output", type=Path, default=DEFAULT_REPAIR_OUTPUT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def advance_residual_focus_results(
    *,
    action_plan: Path,
    gate_output: Path,
    pool_output: Path,
    status_output: Path,
    monitor_dir: Path,
    artifact_root: Path,
    pool_eval_output: Path = DEFAULT_POOL_EVAL_OUTPUT,
    repair_output: Path = DEFAULT_REPAIR_OUTPUT,
) -> dict[str, Any]:
    initial_status = residual_focus_status(
        action_plan=action_plan,
        gate=gate_output,
        pool_plan=pool_output,
        pool_eval=pool_eval_output,
        repair_plan=repair_output,
        monitor_dir=monitor_dir,
        artifact_root=artifact_root,
    )
    ran_gate = False
    ran_pool_planner = False
    ran_pool_evaluator = False
    ran_source_selection_summary = False
    gate_report: dict[str, Any] = _read_json_or_empty(gate_output)
    pool_report: dict[str, Any] = _read_json_or_empty(pool_output)
    pool_eval_report: dict[str, Any] = _read_json_or_empty(pool_eval_output)
    repair_report: dict[str, Any] = _read_json_or_empty(repair_output)
    source_selection_summary_report = _source_selection_summary_when_ready(
        action_plan,
        artifact_root=artifact_root,
    )
    if source_selection_summary_report.get("status") in {"pass", "hold"}:
        ran_source_selection_summary = bool(source_selection_summary_report.get("wrote_summary", False))

    if initial_status["status"] == "outputs_ready_gate_needed":
        gate_report = gate_residual_focus_262k(action_plan=action_plan)
        _write_json(gate_output, gate_report)
        ran_gate = True

    if gate_report.get("status") in {"pass", "fail"}:
        pool_report = plan_residual_guided_diverse_pool(residual_focus_gate=gate_output)
        _write_json(pool_output, pool_report)
        ran_pool_planner = True

    if pool_report.get("should_run_pool") is True:
        candidate_pool_report = _evaluate_pool3_when_artifacts_ready(
            action_plan=action_plan,
            pool_plan=pool_output,
            pool_report=pool_report,
        )
        pool_eval_report = candidate_pool_report
        _write_json(pool_eval_output, pool_eval_report)
        if pool_eval_report.get("status") == "pass":
            ran_pool_evaluator = True

    if gate_report.get("status") == "fail":
        repair_report = plan_residual_focus_repair(summary=gate_output)
        _write_json(repair_output, repair_report)
    elif pool_eval_report.get("status") == "hold":
        repair_report = plan_residual_focus_repair(summary=pool_eval_output)
        _write_json(repair_output, repair_report)

    final_status = residual_focus_status(
        action_plan=action_plan,
        gate=gate_output,
        pool_plan=pool_output,
        pool_eval=pool_eval_output,
        repair_plan=repair_output,
        monitor_dir=monitor_dir,
        artifact_root=artifact_root,
    )
    _write_json(status_output, final_status)
    status, decision = _advance_status(
        final_status,
        ran_gate=ran_gate,
        ran_pool_planner=ran_pool_planner,
        ran_pool_evaluator=ran_pool_evaluator,
        pool_eval_report=pool_eval_report,
    )
    return {
        "status": status,
        "decision": decision,
        "ran_gate": ran_gate,
        "ran_pool_planner": ran_pool_planner,
        "ran_pool_evaluator": ran_pool_evaluator,
        "ran_source_selection_summary": ran_source_selection_summary,
        "initial_status": initial_status["status"],
        "final_status": final_status["status"],
        "gate_status": final_status["gate_status"],
        "gate_decision": final_status["gate_decision"],
        "pool_status": final_status["pool_status"],
        "pool_eval_status": str(pool_eval_report.get("status", "")),
        "pool_eval_decision": str(pool_eval_report.get("decision", "")),
        "repair_plan": final_status["repair_plan"],
        "repair_status": final_status["repair_status"],
        "repair_decision": final_status["repair_decision"],
        "source_selection_summary_output": str(source_selection_summary_report.get("output", "")),
        "source_selection_summary_status": str(source_selection_summary_report.get("status", "")),
        "source_selection_summary_decision": str(source_selection_summary_report.get("decision", "")),
        "source_selection_summary_missing_report_count": int(
            source_selection_summary_report.get("missing_report_count", 0)
        ),
        "source_selection_report_count": int(source_selection_summary_report.get("report_count", 0)),
        "source_selection_existing_report_count": int(
            source_selection_summary_report.get("existing_report_count", 0)
        ),
        "source_selection_missing_report_count": int(
            source_selection_summary_report.get("missing_report_count", 0)
        ),
        "source_selection_missing_reports": [
            str(path) for path in source_selection_summary_report.get("missing_reports", [])
        ],
        "missing_pool3_score_artifact_count": len(pool_eval_report.get("missing_score_artifacts", [])),
        "missing_pool3_score_artifacts": [str(path) for path in pool_eval_report.get("missing_score_artifacts", [])],
        "should_run_pool": final_status["should_run_pool"],
        "missing_output_count": final_status["missing_output_count"],
        "next_action": _advance_next_action(final_status, pool_eval_report),
        "claim_scope": (
            "local residual-focus postprocess advancement only; does not SSH, sync, launch "
            "remote jobs, or make a formal/breakthrough SPN/PRESENT claim"
        ),
    }


def _source_selection_summary_when_ready(action_plan: Path, *, artifact_root: Path) -> dict[str, Any]:
    plan = _read_json_or_empty(action_plan)
    if not plan:
        return {
            "status": "pending",
            "decision": "wait_for_residual_focus_action_plan",
            "output": "",
            "missing_report_count": 0,
        }
    output = Path(
        str(
            plan.get(
                "source_selection_summary_output",
                artifact_root / "residual_axis_spectrum_summary.json",
            )
        )
    )
    reports = _source_selection_report_paths(plan)
    existing = [path for path in reports if path.exists()]
    missing = [str(path) for path in reports if not path.exists()]
    if missing or not reports:
        return {
            "status": "pending",
            "decision": "wait_for_train_axis_spectrum_reports",
            "output": str(output),
            "report_count": len(reports),
            "existing_report_count": len(existing),
            "missing_reports": missing,
            "missing_report_count": len(missing),
        }
    summary = summarize_residual_axis_spectrum(
        spectrum_reports=reports,
        min_report_support=2,
    )
    _write_json(output, summary)
    return {
        "status": str(summary.get("status", "")),
        "decision": str(summary.get("decision", "")),
        "output": str(output),
        "report_count": len(reports),
        "existing_report_count": len(existing),
        "missing_reports": [],
        "missing_report_count": 0,
        "wrote_summary": True,
    }


def _source_selection_report_paths(plan: dict[str, Any]) -> list[Path]:
    reports: list[Path] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        outputs = seed_plan.get("source_selection_outputs", {})
        if not isinstance(outputs, dict):
            continue
        for key in ("train_residual_loss_axis_spectrum", "train_hard_error_axis_spectrum"):
            value = outputs.get(key)
            if value:
                reports.append(Path(str(value)))
    return reports


def _advance_status(
    final_status: dict[str, Any],
    *,
    ran_gate: bool,
    ran_pool_planner: bool,
    ran_pool_evaluator: bool,
    pool_eval_report: dict[str, Any],
) -> tuple[str, str]:
    if ran_pool_evaluator and pool_eval_report.get("status") == "pass":
        return "pass", "residual_guided_pool_evaluated"
    if pool_eval_report.get("status") == "hold":
        return "hold", "repair_residual_guided_pool3_before_scaleup"
    if pool_eval_report.get("status") == "pending" and pool_eval_report.get("decision") == "wait_for_pool3_score_artifacts":
        return "pending", "wait_for_pool3_score_artifacts"
    if final_status["should_run_pool"]:
        return "pass", "residual_guided_pool_ready"
    if final_status["gate_status"] == "fail":
        return "hold", "repair_residual_focus_before_pool"
    if ran_gate or ran_pool_planner:
        return "processed", "residual_focus_postprocess_updated"
    return "pending", "wait_for_residual_focus_outputs"


def _advance_next_action(final_status: dict[str, Any], pool_eval_report: dict[str, Any]) -> dict[str, Any]:
    if pool_eval_report.get("status") == "pending" and pool_eval_report.get("decision") == "wait_for_pool3_score_artifacts":
        return {
            "branch": "wait_for_pool3_score_artifacts",
            "should_launch_remote": False,
        }
    if pool_eval_report.get("status") == "hold":
        return {
            "branch": "repair_residual_guided_pool3_before_scaleup",
            "should_launch_remote": False,
        }
    return final_status["next_action"]


def _evaluate_pool3_when_artifacts_ready(
    *,
    action_plan: Path,
    pool_plan: Path,
    pool_report: dict[str, Any],
) -> dict[str, Any]:
    plan = _read_json_or_empty(action_plan)
    selected = str(pool_report.get("selected_residual_candidate", ""))
    if not selected:
        return {
            "status": "pending",
            "decision": "wait_for_pool3_selected_candidate",
            "missing_score_artifacts": [],
            "claim_scope": "local Pool 3 evaluator guard only; selected residual candidate is not available yet",
        }
    seed_reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        paths = _pool3_artifact_paths(seed_plan, selected_residual_candidate=selected)
        seed_missing = [str(path) for path in paths.values() if not path.exists()]
        missing.extend(seed_missing)
        if seed_missing:
            continue
        seed_report = evaluate_residual_guided_diverse_pool(
            pool_plan=pool_plan,
            trail_position_artifact=paths["trail_position_artifact"],
            raw117_artifact=paths["raw117_artifact"],
            residual_focus_artifact=paths["residual_focus_artifact"],
            uniform_control_artifact=paths["uniform_control_artifact"],
            labelshuffle_control_artifact=paths["labelshuffle_control_artifact"],
        )
        seed_report["seed"] = int(seed_plan.get("seed", len(seed_reports)))
        seed_reports.append(seed_report)
    if missing or not seed_reports:
        return {
            "status": "pending",
            "decision": "wait_for_pool3_score_artifacts",
            "selected_residual_candidate": selected,
            "missing_score_artifacts": missing,
            "seed_report_count": len(seed_reports),
            "claim_scope": (
                "local Pool 3 evaluator guard only; fixed-fusion evaluation waits until "
                "all per-seed score artifacts are present and aligned"
            ),
        }
    decisions = {str(report.get("decision", "")) for report in seed_reports}
    return {
        "status": "pass" if all(report.get("status") == "pass" for report in seed_reports) else "hold",
        "decision": (
            "residual_guided_pool3_fixed_fusion_evaluated"
            if decisions == {"support_residual_guided_pool3_fixed_fusion"}
            else "residual_guided_pool3_fixed_fusion_mixed_or_controlled"
        ),
        "selected_residual_candidate": selected,
        "seed_reports": seed_reports,
        "claim_scope": (
            "application-level medium diagnostic fixed-fusion summary across retrieved "
            "residual-focus seeds; not formal SPN/PRESENT evidence and not a breakthrough claim"
        ),
    }


def _pool3_artifact_paths(
    seed_plan: dict[str, Any],
    *,
    selected_residual_candidate: str,
) -> dict[str, Path]:
    artifact_root = Path(str(seed_plan.get("artifact_root", "")))
    return {
        "trail_position_artifact": Path(str(seed_plan.get("validation_trail_position_scores", ""))),
        "raw117_artifact": artifact_root / "validation_raw117_scores",
        "residual_focus_artifact": artifact_root / f"residual_{selected_residual_candidate}_validation_scores",
        "uniform_control_artifact": artifact_root / "residual_uniform_validation_scores",
        "labelshuffle_control_artifact": artifact_root / "residual_focus10_labelshuffle_validation_scores",
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = advance_residual_focus_results(
        action_plan=args.action_plan,
        gate_output=args.gate_output,
        pool_output=args.pool_output,
        pool_eval_output=args.pool_eval_output,
        repair_output=args.repair_output,
        status_output=args.status_output,
        monitor_dir=args.monitor_dir,
        artifact_root=args.artifact_root,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
