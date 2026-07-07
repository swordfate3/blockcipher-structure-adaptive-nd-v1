from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_GATE = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_POOL_PLAN = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
DEFAULT_POOL_EVAL = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json")
DEFAULT_REPAIR_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_repair_plan.json")
DEFAULT_MONITOR_DIR = Path("outputs/remote_results/i1_present_r8_residual_focus_262k_retry1/monitor")
DEFAULT_ARTIFACT_ROOT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_status.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local-only status for the PRESENT r8 residual-focus 262k route."
    )
    parser.add_argument("--action-plan", type=Path, default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--pool-plan", type=Path, default=DEFAULT_POOL_PLAN)
    parser.add_argument("--pool-eval", type=Path, default=DEFAULT_POOL_EVAL)
    parser.add_argument("--repair-plan", type=Path, default=DEFAULT_REPAIR_PLAN)
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def residual_focus_status(
    *,
    action_plan: Path,
    gate: Path,
    pool_plan: Path,
    monitor_dir: Path,
    artifact_root: Path,
    pool_eval: Path = DEFAULT_POOL_EVAL,
    repair_plan: Path = DEFAULT_REPAIR_PLAN,
) -> dict[str, Any]:
    action = _read_json_or_empty(action_plan)
    gate_report = _read_json_or_empty(gate)
    pool = _read_json_or_empty(pool_plan)
    pool_eval_report = _read_json_or_empty(pool_eval)
    repair_report = _read_json_or_empty(repair_plan)
    source_selection_summary = _source_selection_summary(action, artifact_root)
    source_selection_reports = _source_selection_report_paths(action)
    existing_source_selection_reports = [path for path in source_selection_reports if path.exists()]
    missing_source_selection_reports = [
        str(path) for path in source_selection_reports if not path.exists()
    ]
    planned_outputs = _planned_outputs(action)
    existing_outputs = [path for path in planned_outputs if path.exists()]
    missing_outputs = [str(path) for path in planned_outputs if not path.exists()]
    monitor_lines = _read_lines(monitor_dir / "monitor.log")
    progress_rows = _progress_rows(artifact_root)
    latest_progress = _latest_progress(progress_rows)
    gate_status = str(gate_report.get("status", "missing" if not gate.exists() else ""))
    pool_status = str(pool.get("status", "missing" if not pool_plan.exists() else ""))
    pool_eval_status = str(pool_eval_report.get("status", "missing" if not pool_eval.exists() else ""))
    pool_eval_decision = str(pool_eval_report.get("decision", ""))
    repair_status = str(repair_report.get("status", "missing" if not repair_plan.exists() else ""))
    repair_primary_branch = str(repair_report.get("primary_repair_branch", ""))
    repair_source_summary = str(repair_report.get("source_summary", ""))
    repair_matches_current_context = repair_source_summary in {str(gate), str(pool_plan), str(pool_eval)}
    effective_repair_status = (
        "stale" if repair_status == "ready" and not repair_matches_current_context else repair_status
    )
    should_run_pool = bool(pool.get("should_run_pool", False))
    missing_pool3_score_artifacts = [
        str(path) for path in pool_eval_report.get("missing_score_artifacts", [])
    ]
    status, branch = _status_and_branch(
        gate_status=gate_status,
        pool_status=pool_status,
        pool_eval_status=pool_eval_status,
        pool_eval_decision=pool_eval_decision,
        repair_status=effective_repair_status,
        repair_primary_branch=repair_primary_branch,
        repair_matches_current_context=repair_matches_current_context,
        should_run_pool=should_run_pool,
        planned_outputs=planned_outputs,
        missing_outputs=missing_outputs,
    )
    return {
        "status": status,
        "action_plan": str(action_plan),
        "gate": str(gate),
        "gate_status": gate_status,
        "gate_decision": str(gate_report.get("decision", "")),
        "pool_plan": str(pool_plan),
        "pool_status": pool_status,
        "should_run_pool": should_run_pool,
        "pool_eval": str(pool_eval),
        "pool_eval_status": pool_eval_status,
        "pool_eval_decision": pool_eval_decision,
        "repair_plan": str(repair_plan),
        "repair_status": effective_repair_status,
        "repair_decision": str(repair_report.get("decision", "")),
        "repair_source_summary": repair_source_summary,
        "repair_context_current": repair_matches_current_context,
        "repair_primary_branch": repair_primary_branch,
        "repair_hints": [str(hint) for hint in repair_report.get("repair_hints", [])],
        "missing_pool3_score_artifact_count": len(missing_pool3_score_artifacts),
        "missing_pool3_score_artifacts": missing_pool3_score_artifacts,
        "source_selection_summary_output": source_selection_summary["output"],
        "source_selection_summary_status": source_selection_summary["status"],
        "source_selection_summary_decision": source_selection_summary["decision"],
        "source_selection_recommended_feature_prefixes": source_selection_summary[
            "recommended_feature_prefixes"
        ],
        "source_selection_selected_groups": source_selection_summary["selected_groups"],
        "source_selection_report_count": len(source_selection_reports),
        "source_selection_existing_report_count": len(existing_source_selection_reports),
        "source_selection_missing_report_count": len(missing_source_selection_reports),
        "source_selection_missing_reports": missing_source_selection_reports,
        "monitor_dir": str(monitor_dir),
        "latest_monitor_event": _latest_monitor_event(monitor_lines),
        "latest_progress": latest_progress,
        "progress_summary": _progress_summary(latest_progress),
        "progress_by_seed_split": _progress_by_seed_split(progress_rows),
        "planned_output_count": len(planned_outputs),
        "existing_planned_output_count": len(existing_outputs),
        "missing_output_count": len(missing_outputs),
        "missing_outputs": missing_outputs,
        "next_action": {
            "branch": branch,
            "should_launch_remote": False,
        },
        "claim_scope": (
            "local residual-focus status summary only; does not SSH, launch remote jobs, "
            "run gates, or prove a medium/formal SPN/PRESENT claim"
        ),
    }


def _source_selection_report_paths(action_plan: dict[str, Any]) -> list[Path]:
    reports: list[Path] = []
    for seed_plan in action_plan.get("seeds", []):
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


def _source_selection_summary(action_plan: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    output = Path(
        str(
            action_plan.get(
                "source_selection_summary_output",
                artifact_root / "residual_axis_spectrum_summary.json",
            )
        )
    )
    if not output.exists():
        return {
            "output": str(output),
            "status": "missing",
            "decision": "wait_for_train_axis_spectrum_summary",
            "recommended_feature_prefixes": [],
            "selected_groups": [],
        }
    payload = _read_json_or_empty(output)
    return {
        "output": str(output),
        "status": str(payload.get("status", "")),
        "decision": str(payload.get("decision", "")),
        "recommended_feature_prefixes": [
            str(prefix) for prefix in payload.get("recommended_feature_prefixes", [])
        ],
        "selected_groups": [str(group) for group in payload.get("selected_groups", [])],
    }


def _status_and_branch(
    *,
    gate_status: str,
    pool_status: str,
    pool_eval_status: str,
    pool_eval_decision: str,
    repair_status: str,
    repair_primary_branch: str,
    repair_matches_current_context: bool,
    should_run_pool: bool,
    planned_outputs: list[Path],
    missing_outputs: list[str],
) -> tuple[str, str]:
    if (
        repair_status == "ready"
        and repair_primary_branch
        and repair_matches_current_context
        and (gate_status == "fail" or pool_status == "hold" or pool_eval_status == "hold")
    ):
        return "repair_ready", repair_primary_branch
    if pool_eval_status == "pending" and pool_eval_decision == "wait_for_pool3_score_artifacts":
        return "pool3_scores_pending", "wait_for_pool3_score_artifacts"
    if pool_eval_status == "pass":
        return "pool_evaluated", "document_residual_guided_pool3_fixed_fusion"
    if pool_eval_status == "hold":
        return "pool3_control_hold", "repair_residual_guided_pool3_before_scaleup"
    if should_run_pool and pool_status == "pass":
        return "pool_ready", "instantiate_residual_guided_pool3_fixed_fusion"
    if gate_status == "pass":
        return "gate_passed_pool_plan_needed", "run_residual_guided_pool_planner"
    if gate_status == "fail":
        return "gate_failed", "repair_residual_focus_controls_before_scaleup"
    if planned_outputs and not missing_outputs:
        return "outputs_ready_gate_needed", "run_residual_focus_gate"
    return "running", "wait_for_residual_focus_outputs"


def _planned_outputs(action_plan: dict[str, Any]) -> list[Path]:
    outputs: list[Path] = []
    for seed in action_plan.get("seeds", []):
        if not isinstance(seed, dict):
            continue
        planned = seed.get("planned_outputs", {})
        if isinstance(planned, dict):
            outputs.extend(Path(str(value)) for value in planned.values())
    return sorted(set(outputs))


def _progress_rows(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(artifact_root.glob("**/*progress*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _latest_progress(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest_row: dict[str, Any] | None = None
    latest_time = float("-inf")
    for row in rows:
        row_time = float(row.get("time", 0.0))
        if latest_row is None or row_time >= latest_time:
            latest_row = row
            latest_time = row_time
    return latest_row


def _progress_by_seed_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[int | None, str], dict[str, Any]] = {}
    latest_time_by_key: dict[tuple[int | None, str], float] = {}
    for row in rows:
        key = (_int_or_none(row.get("seed")), str(row.get("split", "")))
        row_time = float(row.get("time", 0.0))
        if key not in latest_time_by_key or row_time >= latest_time_by_key[key]:
            latest_by_key[key] = row
            latest_time_by_key[key] = row_time
    return [
        _progress_summary(latest_by_key[key])
        for key in sorted(latest_by_key, key=lambda item: (-1 if item[0] is None else item[0], item[1]))
    ]


def _progress_summary(progress: dict[str, Any] | None) -> dict[str, Any]:
    if not progress:
        return {}
    rows_done = _int_or_none(progress.get("rows_done"))
    total_rows = _int_or_none(progress.get("total_rows"))
    class_rows_done = _int_or_none(progress.get("class_rows_done"))
    class_total = _int_or_none(progress.get("class_total"))
    summary: dict[str, Any] = {
        "event": str(progress.get("event", "")),
        "stage": str(progress.get("stage", "")),
        "seed": _int_or_none(progress.get("seed")),
        "split": str(progress.get("split", "")),
        "samples_per_class": _int_or_none(progress.get("samples_per_class")),
        "rows_done": rows_done,
        "total_rows": total_rows,
        "rows_remaining": _remaining(rows_done, total_rows),
        "total_progress_fraction": _fraction(rows_done, total_rows),
        "class_rows_done": class_rows_done,
        "class_total": class_total,
        "class_rows_remaining": _remaining(class_rows_done, class_total),
        "class_progress_fraction": _fraction(class_rows_done, class_total),
    }
    return summary


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _remaining(done: int | None, total: int | None) -> int | None:
    if done is None or total is None:
        return None
    return max(0, total - done)


def _fraction(done: int | None, total: int | None) -> float | None:
    if done is None or total is None or total <= 0:
        return None
    return done / total


def _latest_monitor_event(lines: list[str]) -> str | None:
    if not lines:
        return None
    latest = lines[-1].strip()
    parts = latest.split(" ", 1)
    return parts[1] if len(parts) == 2 else latest


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = residual_focus_status(
        action_plan=args.action_plan,
        gate=args.gate,
        pool_plan=args.pool_plan,
        pool_eval=args.pool_eval,
        repair_plan=args.repair_plan,
        monitor_dir=args.monitor_dir,
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
