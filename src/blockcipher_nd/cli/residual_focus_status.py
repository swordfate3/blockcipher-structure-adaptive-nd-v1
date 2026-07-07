from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_GATE = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_POOL_PLAN = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
DEFAULT_POOL_EVAL = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json")
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
) -> dict[str, Any]:
    action = _read_json_or_empty(action_plan)
    gate_report = _read_json_or_empty(gate)
    pool = _read_json_or_empty(pool_plan)
    pool_eval_report = _read_json_or_empty(pool_eval)
    planned_outputs = _planned_outputs(action)
    existing_outputs = [path for path in planned_outputs if path.exists()]
    missing_outputs = [str(path) for path in planned_outputs if not path.exists()]
    monitor_lines = _read_lines(monitor_dir / "monitor.log")
    latest_progress = _latest_progress(artifact_root)
    gate_status = str(gate_report.get("status", "missing" if not gate.exists() else ""))
    pool_status = str(pool.get("status", "missing" if not pool_plan.exists() else ""))
    pool_eval_status = str(pool_eval_report.get("status", "missing" if not pool_eval.exists() else ""))
    pool_eval_decision = str(pool_eval_report.get("decision", ""))
    should_run_pool = bool(pool.get("should_run_pool", False))
    missing_pool3_score_artifacts = [
        str(path) for path in pool_eval_report.get("missing_score_artifacts", [])
    ]
    status, branch = _status_and_branch(
        gate_status=gate_status,
        pool_status=pool_status,
        pool_eval_status=pool_eval_status,
        pool_eval_decision=pool_eval_decision,
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
        "missing_pool3_score_artifact_count": len(missing_pool3_score_artifacts),
        "missing_pool3_score_artifacts": missing_pool3_score_artifacts,
        "monitor_dir": str(monitor_dir),
        "latest_monitor_event": _latest_monitor_event(monitor_lines),
        "latest_progress": latest_progress,
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


def _status_and_branch(
    *,
    gate_status: str,
    pool_status: str,
    pool_eval_status: str,
    pool_eval_decision: str,
    should_run_pool: bool,
    planned_outputs: list[Path],
    missing_outputs: list[str],
) -> tuple[str, str]:
    if pool_eval_status == "pending" and pool_eval_decision == "wait_for_pool3_score_artifacts":
        return "pool3_scores_pending", "wait_for_pool3_score_artifacts"
    if pool_eval_status == "pass":
        return "pool_evaluated", "document_residual_guided_pool3_fixed_fusion"
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


def _latest_progress(artifact_root: Path) -> dict[str, Any] | None:
    progress_files = sorted(artifact_root.glob("**/*progress*.jsonl"))
    latest_row: dict[str, Any] | None = None
    latest_time = float("-inf")
    for path in progress_files:
        for row in _read_jsonl(path):
            row_time = float(row.get("time", 0.0)) if isinstance(row, dict) else 0.0
            if latest_row is None or row_time >= latest_time:
                latest_row = row
                latest_time = row_time
    return latest_row


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
        monitor_dir=args.monitor_dir,
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
