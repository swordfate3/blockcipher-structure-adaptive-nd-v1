from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.advance_residual_focus_results import (
    DEFAULT_ACTION_PLAN,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_GATE_OUTPUT,
    DEFAULT_MONITOR_DIR,
    DEFAULT_OUTPUT as DEFAULT_ADVANCE_OUTPUT,
    DEFAULT_POOL_EVAL_OUTPUT,
    DEFAULT_POOL_OUTPUT,
    DEFAULT_REPAIR_OUTPUT,
    DEFAULT_STATUS_OUTPUT,
    advance_residual_focus_results,
)


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_watch_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only watcher for residual-focus retrieved outputs. It repeatedly "
            "runs advance-residual-focus-results and stops when the postprocess "
            "reaches a pass/hold terminal state."
        )
    )
    parser.add_argument("--action-plan", type=Path, default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--gate-output", type=Path, default=DEFAULT_GATE_OUTPUT)
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL_OUTPUT)
    parser.add_argument("--pool-eval-output", type=Path, default=DEFAULT_POOL_EVAL_OUTPUT)
    parser.add_argument("--repair-output", type=Path, default=DEFAULT_REPAIR_OUTPUT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--advance-output", type=Path, default=DEFAULT_ADVANCE_OUTPUT)
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def watch_residual_focus_results(
    *,
    action_plan: Path,
    gate_output: Path,
    pool_output: Path,
    status_output: Path,
    advance_output: Path,
    monitor_dir: Path,
    artifact_root: Path,
    pool_eval_output: Path = DEFAULT_POOL_EVAL_OUTPUT,
    repair_output: Path = DEFAULT_REPAIR_OUTPUT,
    output: Path | None = None,
    max_iterations: int = 0,
    sleep_seconds: float = 900.0,
) -> dict[str, Any]:
    iteration = 0
    last_advance: dict[str, Any] | None = None
    while True:
        iteration += 1
        last_advance = advance_residual_focus_results(
            action_plan=action_plan,
            gate_output=gate_output,
            pool_output=pool_output,
            pool_eval_output=pool_eval_output,
            repair_output=repair_output,
            status_output=status_output,
            monitor_dir=monitor_dir,
            artifact_root=artifact_root,
        )
        _write_json(advance_output, last_advance)
        pending_report = _watch_report(last_advance, iteration_count=iteration, terminal=False)
        if output is not None:
            _write_json(output, pending_report)
        if _is_terminal(last_advance):
            terminal_report = _watch_report(last_advance, iteration_count=iteration, terminal=True)
            if output is not None:
                _write_json(output, terminal_report)
            return terminal_report
        if max_iterations > 0 and iteration >= max_iterations:
            return _watch_report(last_advance, iteration_count=iteration, terminal=False)
        time.sleep(max(0.0, sleep_seconds))


def _is_terminal(report: dict[str, Any]) -> bool:
    return str(report.get("status", "")) in {"pass", "hold"}


def _watch_report(report: dict[str, Any], *, iteration_count: int, terminal: bool) -> dict[str, Any]:
    return {
        "status": str(report.get("status", "unknown")),
        "decision": str(report.get("decision", "")),
        "iteration_count": iteration_count,
        "terminal": terminal,
        "ran_gate": bool(report.get("ran_gate", False)),
        "ran_pool_planner": bool(report.get("ran_pool_planner", False)),
        "ran_pool_evaluator": bool(report.get("ran_pool_evaluator", False)),
        "pool_eval_status": str(report.get("pool_eval_status", "")),
        "pool_eval_decision": str(report.get("pool_eval_decision", "")),
        "missing_pool3_score_artifact_count": int(report.get("missing_pool3_score_artifact_count", 0)),
        "missing_pool3_score_artifacts": [
            str(path) for path in report.get("missing_pool3_score_artifacts", [])
        ],
        "latest_monitor_event": str(report.get("latest_monitor_event", "")),
        "latest_progress": report.get("latest_progress"),
        "progress_summary": report.get("progress_summary", {}),
        "progress_by_seed_split": report.get("progress_by_seed_split", []),
        "repair_plan": str(report.get("repair_plan", "")),
        "repair_status": str(report.get("repair_status", "")),
        "repair_decision": str(report.get("repair_decision", "")),
        "repair_source_summary": str(report.get("repair_source_summary", "")),
        "repair_context_current": bool(report.get("repair_context_current", False)),
        "repair_primary_branch": str(report.get("repair_primary_branch", "")),
        "source_selection_report_count": int(report.get("source_selection_report_count", 0)),
        "source_selection_existing_report_count": int(report.get("source_selection_existing_report_count", 0)),
        "source_selection_missing_report_count": int(report.get("source_selection_missing_report_count", 0)),
        "source_selection_missing_reports": [
            str(path) for path in report.get("source_selection_missing_reports", [])
        ],
        "source_selection_summary_status": str(report.get("source_selection_summary_status", "")),
        "source_selection_summary_decision": str(report.get("source_selection_summary_decision", "")),
        "source_selection_summary_output": str(report.get("source_selection_summary_output", "")),
        "planned_output_count": int(report.get("planned_output_count", 0)),
        "existing_planned_output_count": int(report.get("existing_planned_output_count", 0)),
        "missing_output_count": int(report.get("missing_output_count", 0)),
        "missing_outputs": [str(path) for path in report.get("missing_outputs", [])],
        "next_action": report.get("next_action", {}),
        "claim_scope": (
            "local residual-focus watch report only; does not SSH, sync, launch remote jobs, "
            "or make a formal/breakthrough SPN/PRESENT claim"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = watch_residual_focus_results(
        action_plan=args.action_plan,
        gate_output=args.gate_output,
        pool_output=args.pool_output,
        pool_eval_output=args.pool_eval_output,
        repair_output=args.repair_output,
        status_output=args.status_output,
        advance_output=args.advance_output,
        output=args.output,
        monitor_dir=args.monitor_dir,
        artifact_root=args.artifact_root,
        max_iterations=args.max_iterations,
        sleep_seconds=args.sleep_seconds,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
