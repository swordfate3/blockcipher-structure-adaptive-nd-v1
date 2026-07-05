from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.plan_next_action import plan_next_action


BRANCH_PRIORITIES = {
    "r9_1m_seed0_plan": 100,
    "r9_seed1_or_curriculum_scale_plan": 90,
    "r8_pairset_seed1_or_frozen_control": 80,
    "r8_pairset_weak_positive_review": 70,
    "r9_variance_or_aggregation_review": 60,
    "stop_from_scratch_r9_r10": 55,
    "baseline_best_no_candidate_scale": 50,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read multiple postprocess summaries and select one launchable "
            "next action without touching remote state."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Postprocess summary JSON path. Repeat for multiple candidates.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def arbitrate_next_actions(summary_paths: list[Path]) -> dict[str, Any]:
    reports = [_candidate_report(index, path) for index, path in enumerate(summary_paths)]
    ready = [report for report in reports if report["launchable"]]
    ready.sort(key=lambda item: (-item["priority"], item["index"], item["summary"]))

    selected = ready[0] if ready else None
    deferred = []
    if selected is not None:
        for report in ready[1:]:
            deferred.append(
                {
                    **report,
                    "defer_reason": (
                        f"lower_priority_than_selected:{selected['branch']} "
                        f"({report['priority']} < {selected['priority']})"
                    ),
                }
            )

    return {
        "status": "selected" if selected is not None else "no_launchable_action",
        "selected": selected,
        "deferred": deferred,
        "not_ready": [report for report in reports if not report["launchable"]],
        "policy": {
            "main_thread": "do_not_launch_blindly; launch only the selected plan-aligned branch",
            "priority_order": BRANCH_PRIORITIES,
            "tie_breaker": "input_order",
        },
    }


def _candidate_report(index: int, summary_path: Path) -> dict[str, Any]:
    try:
        report = plan_next_action(summary_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "index": index,
            "summary": str(summary_path),
            "status": "fail",
            "launchable": False,
            "priority": -1,
            "branch": None,
            "run_id": None,
            "decision": None,
            "reason": f"unreadable_or_invalid_summary:{exc}",
            "errors": [str(exc)],
        }

    branch = report.get("branch")
    priority = BRANCH_PRIORITIES.get(str(branch), 0)
    launchable = (
        report.get("status") == "pass"
        and bool(report.get("should_launch_remote"))
        and bool(report.get("readiness_pass"))
        and not bool(report.get("requires_implementation"))
        and priority > 0
    )
    reason = _readiness_reason(report, priority=priority)
    next_action = report.get("next_action")
    if not isinstance(next_action, dict):
        next_action = {}
    candidate_route_readiness = _candidate_route_readiness(summary_path)
    return {
        "index": index,
        "summary": str(summary_path),
        "status": report.get("status"),
        "launchable": launchable,
        "priority": priority,
        "branch": branch,
        "run_id": next_action.get("run_id") or report.get("run_id"),
        "decision": report.get("decision"),
        "action": report.get("action"),
        "claim_scope": report.get("claim_scope"),
        "launch_remote_config": next_action.get("launch_remote_config"),
        "readiness_pass": report.get("readiness_pass"),
        "requires_implementation": report.get("requires_implementation"),
        "should_launch_remote": report.get("should_launch_remote"),
        "reason": reason,
        "launch_checklist": report.get("launch_checklist", []),
        "candidate_route_readiness": candidate_route_readiness,
        "errors": report.get("errors", []),
    }


def _readiness_reason(report: dict[str, Any], *, priority: int) -> str:
    if report.get("status") != "pass":
        return "plan_next_action_status_not_pass"
    if not report.get("should_launch_remote"):
        return "summary_does_not_request_remote_launch"
    if report.get("requires_implementation"):
        return "requires_implementation_before_launch"
    if not report.get("readiness_pass"):
        return "remote_readiness_or_launch_artifacts_not_ready"
    if priority <= 0:
        return "branch_not_in_high_round_priority_policy"
    return "launchable"


def _candidate_route_readiness(summary_path: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    path = summary.get("candidate_route_readiness")
    if not isinstance(path, str) or not path:
        return None
    try:
        readiness = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unavailable", "path": path}
    routes = readiness.get("candidate_routes")
    if not isinstance(routes, dict):
        return {"status": readiness.get("status"), "path": path, "candidate_routes": {}}
    return {
        "status": readiness.get("status"),
        "path": path,
        "policy": readiness.get("policy"),
        "candidate_routes": {
            str(name): _compact_candidate_route(value)
            for name, value in routes.items()
            if isinstance(value, dict)
        },
    }


def _compact_candidate_route(route: dict[str, Any]) -> dict[str, Any]:
    next_action = route.get("next_action")
    if not isinstance(next_action, dict):
        next_action = {}
    readiness_reports = route.get("readiness_reports")
    if not isinstance(readiness_reports, list):
        readiness_reports = []
    return {
        "status": route.get("status"),
        "branch": route.get("branch"),
        "should_launch_remote": route.get("should_launch_remote"),
        "requires_implementation": route.get("requires_implementation"),
        "readiness_pass": route.get("readiness_pass"),
        "remote_readiness_pass": route.get("remote_readiness_pass"),
        "launch_artifacts_pass": route.get("launch_artifacts_pass"),
        "run_id": next_action.get("run_id"),
        "launch_remote_config": next_action.get("launch_remote_config"),
        "readiness_reports": [
            {
                "role": item.get("role"),
                "config": item.get("config"),
                "readiness_status": (item.get("readiness") or {}).get("status")
                if isinstance(item.get("readiness"), dict)
                else None,
                "launch_artifacts_status": (item.get("launch_artifacts") or {}).get("status")
                if isinstance(item.get("launch_artifacts"), dict)
                else None,
            }
            for item in readiness_reports
            if isinstance(item, dict)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = arbitrate_next_actions(args.summary)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
