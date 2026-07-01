from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a postprocess summary and emit the local next-action readiness plan."
    )
    parser.add_argument("--summary", required=True, type=Path, help="Postprocess summary JSON path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def plan_next_action(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    next_action = summary.get("next_action", {})
    if not isinstance(next_action, dict):
        next_action = {}

    readiness_reports: list[dict[str, Any]] = []
    for role, key in [
        ("stage_a", "stage_a_remote_config"),
        ("primary", "launch_remote_config"),
    ]:
        config = next_action.get(key)
        if not isinstance(config, str) or not config:
            continue
        readiness_reports.append(
            {
                "role": role,
                "config": config,
                "readiness": _readiness_report(Path(config)),
            }
        )

    should_launch_remote = bool(next_action.get("should_launch_remote"))
    readiness_statuses = [report["readiness"]["status"] for report in readiness_reports]
    readiness_pass = bool(readiness_reports) and all(status == "pass" for status in readiness_statuses)
    launch_checklist = _launch_checklist(
        should_launch_remote=should_launch_remote,
        readiness_pass=readiness_pass,
        next_action=next_action,
        readiness_reports=readiness_reports,
    )
    return {
        "status": "pass" if (not should_launch_remote or readiness_pass) else "fail",
        "summary": str(summary_path),
        "run_id": summary.get("run_id"),
        "decision": summary.get("decision"),
        "action": summary.get("action"),
        "branch": next_action.get("branch"),
        "should_launch_remote": should_launch_remote,
        "requires_implementation": bool(next_action.get("requires_implementation")),
        "readiness_pass": readiness_pass,
        "readiness_reports": readiness_reports,
        "launch_checklist": launch_checklist,
        "next_action": next_action,
        "claim_scope": summary.get("claim_scope"),
        "errors": _errors(should_launch_remote=should_launch_remote, readiness_reports=readiness_reports),
    }


def _errors(*, should_launch_remote: bool, readiness_reports: list[dict[str, Any]]) -> list[str]:
    if not should_launch_remote:
        return []
    if not readiness_reports:
        return ["next_action requests remote launch but no launch_remote_config was provided"]
    errors: list[str] = []
    for report in readiness_reports:
        readiness = report["readiness"]
        if readiness["status"] != "pass":
            errors.append(f"{report['role']} readiness failed: {readiness.get('errors', [])}")
    return errors


def _launch_checklist(
    *,
    should_launch_remote: bool,
    readiness_pass: bool,
    next_action: dict[str, Any],
    readiness_reports: list[dict[str, Any]],
) -> list[str]:
    if not should_launch_remote:
        return []
    if not readiness_pass:
        return ["Do not launch until all readiness reports pass."]
    configs = [report["config"] for report in readiness_reports]
    run_id = next_action.get("run_id") or "<next-run-id>"
    monitor_owner = next_action.get("monitor_owner") or "tmux watcher or sub-agent"
    return [
        f"Launch ready config(s) from the pushed commit: {', '.join(configs)}.",
        "Use the generated cmd.exe /c launcher under G:\\lxy; do not use cmd.exe /k.",
        f"Expected remote run id: {run_id}.",
        f"Hand off monitoring and retrieval to {monitor_owner}; do not SSH-poll from the main thread.",
        "After retrieval, run route-specific postprocess, update docs/experiments, commit, and push.",
    ]


def _readiness_report(config_path: Path) -> dict[str, Any]:
    try:
        return remote_readiness_report(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "config": str(config_path),
            "errors": [str(exc)],
            "warnings": [],
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_next_action(args.summary)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
