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
    implementation_checklist = _implementation_checklist(
        requires_implementation=bool(next_action.get("requires_implementation")),
        should_launch_remote=should_launch_remote,
        next_action=next_action,
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
        "implementation_checklist": implementation_checklist,
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


def _implementation_checklist(
    *,
    requires_implementation: bool,
    should_launch_remote: bool,
    next_action: dict[str, Any],
) -> list[str]:
    if not requires_implementation:
        return []
    branch = next_action.get("branch") or "<next-branch>"
    plan_doc = (
        next_action.get("next_plan_doc")
        or next_action.get("fallback_plan")
        or _default_plan_doc_from_options(next_action)
        or _default_plan_doc_for_branch(str(branch))
        or "docs/experiments/<plan>.md"
    )
    suggested_plan = (
        next_action.get("suggested_plan_config")
        or _default_plan_config_for_branch(str(branch))
        or "<next experiment config>"
    )
    checklist = [
        f"Prepare branch `{branch}` before any remote launch.",
        f"Update or create the experiment plan in `{plan_doc}`.",
        f"Create and review `{suggested_plan}` with one attributable hypothesis.",
        "Run the route smoke/readiness checks locally.",
        "Commit and push the docs/config/code changes before launching from GitHub.",
    ]
    if should_launch_remote:
        checklist.insert(0, "Do not launch yet: implementation assets are not ready.")
    return checklist


def _default_plan_doc_for_branch(branch: str) -> str | None:
    return {
        "candidate_trail_consistency": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
        "candidate_trail_seed1_confirmation": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
        "candidate_trail_seed1_variance_check": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
        "bit_transition_spectrum_seed0": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
        "transition_spectrum_seed1_confirmation": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
        "transition_spectrum_variance_check": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
        "stop_transition_spectrum_route": "docs/experiments/innovation1-trail-family-consistency-plan.md",
        "trail_family_seed1_confirmation": "docs/experiments/innovation1-trail-family-consistency-plan.md",
        "trail_family_variance_check": "docs/experiments/innovation1-trail-family-consistency-plan.md",
        "stop_trail_family_route": "docs/experiments/innovation1-pairset-aggregation-control-plan.md",
    }.get(branch)


def _default_plan_doc_from_options(next_action: dict[str, Any]) -> str | None:
    options = next_action.get("fallback_plan_options")
    if not isinstance(options, list):
        return None
    for option in options:
        if isinstance(option, str) and option.startswith("docs/experiments/"):
            return option
    return None


def _default_plan_config_for_branch(branch: str) -> str | None:
    return {
        "bit_transition_spectrum_seed0": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0.json"
        ),
        "transition_spectrum_seed1_confirmation": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
        ),
        "transition_spectrum_variance_check": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
        ),
        "stop_transition_spectrum_route": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_trail_family_r7_262k_seed0.json"
        ),
        "trail_family_seed1_confirmation": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_trail_family_r7_262k_seed1.json"
        ),
        "trail_family_variance_check": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_trail_family_r7_262k_seed1.json"
        ),
        "stop_trail_family_route": (
            "configs/experiment/innovation1/"
            "innovation1_spn_present_pairset_aggregation_control_r7_262k.csv"
        ),
    }.get(branch)


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
