from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_GATE_REPORT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_readiness.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the PRESENT r8 residual-focus 262144/class action plan "
            "is safe and ready to execute or consume."
        )
    )
    parser.add_argument("--action-plan", type=Path, default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def audit_residual_focus_262k_readiness(
    *,
    action_plan: Path,
    gate_report: Path,
) -> dict[str, Any]:
    plan = _load_json(action_plan)
    gate = _load_json(gate_report)
    commands = [str(item) for item in plan.get("commands", [])]
    control_commands = [str(item) for item in plan.get("control_commands", [])]
    all_commands = commands + control_commands
    unsafe_findings = _unsafe_command_findings(all_commands)
    remote_checkpoint_seeds = _remote_checkpoint_seeds(plan)
    missing_outputs = [str(path) for path in gate.get("missing_outputs", [])]
    blockers: list[str] = []
    if plan.get("status") != "pass":
        blockers.append("action_plan_not_ready")
    if missing_outputs:
        blockers.append("gate_missing_outputs")
    if remote_checkpoint_seeds:
        blockers.append("remote_checkpoint_reference_requires_remote_or_retrieved_checkpoint")
    blockers.extend(finding["reason"] for finding in unsafe_findings)
    status = "fail" if unsafe_findings else "pending" if blockers else "pass"
    return {
        "status": status,
        "decision": (
            "residual_focus_262k_execution_plan_unsafe"
            if status == "fail"
            else "residual_focus_262k_execution_not_ready"
            if status == "pending"
            else "residual_focus_262k_ready_for_result_gate"
        ),
        "action_plan": str(action_plan),
        "gate_report": str(gate_report),
        "action_plan_status": str(plan.get("status", "")),
        "gate_status": str(gate.get("status", "")),
        "source_gate_assessment": str(plan.get("source_gate_assessment", "")),
        "command_count": len(commands),
        "control_command_count": len(control_commands),
        "remote_checkpoint_seed_count": len(remote_checkpoint_seeds),
        "remote_checkpoint_seeds": remote_checkpoint_seeds,
        "missing_outputs_count": len(missing_outputs),
        "missing_outputs": missing_outputs,
        "unsafe_command_count": len(unsafe_findings),
        "unsafe_commands": unsafe_findings,
        "blockers": sorted(set(blockers)),
        "should_run_local_commands": status == "pass" and not remote_checkpoint_seeds,
        "next_action": {
            "branch": _next_branch(
                status=status,
                remote_checkpoint_seeds=remote_checkpoint_seeds,
                missing_outputs=missing_outputs,
            ),
            "should_launch_remote": False,
        },
        "claim_scope": (
            "readiness audit only; does not launch remote jobs, does not SSH-poll, "
            "does not run residual-focus commands, and does not prove a medium or formal claim"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _remote_checkpoint_seeds(plan: dict[str, Any]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict) or not bool(seed_plan.get("remote_checkpoint_reference")):
            continue
        seeds.append(
            {
                "seed": int(seed_plan.get("seed", -1)),
                "train_trail_position_checkpoint": str(seed_plan.get("train_trail_position_checkpoint", "")),
            }
        )
    return seeds


def _unsafe_command_findings(commands: list[str]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for index, command in enumerate(commands):
        lowered = command.lower()
        if "cmd.exe /k" in lowered:
            findings.append(
                {
                    "index": str(index),
                    "reason": "unsafe_command_contains_cmd_exe_k",
                    "command": command,
                }
            )
        if " ssh " in f" {lowered} ":
            findings.append(
                {
                    "index": str(index),
                    "reason": "unsafe_command_contains_ssh",
                    "command": command,
                }
            )
    return findings


def _next_branch(
    *,
    status: str,
    remote_checkpoint_seeds: list[dict[str, Any]],
    missing_outputs: list[str],
) -> str:
    if status == "fail":
        return "repair_residual_focus_execution_plan"
    if remote_checkpoint_seeds:
        return "publish_source_then_run_remote_or_retrieve_checkpoints"
    if missing_outputs:
        return "finish_residual_focus_262k_outputs"
    return "run_residual_focus_262k_gate"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_residual_focus_262k_readiness(
        action_plan=args.action_plan,
        gate_report=args.gate_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
