from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.gate_residual_focus_262k import gate_residual_focus_262k
from blockcipher_nd.cli.plan_residual_guided_diverse_pool import plan_residual_guided_diverse_pool
from blockcipher_nd.cli.residual_focus_status import residual_focus_status


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_GATE_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_POOL_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
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
) -> dict[str, Any]:
    initial_status = residual_focus_status(
        action_plan=action_plan,
        gate=gate_output,
        pool_plan=pool_output,
        monitor_dir=monitor_dir,
        artifact_root=artifact_root,
    )
    ran_gate = False
    ran_pool_planner = False
    gate_report: dict[str, Any] = _read_json_or_empty(gate_output)
    pool_report: dict[str, Any] = _read_json_or_empty(pool_output)

    if initial_status["status"] == "outputs_ready_gate_needed":
        gate_report = gate_residual_focus_262k(action_plan=action_plan)
        _write_json(gate_output, gate_report)
        ran_gate = True

    if gate_report.get("status") in {"pass", "fail"}:
        pool_report = plan_residual_guided_diverse_pool(residual_focus_gate=gate_output)
        _write_json(pool_output, pool_report)
        ran_pool_planner = True

    final_status = residual_focus_status(
        action_plan=action_plan,
        gate=gate_output,
        pool_plan=pool_output,
        monitor_dir=monitor_dir,
        artifact_root=artifact_root,
    )
    _write_json(status_output, final_status)
    status, decision = _advance_status(final_status, ran_gate=ran_gate, ran_pool_planner=ran_pool_planner)
    return {
        "status": status,
        "decision": decision,
        "ran_gate": ran_gate,
        "ran_pool_planner": ran_pool_planner,
        "initial_status": initial_status["status"],
        "final_status": final_status["status"],
        "gate_status": final_status["gate_status"],
        "gate_decision": final_status["gate_decision"],
        "pool_status": final_status["pool_status"],
        "should_run_pool": final_status["should_run_pool"],
        "missing_output_count": final_status["missing_output_count"],
        "next_action": final_status["next_action"],
        "claim_scope": (
            "local residual-focus postprocess advancement only; does not SSH, sync, launch "
            "remote jobs, or make a formal/breakthrough SPN/PRESENT claim"
        ),
    }


def _advance_status(
    final_status: dict[str, Any],
    *,
    ran_gate: bool,
    ran_pool_planner: bool,
) -> tuple[str, str]:
    if final_status["should_run_pool"]:
        return "pass", "residual_guided_pool_ready"
    if final_status["gate_status"] == "fail":
        return "hold", "repair_residual_focus_before_pool"
    if ran_gate or ran_pool_planner:
        return "processed", "residual_focus_postprocess_updated"
    return "pending", "wait_for_residual_focus_outputs"


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
        status_output=args.status_output,
        monitor_dir=args.monitor_dir,
        artifact_root=args.artifact_root,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
