from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.evaluate_residual_guided_diverse_pool import (
    evaluate_residual_guided_diverse_pool,
)
from blockcipher_nd.cli.gate_residual_focus_262k import gate_residual_focus_262k
from blockcipher_nd.cli.plan_residual_guided_diverse_pool import (
    DEFAULT_SOURCE_SELECTION_SUMMARY as DEFAULT_POOL_SOURCE_SELECTION_SUMMARY,
    plan_residual_guided_diverse_pool,
)
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
    source_selection_command_report = _run_source_selection_commands_when_ready(
        action_plan,
        initial_status=initial_status,
    )
    source_selected_command_report = {"ran_commands": False, "run_count": 0, "results": []}
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
        source_selection_summary = Path(
            str(source_selection_summary_report.get("output") or DEFAULT_POOL_SOURCE_SELECTION_SUMMARY)
        )
        pool_report = plan_residual_guided_diverse_pool(
            residual_focus_gate=gate_output,
            source_selection_summary=source_selection_summary,
        )
        _write_json(pool_output, pool_report)
        ran_pool_planner = True

    if pool_report.get("should_run_pool") is True:
        source_selected_command_report = _run_source_selected_commands_when_ready(
            action_plan,
            pool_report=pool_report,
        )
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
    repair_fields = _advance_repair_fields(final_status)
    return {
        "status": status,
        "decision": decision,
        "ran_gate": ran_gate,
        "ran_pool_planner": ran_pool_planner,
        "ran_pool_evaluator": ran_pool_evaluator,
        "ran_source_selection_commands": bool(source_selection_command_report["ran_commands"]),
        "source_selection_command_run_count": int(source_selection_command_report["run_count"]),
        "source_selection_command_results": source_selection_command_report["results"],
        "ran_source_selected_commands": bool(source_selected_command_report["ran_commands"]),
        "source_selected_command_run_count": int(source_selected_command_report["run_count"]),
        "source_selected_command_results": source_selected_command_report["results"],
        "ran_source_selection_summary": ran_source_selection_summary,
        "initial_status": initial_status["status"],
        "final_status": final_status["status"],
        "gate_status": final_status["gate_status"],
        "gate_decision": final_status["gate_decision"],
        "pool_status": final_status["pool_status"],
        "pool_eval_status": str(pool_eval_report.get("status", "")),
        "pool_eval_decision": str(pool_eval_report.get("decision", "")),
        "repair_plan": final_status["repair_plan"],
        "repair_status": repair_fields["repair_status"],
        "repair_active": repair_fields["repair_active"],
        "repair_stale_reason": repair_fields["repair_stale_reason"],
        "repair_decision": repair_fields["repair_decision"],
        "repair_source_summary": repair_fields["repair_source_summary"],
        "repair_context_current": repair_fields["repair_context_current"],
        "repair_primary_branch": repair_fields["repair_primary_branch"],
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
        "latest_monitor_event": final_status["latest_monitor_event"],
        "latest_progress": final_status["latest_progress"],
        "progress_summary": final_status["progress_summary"],
        "progress_by_seed_split": final_status["progress_by_seed_split"],
        "planned_output_count": final_status["planned_output_count"],
        "existing_planned_output_count": final_status["existing_planned_output_count"],
        "missing_output_count": final_status["missing_output_count"],
        "missing_outputs": [str(path) for path in final_status["missing_outputs"]],
        "next_action": _advance_next_action(
            final_status,
            pool_eval_report,
            action_plan=action_plan,
            gate_output=gate_output,
            pool_output=pool_output,
            pool_eval_output=pool_eval_output,
            repair_output=repair_output,
            status_output=status_output,
            monitor_dir=monitor_dir,
            artifact_root=artifact_root,
        ),
        "claim_scope": (
            "local residual-focus postprocess advancement only; does not SSH, sync, launch "
            "remote jobs, or make a formal/breakthrough SPN/PRESENT claim"
        ),
    }


def _advance_repair_fields(final_status: dict[str, Any]) -> dict[str, Any]:
    repair_status = str(final_status.get("repair_status", ""))
    repair_active = bool(final_status.get("repair_active", False))
    repair_context_current = bool(final_status.get("repair_context_current", False))
    if repair_status == "stale" and not repair_active and not repair_context_current:
        return {
            "repair_status": repair_status,
            "repair_active": False,
            "repair_stale_reason": str(final_status.get("repair_stale_reason", "")),
            "repair_decision": "",
            "repair_source_summary": "",
            "repair_context_current": False,
            "repair_primary_branch": "",
        }
    return {
        "repair_status": repair_status,
        "repair_active": repair_active,
        "repair_stale_reason": str(final_status.get("repair_stale_reason", "")),
        "repair_decision": str(final_status.get("repair_decision", "")),
        "repair_source_summary": str(final_status.get("repair_source_summary", "")),
        "repair_context_current": repair_context_current,
        "repair_primary_branch": str(final_status.get("repair_primary_branch", "")),
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


def _run_source_selection_commands_when_ready(
    action_plan: Path,
    *,
    initial_status: dict[str, Any],
) -> dict[str, Any]:
    if int(initial_status.get("missing_output_count", 0)) != 0:
        return {"ran_commands": False, "run_count": 0, "results": []}
    plan = _read_json_or_empty(action_plan)
    missing_reports = [
        path for path in _source_selection_report_paths(plan) if not path.exists()
    ]
    if not missing_reports:
        return {"ran_commands": False, "run_count": 0, "results": []}
    commands = _source_selection_commands(plan)
    results = [
        _run_local_action_command(
            command,
            allowed_scripts={"scripts/analyze-residual-bucket-axis-spectrum"},
        )
        for command in commands
    ]
    return {
        "ran_commands": bool(results),
        "run_count": len(results),
        "results": results,
    }


def _run_source_selected_commands_when_ready(
    action_plan: Path,
    *,
    pool_report: dict[str, Any],
) -> dict[str, Any]:
    if not _pool_plan_requires_source_selected(pool_report):
        return {"ran_commands": False, "run_count": 0, "results": []}
    plan = _read_json_or_empty(action_plan)
    missing_outputs = [path for path in _source_selected_output_paths(plan) if not path.exists()]
    if not missing_outputs:
        return {"ran_commands": False, "run_count": 0, "results": []}
    commands = _source_selected_commands(plan)
    results = [
        _run_local_action_command(
            command,
            allowed_scripts={"scripts/fit-residual-correction-feature-expert"},
        )
        for command in commands
    ]
    return {
        "ran_commands": bool(results),
        "run_count": len(results),
        "results": results,
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


def _source_selected_output_paths(plan: dict[str, Any]) -> list[Path]:
    outputs: list[Path] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        planned = seed_plan.get("source_selected_outputs", {})
        if not isinstance(planned, dict):
            continue
        outputs.extend(Path(str(value)) for value in planned.values())
    return outputs


def _source_selection_commands(plan: dict[str, Any]) -> list[str]:
    commands = [str(command) for command in plan.get("source_selection_commands", [])]
    if commands:
        return commands
    nested: list[str] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        nested.extend(str(command) for command in seed_plan.get("source_selection_commands", []))
    return nested


def _source_selected_commands(plan: dict[str, Any]) -> list[str]:
    commands = [str(command) for command in plan.get("source_selected_commands", [])]
    if commands:
        return commands
    nested: list[str] = []
    for seed_plan in plan.get("seeds", []):
        if not isinstance(seed_plan, dict):
            continue
        nested.extend(str(command) for command in seed_plan.get("source_selected_commands", []))
    return nested


def _run_local_action_command(command: str, *, allowed_scripts: set[str]) -> dict[str, Any]:
    tokens = shlex.split(command)
    _validate_local_action_tokens(tokens, allowed_scripts=allowed_scripts)
    env = os.environ.copy()
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        key, value = tokens.pop(0).split("=", 1)
        env[key] = value
    completed = subprocess.run(
        tokens,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "command": _redacted_command(tokens),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def _validate_local_action_tokens(tokens: list[str], *, allowed_scripts: set[str]) -> None:
    lowered = [token.lower() for token in tokens]
    forbidden = ("ssh", "scp", "cmd.exe", "g:\\lxy")
    if any(any(item in token for item in forbidden) for token in lowered):
        raise ValueError("source-selection command contains forbidden remote token")
    command_tokens = list(tokens)
    while command_tokens and "=" in command_tokens[0] and not command_tokens[0].startswith("-"):
        command_tokens.pop(0)
    if len(command_tokens) < 3 or command_tokens[:2] != ["uv", "run"]:
        raise ValueError("source-selection command must start with UV_CACHE_DIR=/tmp/uv-cache uv run")
    if command_tokens[2] not in allowed_scripts:
        raise ValueError(f"local action command must run one of {sorted(allowed_scripts)}")


def _redacted_command(tokens: list[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


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


def _advance_next_action(
    final_status: dict[str, Any],
    pool_eval_report: dict[str, Any],
    *,
    action_plan: Path,
    gate_output: Path,
    pool_output: Path,
    pool_eval_output: Path,
    repair_output: Path,
    status_output: Path,
    monitor_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    local_command = _advance_local_command(
        action_plan=action_plan,
        gate_output=gate_output,
        pool_output=pool_output,
        pool_eval_output=pool_eval_output,
        repair_output=repair_output,
        status_output=status_output,
        monitor_dir=monitor_dir,
        artifact_root=artifact_root,
    )
    if pool_eval_report.get("status") == "pending" and pool_eval_report.get("decision") == "wait_for_pool3_score_artifacts":
        return {
            "branch": "wait_for_pool3_score_artifacts",
            "should_launch_remote": False,
            "local_command": local_command,
        }
    if pool_eval_report.get("status") == "hold":
        return {
            "branch": "repair_residual_guided_pool3_before_scaleup",
            "should_launch_remote": False,
            "local_command": local_command,
        }
    next_action = dict(final_status["next_action"])
    next_action["local_command"] = local_command
    return next_action


def _advance_local_command(
    *,
    action_plan: Path,
    gate_output: Path,
    pool_output: Path,
    pool_eval_output: Path,
    repair_output: Path,
    status_output: Path,
    monitor_dir: Path,
    artifact_root: Path,
) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/advance-residual-focus-results",
        "--action-plan",
        str(action_plan),
        "--gate-output",
        str(gate_output),
        "--pool-output",
        str(pool_output),
        "--pool-eval-output",
        str(pool_eval_output),
        "--repair-output",
        str(repair_output),
        "--status-output",
        str(status_output),
        "--monitor-dir",
        str(monitor_dir),
        "--artifact-root",
        str(artifact_root),
    ]
    return " ".join(shlex.quote(part) for part in parts)


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
        paths = _pool3_artifact_paths(
            seed_plan,
            selected_residual_candidate=selected,
            include_source_selected=_pool_plan_requires_source_selected(pool_report),
        )
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
            source_selected_residual_artifact=paths.get("source_selected_residual_artifact"),
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
    include_source_selected: bool = False,
) -> dict[str, Path]:
    artifact_root = Path(str(seed_plan.get("artifact_root", "")))
    paths = {
        "trail_position_artifact": Path(str(seed_plan.get("validation_trail_position_scores", ""))),
        "raw117_artifact": artifact_root / "validation_raw117_scores",
        "residual_focus_artifact": artifact_root / f"residual_{selected_residual_candidate}_validation_scores",
        "uniform_control_artifact": artifact_root / "residual_uniform_validation_scores",
        "labelshuffle_control_artifact": artifact_root / "residual_focus10_labelshuffle_validation_scores",
    }
    if include_source_selected:
        paths["source_selected_residual_artifact"] = (
            artifact_root / f"residual_{selected_residual_candidate}_source_selected_validation_scores"
        )
    return paths


def _pool_plan_requires_source_selected(pool_report: dict[str, Any]) -> bool:
    return "trail_position + raw117 + source_selected_residual_focus" in [
        str(fusion) for fusion in pool_report.get("planned_fixed_fusions", [])
    ]


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
