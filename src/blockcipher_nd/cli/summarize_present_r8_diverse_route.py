from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.monitor_health import monitor_health_report
from blockcipher_nd.cli.residual_focus_status import (
    DEFAULT_ACTION_PLAN as DEFAULT_RESIDUAL_ACTION_PLAN,
    DEFAULT_ARTIFACT_ROOT as DEFAULT_RESIDUAL_ARTIFACT_ROOT,
    DEFAULT_GATE as DEFAULT_RESIDUAL_GATE,
    DEFAULT_MONITOR_DIR as DEFAULT_RESIDUAL_MONITOR_DIR,
    DEFAULT_REPAIR_PLAN as DEFAULT_RESIDUAL_REPAIR_PLAN,
    residual_focus_status,
)


DEFAULT_RESIDUAL_STATUS = Path("outputs/local_audits/i1_present_r8_residual_focus_status.json")
DEFAULT_POOL_PLAN = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")
DEFAULT_POOL_EVAL = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json")
DEFAULT_STATE_TOKEN_PLAN = Path("outputs/local_audits/i1_present_r8_state_token_residual_expert_plan_with_controls.json")
DEFAULT_BUCKET_RESIDUAL_PLAN = Path("outputs/local_audits/i1_present_r8_bucket_residual_262k_action_plan.json")
DEFAULT_BUCKET_RESIDUAL_CONTROL_GATE = Path("outputs/local_audits/i1_present_r8_bucket_residual_controls_gate.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_diverse_route_summary.json")
DEFAULT_MONITOR_HEALTH_RUN_ID = "i1_present_r8_residual_focus_262k_retry1"
DEFAULT_MONITOR_HEALTH_ROOT = Path("outputs/remote_results")
DEFAULT_MONITOR_HEALTH_PROGRESS_ROOT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the local-only PRESENT r8 diverse expert route state across "
            "residual-focus, Pool 3, and state-token control gates."
        )
    )
    parser.add_argument("--residual-status", type=Path, default=DEFAULT_RESIDUAL_STATUS)
    parser.add_argument("--pool-plan", type=Path, default=DEFAULT_POOL_PLAN)
    parser.add_argument("--pool-eval", type=Path, default=DEFAULT_POOL_EVAL)
    parser.add_argument("--state-token-plan", type=Path, default=DEFAULT_STATE_TOKEN_PLAN)
    parser.add_argument("--bucket-residual-plan", type=Path, default=DEFAULT_BUCKET_RESIDUAL_PLAN)
    parser.add_argument("--bucket-residual-control-gate", type=Path, default=DEFAULT_BUCKET_RESIDUAL_CONTROL_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-residual-status",
        action="store_true",
        help="Recompute and write the residual-focus status before summarizing routes.",
    )
    parser.add_argument("--residual-action-plan", type=Path, default=DEFAULT_RESIDUAL_ACTION_PLAN)
    parser.add_argument("--residual-gate", type=Path, default=DEFAULT_RESIDUAL_GATE)
    parser.add_argument("--residual-repair-plan", type=Path, default=DEFAULT_RESIDUAL_REPAIR_PLAN)
    parser.add_argument("--residual-monitor-dir", type=Path, default=DEFAULT_RESIDUAL_MONITOR_DIR)
    parser.add_argument("--residual-artifact-root", type=Path, default=DEFAULT_RESIDUAL_ARTIFACT_ROOT)
    parser.add_argument(
        "--include-monitor-health",
        action="store_true",
        help="Embed a compact local monitor-health summary for the residual-focus run.",
    )
    parser.add_argument("--monitor-health-run-id", default=DEFAULT_MONITOR_HEALTH_RUN_ID)
    parser.add_argument("--monitor-health-root", type=Path, default=DEFAULT_MONITOR_HEALTH_ROOT)
    parser.add_argument(
        "--monitor-health-progress-root",
        type=Path,
        action="append",
        default=[],
        help="Optional local progress root to pass through to monitor-health.",
    )
    parser.add_argument("--monitor-health-stale-after-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def summarize_present_r8_diverse_route(
    *,
    residual_status_path: Path,
    pool_plan_path: Path,
    pool_eval_path: Path,
    state_token_plan_path: Path,
    bucket_residual_plan_path: Path,
    bucket_residual_control_gate_path: Path,
    monitor_health: dict[str, Any] | None = None,
    advance_command: str | None = None,
) -> dict[str, Any]:
    residual = _load_json(residual_status_path)
    pool_plan = _load_json_or_empty(pool_plan_path)
    pool_eval = _load_json_or_empty(pool_eval_path)
    state_token = _load_json_or_empty(state_token_plan_path)
    bucket_residual_plan = _load_json_or_empty(bucket_residual_plan_path)
    bucket_residual_control_gate = _load_json_or_empty(bucket_residual_control_gate_path)
    pool3_route = _pool3_route(residual, pool_plan, pool_eval)
    state_token_route = _state_token_route(state_token)
    bucket_residual_route = _bucket_residual_route(bucket_residual_plan, bucket_residual_control_gate)
    linear_combo_route = _linear_combo_integral_residual_route(residual)
    status, decision, selected_next_action = _route_decision(
        residual=residual,
        pool3_route=pool3_route,
        state_token_route=state_token_route,
    )
    return {
        "status": status,
        "decision": decision,
        "selected_next_action": selected_next_action,
        "should_launch_remote": False,
        "inputs": {
            "residual_status": str(residual_status_path),
            "pool_plan": str(pool_plan_path),
            "pool_eval": str(pool_eval_path),
            "state_token_plan": str(state_token_plan_path),
            "bucket_residual_plan": str(bucket_residual_plan_path),
            "bucket_residual_control_gate": str(bucket_residual_control_gate_path),
        },
        "residual_focus": {
            "status": str(residual.get("status", "")),
            "gate_status": str(residual.get("gate_status", "")),
            "gate_decision": str(residual.get("gate_decision", "")),
            "missing_output_count": int(residual.get("missing_output_count", 0)),
            "repair_active": bool(residual.get("repair_active", False)),
            "repair_status": str(residual.get("repair_status", "")),
            "repair_stale_reason": str(residual.get("repair_stale_reason", "")),
            "repair_context_current": bool(residual.get("repair_context_current", False)),
            "source_selection_summary_status": str(
                residual.get("source_selection_summary_status", "")
            ),
            "source_selection_summary_decision": str(
                residual.get("source_selection_summary_decision", "")
            ),
            "source_selection_report_count": int(residual.get("source_selection_report_count", 0)),
            "source_selection_existing_report_count": int(
                residual.get("source_selection_existing_report_count", 0)
            ),
            "source_selection_missing_report_count": int(
                residual.get("source_selection_missing_report_count", 0)
            ),
            "source_selection_recommended_feature_prefixes": [
                str(prefix)
                for prefix in residual.get("source_selection_recommended_feature_prefixes", [])
            ],
            "source_selection_selected_groups": [
                str(group) for group in residual.get("source_selection_selected_groups", [])
            ],
            "latest_monitor_event": str(residual.get("latest_monitor_event", "")),
            "progress_summary": _dict_value(residual.get("progress_summary")),
            "progress_by_seed_split": [
                _dict_value(row) for row in residual.get("progress_by_seed_split", [])
            ],
            "planned_output_count": int(residual.get("planned_output_count", 0)),
            "existing_planned_output_count": int(residual.get("existing_planned_output_count", 0)),
            "monitor_health_command": _residual_monitor_health_command(),
            "monitor_health_summary": _compact_monitor_health(monitor_health),
            "advance_command": advance_command or _residual_advance_command(),
            "next_action": _compact_next_action(residual.get("next_action")),
        },
        "candidate_routes": {
            "pool3_residual_guided": pool3_route,
            "state_token_residual": state_token_route,
            "bucket_conditioned_residual": bucket_residual_route,
            "linear_combo_integral_residual": linear_combo_route,
        },
        "policy": {
            "primary_gate": "residual_focus_262k_before_diverse_pool",
            "pool3_priority": "prefer residual-guided Pool 3 when residual-focus passes and controls are available",
            "state_token_policy": "do not promote when coordinate/value-only controls hold",
            "bucket_residual_policy": "track as a genuinely different third-family migration candidate without bypassing residual-focus or control gates",
            "linear_combo_integral_policy": (
                "track as a literature-guided SPN/integral residual candidate; select only from train "
                "source-selection summaries and do not launch while residual-focus is pending"
            ),
            "remote_policy": "local summary only; never SSH, launch, or scale by itself",
        },
        "claim_scope": (
            "local PRESENT r8 diverse-route arbitration only; does not train, SSH, launch remote jobs, "
            "change labels or negative mode, or make a medium/formal SPN/PRESENT claim"
        ),
    }


def _route_decision(
    *,
    residual: dict[str, Any],
    pool3_route: dict[str, Any],
    state_token_route: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    residual_status = str(residual.get("status", ""))
    gate_status = str(residual.get("gate_status", ""))
    gate_decision = str(residual.get("gate_decision", ""))
    if gate_status in {"fail", "hold"} or gate_decision.startswith("hold_") or residual_status == "repair_ready":
        next_action = _compact_next_action(residual.get("next_action"))
        return (
            "hold",
            next_action["branch"] or "repair_residual_focus_before_diverse_pool",
            next_action,
        )
    if residual_status == "outputs_ready_gate_needed":
        next_action = _compact_next_action(residual.get("next_action"))
        return (
            "pending",
            next_action["branch"] or "run_residual_focus_gate",
            next_action,
        )
    if pool3_route["status"] == "ready":
        return (
            "ready",
            "instantiate_residual_guided_pool3_fixed_fusion",
            _compact_next_action(pool3_route.get("next_action")),
        )
    if pool3_route["status"] == "evaluated":
        next_action = _compact_next_action(pool3_route.get("next_action"))
        return (
            "pass",
            next_action["branch"] or "document_residual_guided_pool3_fixed_fusion",
            next_action,
        )
    if pool3_route["status"] == "control_hold":
        next_action = _compact_next_action(pool3_route.get("next_action"))
        return (
            "hold",
            next_action["branch"] or "repair_residual_guided_pool3_before_scaleup",
            next_action,
        )
    if pool3_route["status"] == "waiting_for_pool3_score_artifacts":
        return (
            "pending",
            "wait_for_pool3_score_artifacts",
            _compact_next_action(pool3_route.get("next_action")),
        )
    if pool3_route["status"] == "waiting_for_pool_plan":
        next_action = _compact_next_action(pool3_route.get("next_action"))
        return (
            "pending",
            next_action["branch"] or "run_residual_guided_pool_planner",
            next_action,
        )
    return (
        "pending",
        "wait_for_residual_focus_outputs",
        _compact_next_action(residual.get("next_action")),
    )


def _pool3_route(
    residual: dict[str, Any],
    pool_plan: dict[str, Any],
    pool_eval: dict[str, Any],
) -> dict[str, Any]:
    gate_status = str(residual.get("gate_status", ""))
    pool_status = str(pool_plan.get("status", residual.get("pool_status", "")))
    pool_eval_status = str(pool_eval.get("status", residual.get("pool_eval_status", "")))
    if gate_status != "pass":
        return {
            "status": "blocked_by_residual_focus",
            "reason": "residual_focus_gate_not_passed",
            "next_action": _compact_next_action(residual.get("next_action")),
        }
    if pool_eval_status == "pass":
        return {
            "status": "evaluated",
            "decision": str(pool_eval.get("decision", "")),
            "next_action": _compact_next_action(pool_eval.get("next_action")),
        }
    if pool_eval_status in {"hold", "fail"}:
        return {
            "status": "control_hold",
            "decision": str(pool_eval.get("decision", "")),
            "next_action": _compact_next_action(pool_eval.get("next_action")),
        }
    if pool_eval_status == "pending" and pool_eval.get("decision") == "wait_for_pool3_score_artifacts":
        return {
            "status": "waiting_for_pool3_score_artifacts",
            "missing_score_artifacts": [str(path) for path in pool_eval.get("missing_score_artifacts", [])],
            "next_action": _compact_next_action(pool_eval.get("next_action")),
        }
    if pool_status == "pass" and pool_plan.get("should_run_pool") is True:
        return {
            "status": "ready",
            "decision": str(pool_plan.get("decision", "")),
            "selected_residual_candidate": str(pool_plan.get("selected_residual_candidate", "")),
            "expert_families": _string_list(pool_plan.get("expert_families")),
            "control_families": _string_list(pool_plan.get("control_families")),
            "planned_fixed_fusions": _string_list(pool_plan.get("planned_fixed_fusions")),
            "source_selection_status": str(pool_plan.get("source_selection_status", "")),
            "source_selection_decision": str(pool_plan.get("source_selection_decision", "")),
            "source_selection_recommended_feature_prefixes": _string_list(
                pool_plan.get("source_selection_recommended_feature_prefixes")
            ),
            "source_selection_selected_groups": _string_list(
                pool_plan.get("source_selection_selected_groups")
            ),
            "next_action": _compact_next_action(pool_plan.get("next_action")),
        }
    return {
        "status": "waiting_for_pool_plan",
        "decision": str(pool_plan.get("decision", "")),
        "next_action": _compact_next_action(residual.get("next_action")),
    }


def _state_token_route(state_token: dict[str, Any]) -> dict[str, Any]:
    status = str(state_token.get("status", "missing" if not state_token else ""))
    control_status = str(state_token.get("state_token_control_status", ""))
    if status == "hold" or control_status == "hold":
        return {
            "status": "blocked_by_controls",
            "decision": str(state_token.get("decision", "")),
            "control_status": control_status,
            "failing_seed_count": int(state_token.get("state_token_failing_seed_count", 0)),
            "failing_control_event_count": int(state_token.get("state_token_failing_control_event_count", 0)),
            "next_action": _compact_next_action(state_token.get("next_action")),
        }
    return {
        "status": status or "missing",
        "decision": str(state_token.get("decision", "")),
        "control_status": control_status,
        "next_action": _compact_next_action(state_token.get("next_action")),
    }


def _bucket_residual_route(plan: dict[str, Any], control_gate: dict[str, Any]) -> dict[str, Any]:
    if not plan and not control_gate:
        return {
            "status": "missing",
            "plan_status": "missing",
            "control_status": "missing",
            "next_action": {"branch": "", "should_launch_remote": False},
        }
    plan_status = str(plan.get("status", "missing" if not plan else ""))
    control_status = str(control_gate.get("status", "missing" if not control_gate else ""))
    route = {
        "status": _bucket_residual_status(plan_status, control_status, plan),
        "plan_status": plan_status or "missing",
        "plan_decision": str(plan.get("decision", "")),
        "source_status": str(plan.get("source_status", "")),
        "source_decision": str(plan.get("source_decision", "")),
        "control_status": control_status or "missing",
        "control_decision": str(control_gate.get("decision", "")),
        "should_run": bool(plan.get("should_run", False)),
        "reason": str(plan.get("reason", "")),
        "missing": [str(item) for item in plan.get("missing", [])],
        "errors": [str(item) for item in control_gate.get("errors", [])],
        "seed_count": int(control_gate.get("seed_count", 0)),
        "min_three_vs_two_auc_delta": _optional_float(control_gate.get("min_three_vs_two_auc_delta")),
        "min_bucket_vs_nobucket_auc_delta": _optional_float(
            control_gate.get("min_bucket_vs_nobucket_auc_delta")
        ),
        "next_action": _compact_next_action(control_gate.get("next_action")),
        "claim_scope": (
            "local bucket-conditioned residual route tracking only; controls are local diagnostics and "
            "262144/class migration still waits for retrieved score artifacts"
        ),
    }
    if not route["next_action"]["branch"]:
        route["next_action"] = _compact_bucket_next_action(plan.get("next_action"))
    return route


def _bucket_residual_status(plan_status: str, control_status: str, plan: dict[str, Any]) -> str:
    if control_status in {"fail", "hold"}:
        return "blocked_by_controls"
    if plan_status == "pass" and control_status == "pass" and plan.get("should_run") is True:
        return "ready_262k_migration_plan"
    if plan_status == "pending":
        return "pending_262k_artifacts"
    if plan_status in {"fail", "hold"}:
        return "blocked_by_plan"
    if control_status == "pass":
        return "controls_pass_plan_missing"
    return plan_status or control_status or "missing"


def _linear_combo_integral_residual_route(residual: dict[str, Any]) -> dict[str, Any]:
    gate_status = str(residual.get("gate_status", ""))
    source_status = str(residual.get("source_selection_summary_status", ""))
    source_decision = str(residual.get("source_selection_summary_decision", ""))
    next_action = _compact_next_action(residual.get("next_action"))
    base = {
        "decision": "wait_for_residual_focus_outputs_before_linear_combo_integral_expert",
        "selection_split": "train_only",
        "requires_source_selection": True,
        "source_selection_summary_status": source_status,
        "source_selection_summary_decision": source_decision,
        "recommended_feature_prefixes": _string_list(
            residual.get("source_selection_recommended_feature_prefixes")
        ),
        "selected_groups": _string_list(residual.get("source_selection_selected_groups")),
        "next_action": next_action,
        "claim_scope": (
            "route-arbitration candidate only; inspired by SPN/integral representation work and "
            "requires retrieved residual-focus artifacts before any scoring or remote launch"
        ),
    }
    if gate_status != "pass":
        return {
            **base,
            "status": "blocked_by_residual_focus",
            "reason": "residual_focus_gate_not_passed",
        }
    if source_status != "pass":
        return {
            **base,
            "status": "waiting_for_train_source_selection",
            "decision": "wait_for_train_axis_spectrum_before_linear_combo_integral_expert",
            "reason": "train_only_source_selection_not_ready",
        }
    return {
        **base,
        "status": "planned_after_source_selection",
        "decision": "plan_linear_combo_integral_residual_expert",
        "reason": "residual_focus_passed_and_train_source_selection_ready",
    }


def _residual_monitor_health_command() -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/monitor-health "
        "--run-id i1_present_r8_residual_focus_262k_retry1 "
        "--recent-lines 10 "
        "--stale-after-seconds 3600 "
        "--progress-root outputs/local_audits/i1_present_r8_residual_focus_262k"
    )


def _residual_advance_command(
    *,
    action_plan: Path = DEFAULT_RESIDUAL_ACTION_PLAN,
    gate_output: Path = DEFAULT_RESIDUAL_GATE,
    pool_output: Path = DEFAULT_POOL_PLAN,
    pool_eval_output: Path = DEFAULT_POOL_EVAL,
    repair_output: Path = DEFAULT_RESIDUAL_REPAIR_PLAN,
    status_output: Path = DEFAULT_RESIDUAL_STATUS,
    monitor_dir: Path = DEFAULT_RESIDUAL_MONITOR_DIR,
    artifact_root: Path = DEFAULT_RESIDUAL_ARTIFACT_ROOT,
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


def _compact_monitor_health(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {
        "status": str(report.get("status", "")),
        "needs_main_thread_intervention": bool(report.get("needs_main_thread_intervention", False)),
        "results_jsonl_exists": bool(report.get("results_jsonl_exists", False)),
        "results_jsonl_line_count": int(report.get("results_jsonl_line_count", 0)),
        "heartbeat": _dict_value(report.get("heartbeat")),
        "progress_summary": _compact_monitor_progress(report.get("progress_summary")),
    }


def _compact_monitor_progress(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = [
        "path",
        "exists",
        "source_kind",
        "line_count",
        "parsed_line_count",
        "latest_event",
        "latest_split",
        "latest_total_rows",
        "latest_samples_per_class",
        "stage",
        "model",
        "cache_event",
        "cache_split",
        "cache_rows_done",
        "cache_total_rows",
        "cache_rows_remaining",
        "cache_class_rows_done",
        "cache_class_total",
        "cache_class_rows_remaining",
        "cache_chunk_rows",
        "cache_chunk_index",
        "cache_class_chunk_index",
        "cache_total_progress_percent",
        "cache_class_progress_percent",
        "cache_positive_class_rows_done",
        "cache_positive_class_total",
        "cache_positive_class_rows_remaining",
        "cache_positive_class_progress_percent",
        "cache_negative_class_rows_done",
        "cache_negative_class_total",
        "cache_negative_class_rows_remaining",
        "cache_negative_class_progress_percent",
        "cache_rows_per_second",
        "cache_rate_window_seconds",
        "cache_rate_window_rows",
        "cache_eta_seconds",
        "epoch",
        "epochs",
        "val_auc",
        "best_checkpoint_metric",
    ]
    return {key: value.get(key) for key in keys if key in value}


def _compact_next_action(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"branch": "", "should_launch_remote": False}
    return {
        "branch": str(value.get("branch", "")),
        "should_launch_remote": bool(value.get("should_launch_remote", False)),
    }


def _compact_bucket_next_action(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return _compact_next_action(value)
    if isinstance(value, str) and value:
        return {"branch": value, "should_launch_remote": False}
    return {"branch": "", "should_launch_remote": False}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.refresh_residual_status:
        residual_status_report = residual_focus_status(
            action_plan=args.residual_action_plan,
            gate=args.residual_gate,
            pool_plan=args.pool_plan,
            pool_eval=args.pool_eval,
            repair_plan=args.residual_repair_plan,
            monitor_dir=args.residual_monitor_dir,
            artifact_root=args.residual_artifact_root,
        )
        _write_json(args.residual_status, residual_status_report)
    monitor_health = None
    if args.include_monitor_health:
        progress_roots = args.monitor_health_progress_root or [DEFAULT_MONITOR_HEALTH_PROGRESS_ROOT]
        monitor_health = monitor_health_report(
            run_id=args.monitor_health_run_id,
            root=args.monitor_health_root,
            recent_lines=10,
            stale_after_seconds=args.monitor_health_stale_after_seconds,
            progress_roots=progress_roots,
        )
    report = summarize_present_r8_diverse_route(
        residual_status_path=args.residual_status,
        pool_plan_path=args.pool_plan,
        pool_eval_path=args.pool_eval,
        state_token_plan_path=args.state_token_plan,
        bucket_residual_plan_path=args.bucket_residual_plan,
        bucket_residual_control_gate_path=args.bucket_residual_control_gate,
        monitor_health=monitor_health,
        advance_command=_residual_advance_command(
            action_plan=args.residual_action_plan,
            gate_output=args.residual_gate,
            pool_output=args.pool_plan,
            pool_eval_output=args.pool_eval,
            repair_output=args.residual_repair_plan,
            status_output=args.residual_status,
            monitor_dir=args.residual_monitor_dir,
            artifact_root=args.residual_artifact_root,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
