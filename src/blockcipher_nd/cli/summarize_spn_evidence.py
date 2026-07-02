from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.monitor_health import _progress_summary, monitor_health_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize retrieved Innovation 1 SPN evidence from local remote-result artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/remote_results"),
        help="Local remote-result artifact root.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def summarize_spn_evidence(root: Path) -> dict[str, Any]:
    summaries = [_load_summary(path) for path in sorted(root.glob("*/*_postprocess_summary.json"))]
    route_states = _route_states(summaries)
    routes = sorted((_route_summary(summary, route_states) for summary in summaries), key=_route_sort_key)
    strongest_route = _strongest_route(routes)
    active = _active_recommendation(root, routes)
    return {
        "status": "pass",
        "root": str(root),
        "summaries_scanned": len(summaries),
        "routes": routes,
        "strongest_route": strongest_route,
        "active_recommendation": active,
        "claim_guardrails": [
            "Do not call 262144/class SPN/PRESENT runs formal evidence.",
            "Do not claim breakthrough/SOTA from this summary.",
            "Treat missing or running artifacts as incomplete evidence.",
        ],
    }


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    data["_summary_path"] = str(path)
    return data


def _route_summary(summary: dict[str, Any], route_states: dict[str, str]) -> dict[str, Any]:
    run_id = str(summary.get("run_id") or _infer_run_id_from_path(summary))
    raw_next_action = _compact_next_action(summary.get("next_action"))
    route_state = route_states.get(run_id, "current_or_historical")
    return {
        "run_id": run_id,
        "decision": summary.get("decision"),
        "status": summary.get("status"),
        "validation_status": summary.get("validation_status"),
        "route_state": route_state,
        "evidence_scale": _evidence_scale(summary, run_id),
        "claim_scope": summary.get("claim_scope"),
        "metrics": _metrics(summary),
        "summary": summary.get("summary") or summary.get("_summary_path"),
        "next_action": raw_next_action,
        "effective_next_action": _effective_next_action(route_state, raw_next_action),
    }


def _route_states(summaries: list[dict[str, Any]]) -> dict[str, str]:
    decisions = {str(summary.get("decision") or "") for summary in summaries}
    has_transition_spectrum_decision = any(
        decision.startswith(("support_transition", "weak_transition", "stop_transition"))
        for decision in decisions
    )
    has_candidate_trail_decision = any(
        decision.startswith(("support_candidate", "weak_candidate", "stop_candidate"))
        for decision in decisions
    )
    has_topology_stop = "stop_topology_aware_network_route" in decisions
    has_invp_attribution_support = "support_invp_structural_attribution" in decisions
    states: dict[str, str] = {}
    for summary in summaries:
        run_id = str(summary.get("run_id") or _infer_run_id_from_path(summary))
        decision = str(summary.get("decision") or "")
        if decision in {"launch_invp_seed1_confirmation", "confirm_invp_route"} and has_invp_attribution_support:
            states[run_id] = "superseded"
        elif decision == "weak_ddt_graph_signal" and (has_topology_stop or has_candidate_trail_decision):
            states[run_id] = "superseded"
        elif decision == "weak_topology_aware_network_signal" and has_topology_stop:
            states[run_id] = "superseded"
        elif decision.startswith(("support_candidate", "weak_candidate", "stop_candidate")) and has_transition_spectrum_decision:
            states[run_id] = "superseded"
        else:
            states[run_id] = "current_or_historical"
    return states


def _effective_next_action(route_state: str, raw_next_action: dict[str, Any]) -> dict[str, Any]:
    if route_state == "superseded":
        return {
            "should_launch_remote": False,
            "reason": "superseded_by_later_route_decision",
        }
    return raw_next_action


def _infer_run_id_from_path(summary: dict[str, Any]) -> str:
    path = Path(str(summary.get("_summary_path", "")))
    name = path.name
    suffix = "_postprocess_summary.json"
    return name[: -len(suffix)] if name.endswith(suffix) else path.parent.name


def _evidence_scale(summary: dict[str, Any], run_id: str) -> str:
    text = " ".join(
        str(value)
        for value in [
            run_id,
            summary.get("claim_scope"),
            summary.get("plan"),
            summary.get("results"),
        ]
        if value is not None
    )
    if "1000000/class" in text or "_1m_" in text or "r7_1m" in text:
        if "attribution" in run_id or "attribution-control" in text:
            return "paper_scale_attribution_control"
        return "paper_scale_single_seed"
    if "262144/class" in text or "_262k" in text or "r7_262k" in text:
        return "medium_diagnostic"
    if "65536/class" in text or "_64k" in text or "_65k" in text:
        return "medium_diagnostic"
    if "smoke" in text.lower():
        return "smoke"
    return "unknown"


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "accuracy",
        "calibrated_accuracy",
        "auc",
        "auc_delta",
        "auc_delta_vs_paligned_mcnd_1m",
        "invp_seed0_auc",
        "invp_seed1_auc",
        "invp_min_auc",
        "invp_mean_auc",
        "max_control_auc",
        "attribution_margin",
        "margin_vs_best_control_auc",
        "margin_vs_invp_auc",
        "margin_vs_shuffled_auc",
        "best_candidate_auc",
        "anchor_auc",
        "shuffled_auc",
    ]
    return {key: summary[key] for key in keys if key in summary}


def _compact_next_action(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in [
            "branch",
            "reason",
            "should_launch_remote",
            "requires_implementation",
            "readiness_command",
            "fallback_plan",
        ]
        if key in value
    }


def _route_sort_key(route: dict[str, Any]) -> tuple[int, str]:
    scale_rank = {
        "paper_scale_attribution_control": 0,
        "paper_scale_single_seed": 1,
        "medium_diagnostic": 2,
        "smoke": 3,
        "unknown": 4,
    }
    return (scale_rank.get(str(route["evidence_scale"]), 5), str(route["run_id"]))


def _strongest_route(routes: list[dict[str, Any]]) -> dict[str, Any]:
    attribution = [
        route
        for route in routes
        if route["decision"] == "support_invp_structural_attribution"
        and route["status"] == "pass"
        and route["validation_status"] in {"pass", "not_run", None}
    ]
    if attribution:
        route = attribution[0]
        metrics = route["metrics"]
        return {
            "route": "present_nibble_invp_only_spn_only",
            "decision": route["decision"],
            "evidence_level": "two_seed_1000000_class_positive_with_attribution_control",
            "summary": route["summary"],
            "invp_seed0_auc": metrics.get("invp_seed0_auc"),
            "invp_seed1_auc": metrics.get("invp_seed1_auc"),
            "invp_min_auc": metrics.get("invp_min_auc"),
            "max_control_auc": metrics.get("max_control_auc"),
            "attribution_margin": metrics.get("attribution_margin"),
            "allowed_claim": (
                "InvP/P-layer aligned SPN view has stable positive evidence over the local "
                "same-protocol Zhang/Wang anchor, with attribution-control support."
            ),
        }
    paper_scale = [route for route in routes if route["evidence_scale"] == "paper_scale_single_seed"]
    if paper_scale:
        best = max(paper_scale, key=lambda route: _numeric_metric(route, "auc"))
        return {
            "route": "unknown_paper_scale_best_auc",
            "decision": best["decision"],
            "evidence_level": "paper_scale_single_seed",
            "summary": best["summary"],
            "auc": best["metrics"].get("auc"),
            "allowed_claim": "Paper-scale single-seed evidence only; needs confirmation.",
        }
    return {
        "route": None,
        "decision": None,
        "evidence_level": "insufficient_retrieved_evidence",
        "allowed_claim": "No retrieved SPN route summary was strong enough for a route-level claim.",
    }


def _numeric_metric(route: dict[str, Any], key: str) -> float:
    value = route.get("metrics", {}).get(key)
    return float(value) if isinstance(value, int | float) else float("-inf")


def _active_recommendation(root: Path, routes: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_running = _candidate_trail_running(root)
    if candidate_running is not None:
        return candidate_running
    transition_summaries = [
        route for route in routes if _is_transition_spectrum_decision(str(route["decision"] or ""))
    ]
    if transition_summaries:
        newest = sorted(transition_summaries, key=lambda route: str(route["run_id"]))[-1]
        return _recommend_from_transition_decision(newest)
    candidate_summaries = [
        route for route in routes if str(route["decision"] or "").endswith("candidate_trail_route")
    ]
    if candidate_summaries:
        newest = sorted(candidate_summaries, key=lambda route: str(route["run_id"]))[-1]
        return _recommend_from_candidate_decision(newest)
    return {
        "branch": "review_research_plan",
        "reason": "no active candidate-trail run or candidate-trail summary found",
        "should_launch_remote": False,
    }


def _candidate_trail_running(root: Path) -> dict[str, Any] | None:
    for run_root in sorted(root.glob("i1_candidate_trail_consistency*")):
        summaries = list(run_root.glob("*_postprocess_summary.json"))
        if summaries:
            continue
        monitor_log = run_root / "monitor" / "monitor.log"
        recent_lines = _tail_lines(monitor_log, 8)
        health = _candidate_monitor_health(root, run_root.name)
        if health["postprocess_allowed"]:
            return {
                "branch": "postprocess_candidate_trail_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "candidate-trail results are ready locally and need postprocess before branch decisions",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": health["needs_main_thread_intervention"],
                "postprocess_allowed": True,
                "postprocess_command": health["postprocess_command"],
                "results_jsonl": health["results_jsonl"],
                "results_jsonl_line_count": health["results_jsonl_line_count"],
                "expected_rows": health["expected_rows"],
                "progress_summary": _active_progress_summary(run_root),
                "conditional_followup": _candidate_conditional_followup(),
                "monitor_health_command": _candidate_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _candidate_postprocess_command(run_root),
                "main_thread_policy": _candidate_main_thread_policy("postprocess"),
            }
        if health["status"] in {"running", "stale_monitor", "launch_stalled", "unknown"} and any(
            "running" in line for line in recent_lines
        ):
            return {
                "branch": "wait_for_candidate_trail_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "candidate-trail run has monitor activity but no postprocess summary yet",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": health["needs_main_thread_intervention"],
                "postprocess_allowed": health["postprocess_allowed"],
                "progress_summary": _active_progress_summary(run_root),
                "conditional_followup": _candidate_conditional_followup(),
                "monitor_health_command": _candidate_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _candidate_postprocess_command(run_root),
                "main_thread_policy": _candidate_main_thread_policy("waiting"),
            }
    return None


def _candidate_monitor_health(root: Path, run_id: str) -> dict[str, Any]:
    plan_path = _candidate_plan_path(run_id)
    return monitor_health_report(
        run_id=run_id,
        root=root,
        tmux_session=_candidate_tmux_session(run_id),
        plan_path=plan_path,
        plan_doc_paths=[Path("docs/experiments/innovation1-candidate-trail-consistency-plan.md")],
        expected_rows=4,
        postprocess_kind="candidate_trail",
    )


def _active_progress_summary(run_root: Path) -> dict[str, Any]:
    progress = _progress_summary(run_root)
    keys = [
        "path",
        "exists",
        "line_count",
        "parsed_line_count",
        "latest_event",
        "cache_rows_done",
        "cache_total_rows",
        "cache_class_rows_done",
        "cache_class_total",
        "cache_chunk_rows",
        "cache_chunk_index",
        "cache_class_chunk_index",
        "cache_total_progress_percent",
        "cache_class_progress_percent",
        "cache_rows_per_second",
        "cache_eta_seconds",
        "model",
        "epoch",
        "epochs",
        "val_auc",
        "best_checkpoint_metric",
    ]
    return {key: progress.get(key) for key in keys if key in progress}


def _candidate_monitor_health_command(run_id: str) -> str:
    plan_path = _candidate_plan_path(run_id)
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {run_id} "
        f"--tmux-session {_candidate_tmux_session(run_id)} "
        f"--plan {plan_path} "
        "--plan-doc docs/experiments/innovation1-candidate-trail-consistency-plan.md "
        "--expected-rows 4 --postprocess-kind candidate_trail"
    )


def _candidate_postprocess_command(run_root: Path) -> str:
    run_id = run_root.name
    plan_path = _candidate_plan_path(run_id)
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-candidate-trail "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {plan_path} "
        "--expected-rows 4 "
        "--update-plan-doc docs/experiments/innovation1-candidate-trail-consistency-plan.md"
    )


def _candidate_conditional_followup() -> dict[str, Any]:
    config = Path(
        "configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json"
    )
    readiness = _remote_readiness(config)
    return {
        "branch": "candidate_trail_seed1_confirmation_or_variance_check",
        "run_id": "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702",
        "launch_remote_config": str(config),
        "launch_gate": "support_candidate_trail_route or weak_candidate_trail_signal",
        "readiness_pass": readiness.get("status") == "pass",
        "readiness": readiness,
        "should_launch_now": False,
    }


def _remote_readiness(config: Path) -> dict[str, Any]:
    try:
        return remote_readiness_report(config)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "config": str(config),
            "errors": [str(exc)],
            "warnings": [],
        }


def _candidate_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run the listed postprocess_when_ready_command",
                "validate gate artifacts and update docs/experiments",
                "commit and push postprocess documentation before following the gated branch",
            ],
            "forbidden_until_gate": [
                "launch candidate-trail seed1",
                "launch bit-transition-spectrum seed0",
                "launch pair-set aggregation control",
                "make route-level or breakthrough claims",
            ],
            "gate_condition": (
                "postprocess summary exists, validates against the plan, and emits a decision "
                "such as support_candidate_trail_route, weak_candidate_trail_signal, or "
                "stop_candidate_trail_route"
            ),
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "improve local planning, readiness, or postprocess tooling without changing the active run",
            "wait for the watcher or sub-agent to retrieve result artifacts",
        ],
        "forbidden_until_gate": [
            "launch candidate-trail seed1",
            "launch bit-transition-spectrum seed0",
            "launch pair-set aggregation control",
            "SSH-poll or tmux-loop from the main thread",
            "make route-level or breakthrough claims",
        ],
        "gate_condition": (
            "watcher retrieves the expected candidate-trail JSONL rows and "
            "postprocess_allowed becomes true"
        ),
    }


def _candidate_plan_path(run_id: str) -> Path:
    seed = _candidate_seed_label(run_id)
    return Path(
        "configs/experiment/innovation1/"
        f"innovation1_spn_present_candidate_trail_consistency_r7_262k_{seed}.json"
    )


def _candidate_tmux_session(run_id: str) -> str:
    seed = _candidate_seed_label(run_id)
    return f"monitor_i1_candidate_trail_{seed}_20260702"


def _candidate_seed_label(run_id: str) -> str:
    return "seed1" if "_seed1_" in run_id else "seed0"


def _recommend_from_candidate_decision(route: dict[str, Any]) -> dict[str, Any]:
    decision = route["decision"]
    next_action = _candidate_decision_next_action(str(decision), route["next_action"])
    return {
        "branch": next_action["branch"],
        "run_id": route["run_id"],
        "decision": decision,
        "should_launch_remote": False,
        "reason": "candidate-trail summary is available; follow its gated branch after docs/update checks",
        "next_action": next_action,
    }


def _candidate_decision_next_action(decision: str, raw_next_action: dict[str, Any]) -> dict[str, Any]:
    seed1_config = "configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json"
    seed1_run_id = "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702"
    transition_seed0_config = (
        "configs/remote/"
        "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702.json"
    )
    transition_seed0_run_id = "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    if decision == "support_candidate_trail_route":
        return {
            **raw_next_action,
            "branch": "candidate_trail_seed1_confirmation",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1.json"
            ),
            "launch_remote_config": seed1_config,
            "suggested_feature_cache_workers": 4,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {seed1_config}"
            ),
            "run_id": seed1_run_id,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "weak_candidate_trail_signal":
        return {
            **raw_next_action,
            "branch": "candidate_trail_seed1_variance_check",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1.json"
            ),
            "launch_remote_config": seed1_config,
            "suggested_feature_cache_workers": 4,
            "fallback_plan": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {seed1_config}"
            ),
            "run_id": seed1_run_id,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "stop_candidate_trail_route":
        return {
            **raw_next_action,
            "branch": "bit_transition_spectrum_seed0",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0.json"
            ),
            "launch_remote_config": transition_seed0_config,
            "suggested_feature_cache_workers": 4,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {transition_seed0_config}"
            ),
            "run_id": transition_seed0_run_id,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    return {
        **raw_next_action,
        "branch": "manual_review",
        "should_launch_remote": False,
        "requires_implementation": False,
    }


def _recommend_from_transition_decision(route: dict[str, Any]) -> dict[str, Any]:
    decision = route["decision"]
    next_action = _transition_decision_next_action(str(decision), route["next_action"])
    return {
        "branch": next_action["branch"],
        "run_id": route["run_id"],
        "decision": decision,
        "should_launch_remote": False,
        "reason": "transition-spectrum summary is available; follow its gated branch after docs/update checks",
        "next_action": next_action,
    }


def _transition_decision_next_action(decision: str, raw_next_action: dict[str, Any]) -> dict[str, Any]:
    if decision == "support_transition_spectrum_route":
        return {
            **raw_next_action,
            "branch": "transition_spectrum_seed1_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "next_plan_doc": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
            ),
            "suggested_feature_cache_workers": 4,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                "--config configs/remote/<transition-spectrum-seed1-config>.json"
            ),
        }
    if decision == "weak_transition_spectrum_signal":
        return {
            **raw_next_action,
            "branch": "transition_spectrum_seed1_variance_check",
            "should_launch_remote": False,
            "requires_implementation": True,
            "next_plan_doc": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
            ),
            "suggested_feature_cache_workers": 4,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                "--config configs/remote/<transition-spectrum-seed1-config>.json"
            ),
        }
    if decision == "stop_transition_spectrum_route":
        return {
            **raw_next_action,
            "branch": "new_spn_hypothesis_plan",
            "should_launch_remote": False,
            "requires_implementation": True,
            "next_plan_doc": "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "fallback_plan_options": [
                "docs/experiments/innovation1-trail-family-consistency-plan.md",
                "docs/research/spn_structured_nn_research_plan.md",
            ],
        }
    return {
        **raw_next_action,
        "branch": "manual_review",
        "should_launch_remote": False,
        "requires_implementation": False,
    }


def _is_transition_spectrum_decision(decision: str) -> bool:
    return decision.startswith(("support_transition", "weak_transition", "stop_transition"))


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize_spn_evidence(args.root)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
