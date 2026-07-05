from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.monitor_health import _progress_summary, monitor_health_report


HIGH_ROUND_RUNS = [
    {
        "run_id": "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        "branch": "r9_weak_probe_seed0",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_262k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r9-weak-probe-plan.md",
        "expected_rows": 3,
        "postprocess_kind": "r9_weak_probe",
        "postprocess_script": "scripts/postprocess-r9-weak-probe",
    },
    {
        "run_id": "i1_present_r8_pairset_1m_seed0_gpu1_20260705",
        "branch": "r8_pairset_1m_seed0",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_pairset_r8_1m_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r8-round-extension-ladder-plan.md",
        "expected_rows": 2,
        "postprocess_kind": "r8_pairset_1m",
        "postprocess_script": "scripts/postprocess-r8-pairset-1m",
    },
]

FOLLOWUP_RUNS = [
    {
        "run_id": "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705",
        "branch": "r9_curriculum_from_r8_262k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md",
        "expected_rows": 2,
        "postprocess_kind": "r9_weak_probe",
        "postprocess_script": "scripts/postprocess-r9-weak-probe",
        "wait_branch": "wait_for_r9_curriculum_result",
        "postprocess_branch": "postprocess_r9_curriculum_result",
        "route_label": "r9 curriculum",
    },
    {
        "run_id": "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705",
        "branch": "r9_difference_screen_65k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r9-difference-screen-plan.md",
        "expected_rows": 7,
        "postprocess_kind": "difference_screen",
        "postprocess_script": "scripts/postprocess-difference-screen",
        "wait_branch": "wait_for_r9_difference_screen_result",
        "postprocess_branch": "postprocess_r9_difference_screen_result",
        "route_label": "r9 difference screen",
    },
    {
        "run_id": "i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705",
        "branch": "r8_integral_inverse_feature_65k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md",
        "expected_rows": 3,
        "postprocess_kind": "integral_inverse_feature",
        "postprocess_script": "scripts/postprocess-integral-inverse-feature",
        "wait_branch": "wait_for_integral_inverse_feature_result",
        "postprocess_branch": "postprocess_integral_inverse_feature_result",
        "route_label": "integral/inverse feature",
    },
    {
        "run_id": "i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705",
        "branch": "pair_mixer_r8_262k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-pair-mixer-consistency-plan.md",
        "expected_rows": 2,
        "postprocess_kind": "pair_mixer",
        "postprocess_script": "scripts/postprocess-pair-mixer-consistency",
        "wait_branch": "wait_for_pair_mixer_result",
        "postprocess_branch": "postprocess_pair_mixer_result",
        "route_label": "pair-mixer",
    },
    {
        "run_id": "i1_present_r8_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705",
        "branch": "pair_evidence_pooling_r8_65k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_r8_65k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-pair-evidence-pooling-screen-plan.md",
        "expected_rows": 4,
        "postprocess_kind": "pair_evidence_pooling",
        "postprocess_script": "scripts/postprocess-pair-evidence-pooling",
        "wait_branch": "wait_for_pair_evidence_pooling_result",
        "postprocess_branch": "postprocess_pair_evidence_pooling_result",
        "route_label": "pair-evidence pooling",
    },
    {
        "run_id": "i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705",
        "branch": "pair_evidence_pooling_r9_65k",
        "plan": "configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0.csv",
        "plan_doc": "docs/experiments/innovation1-present-r9-pair-evidence-pooling-screen-plan.md",
        "expected_rows": 4,
        "postprocess_kind": "pair_evidence_pooling",
        "postprocess_script": "scripts/postprocess-pair-evidence-pooling",
        "wait_branch": "wait_for_pair_evidence_pooling_result",
        "postprocess_branch": "postprocess_pair_evidence_pooling_result",
        "route_label": "pair-evidence pooling",
    },
]


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
    metrics = {key: summary[key] for key in keys if key in summary}

    for prefix, value in [
        ("baseline", summary.get("baseline")),
        ("best_candidate", summary.get("best_candidate")),
        ("best_overall", summary.get("best_overall")),
    ]:
        if not isinstance(value, dict):
            continue
        for key in ["accuracy", "calibrated_accuracy", "auc", "loss"]:
            if key in value:
                metrics[f"{prefix}_{key}"] = value[key]
        if "model" in value:
            metrics[f"{prefix}_model"] = value["model"]

    nested_keys = [
        "candidate_delta_vs_baseline_auc",
        "delta_vs_baseline_auc",
        "delta_vs_baseline",
    ]
    for key in nested_keys:
        if key in summary:
            metrics[key] = summary[key]
    return metrics


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
    high_round_running = _high_round_running(root)
    if high_round_running is not None:
        return high_round_running
    high_round_arbitration = _high_round_arbitration_recommendation(routes)
    if high_round_arbitration is not None:
        return high_round_arbitration
    followup_running = _followup_running(root)
    if followup_running is not None:
        return followup_running
    candidate_running = _candidate_trail_running(root)
    if candidate_running is not None:
        return candidate_running
    trail_family_running = _trail_family_running(root)
    if trail_family_running is not None:
        return trail_family_running
    sbox_prior_running = _sbox_prior_running(root)
    if sbox_prior_running is not None:
        return sbox_prior_running
    active_auxiliary_running = _active_auxiliary_running(root)
    if active_auxiliary_running is not None:
        return active_auxiliary_running
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


def _high_round_running(root: Path) -> dict[str, Any] | None:
    active_runs: list[dict[str, Any]] = []
    ready_runs: list[dict[str, Any]] = []
    for spec in HIGH_ROUND_RUNS:
        run_id = str(spec["run_id"])
        run_root = root / run_id
        if not run_root.exists() or list(run_root.glob("*_postprocess_summary.json")):
            continue
        health = _high_round_monitor_health(root, spec)
        monitor_log = run_root / "monitor" / "monitor.log"
        entry = {
            "branch": spec["branch"],
            "run_id": run_id,
            "status": health["status"],
            "monitor_log": str(monitor_log),
            "recent_monitor_lines": _tail_lines(monitor_log, 8),
            "heartbeat": health["heartbeat"],
            "needs_main_thread_intervention": health["needs_main_thread_intervention"],
            "postprocess_allowed": health["postprocess_allowed"],
            "postprocess_command": health["postprocess_command"],
            "results_jsonl": health["results_jsonl"],
            "results_jsonl_line_count": health["results_jsonl_line_count"],
            "expected_rows": health["expected_rows"],
            "progress_summary": _active_progress_summary(run_root),
            "monitor_health_command": _high_round_monitor_health_command(spec),
            "postprocess_when_ready_command": _high_round_postprocess_command(spec),
        }
        if health["postprocess_allowed"]:
            ready_runs.append(entry)
        elif _is_active_high_round_health(health, entry["recent_monitor_lines"]):
            active_runs.append(entry)

    if ready_runs:
        return {
            "branch": "postprocess_high_round_result",
            "status": "result_ready",
            "should_launch_remote": False,
            "reason": "one or more high-round results are ready locally and need postprocess before arbitration",
            "ready_runs": ready_runs,
            "active_runs": active_runs,
            "main_thread_policy": _high_round_main_thread_policy("postprocess"),
        }
    if active_runs:
        return {
            "branch": "wait_for_high_round_results",
            "status": "running",
            "should_launch_remote": False,
            "reason": "high-round r8/r9 watcher artifacts are still running; do not branch before postprocess gates",
            "active_runs": active_runs,
            "main_thread_policy": _high_round_main_thread_policy("waiting"),
        }
    return None


def _followup_running(root: Path) -> dict[str, Any] | None:
    for spec in FOLLOWUP_RUNS:
        run_id = str(spec["run_id"])
        run_root = root / run_id
        if not run_root.exists() or list(run_root.glob("*_postprocess_summary.json")):
            continue
        health = _followup_monitor_health(root, spec)
        monitor_log = run_root / "monitor" / "monitor.log"
        recent_lines = _tail_lines(monitor_log, 8)
        entry = {
            "branch": spec["branch"],
            "run_id": run_id,
            "status": health["status"],
            "should_launch_remote": False,
            "monitor_log": str(monitor_log),
            "recent_monitor_lines": recent_lines,
            "heartbeat": health["heartbeat"],
            "needs_main_thread_intervention": health["needs_main_thread_intervention"],
            "postprocess_allowed": health["postprocess_allowed"],
            "postprocess_command": health["postprocess_command"],
            "results_jsonl": health["results_jsonl"],
            "results_jsonl_line_count": health["results_jsonl_line_count"],
            "expected_rows": health["expected_rows"],
            "progress_summary": _active_progress_summary(run_root),
            "monitor_health_command": _followup_monitor_health_command(spec),
            "postprocess_when_ready_command": _followup_postprocess_command(spec),
        }
        if health["postprocess_allowed"]:
            return {
                **entry,
                "branch": str(spec["postprocess_branch"]),
                "reason": (
                    f"{spec['route_label']} results are ready locally and need "
                    "postprocess before branch decisions"
                ),
                "main_thread_policy": _followup_main_thread_policy("postprocess"),
            }
        if _is_active_high_round_health(health, recent_lines):
            branch = (
                f"diagnose_{spec['postprocess_kind']}_launch"
                if health["needs_main_thread_intervention"]
                else str(spec["wait_branch"])
            )
            return {
                **entry,
                "branch": branch,
                "reason": (
                    f"{spec['route_label']} run has monitor activity but no "
                    "postprocess summary yet"
                ),
                "main_thread_policy": _followup_main_thread_policy("waiting"),
            }
    return None


def _high_round_arbitration_recommendation(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    high_round_routes = [
        route
        for route in routes
        if str(route.get("decision") or "").startswith(
            (
                "strong_r9",
                "r9_weak",
                "near_random_r9",
                "stop_from_scratch_r9",
                "baseline_best_or_candidate",
                "support_r8_pairset_1m",
                "weak_r8_pairset_1m",
                "stop_or_rethink_r8_pairset",
            )
        )
    ]
    summaries = [str(route["summary"]) for route in high_round_routes if route.get("summary")]
    if len(summaries) < 2:
        return None
    return {
        "branch": "arbitrate_high_round_next_actions",
        "status": "ready_for_arbitration",
        "should_launch_remote": False,
        "reason": "multiple high-round postprocess summaries exist; arbitrate before launching any next branch",
        "summary_count": len(summaries),
        "summaries": summaries,
        "arbitration_command": _high_round_arbitration_command(summaries),
        "main_thread_policy": _high_round_main_thread_policy("arbitrate"),
    }


def _high_round_monitor_health(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    return monitor_health_report(
        run_id=str(spec["run_id"]),
        root=root,
        plan_path=Path(str(spec["plan"])),
        plan_doc_paths=[Path(str(spec["plan_doc"]))],
        expected_rows=int(spec["expected_rows"]),
        postprocess_kind=str(spec["postprocess_kind"]),
    )


def _high_round_monitor_health_command(spec: dict[str, Any]) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {spec['run_id']} "
        f"--plan {spec['plan']} "
        f"--plan-doc {spec['plan_doc']} "
        f"--expected-rows {spec['expected_rows']} "
        f"--postprocess-kind {spec['postprocess_kind']}"
    )


def _high_round_postprocess_command(spec: dict[str, Any]) -> str:
    run_id = str(spec["run_id"])
    return (
        f"UV_CACHE_DIR=/tmp/uv-cache uv run python {spec['postprocess_script']} "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {spec['plan']} "
        f"--expected-rows {spec['expected_rows']} "
        f"--update-plan-doc {spec['plan_doc']}"
    )


def _high_round_arbitration_command(summaries: list[str]) -> str:
    summary_args = " ".join(f"--summary {summary}" for summary in summaries)
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/arbitrate-next-actions "
        f"{summary_args} "
        "--output outputs/remote_results/high_round_next_action_arbitration.json"
    )


def _followup_monitor_health(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    return monitor_health_report(
        run_id=str(spec["run_id"]),
        root=root,
        plan_path=Path(str(spec["plan"])),
        plan_doc_paths=[Path(str(spec["plan_doc"]))],
        expected_rows=int(spec["expected_rows"]),
        postprocess_kind=str(spec["postprocess_kind"]),
    )


def _followup_monitor_health_command(spec: dict[str, Any]) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {spec['run_id']} "
        f"--plan {spec['plan']} "
        f"--plan-doc {spec['plan_doc']} "
        f"--expected-rows {spec['expected_rows']} "
        f"--postprocess-kind {spec['postprocess_kind']}"
    )


def _followup_postprocess_command(spec: dict[str, Any]) -> str:
    run_id = str(spec["run_id"])
    return (
        f"UV_CACHE_DIR=/tmp/uv-cache uv run python {spec['postprocess_script']} "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {spec['plan']} "
        f"--expected-rows {spec['expected_rows']} "
        f"--update-plan-doc {spec['plan_doc']}"
    )


def _is_active_high_round_health(health: dict[str, Any], recent_lines: list[str]) -> bool:
    active_statuses = {
        "running",
        "stale_monitor",
        "launch_stalled",
        "remote_artifacts_missing",
        "unknown",
    }
    has_activity = bool(recent_lines) or bool(health.get("heartbeat", {}).get("newest_timestamp"))
    return str(health.get("status")) in active_statuses and has_activity


def _high_round_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run each listed high-round postprocess command",
                "update docs/experiments through the postprocess command",
                "commit and push result documentation before branch launch",
                "run arbitrate-next-actions if more than one high-round summary is available",
            ],
            "forbidden_until_gate": [
                "launch r8 seed1",
                "launch r9 seed1",
                "launch r9 1M",
                "launch r9 curriculum",
                "launch r9 difference screen",
                "make r8/r9 route-level or breakthrough claims",
            ],
            "gate_condition": "route-specific postprocess summaries exist, validate against plans, and emit next_action readiness",
        }
    if state == "arbitrate":
        return {
            "allowed_actions": [
                "run the listed arbitrate-next-actions command",
                "launch only the selected branch after checking readiness and committing documentation",
            ],
            "forbidden_until_gate": [
                "launch multiple high-round follow-up branches in parallel",
                "launch a lower-priority branch while a stronger r9 branch is selected",
                "make formal high-round claims before 1M/class multi-seed evidence",
            ],
            "gate_condition": "arbitration report selects one launchable branch and deferred branches are recorded",
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "improve local planning, readiness, or postprocess tooling without changing active remote runs",
            "wait for watcher/sub-agent retrieval",
        ],
        "forbidden_until_gate": [
            "SSH-poll or tmux-loop from the main thread",
            "launch r8/r9/r10 follow-up branches",
            "make route-level or breakthrough claims",
        ],
        "gate_condition": "watcher retrieves expected high-round JSONL rows and postprocess_allowed becomes true",
    }


def _followup_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run the listed route-specific postprocess command",
                "update docs/experiments through the postprocess command",
                "commit and push result documentation before any branch launch",
                "feed the postprocess summary into the normal next-action gate",
            ],
            "forbidden_until_gate": [
                "launch pair-mixer seed/scale follow-up",
                "launch pair-evidence pooling confirmation",
                "make route-level claims from a screen or medium diagnostic",
            ],
            "gate_condition": "postprocess summary validates against the plan and emits a next_action decision",
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "wait for watcher/sub-agent retrieval",
            "improve local postprocess or readiness tooling without changing active remote runs",
        ],
        "forbidden_until_gate": [
            "SSH-poll or tmux-loop from the main thread",
            "launch another follow-up branch in parallel",
            "interpret partial JSONL or empty result files as model evidence",
        ],
        "gate_condition": "watcher retrieves the expected JSONL rows and postprocess_allowed becomes true",
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


def _trail_family_running(root: Path) -> dict[str, Any] | None:
    for run_root in sorted(root.glob("i1_trail_family*")):
        summaries = list(run_root.glob("*_postprocess_summary.json"))
        if summaries:
            continue
        monitor_log = run_root / "monitor" / "monitor.log"
        recent_lines = _tail_lines(monitor_log, 8)
        health = _trail_family_monitor_health(root, run_root.name)
        if health["postprocess_allowed"]:
            return {
                "branch": "postprocess_trail_family_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "trail-family results are ready locally and need postprocess before branch decisions",
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
                "conditional_followup": _trail_family_conditional_followup(),
                "deferred_candidate_queue": _trail_family_deferred_candidate_queue(),
                "monitor_health_command": _trail_family_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _trail_family_postprocess_command(run_root),
                "main_thread_policy": _trail_family_main_thread_policy("postprocess"),
            }
        if health["status"] in {"running", "stale_monitor", "launch_stalled", "unknown"} and any(
            "running" in line for line in recent_lines
        ):
            return {
                "branch": "wait_for_trail_family_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "trail-family run has monitor activity but no postprocess summary yet",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": health["needs_main_thread_intervention"],
                "postprocess_allowed": health["postprocess_allowed"],
                "progress_summary": _active_progress_summary(run_root),
                "conditional_followup": _trail_family_conditional_followup(),
                "deferred_candidate_queue": _trail_family_deferred_candidate_queue(),
                "monitor_health_command": _trail_family_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _trail_family_postprocess_command(run_root),
                "main_thread_policy": _trail_family_main_thread_policy("waiting"),
            }
    return None


def _active_auxiliary_running(root: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for run_root in sorted(root.glob("i1_active_auxiliary*"), reverse=True):
        summaries = list(run_root.glob("*_postprocess_summary.json"))
        if summaries:
            continue
        monitor_log = run_root / "monitor" / "monitor.log"
        recent_lines = _tail_lines(monitor_log, 8)
        health = _active_auxiliary_monitor_health(root, run_root.name)
        if health["status"] == "failed":
            candidates.append({
                "branch": "active_auxiliary_failed",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "active-auxiliary run failed; write failure or repair plan before launching another branch",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": True,
                "postprocess_allowed": False,
                "failed_markers": health["failed_markers"],
                "results_jsonl": health["results_jsonl"],
                "results_jsonl_line_count": health["results_jsonl_line_count"],
                "expected_rows": health["expected_rows"],
                "progress_summary": _active_progress_summary(run_root),
                "conditional_followup": _active_auxiliary_conditional_followup(),
                "monitor_health_command": _active_auxiliary_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _active_auxiliary_postprocess_command(run_root),
                "main_thread_policy": _active_auxiliary_main_thread_policy("failed"),
            })
            continue
        if health["postprocess_allowed"]:
            candidates.append({
                "branch": "postprocess_active_auxiliary_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "active-auxiliary results are ready locally and need postprocess before branch decisions",
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
                "conditional_followup": _active_auxiliary_conditional_followup(),
                "monitor_health_command": _active_auxiliary_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _active_auxiliary_postprocess_command(run_root),
                "main_thread_policy": _active_auxiliary_main_thread_policy("postprocess"),
            })
            continue
        active_statuses = {
            "running",
            "stale_monitor",
            "launch_stalled",
            "remote_artifacts_missing",
            "unknown",
        }
        has_monitor_activity = bool(recent_lines) or bool(health.get("heartbeat", {}).get("newest_timestamp"))
        if health["status"] in active_statuses and has_monitor_activity:
            branch = (
                "diagnose_active_auxiliary_launch"
                if health["needs_main_thread_intervention"]
                else "wait_for_active_auxiliary_result"
            )
            candidates.append({
                "branch": branch,
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "active-auxiliary run has monitor activity but no postprocess summary yet",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": health["needs_main_thread_intervention"],
                "postprocess_allowed": health["postprocess_allowed"],
                "progress_summary": _active_progress_summary(run_root),
                "conditional_followup": _active_auxiliary_conditional_followup(),
                "monitor_health_command": _active_auxiliary_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _active_auxiliary_postprocess_command(run_root),
                "main_thread_policy": _active_auxiliary_main_thread_policy("waiting"),
            })
    if not candidates:
        return None
    return sorted(candidates, key=_active_auxiliary_candidate_rank, reverse=True)[0]


def _sbox_prior_running(root: Path) -> dict[str, Any] | None:
    for run_root in sorted(root.glob("i1_sbox_prior_gate*"), reverse=True):
        summaries = list(run_root.glob("*_postprocess_summary.json"))
        if summaries:
            continue
        monitor_log = run_root / "monitor" / "monitor.log"
        recent_lines = _tail_lines(monitor_log, 8)
        health = _sbox_prior_monitor_health(root, run_root.name)
        if health["postprocess_allowed"]:
            return {
                "branch": "postprocess_sbox_prior_result",
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "S-box prior gate results are ready locally and need postprocess before branch decisions",
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
                "monitor_health_command": _sbox_prior_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _sbox_prior_postprocess_command(run_root),
                "main_thread_policy": _sbox_prior_main_thread_policy("postprocess"),
            }
        active_statuses = {
            "running",
            "stale_monitor",
            "launch_stalled",
            "remote_artifacts_missing",
            "unknown",
        }
        has_monitor_activity = bool(recent_lines) or bool(health.get("heartbeat", {}).get("newest_timestamp"))
        if health["status"] in active_statuses and has_monitor_activity:
            branch = (
                "diagnose_sbox_prior_launch"
                if health["needs_main_thread_intervention"]
                else "wait_for_sbox_prior_result"
            )
            return {
                "branch": branch,
                "run_id": run_root.name,
                "status": health["status"],
                "should_launch_remote": False,
                "reason": "S-box prior gate run has monitor activity but no postprocess summary yet",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "heartbeat": health["heartbeat"],
                "needs_main_thread_intervention": health["needs_main_thread_intervention"],
                "postprocess_allowed": health["postprocess_allowed"],
                "progress_summary": _active_progress_summary(run_root),
                "monitor_health_command": _sbox_prior_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _sbox_prior_postprocess_command(run_root),
                "main_thread_policy": _sbox_prior_main_thread_policy("waiting"),
            }
    return None


def _active_auxiliary_candidate_rank(candidate: dict[str, Any]) -> tuple[int, str]:
    if candidate["branch"] == "postprocess_active_auxiliary_result":
        priority = 4
    elif not bool(candidate.get("needs_main_thread_intervention")):
        priority = 3
    elif candidate["branch"] == "diagnose_active_auxiliary_launch":
        priority = 2
    else:
        priority = 1
    return priority, str(candidate["run_id"])


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


def _trail_family_monitor_health(root: Path, run_id: str) -> dict[str, Any]:
    return monitor_health_report(
        run_id=run_id,
        root=root,
        tmux_session=_trail_family_tmux_session(run_id),
        plan_path=_trail_family_plan_path(run_id),
        plan_doc_paths=[Path("docs/experiments/innovation1-trail-family-consistency-plan.md")],
        expected_rows=4,
        postprocess_kind="trail_family",
    )


def _active_auxiliary_monitor_health(root: Path, run_id: str) -> dict[str, Any]:
    return monitor_health_report(
        run_id=run_id,
        root=root,
        tmux_session=_active_auxiliary_tmux_session(run_id),
        plan_path=_active_auxiliary_plan_path(run_id),
        plan_doc_paths=[Path("docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md")],
        expected_rows=3,
        postprocess_kind="active_auxiliary",
    )


def _sbox_prior_monitor_health(root: Path, run_id: str) -> dict[str, Any]:
    return monitor_health_report(
        run_id=run_id,
        root=root,
        tmux_session=_sbox_prior_tmux_session(run_id),
        plan_path=_sbox_prior_plan_path(run_id),
        plan_doc_paths=[Path("docs/experiments/innovation1-sbox-transition-prior-gate-plan.md")],
        expected_rows=4,
        postprocess_kind="sbox_prior",
    )


def _active_progress_summary(run_root: Path) -> dict[str, Any]:
    progress = _progress_summary(run_root)
    keys = [
        "path",
        "exists",
        "line_count",
        "parsed_line_count",
        "latest_event",
        "latest_split",
        "latest_total_rows",
        "latest_samples_per_class",
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
        "cache_rows_per_second",
        "cache_rate_window_seconds",
        "cache_rate_window_rows",
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


def _trail_family_monitor_health_command(run_id: str) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {run_id} "
        f"--tmux-session {_trail_family_tmux_session(run_id)} "
        f"--plan {_trail_family_plan_path(run_id)} "
        "--plan-doc docs/experiments/innovation1-trail-family-consistency-plan.md "
        "--expected-rows 4 --postprocess-kind trail_family"
    )


def _active_auxiliary_monitor_health_command(run_id: str) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {run_id} "
        f"--tmux-session {_active_auxiliary_tmux_session(run_id)} "
        f"--plan {_active_auxiliary_plan_path(run_id)} "
        "--plan-doc docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md "
        "--expected-rows 3 --postprocess-kind active_auxiliary"
    )


def _sbox_prior_monitor_health_command(run_id: str) -> str:
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {run_id} "
        f"--tmux-session {_sbox_prior_tmux_session(run_id)} "
        f"--plan {_sbox_prior_plan_path(run_id)} "
        "--plan-doc docs/experiments/innovation1-sbox-transition-prior-gate-plan.md "
        "--expected-rows 4 --postprocess-kind sbox_prior"
    )


def _trail_family_postprocess_command(run_root: Path) -> str:
    run_id = run_root.name
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-trail-family "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {_trail_family_plan_path(run_id)} "
        "--expected-rows 4 "
        "--update-plan-doc docs/experiments/innovation1-trail-family-consistency-plan.md"
    )


def _active_auxiliary_postprocess_command(run_root: Path) -> str:
    run_id = run_root.name
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-active-auxiliary "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {_active_auxiliary_plan_path(run_id)} "
        "--expected-rows 3 "
        "--update-plan-doc docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md"
    )


def _sbox_prior_postprocess_command(run_root: Path) -> str:
    run_id = run_root.name
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-sbox-prior "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        f"--plan {_sbox_prior_plan_path(run_id)} "
        "--expected-rows 4 "
        "--update-plan-doc docs/experiments/innovation1-sbox-transition-prior-gate-plan.md"
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


def _trail_family_conditional_followup() -> dict[str, Any]:
    config = Path("configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json")
    readiness = _remote_readiness(config)
    return {
        "branch": "trail_family_seed1_confirmation_or_variance_check",
        "run_id": "i1_trail_family_r7_262k_seed1_gpu1_20260702",
        "launch_remote_config": str(config),
        "launch_gate": "support_trail_family_route or weak_trail_family_signal",
        "readiness_pass": readiness.get("status") == "pass",
        "readiness": readiness,
        "should_launch_now": False,
        "fallback_if_stop": "active_auxiliary_seed0",
        "fallback_after_active_auxiliary_stop": "sbox_transition_prior_gate_seed0",
    }


def _active_auxiliary_conditional_followup() -> dict[str, Any]:
    seed1_config = Path(
        "configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
    )
    fallback_config = Path(
        "configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
    )
    seed1_readiness = _remote_readiness(seed1_config)
    fallback_readiness = _remote_readiness(fallback_config)
    return {
        "branch": "active_auxiliary_seed1_confirmation_or_sbox_prior_fallback",
        "run_id": "i1_active_auxiliary_r7_262k_seed1_gpu1_20260703",
        "launch_remote_config": str(seed1_config),
        "launch_gate": "support_active_auxiliary_route or weak_active_auxiliary_signal",
        "readiness_pass": seed1_readiness.get("status") == "pass",
        "readiness": seed1_readiness,
        "should_launch_now": False,
        "fallback_if_stop": "sbox_transition_prior_gate_seed0",
        "fallback_run_id": "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703",
        "fallback_remote_config": str(fallback_config),
        "fallback_readiness_status": str(fallback_readiness.get("status", "unknown")),
    }


def _trail_family_deferred_candidate_queue() -> list[dict[str, str]]:
    candidates = [
        {
            "branch": "trail_family_seed1_confirmation_or_variance_check",
            "launch_gate": "support_trail_family_route or weak_trail_family_signal",
            "run_id": "i1_trail_family_r7_262k_seed1_gpu1_20260702",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json"
            ),
            "plan_doc": "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "status": "prepared_conditional",
        },
        {
            "branch": "active_auxiliary_seed0",
            "launch_gate": "stop/tied trail-family gate or explicit user selection",
            "run_id": "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json"
            ),
            "plan_doc": "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md",
            "status": "prepared_deferred",
        },
        {
            "branch": "sbox_transition_prior_gate_seed0",
            "launch_gate": "stop/tied trail-family gate after active-auxiliary is not selected",
            "run_id": "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
            ),
            "plan_doc": "docs/experiments/innovation1-sbox-transition-prior-gate-plan.md",
            "status": "prepared_deferred",
        },
    ]
    return [_with_readiness_status(candidate) for candidate in candidates]


def _with_readiness_status(candidate: dict[str, str]) -> dict[str, str]:
    config = candidate.get("launch_remote_config")
    if config:
        candidate = dict(candidate)
        candidate["readiness_status"] = str(_remote_readiness(Path(config)).get("status", "unknown"))
    return candidate


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


def _trail_family_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run the listed postprocess_when_ready_command",
                "validate gate artifacts and update docs/experiments",
                "commit and push postprocess documentation before following the gated branch",
            ],
            "forbidden_until_gate": [
                "launch trail-family seed1",
                "launch active-auxiliary seed0",
                "launch S-box transition prior seed0",
                "make route-level or breakthrough claims",
            ],
            "gate_condition": (
                "postprocess summary exists, validates against the plan, and emits a decision "
                "such as support_trail_family_route, weak_trail_family_signal, or stop_trail_family_route"
            ),
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "improve local planning, readiness, or postprocess tooling without changing the active run",
            "wait for the watcher or sub-agent to retrieve result artifacts",
        ],
        "forbidden_until_gate": [
            "launch trail-family seed1",
            "launch active-auxiliary seed0",
            "launch S-box transition prior seed0",
            "SSH-poll or tmux-loop from the main thread",
            "make route-level or breakthrough claims",
        ],
        "gate_condition": "watcher retrieves the expected trail-family JSONL rows and postprocess_allowed becomes true",
    }


def _active_auxiliary_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run the listed postprocess_when_ready_command",
                "validate gate artifacts and update docs/experiments",
                "commit and push postprocess documentation before following the gated branch",
            ],
            "forbidden_until_gate": [
                "launch active-auxiliary seed1",
                "launch S-box transition prior seed0",
                "make route-level or breakthrough claims",
            ],
            "gate_condition": (
                "postprocess summary exists, validates against the plan, and emits a decision "
                "such as support_active_auxiliary_route, weak_active_auxiliary_signal, or "
                "stop_active_auxiliary_route"
            ),
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "improve local planning, readiness, or postprocess tooling without changing the active run",
            "wait for the watcher or sub-agent to retrieve result artifacts",
        ],
        "forbidden_until_gate": [
            "launch active-auxiliary seed1",
            "launch S-box transition prior seed0",
            "SSH-poll or tmux-loop from the main thread",
            "make route-level or breakthrough claims",
        ],
        "gate_condition": "watcher retrieves the expected active-auxiliary JSONL rows and postprocess_allowed becomes true",
    }


def _sbox_prior_main_thread_policy(state: str) -> dict[str, Any]:
    if state == "postprocess":
        return {
            "allowed_actions": [
                "run the listed postprocess_when_ready_command",
                "validate gate artifacts and update docs/experiments",
                "commit and push postprocess documentation before following the gated branch",
            ],
            "forbidden_until_gate": [
                "launch S-box transition prior seed1",
                "launch a replacement SPN route",
                "make route-level or breakthrough claims",
            ],
            "gate_condition": (
                "postprocess summary exists, validates against the plan, and emits a decision "
                "such as support_sbox_prior_route, weak_sbox_prior_signal, or stop_sbox_prior_route"
            ),
        }
    return {
        "allowed_actions": [
            "perform bounded local status checks from retrieved artifacts",
            "improve local planning, readiness, or postprocess tooling without changing the active run",
            "wait for the watcher or sub-agent to retrieve result artifacts",
        ],
        "forbidden_until_gate": [
            "launch S-box transition prior seed1",
            "launch a replacement SPN route",
            "SSH-poll or tmux-loop from the main thread",
            "make route-level or breakthrough claims",
        ],
        "gate_condition": "watcher retrieves the expected S-box prior JSONL rows and postprocess_allowed becomes true",
    }


def _trail_family_plan_path(run_id: str) -> Path:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    return Path(
        "configs/experiment/innovation1/"
        f"innovation1_spn_present_trail_family_r7_262k_{seed}.json"
    )


def _trail_family_tmux_session(run_id: str) -> str:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    return f"monitor_i1_trail_family_{seed}_20260702"


def _active_auxiliary_plan_path(run_id: str) -> Path:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    retry = "_retry1" if "_retry1_" in run_id else ""
    return Path(
        "configs/experiment/innovation1/"
        f"innovation1_spn_present_active_auxiliary_r7_262k_{seed}{retry}.json"
    )


def _active_auxiliary_tmux_session(run_id: str) -> str:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    if "_retry1_" in run_id:
        return f"monitor_i1_active_auxiliary_{seed}_retry1_20260704"
    return f"monitor_i1_active_auxiliary_{seed}_20260703"


def _sbox_prior_plan_path(run_id: str) -> Path:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    return Path(
        "configs/experiment/innovation1/"
        f"innovation1_spn_present_sbox_transition_prior_gate_r7_262k_{seed}.csv"
    )


def _sbox_prior_tmux_session(run_id: str) -> str:
    seed = "seed1" if "_seed1_" in run_id else "seed0"
    return f"monitor_i1_sbox_prior_gate_{seed}_20260704"


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
    seed1_config = (
        "configs/remote/"
        "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702.json"
    )
    seed1_run_id = "i1_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702"
    trail_family_seed0_config = (
        "configs/remote/"
        "innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json"
    )
    trail_family_seed0_run_id = "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    if decision == "support_transition_spectrum_route":
        return {
            **raw_next_action,
            "branch": "transition_spectrum_seed1_confirmation",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
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
    if decision == "weak_transition_spectrum_signal":
        return {
            **raw_next_action,
            "branch": "transition_spectrum_seed1_variance_check",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
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
    if decision == "stop_transition_spectrum_route":
        return {
            **raw_next_action,
            "branch": "trail_family_seed0",
            "should_launch_remote": True,
            "requires_implementation": False,
            "next_plan_doc": "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_trail_family_r7_262k_seed0.json"
            ),
            "launch_remote_config": trail_family_seed0_config,
            "suggested_feature_cache_workers": 4,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {trail_family_seed0_config}"
            ),
            "run_id": trail_family_seed0_run_id,
            "monitor_owner": "tmux watcher or sub-agent",
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
