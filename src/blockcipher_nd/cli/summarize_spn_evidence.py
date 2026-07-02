from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.monitor_health import _progress_summary


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
        if any("running" in line for line in recent_lines):
            return {
                "branch": "wait_for_candidate_trail_result",
                "run_id": run_root.name,
                "status": "running",
                "should_launch_remote": False,
                "reason": "candidate-trail run has monitor activity but no postprocess summary yet",
                "monitor_log": str(monitor_log),
                "recent_monitor_lines": recent_lines,
                "progress_summary": _active_progress_summary(run_root),
                "monitor_health_command": _candidate_monitor_health_command(run_root.name),
                "postprocess_when_ready_command": _candidate_postprocess_command(run_root),
            }
    return None


def _active_progress_summary(run_root: Path) -> dict[str, Any]:
    progress = _progress_summary(run_root)
    keys = [
        "path",
        "exists",
        "latest_event",
        "cache_rows_done",
        "cache_total_rows",
        "cache_class_rows_done",
        "cache_class_total",
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
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health "
        f"--root outputs/remote_results --run-id {run_id} "
        "--tmux-session monitor_i1_candidate_trail_seed0_20260702 "
        "--plan configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed0.json "
        "--plan-doc docs/experiments/innovation1-candidate-trail-consistency-plan.md "
        "--expected-rows 4 --postprocess-kind candidate_trail"
    )


def _candidate_postprocess_command(run_root: Path) -> str:
    run_id = run_root.name
    return (
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-candidate-trail "
        f"--results outputs/remote_results/{run_id}/results/{run_id}.jsonl "
        f"--output-dir outputs/remote_results/{run_id} "
        f"--run-id {run_id} "
        "--plan configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed0.json "
        "--expected-rows 4 "
        "--update-plan-doc docs/experiments/innovation1-candidate-trail-consistency-plan.md"
    )


def _recommend_from_candidate_decision(route: dict[str, Any]) -> dict[str, Any]:
    decision = route["decision"]
    if decision == "support_candidate_trail_route":
        branch = "candidate_trail_seed1_confirmation"
    elif decision == "weak_candidate_trail_signal":
        branch = "candidate_trail_seed1_variance_check"
    elif decision == "stop_candidate_trail_route":
        branch = "bit_transition_spectrum_seed0"
    else:
        branch = "manual_review"
    return {
        "branch": branch,
        "run_id": route["run_id"],
        "decision": decision,
        "should_launch_remote": False,
        "reason": "candidate-trail summary is available; follow its gated branch after docs/update checks",
        "next_action": route["next_action"],
    }


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
