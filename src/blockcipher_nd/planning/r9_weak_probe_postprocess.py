from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.next_action_readiness import next_action_readiness_report
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


BASELINE_MODEL = "present_zhang_wang_keras_mcnd"
INVP_MODEL = "present_nibble_invp_only_spn_only"
PAIR_MODEL = "present_nibble_invp_pair_consistency_spn_only"
R9_SEED1_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_r9_weak_probe_262k_seed1_gpu0_20260705.json"
)
R9_SEED1_RUN_ID = "i1_present_r9_weak_probe_262k_seed1_gpu0_20260705"
R9_1M_SEED0_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_r9_1m_seed0_gpu0_20260705.json"
)
R9_1M_SEED0_RUN_ID = "i1_present_r9_1m_seed0_gpu0_20260705"
R9_CURRICULUM_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json"
)
R9_CURRICULUM_RUN_ID = "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705"

DEFAULT_NEAR_RANDOM_AUC_CEILING = 0.505
DEFAULT_WEAK_TRACE_AUC_CEILING = 0.52
DEFAULT_STRONG_AUC_THRESHOLD = 0.55
DEFAULT_BASELINE_MARGIN = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a PRESENT r9 weak-probe diagnostic result.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument("--near-random-auc-ceiling", type=float, default=DEFAULT_NEAR_RANDOM_AUC_CEILING)
    parser.add_argument("--weak-trace-auc-ceiling", type=float, default=DEFAULT_WEAK_TRACE_AUC_CEILING)
    parser.add_argument("--strong-auc-threshold", type=float, default=DEFAULT_STRONG_AUC_THRESHOLD)
    parser.add_argument("--baseline-margin", type=float, default=DEFAULT_BASELINE_MARGIN)
    parser.add_argument("--claim-scope", default=None)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_r9_weak_probe_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int = 3,
    near_random_auc_ceiling: float = DEFAULT_NEAR_RANDOM_AUC_CEILING,
    weak_trace_auc_ceiling: float = DEFAULT_WEAK_TRACE_AUC_CEILING,
    strong_auc_threshold: float = DEFAULT_STRONG_AUC_THRESHOLD,
    baseline_margin: float = DEFAULT_BASELINE_MARGIN,
    claim_scope: str | None = None,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    curves_path = output_dir / f"{run_id}_curves.svg"
    history_path = output_dir / f"{run_id}_history.csv"
    plot_report = plot_jsonl_training_curves(results_path, curves_path, title=run_id)
    plot_report["history_csv"] = write_history_csv(results_path, history_path)

    gate_report = gate_r9_weak_probe_result(
        results_path,
        expected_rows=expected_rows,
        near_random_auc_ceiling=near_random_auc_ceiling,
        weak_trace_auc_ceiling=weak_trace_auc_ceiling,
        strong_auc_threshold=strong_auc_threshold,
        baseline_margin=baseline_margin,
        claim_scope=claim_scope,
    )
    gate_path = output_dir / f"{run_id}_r9_weak_probe_gate.json"
    _write_json(gate_path, gate_report)

    validation_status = validation_report["status"] if validation_report is not None else "not_run"
    status = "pass" if gate_report["status"] == "pass" and validation_status in {"pass", "not_run"} else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path) if plan_path is not None else None,
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path) if validation_path is not None else None,
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "r9_weak_probe_gate": str(gate_path),
        "validation_status": validation_status,
        "r9_weak_probe_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "baseline": gate_report["baseline"],
        "best_candidate": gate_report["best_candidate"],
        "best_overall": gate_report["best_overall"],
        "candidate_delta_vs_baseline_auc": gate_report["candidate_delta_vs_baseline_auc"],
        "ranking": gate_report["ranking"],
        "thresholds": gate_report["thresholds"],
        "claim_scope": gate_report["claim_scope"],
        "plot_report": plot_report,
    }
    report["next_action"] = _next_action(report)
    report["next_steps"] = _next_steps(report)

    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    next_action_readiness_path = output_dir / f"{run_id}_next_action_readiness.json"
    report["summary"] = str(summary_path)
    report["summary_markdown"] = str(markdown_path)
    report["next_action_readiness"] = str(next_action_readiness_path)

    update_paths = plan_doc_paths or []
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_r9_weak_probe_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    _write_json(next_action_readiness_path, _next_action_readiness_report(report, summary_path))
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_r9_weak_probe_result(
    results_path: Path,
    *,
    expected_rows: int = 3,
    near_random_auc_ceiling: float = DEFAULT_NEAR_RANDOM_AUC_CEILING,
    weak_trace_auc_ceiling: float = DEFAULT_WEAK_TRACE_AUC_CEILING,
    strong_auc_threshold: float = DEFAULT_STRONG_AUC_THRESHOLD,
    baseline_margin: float = DEFAULT_BASELINE_MARGIN,
    claim_scope: str | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    entries = [_entry(row) for row in rows]
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")

    by_model = {entry["model"]: entry for entry in entries}
    baseline = by_model.get(BASELINE_MODEL)
    candidates = [entry for entry in entries if entry["model"] in {INVP_MODEL, PAIR_MODEL}]
    if baseline is None:
        errors.append(f"missing baseline model {BASELINE_MODEL}")
    for model in (INVP_MODEL, PAIR_MODEL):
        if model not in by_model:
            errors.append(f"missing candidate model {model}")
    for entry in entries:
        if not isinstance(entry.get("auc"), (int, float)):
            errors.append(f"missing auc for {entry['model']}")

    ranking = sorted(entries, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    candidate_ranking = sorted(candidates, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    best_overall = ranking[0] if ranking else None
    best_candidate = candidate_ranking[0] if candidate_ranking else None
    candidate_delta = _auc_delta(best_candidate, baseline)

    if errors:
        decision = "invalid_r9_weak_probe_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "The r9 weak-probe result is incomplete or missing required metrics."
        status = "fail"
    else:
        status = "pass"
        best_candidate_auc = float(best_candidate["auc"])
        if best_candidate_auc <= near_random_auc_ceiling:
            decision = "stop_from_scratch_r9_r10_plan_curriculum_or_difference_search"
            action = "do_not_scale_from_scratch_r9; prepare curriculum_or_difference_search"
            interpretation = "The best r9 SPN candidate is near random under this protocol."
        elif best_candidate_auc <= weak_trace_auc_ceiling:
            decision = "near_random_r9_weak_trace_check_variance_or_aggregation"
            action = "do_not_launch_1m; prefer variance_check_or_application_level_aggregation"
            interpretation = "The best r9 SPN candidate shows only a weak trace below the promotion band."
        elif candidate_delta is not None and best_candidate_auc > strong_auc_threshold and candidate_delta >= baseline_margin:
            decision = "strong_r9_diagnostic_prepare_1m_seed0"
            action = "prepare_r9_1m_seed0_plan_before_any_formal_claim"
            interpretation = "The best r9 SPN candidate is a strong single-seed diagnostic above baseline."
        elif candidate_delta is not None and candidate_delta > 0:
            decision = "r9_weak_positive_prepare_seed1_or_curriculum_scale"
            action = "prepare_seed1_or_curriculum_scale_after_documenting_medium_diagnostic"
            interpretation = "The best r9 SPN candidate beats baseline, but not by the strong diagnostic gate."
        else:
            decision = "baseline_best_or_candidate_not_above_baseline"
            action = "do_not_branch_to_candidate_scale; inspect curriculum_or_difference_search"
            interpretation = "The r9 baseline is tied or better than the SPN candidates at this scale."

    return {
        "status": status,
        "results": str(results_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "errors": errors,
        "baseline": baseline,
        "best_candidate": best_candidate,
        "best_overall": best_overall,
        "candidate_delta_vs_baseline_auc": candidate_delta,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "thresholds": {
            "near_random_auc_ceiling": near_random_auc_ceiling,
            "weak_trace_auc_ceiling": weak_trace_auc_ceiling,
            "strong_auc_threshold": strong_auc_threshold,
            "baseline_margin": baseline_margin,
        },
        "claim_scope": claim_scope
        or (
            "PRESENT r9 262144/class single-seed weak-probe diagnostic only; "
            "not paper-scale, formal multi-seed, or breakthrough evidence"
        ),
    }


def update_plan_doc_with_r9_weak_probe_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved r9 Weak-Probe Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- r9-weak-probe-postprocess:{report['run_id']}:start -->"
    end = f"<!-- r9-weak-probe-postprocess:{report['run_id']}:end -->"
    block = f"{start}\n{_plan_doc_result_section(report)}\n{end}"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    plan_doc_path.write_text(text, encoding="utf-8")


def _next_action(report: dict[str, Any]) -> dict[str, Any]:
    if report["status"] != "pass":
        return {
            "branch": "invalid",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": "postprocess_status_failed",
        }
    decision = str(report["decision"])
    if decision == "strong_r9_diagnostic_prepare_1m_seed0":
        return {
            "branch": "r9_1m_seed0_plan",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "selected_model": (report.get("best_candidate") or {}).get("model", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-r9-weak-probe-plan.md",
            "launch_remote_config": R9_1M_SEED0_REMOTE_CONFIG,
            "suggested_remote_config": R9_1M_SEED0_REMOTE_CONFIG,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R9_1M_SEED0_REMOTE_CONFIG}"
            ),
            "run_id": R9_1M_SEED0_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "r9_weak_positive_prepare_seed1_or_curriculum_scale":
        return {
            "branch": "r9_seed1_or_curriculum_scale_plan",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "selected_model": (report.get("best_candidate") or {}).get("model", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-r9-weak-probe-plan.md",
            "launch_remote_config": R9_SEED1_REMOTE_CONFIG,
            "suggested_remote_config": R9_SEED1_REMOTE_CONFIG,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R9_SEED1_REMOTE_CONFIG}"
            ),
            "run_id": R9_SEED1_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
            "candidate_next_routes": ["r9_seed1_diagnostic", "r8_to_r9_curriculum_scale"],
        }
    if decision == "near_random_r9_weak_trace_check_variance_or_aggregation":
        return {
            "branch": "r9_variance_or_aggregation_review",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md",
            "launch_remote_config": R9_CURRICULUM_REMOTE_CONFIG,
            "suggested_remote_config": R9_CURRICULUM_REMOTE_CONFIG,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R9_CURRICULUM_REMOTE_CONFIG}"
            ),
            "run_id": R9_CURRICULUM_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
            "candidate_next_routes": ["r8_to_r9_curriculum", "seed_variance_review", "application_level_aggregation"],
        }
    if decision == "stop_from_scratch_r9_r10_plan_curriculum_or_difference_search":
        return {
            "branch": "stop_from_scratch_r9_r10",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md",
            "launch_remote_config": R9_CURRICULUM_REMOTE_CONFIG,
            "suggested_remote_config": R9_CURRICULUM_REMOTE_CONFIG,
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R9_CURRICULUM_REMOTE_CONFIG}"
            ),
            "run_id": R9_CURRICULUM_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
            "fallback_hypotheses": ["r8_to_r9_curriculum", "r9_difference_screen", "r8_integral_inverse_feature"],
        }
    if decision == "baseline_best_or_candidate_not_above_baseline":
        return {
            "branch": "baseline_best_no_candidate_scale",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": ["r8_to_r9_curriculum", "r9_difference_screen"],
        }
    return {
        "branch": "manual_review",
        "should_launch_remote": False,
        "requires_implementation": False,
        "reason": decision,
    }


def _next_steps(report: dict[str, Any]) -> list[str]:
    branch = report["next_action"]["branch"]
    if branch == "invalid":
        return ["Inspect validation/gate errors before interpreting r9 metrics."]
    if branch == "r9_1m_seed0_plan":
        return [
            "Record the retrieved result as medium diagnostic only.",
            "Write a lean r9 1M/class seed0 plan for the selected candidate and same-budget baseline.",
            "Do not claim r9 success before paper-scale and seed confirmation evidence.",
        ]
    if branch == "r9_seed1_or_curriculum_scale_plan":
        return [
            "Record the weak positive medium diagnostic result.",
            "Choose between seed1 variance check and r8-to-r9 curriculum scale based on the paired r8 1M result.",
            "Keep protocol fixed unless the next plan explicitly changes the hypothesis.",
        ]
    if branch == "r9_variance_or_aggregation_review":
        return [
            "Do not scale directly to 1M/class from this weak trace.",
            "Use the prepared r8-to-r9 curriculum diagnostic as the next launchable training-path test.",
            "Keep seed variance and aggregation as follow-up options after curriculum is retrieved.",
        ]
    if branch == "stop_from_scratch_r9_r10":
        return [
            "Stop from-scratch r9/r10 scaling under the current protocol.",
            "Use the prepared r8-to-r9 curriculum diagnostic before any r10 attempt.",
            "If curriculum is also near-random, prefer difference search or high-round data-representation screens.",
        ]
    if branch == "baseline_best_no_candidate_scale":
        return [
            "Do not scale the candidate architecture from this result.",
            "Use the result to choose curriculum or data-construction branches instead.",
        ]
    return ["Manual review required before branching."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    baseline = report.get("baseline") or {}
    best_candidate = report.get("best_candidate") or {}
    best_overall = report.get("best_overall") or {}
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Best candidate", best_candidate.get("model", "")),
        ("Best candidate AUC", _format_value(best_candidate.get("auc"))),
        ("Baseline AUC", _format_value(baseline.get("auc"))),
        ("Candidate delta vs baseline AUC", _format_value(report["candidate_delta_vs_baseline_auc"])),
        ("Best overall", best_overall.get("model", "")),
        ("Best overall AUC", _format_value(best_overall.get("auc"))),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next action should launch remote", report["next_action"].get("should_launch_remote", "")),
        ("Next action launch config", report["next_action"].get("launch_remote_config", "")),
        ("Next action readiness command", report["next_action"].get("readiness_command", "")),
        ("Next action readiness", report["next_action_readiness"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("r9 weak-probe gate", report["r9_weak_probe_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} r9 Weak-Probe Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} r9 Weak-Probe Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Ranking",
        "",
        "| Rank | Model | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            f"{index} | `{entry['model']}` | "
            f"{_format_value(entry.get('auc'))} | "
            f"{_format_value(entry.get('calibrated_accuracy'))} | "
            f"{_format_value(entry.get('accuracy'))} | "
            f"{_format_value(entry.get('loss'))} |"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _next_action_readiness_report(report: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    return next_action_readiness_report(summary_path=summary_path, report=report)


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    model = str(row.get("model") or row.get("selected_model") or row.get("model_key") or "")
    return {
        "model": model,
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "calibrated_accuracy": metrics.get("calibrated_accuracy"),
        "loss": metrics.get("loss"),
    }


def _auc_delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float | None:
    if left is None or right is None:
        return None
    left_auc = left.get("auc")
    right_auc = right.get("auc")
    if not isinstance(left_auc, (int, float)) or not isinstance(right_auc, (int, float)):
        return None
    return float(left_auc) - float(right_auc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_r9_weak_probe_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        near_random_auc_ceiling=args.near_random_auc_ceiling,
        weak_trace_auc_ceiling=args.weak_trace_auc_ceiling,
        strong_auc_threshold=args.strong_auc_threshold,
        baseline_margin=args.baseline_margin,
        claim_scope=args.claim_scope,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
