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
PAIRSET_MODEL = "present_nibble_invp_pair_consistency_spn_only"
R8_PAIRSET_1M_SEED1_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
)
R8_PAIRSET_1M_SEED1_RUN_ID = "i1_present_r8_pairset_1m_seed1_gpu1_20260705"
R8_PAIRSET_CONTROL_STAGE_A_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k_gpu0_20260705.json"
)
R8_PAIRSET_CONTROL_STAGE_B_REMOTE_CONFIG = (
    "configs/remote/"
    "innovation1_spn_present_pairset_aggregation_control_r8_262k_gpu0_20260705.json"
)
R8_PAIRSET_CONTROL_STAGE_A_RUN_ID = "i1_pairset_single_pair_scorer_r8_262k_seed0_gpu0_20260705"
R8_PAIRSET_CONTROL_STAGE_B_RUN_ID = "i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705"
DEFAULT_SUPPORT_MARGIN = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a PRESENT r8 pair-set 1M confirmation result.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=2)
    parser.add_argument("--support-margin", type=float, default=DEFAULT_SUPPORT_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_r8_pairset_1m_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int = 2,
    support_margin: float = DEFAULT_SUPPORT_MARGIN,
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

    gate_report = gate_r8_pairset_1m_result(
        results_path,
        expected_rows=expected_rows,
        support_margin=support_margin,
    )
    gate_path = output_dir / f"{run_id}_r8_pairset_1m_gate.json"
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
        "r8_pairset_1m_gate": str(gate_path),
        "validation_status": validation_status,
        "r8_pairset_1m_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "baseline": gate_report["baseline"],
        "pairset": gate_report["pairset"],
        "delta_vs_baseline_auc": gate_report["delta_vs_baseline_auc"],
        "ranking": gate_report["ranking"],
        "support_margin": gate_report["support_margin"],
        "claim_scope": gate_report["claim_scope"],
        "plot_report": plot_report,
    }
    report["next_action"] = _next_action(report)
    report["next_steps"] = _next_steps(report)

    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    next_action_readiness_path = output_dir / f"{run_id}_next_action_readiness.json"
    candidate_route_readiness_path = output_dir / f"{run_id}_candidate_route_readiness.json"
    report["summary"] = str(summary_path)
    report["summary_markdown"] = str(markdown_path)
    report["next_action_readiness"] = str(next_action_readiness_path)
    report["candidate_route_readiness"] = str(candidate_route_readiness_path)

    update_paths = plan_doc_paths or []
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_r8_pairset_1m_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    _write_json(next_action_readiness_path, _next_action_readiness_report(report, summary_path))
    _write_json(candidate_route_readiness_path, _candidate_route_readiness_report(report, summary_path))
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_r8_pairset_1m_result(
    results_path: Path,
    *,
    expected_rows: int = 2,
    support_margin: float = DEFAULT_SUPPORT_MARGIN,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    entries = [_entry(row) for row in rows]
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")

    by_model = {entry["model"]: entry for entry in entries}
    baseline = by_model.get(BASELINE_MODEL)
    pairset = by_model.get(PAIRSET_MODEL)
    if baseline is None:
        errors.append(f"missing baseline model {BASELINE_MODEL}")
    if pairset is None:
        errors.append(f"missing pairset model {PAIRSET_MODEL}")
    for entry in entries:
        if not isinstance(entry.get("auc"), (int, float)):
            errors.append(f"missing auc for {entry['model']}")

    ranking = sorted(entries, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    delta = _auc_delta(pairset, baseline)

    if errors:
        status = "fail"
        decision = "invalid_r8_pairset_1m_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "The r8 pair-set 1M result is incomplete or missing required metrics."
    elif delta is not None and delta >= support_margin:
        status = "pass"
        decision = "support_r8_pairset_1m_confirmation"
        action = "prepare_seed1_or_frozen_aggregation_control_before_formal_claim"
        interpretation = "The pair-set candidate beats the r8 1M baseline by the configured support margin."
    elif delta is not None and delta > 0:
        status = "pass"
        decision = "weak_r8_pairset_1m_positive_needs_seed1_or_controls"
        action = "repeat_seed_or_run_controls_before_scaling_claim"
        interpretation = "The pair-set candidate is positive but below the support margin."
    else:
        status = "pass"
        decision = "stop_or_rethink_r8_pairset_scale"
        action = "do_not_expand_pairset_scale_without_new_evidence"
        interpretation = "The pair-set candidate does not beat the r8 1M baseline."

    return {
        "status": status,
        "results": str(results_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "errors": errors,
        "baseline": baseline,
        "pairset": pairset,
        "delta_vs_baseline_auc": delta,
        "support_margin": support_margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "claim_scope": (
            "PRESENT r8 1000000/class single-seed pair-set confirmation only; "
            "not formal multi-seed route evidence or breakthrough evidence"
        ),
    }


def update_plan_doc_with_r8_pairset_1m_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved r8 Pair-Set 1M Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- r8-pairset-1m-postprocess:{report['run_id']}:start -->"
    end = f"<!-- r8-pairset-1m-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_r8_pairset_1m_confirmation":
        return {
            "branch": "r8_pairset_seed1_or_frozen_control",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "candidate_next_routes": ["r8_pairset_1m_seed1", "r8_pairset_frozen_aggregation_control"],
            "next_plan_doc": "docs/experiments/innovation1-present-r8-round-extension-ladder-plan.md",
            "launch_remote_config": R8_PAIRSET_1M_SEED1_REMOTE_CONFIG,
            "suggested_remote_config": R8_PAIRSET_1M_SEED1_REMOTE_CONFIG,
            "control_stage_a_remote_config": R8_PAIRSET_CONTROL_STAGE_A_REMOTE_CONFIG,
            "control_stage_b_remote_config": R8_PAIRSET_CONTROL_STAGE_B_REMOTE_CONFIG,
            "control_stage_a_run_id": R8_PAIRSET_CONTROL_STAGE_A_RUN_ID,
            "control_stage_b_run_id": R8_PAIRSET_CONTROL_STAGE_B_RUN_ID,
            "control_readiness_commands": _control_readiness_commands(),
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R8_PAIRSET_1M_SEED1_REMOTE_CONFIG}"
            ),
            "run_id": R8_PAIRSET_1M_SEED1_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "weak_r8_pairset_1m_positive_needs_seed1_or_controls":
        return {
            "branch": "r8_pairset_weak_positive_review",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "candidate_next_routes": ["repeat_seed", "frozen_aggregation_control"],
            "next_plan_doc": "docs/experiments/innovation1-present-r8-round-extension-ladder-plan.md",
            "launch_remote_config": R8_PAIRSET_1M_SEED1_REMOTE_CONFIG,
            "suggested_remote_config": R8_PAIRSET_1M_SEED1_REMOTE_CONFIG,
            "control_stage_a_remote_config": R8_PAIRSET_CONTROL_STAGE_A_REMOTE_CONFIG,
            "control_stage_b_remote_config": R8_PAIRSET_CONTROL_STAGE_B_REMOTE_CONFIG,
            "control_stage_a_run_id": R8_PAIRSET_CONTROL_STAGE_A_RUN_ID,
            "control_stage_b_run_id": R8_PAIRSET_CONTROL_STAGE_B_RUN_ID,
            "control_readiness_commands": _control_readiness_commands(),
            "readiness_command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
                f"--config {R8_PAIRSET_1M_SEED1_REMOTE_CONFIG}"
            ),
            "run_id": R8_PAIRSET_1M_SEED1_RUN_ID,
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "stop_or_rethink_r8_pairset_scale":
        return {
            "branch": "stop_r8_pairset_scale",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": ["r9_curriculum", "r9_difference_screen", "r8_integral_inverse_feature"],
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
        return ["Inspect validation/gate errors before interpreting r8 pair-set metrics."]
    if branch == "r8_pairset_seed1_or_frozen_control":
        return [
            "Record this as paper-scale single-seed evidence only.",
            "Prepare seed1 confirmation or frozen aggregation control before any formal route claim.",
            "Use the paired r9 weak-probe gate to decide whether pair-set remains the high-round priority.",
        ]
    if branch == "r8_pairset_weak_positive_review":
        return [
            "Treat this as weak paper-scale positive evidence.",
            "Run seed/control only if r9 weak-probe or attribution priorities keep pair-set selected.",
        ]
    if branch == "stop_r8_pairset_scale":
        return [
            "Do not expand r8 pair-set scale from this result.",
            "Prefer curriculum, difference search, or integral/inverse feature screens.",
        ]
    return ["Manual review required before branching."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    baseline = report.get("baseline") or {}
    pairset = report.get("pairset") or {}
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Pair-set AUC", _format_value(pairset.get("auc"))),
        ("Baseline AUC", _format_value(baseline.get("auc"))),
        ("Delta vs baseline AUC", _format_value(report["delta_vs_baseline_auc"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next action should launch remote", report["next_action"].get("should_launch_remote", "")),
        ("Next action launch config", report["next_action"].get("launch_remote_config", "")),
        ("Next action readiness command", report["next_action"].get("readiness_command", "")),
        ("Next action readiness", report["next_action_readiness"]),
        ("Candidate route readiness", report["candidate_route_readiness"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("r8 pair-set 1M gate", report["r8_pairset_1m_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} r8 Pair-Set 1M Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} r8 Pair-Set 1M Postprocess",
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


def _candidate_route_readiness_report(report: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    next_action = report.get("next_action")
    if not isinstance(next_action, dict):
        next_action = {}
    routes = {
        "r8_pairset_1m_seed1": next_action_readiness_report(
            summary_path=summary_path,
            report=report,
        ),
        "r8_pairset_frozen_aggregation_control": next_action_readiness_report(
            summary_path=summary_path,
            report=report,
            config_keys=(
                ("stage_a", "control_stage_a_remote_config"),
                ("primary", "control_stage_b_remote_config"),
            ),
        ),
    }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in routes.values()) else "fail",
        "summary": str(summary_path),
        "run_id": report.get("run_id"),
        "decision": report.get("decision"),
        "branch": next_action.get("branch"),
        "candidate_routes": routes,
        "policy": (
            "This is readiness only. Launch one selected branch after high-round arbitration; "
            "do not launch seed1 and frozen aggregation control blindly in parallel."
        ),
    }


def _control_readiness_commands() -> list[str]:
    return [
        (
            "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
            f"--config {R8_PAIRSET_CONTROL_STAGE_A_REMOTE_CONFIG}"
        ),
        (
            "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness "
            f"--config {R8_PAIRSET_CONTROL_STAGE_B_REMOTE_CONFIG}"
        ),
    ]


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
    report = postprocess_r8_pairset_1m_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        support_margin=args.support_margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
