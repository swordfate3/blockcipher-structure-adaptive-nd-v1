from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.candidate_trail_gate import (
    DEFAULT_ANCHOR_MODEL,
    DEFAULT_CANDIDATE_MODELS,
    DEFAULT_CANDIDATE_TRAIL_MARGIN,
    DEFAULT_SHUFFLED_MODEL,
    gate_candidate_trail_result,
)
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess candidate-trail consistency results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--candidate-model", action="append", default=[])
    parser.add_argument("--shuffled-model", default=DEFAULT_SHUFFLED_MODEL)
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--anchor-calibrated-accuracy", type=float, default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_CANDIDATE_TRAIL_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_candidate_trail_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_models: tuple[str, ...] = DEFAULT_CANDIDATE_MODELS,
    shuffled_model: str = DEFAULT_SHUFFLED_MODEL,
    anchor_auc: float | None = None,
    anchor_calibrated_accuracy: float | None = None,
    margin: float = DEFAULT_CANDIDATE_TRAIL_MARGIN,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(
            plan_path,
            results_path,
            expected_rows=expected_rows,
        )
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_candidate_trail_result(
        results_path,
        expected_rows=expected_rows,
        anchor_model=anchor_model,
        candidate_models=candidate_models,
        shuffled_model=shuffled_model,
        anchor_auc=anchor_auc,
        anchor_calibrated_accuracy=anchor_calibrated_accuracy,
        margin=margin,
    )
    gate_path = output_dir / f"{run_id}_candidate_trail_gate.json"
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
        "candidate_trail_gate": str(gate_path),
        "validation_status": validation_status,
        "candidate_trail_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "best_candidate_model": gate_report["best_candidate_model"],
        "best_candidate_auc": gate_report["best_candidate_auc"],
        "anchor_auc": gate_report["anchor_auc"],
        "shuffled_auc": gate_report["shuffled_auc"],
        "margin_vs_anchor_auc": gate_report["margin_vs_anchor_auc"],
        "margin_vs_shuffled_auc": gate_report["margin_vs_shuffled_auc"],
        "required_margin": gate_report["required_margin"],
        "best_candidate_calibrated_accuracy": gate_report["best_candidate_calibrated_accuracy"],
        "anchor_calibrated_accuracy": gate_report["anchor_calibrated_accuracy"],
        "calibrated_delta_vs_anchor": gate_report["calibrated_delta_vs_anchor"],
        "claim_scope": gate_report["claim_scope"],
    }
    report["next_action"] = _next_action(report)
    report["next_steps"] = _next_steps(report)

    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    report["summary"] = str(summary_path)
    report["summary_markdown"] = str(markdown_path)

    update_paths = plan_doc_paths or []
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_candidate_trail_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_candidate_trail_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Candidate-Trail Consistency Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- candidate-trail-postprocess:{report['run_id']}:start -->"
    end = f"<!-- candidate-trail-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_candidate_trail_route":
        return {
            "branch": "candidate_trail_seed1_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
        }
    if decision == "weak_candidate_trail_signal":
        return {
            "branch": "candidate_trail_variance_check",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
        }
    if decision == "stop_candidate_trail_route":
        return {
            "branch": "stop_candidate_trail_route",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
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
        return [
            "Do not branch yet; inspect validation_report and candidate_trail_gate errors.",
            "Fix result retrieval, plan alignment, or gate inputs before launching another run.",
        ]
    if branch == "candidate_trail_seed1_confirmation":
        return [
            "Record this as positive medium diagnostic evidence only.",
            "Prepare a gated 262144/class seed1 confirmation before any 1M scale-up.",
            "Do not make formal or breakthrough claims from a single diagnostic seed.",
        ]
    if branch == "candidate_trail_variance_check":
        return [
            "Record this as weak candidate-trail evidence.",
            "Repeat 262144/class or run a variance check before scaling.",
            "Keep InvP-only and shuffled-cell controls in the next matrix.",
        ]
    if branch == "stop_candidate_trail_route":
        return [
            "Record this as tied or negative candidate-trail evidence.",
            "Do not scale candidate-trail consistency as a main route.",
            "Switch to another SPN structure-adaptive hypothesis.",
        ]
    return ["Review the candidate-trail gate manually before launching another experiment."]


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- candidate_trail_status: `{report['candidate_trail_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- best_candidate_model: `{report['best_candidate_model']}`",
            f"- best_candidate_auc: `{_format_value(report['best_candidate_auc'])}`",
            f"- anchor_auc: `{_format_value(report['anchor_auc'])}`",
            f"- shuffled_auc: `{_format_value(report['shuffled_auc'])}`",
            f"- margin_vs_anchor_auc: `{_format_value(report['margin_vs_anchor_auc'])}`",
            f"- margin_vs_shuffled_auc: `{_format_value(report['margin_vs_shuffled_auc'])}`",
            f"- claim_scope: {report['claim_scope']}",
            "",
            "Next Action:",
            "",
            *[f"- {key}: `{value}`" for key, value in report["next_action"].items()],
            "",
            "Next Steps:",
            "",
            *[f"- {step}" for step in report["next_steps"]],
            "",
            "Artifacts:",
            "",
            f"- results: `{report['results']}`",
            f"- validation_report: `{report['validation_report']}`",
            f"- candidate_trail_gate: `{report['candidate_trail_gate']}`",
            f"- summary: `{report['summary']}`",
            *( [f"- plan_docs: `{'; '.join(report['plan_docs'])}`"] if "plan_docs" in report else [] ),
            "",
        ]
    )


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Candidate-trail gate status", report["candidate_trail_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("Best candidate model", report["best_candidate_model"]),
        ("Best candidate AUC", _format_value(report["best_candidate_auc"])),
        ("Anchor AUC", _format_value(report["anchor_auc"])),
        ("Shuffled AUC", _format_value(report["shuffled_auc"])),
        ("Margin vs anchor AUC", _format_value(report["margin_vs_anchor_auc"])),
        ("Margin vs shuffled AUC", _format_value(report["margin_vs_shuffled_auc"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Claim scope", report["claim_scope"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Candidate-trail gate", report["candidate_trail_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Candidate-Trail Consistency Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_models = tuple(args.candidate_model) if args.candidate_model else DEFAULT_CANDIDATE_MODELS
    report = postprocess_candidate_trail_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        anchor_model=args.anchor_model,
        candidate_models=candidate_models,
        shuffled_model=args.shuffled_model,
        anchor_auc=args.anchor_auc,
        anchor_calibrated_accuracy=args.anchor_calibrated_accuracy,
        margin=args.margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
