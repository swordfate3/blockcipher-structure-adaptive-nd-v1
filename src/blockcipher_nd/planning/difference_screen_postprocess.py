from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.difference_screen_gate import (
    DEFAULT_PROMOTION_MARGIN,
    DEFAULT_RANDOM_AUC_CEILING,
    DEFAULT_REFERENCE_DIFFERENCE,
    gate_difference_screen_result,
)
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a PRESENT input-difference screen.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--reference-difference", default=DEFAULT_REFERENCE_DIFFERENCE)
    parser.add_argument("--promotion-margin", type=float, default=DEFAULT_PROMOTION_MARGIN)
    parser.add_argument("--random-auc-ceiling", type=float, default=DEFAULT_RANDOM_AUC_CEILING)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_difference_screen_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    reference_difference: str = DEFAULT_REFERENCE_DIFFERENCE,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    random_auc_ceiling: float = DEFAULT_RANDOM_AUC_CEILING,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_difference_screen_result(
        results_path,
        expected_rows=expected_rows,
        reference_difference=reference_difference,
        promotion_margin=promotion_margin,
        random_auc_ceiling=random_auc_ceiling,
    )
    gate_path = output_dir / f"{run_id}_difference_screen_gate.json"
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
        "difference_screen_gate": str(gate_path),
        "validation_status": validation_status,
        "difference_screen_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "reference_difference": gate_report["reference_difference"],
        "reference": gate_report["reference"],
        "best": gate_report["best"],
        "delta_vs_reference_auc": gate_report["delta_vs_reference_auc"],
        "promotion_margin": gate_report["promotion_margin"],
        "random_auc_ceiling": gate_report["random_auc_ceiling"],
        "ranking": gate_report["ranking"],
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
            update_plan_doc_with_difference_screen_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_difference_screen_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Difference-Screen Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- difference-screen-postprocess:{report['run_id']}:start -->"
    end = f"<!-- difference-screen-postprocess:{report['run_id']}:end -->"
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
    best = report.get("best") or {}
    if decision == "promote_best_difference_to_262k_confirmation":
        return {
            "branch": "r9_difference_262k_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "selected_difference": best.get("difference_id", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-r9-difference-screen-plan.md",
            "implementation_checklist": [
                "Create a 262144/class confirmation CSV for only the selected difference and Zhang/Wang reference.",
                "Keep model_key fixed unless the new experiment explicitly changes the hypothesis.",
                "Run check-remote-readiness before launch and launch from a pushed commit.",
            ],
        }
    if decision == "weak_difference_candidate_repeat_or_confirm_at_262k":
        return {
            "branch": "r9_difference_weak_candidate_review",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "selected_difference": best.get("difference_id", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-r9-difference-screen-plan.md",
            "implementation_checklist": [
                "Decide whether to repeat the 65536/class screen or prepare a 262144/class confirmation.",
                "Do not claim route success from the weak screen margin.",
            ],
        }
    if decision == "keep_current_difference_no_screen_winner":
        return {
            "branch": "return_to_same_difference_model_or_curriculum",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "all_candidates_near_random_stop_difference_screen":
        return {
            "branch": "stop_current_difference_screen",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": [
                "r8_to_r9_curriculum",
                "pair_set_aggregation",
                "topology_aware_architecture",
            ],
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
            "Do not analyze the screen metrics yet; inspect validation and gate errors.",
            "Fix retrieval, plan alignment, or result-row completeness before branching.",
        ]
    if branch == "r9_difference_262k_confirmation":
        return [
            "Record this as a screen-only data-construction signal.",
            "Prepare a lean 262144/class confirmation matrix for the selected difference versus the Zhang/Wang reference.",
            "Keep the model fixed so the confirmation remains attributable to input-difference choice.",
        ]
    if branch == "r9_difference_weak_candidate_review":
        return [
            "Treat the best difference as a weak screen candidate only.",
            "Repeat or confirm at 262144/class only if the active r9/r8 tasks make difference search the chosen branch.",
        ]
    if branch == "return_to_same_difference_model_or_curriculum":
        return [
            "Do not expand the high-round difference-search matrix from this result.",
            "Return to model, curriculum, or pair-set aggregation routes under the current difference.",
        ]
    if branch == "stop_current_difference_screen":
        return [
            "Stop this specific difference-screen branch.",
            "Prefer curriculum, aggregation, or architecture changes before more high-round difference trials.",
        ]
    return ["Manual review required before the next experiment branch."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    best = report.get("best") or {}
    reference = report.get("reference") or {}
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Best difference", best.get("difference_id", "")),
        ("Best AUC", _format_value(best.get("auc"))),
        ("Reference difference", reference.get("difference_id", report["reference_difference"])),
        ("Reference AUC", _format_value(reference.get("auc"))),
        ("Delta vs reference AUC", _format_value(report["delta_vs_reference_auc"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Difference-screen gate", report["difference_screen_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Difference-Screen Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Difference-Screen Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Ranking",
        "",
        "| Rank | Difference | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            f"{index} | `{entry['difference_id']}` | "
            f"{_format_value(entry.get('auc'))} | "
            f"{_format_value(entry.get('calibrated_accuracy'))} | "
            f"{_format_value(entry.get('accuracy'))} | "
            f"{_format_value(entry.get('loss'))} |"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_difference_screen_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        reference_difference=args.reference_difference,
        promotion_margin=args.promotion_margin,
        random_auc_ceiling=args.random_auc_ceiling,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
