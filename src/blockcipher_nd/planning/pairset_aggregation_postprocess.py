from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.pairset_aggregation_gate import gate_pairset_aggregation_control
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess learned pair-set results against frozen single-pair aggregation."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--learned-results", required=True, type=Path)
    parser.add_argument("--frozen-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--learned-model", default="present_nibble_invp_pair_consistency_spn_only")
    parser.add_argument("--anchor-model", default="present_nibble_invp_only_spn_only")
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument(
        "--update-plan-doc",
        type=Path,
        action="append",
        default=[],
        help="Optional experiment plan Markdown file to update. Can be repeated.",
    )
    return parser.parse_args(argv)


def postprocess_pairset_aggregation_control(
    *,
    plan_path: Path,
    learned_results_path: Path,
    frozen_summary_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int | None = None,
    learned_model: str = "present_nibble_invp_pair_consistency_spn_only",
    anchor_model: str = "present_nibble_invp_only_spn_only",
    anchor_auc: float | None = None,
    margin: float = 0.001,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report = validate_result_plan_alignment(
        plan_path,
        learned_results_path,
        expected_rows=expected_rows,
    )
    validation_path = output_dir / f"{run_id}_local_result_gate.json"
    _write_json(validation_path, validation_report)

    curves_path = output_dir / f"{run_id}_curves.svg"
    history_path = output_dir / f"{run_id}_history.csv"
    plot_report = plot_jsonl_training_curves(learned_results_path, curves_path, title=run_id)
    plot_report["history_csv"] = write_history_csv(learned_results_path, history_path)

    gate_report = gate_pairset_aggregation_control(
        learned_results_path,
        frozen_summary_path,
        learned_model=learned_model,
        anchor_model=anchor_model,
        anchor_auc=anchor_auc,
        expected_learned_rows=expected_rows,
        margin=margin,
    )
    gate_path = output_dir / f"{run_id}_pairset_aggregation_gate.json"
    _write_json(gate_path, gate_report)

    status = "pass" if validation_report["status"] == "pass" and gate_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "learned_results": str(learned_results_path),
        "frozen_summary": str(frozen_summary_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "pairset_aggregation_gate": str(gate_path),
        "validation_status": validation_report["status"],
        "pairset_aggregation_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "learned_auc": gate_report["learned_auc"],
        "frozen_auc": gate_report["frozen_auc"],
        "anchor_auc": gate_report["anchor_auc"],
        "margin_vs_frozen_auc": gate_report["margin_vs_frozen_auc"],
        "margin_vs_anchor_auc": gate_report["margin_vs_anchor_auc"],
        "required_margin": gate_report["required_margin"],
        "learned_calibrated_accuracy": gate_report["learned_calibrated_accuracy"],
        "frozen_calibrated_accuracy": gate_report["frozen_calibrated_accuracy"],
        "calibrated_delta_vs_frozen": gate_report["calibrated_delta_vs_frozen"],
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
            update_plan_doc_with_pairset_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_pairset_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Pair-Set Aggregation Control Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- pairset-aggregation-postprocess:{report['run_id']}:start -->"
    end = f"<!-- pairset-aggregation-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_learned_pairset_consistency":
        return {
            "branch": "pairset_seed1_confirmation",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "weak_pairset_consistency_signal":
        return {
            "branch": "pairset_variance_check",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_pairset_consistency_route":
        return {
            "branch": "stop_pairset_main_route",
            "should_launch_remote": False,
            "requires_implementation": True,
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
            "Do not branch yet; inspect validation_report and pairset_aggregation_gate errors.",
            "Fix result retrieval, plan alignment, or frozen aggregation artifacts before launching another run.",
        ]
    if branch == "pairset_seed1_confirmation":
        return [
            "Record this as positive medium diagnostic evidence only.",
            "Prepare a 262144/class seed1 confirmation before any 1M pair-set scale-up.",
            "Do not make formal or breakthrough claims from a single diagnostic seed.",
        ]
    if branch == "pairset_variance_check":
        return [
            "Record this as weak positive pair-set evidence.",
            "Repeat 262144/class or run a variance check before scaling.",
            "Keep InvP-only and frozen aggregation as controls.",
        ]
    if branch == "stop_pairset_main_route":
        return [
            "Record this as tied or negative pair-set evidence.",
            "Do not scale learned pair-set consistency as a main route.",
            "Treat pair-set pooling as aggregation or diagnostic context and switch hypothesis.",
        ]
    return ["Review the pair-set aggregation gate manually before launching another experiment."]


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- pairset_aggregation_status: `{report['pairset_aggregation_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- learned_auc: `{_format_value(report['learned_auc'])}`",
            f"- frozen_auc: `{_format_value(report['frozen_auc'])}`",
            f"- anchor_auc: `{_format_value(report['anchor_auc'])}`",
            f"- margin_vs_frozen_auc: `{_format_value(report['margin_vs_frozen_auc'])}`",
            f"- margin_vs_anchor_auc: `{_format_value(report['margin_vs_anchor_auc'])}`",
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
            f"- learned_results: `{report['learned_results']}`",
            f"- frozen_summary: `{report['frozen_summary']}`",
            f"- validation_report: `{report['validation_report']}`",
            f"- pairset_aggregation_gate: `{report['pairset_aggregation_gate']}`",
            f"- curves: `{report['curves']}`",
            f"- history_csv: `{report['history_csv']}`",
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
        ("Pair-set gate status", report["pairset_aggregation_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("Learned AUC", _format_value(report["learned_auc"])),
        ("Frozen aggregation AUC", _format_value(report["frozen_auc"])),
        ("Anchor AUC", _format_value(report["anchor_auc"])),
        ("Margin vs frozen AUC", _format_value(report["margin_vs_frozen_auc"])),
        ("Margin vs anchor AUC", _format_value(report["margin_vs_anchor_auc"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Claim scope", report["claim_scope"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Learned results JSONL", report["learned_results"]),
        ("Frozen aggregation summary", report["frozen_summary"]),
        ("Validation report", report["validation_report"]),
        ("Pair-set gate", report["pairset_aggregation_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Pair-Set Aggregation Control Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_pairset_aggregation_control(
        plan_path=args.plan,
        learned_results_path=args.learned_results,
        frozen_summary_path=args.frozen_summary,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_rows=args.expected_rows,
        learned_model=args.learned_model,
        anchor_model=args.anchor_model,
        anchor_auc=args.anchor_auc,
        margin=args.margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
