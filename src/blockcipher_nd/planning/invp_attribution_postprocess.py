from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_attribution_gate import gate_invp_attribution_controls
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local post-retrieval validation, plotting, and attribution gating for InvP controls."
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan CSV path.")
    parser.add_argument("--results", required=True, type=Path, help="Retrieved attribution-control JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated reports.")
    parser.add_argument("--run-id", required=True, help="Run id used in output filenames and plot title.")
    parser.add_argument("--expected-rows", type=int, default=2)
    parser.add_argument(
        "--update-plan-doc",
        type=Path,
        action="append",
        default=[],
        help="Optional experiment plan Markdown file to update with the postprocess result. Can be repeated.",
    )
    return parser.parse_args(argv)


def postprocess_invp_attribution_controls(
    *,
    plan_path: Path,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 2,
    plan_doc_path: Path | None = None,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=expected_rows,
    )
    validation_path = output_dir / f"{run_id}_local_result_gate.json"
    _write_json(validation_path, validation_report)

    curves_path = output_dir / f"{run_id}_curves.svg"
    history_path = output_dir / f"{run_id}_history.csv"
    plot_report = plot_jsonl_training_curves(results_path, curves_path, title=run_id)
    plot_report["history_csv"] = write_history_csv(results_path, history_path)

    attribution_report = gate_invp_attribution_controls(
        results_path,
        expected_rows=expected_rows,
    )
    attribution_path = output_dir / f"{run_id}_attribution_gate.json"
    _write_json(attribution_path, attribution_report)

    status = "pass" if validation_report["status"] == "pass" and attribution_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "attribution_gate": str(attribution_path),
        "validation_status": validation_report["status"],
        "attribution_status": attribution_report["status"],
        "decision": attribution_report["decision"],
        "action": attribution_report["action"],
        "interpretation": attribution_report["interpretation"],
        "reference_auc": attribution_report["reference_auc"],
        "invp_seed0_auc": attribution_report["invp_seed0_auc"],
        "invp_seed1_auc": attribution_report["invp_seed1_auc"],
        "invp_min_auc": attribution_report["invp_min_auc"],
        "invp_mean_auc": attribution_report["invp_mean_auc"],
        "max_control_auc": attribution_report["max_control_auc"],
        "attribution_margin": attribution_report["attribution_margin"],
        "required_margin": attribution_report["required_margin"],
        "controls": attribution_report["controls"],
        "claim_scope": attribution_report["claim_scope"],
    }
    report["next_action"] = _next_action(report)
    report["next_steps"] = _next_steps(report)
    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    report["summary"] = str(summary_path)
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    report["summary_markdown"] = str(markdown_path)
    update_paths = _merge_plan_doc_paths(plan_doc_path, plan_doc_paths)
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_attribution_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])
    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_attribution_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    section = _plan_doc_result_section(report)
    header = "## Retrieved Attribution Control Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- invp-attribution-postprocess:{report['run_id']}:start -->"
    end = f"<!-- invp-attribution-postprocess:{report['run_id']}:end -->"
    block = f"{start}\n{section}\n{end}"
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
    if decision == "support_invp_structural_attribution":
        return {
            "branch": "route_level_attribution_summary",
            "should_launch_remote": False,
            "requires_implementation": False,
            "plan_doc": "docs/experiments/innovation1-invp-only-formal-attribution-plan.md",
            "reason": decision,
        }
    if decision == "weak_attribution_support":
        return {
            "branch": "variance_or_additional_controls",
            "should_launch_remote": False,
            "requires_implementation": False,
            "plan_doc": "docs/experiments/innovation1-invp-only-formal-attribution-plan.md",
            "reason": decision,
        }
    if decision == "weaken_invp_structural_attribution":
        return {
            "branch": "new_spn_structure_hypothesis",
            "should_launch_remote": False,
            "requires_implementation": True,
            "plan_doc": "docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md",
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
            "Do not branch yet; inspect validation_report and attribution_gate errors.",
            "Fix result retrieval, plan alignment, or metric availability before launching another run.",
        ]
    if branch == "route_level_attribution_summary":
        return [
            "Update the experiment plan with this attribution-control result.",
            "Write a route-level summary: InvP-only two-seed confirmation plus paper-scale controls.",
            "Continue the active topology-aware network route if the paper needs a stronger method beyond InvP-only.",
            "If topology-aware stops, switch to candidate-trail / transition consistency as the next data-feature branch.",
        ]
    if branch == "variance_or_additional_controls":
        return [
            "Update the experiment plan with weak attribution support.",
            "Do not make a strong attribution claim yet.",
            "Prefer variance analysis or one additional selected control before formalizing InvP-only.",
        ]
    if branch == "new_spn_structure_hypothesis":
        return [
            "Update the experiment plan with the weakened attribution result.",
            "Do not formalize InvP-only as the main SPN structure claim.",
            "Move to the next SPN topology/DDT-aware hypothesis with a new experiment plan.",
        ]
    return ["Review the attribution gate manually before launching another experiment."]


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- attribution_status: `{report['attribution_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- attribution_margin: `{_format_value(report['attribution_margin'])}`",
            f"- max_control_auc: `{_format_value(report['max_control_auc'])}`",
            f"- invp_min_auc: `{_format_value(report['invp_min_auc'])}`",
            f"- claim_scope: {report['claim_scope']}",
            "",
            "Controls:",
            "",
            *_control_lines(report["controls"]),
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
            f"- attribution_gate: `{report['attribution_gate']}`",
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
        ("Attribution status", report["attribution_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("InvP seed0 AUC", _format_value(report["invp_seed0_auc"])),
        ("InvP seed1 AUC", _format_value(report["invp_seed1_auc"])),
        ("InvP min AUC", _format_value(report["invp_min_auc"])),
        ("Max control AUC", _format_value(report["max_control_auc"])),
        ("Attribution margin", _format_value(report["attribution_margin"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Attribution gate", report["attribution_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Attribution Control Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    lines.extend(["", "Control rows:", "", "| Model | AUC | Delta vs Zhang/Wang 1M |", "|---|---:|---:|"])
    for model, control in sorted(report["controls"].items()):
        lines.append(
            f"| `{model}` | `{_format_value(control.get('auc'))}` | "
            f"`{_format_value(control.get('delta_vs_reference_auc'))}` |"
        )
    return "\n".join(lines)


def _control_lines(controls: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model, control in sorted(controls.items()):
        lines.append(
            f"- `{model}`: auc=`{_format_value(control.get('auc'))}`, "
            f"delta_vs_reference=`{_format_value(control.get('delta_vs_reference_auc'))}`"
        )
    return lines


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_plan_doc_paths(plan_doc_path: Path | None, plan_doc_paths: list[Path] | None) -> list[Path]:
    merged: list[Path] = []
    for path in [*(plan_doc_paths or []), plan_doc_path]:
        if path is None:
            continue
        if path not in merged:
            merged.append(path)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_invp_attribution_controls(
        plan_path=args.plan,
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_rows=args.expected_rows,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
