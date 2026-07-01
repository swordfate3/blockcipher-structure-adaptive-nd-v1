from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.planning.topology_aware_gate import gate_topology_aware_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, plot, gate, and document topology-aware network results."
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan CSV path.")
    parser.add_argument("--results", required=True, type=Path, help="Retrieved result JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated reports.")
    parser.add_argument("--run-id", required=True, help="Run id used in output filenames and plot title.")
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument(
        "--update-plan-doc",
        type=Path,
        action="append",
        default=[],
        help="Optional experiment plan Markdown file to update. Can be repeated.",
    )
    return parser.parse_args(argv)


def postprocess_topology_aware_result(
    *,
    plan_path: Path,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 3,
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

    gate_report = gate_topology_aware_result(
        results_path,
        expected_rows=expected_rows,
    )
    gate_path = output_dir / f"{run_id}_topology_aware_gate.json"
    _write_json(gate_path, gate_report)

    status = "pass" if validation_report["status"] == "pass" and gate_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "topology_aware_gate": str(gate_path),
        "validation_status": validation_report["status"],
        "topology_aware_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "margin_vs_invp_auc": gate_report["margin_vs_invp_auc"],
        "margin_vs_shuffled_auc": gate_report["margin_vs_shuffled_auc"],
        "calibrated_delta_vs_invp": gate_report["calibrated_delta_vs_invp"],
        "required_margin": gate_report["required_margin"],
        "models": gate_report["models"],
        "claim_scope": gate_report["claim_scope"],
    }
    report["next_steps"] = _next_steps(report)
    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    report["summary"] = str(summary_path)
    report["summary_markdown"] = str(markdown_path)

    update_paths = _dedupe_paths(plan_doc_paths or [])
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_topology_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_topology_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Topology-Aware Network Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- topology-aware-postprocess:{report['run_id']}:start -->"
    end = f"<!-- topology-aware-postprocess:{report['run_id']}:end -->"
    block = f"{start}\n{_plan_doc_result_section(report)}\n{end}"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    plan_doc_path.write_text(text, encoding="utf-8")


def _next_steps(report: dict[str, Any]) -> list[str]:
    if report["status"] != "pass":
        return [
            "Do not branch yet; inspect validation_report and topology_aware_gate errors.",
            "Fix retrieval, plan alignment, or metric availability before launching another run.",
        ]
    decision = str(report["decision"])
    if decision == "support_topology_aware_network_route":
        return [
            "Record this as medium diagnostic support for true-P topology-aware architecture.",
            "Prepare a 262144/class seed1 confirmation matrix before any 1M scale-up.",
            "Do not make formal or breakthrough claims from this single medium-scale seed.",
        ]
    if decision == "weak_topology_aware_network_signal":
        return [
            "Record this as weak topology-aware network signal.",
            "Run a 262144/class seed1 variance check before deciding whether to scale.",
            "Keep claim scope diagnostic only.",
        ]
    if decision == "stop_topology_aware_network_route":
        return [
            "Record this as tied or negative topology-aware network evidence.",
            "Do not scale this architecture.",
            "Switch to candidate-trail or transition-consistency data representation hypothesis.",
        ]
    return ["Review the topology-aware gate manually before launching another experiment."]


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- topology_aware_status: `{report['topology_aware_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- margin_vs_invp_auc: `{_format_value(report['margin_vs_invp_auc'])}`",
            f"- margin_vs_shuffled_auc: `{_format_value(report['margin_vs_shuffled_auc'])}`",
            f"- calibrated_delta_vs_invp: `{_format_value(report['calibrated_delta_vs_invp'])}`",
            f"- claim_scope: {report['claim_scope']}",
            "",
            "Models:",
            "",
            *_model_lines(report["models"]),
            "",
            "Next Steps:",
            "",
            *[f"- {step}" for step in report["next_steps"]],
            "",
            "Artifacts:",
            "",
            f"- results: `{report['results']}`",
            f"- validation_report: `{report['validation_report']}`",
            f"- topology_aware_gate: `{report['topology_aware_gate']}`",
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
        ("Topology-aware gate status", report["topology_aware_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("Margin vs InvP AUC", _format_value(report["margin_vs_invp_auc"])),
        ("Margin vs shuffled AUC", _format_value(report["margin_vs_shuffled_auc"])),
        ("Calibrated delta vs InvP", _format_value(report["calibrated_delta_vs_invp"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Topology-aware gate", report["topology_aware_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Topology-Aware Network Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    lines.extend(["", "Model rows:", "", "| Model | AUC | Calibrated Accuracy |", "|---|---:|---:|"])
    for model, metrics in sorted(report["models"].items()):
        lines.append(
            f"| `{model}` | `{_format_value(metrics.get('auc'))}` | "
            f"`{_format_value(metrics.get('calibrated_accuracy'))}` |"
        )
    return "\n".join(lines)


def _model_lines(models: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model, metrics in sorted(models.items()):
        lines.append(
            f"- `{model}`: auc=`{_format_value(metrics.get('auc'))}`, "
            f"calibrated_accuracy=`{_format_value(metrics.get('calibrated_accuracy'))}`"
        )
    return lines


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    merged: list[Path] = []
    for path in paths:
        if path not in merged:
            merged.append(path)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_topology_aware_result(
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
