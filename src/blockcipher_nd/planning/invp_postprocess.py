from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_gate import gate_invp_only_result
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local post-retrieval validation, plotting, and branch gating for InvP-only 1M results."
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan CSV path.")
    parser.add_argument("--results", required=True, type=Path, help="Retrieved result JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated reports.")
    parser.add_argument("--run-id", required=True, help="Run id used in output filenames and plot title.")
    parser.add_argument("--expected-rows", type=int, default=1)
    parser.add_argument("--reference-auc", type=float, default=0.793897025948)
    parser.add_argument(
        "--update-plan-doc",
        type=Path,
        default=None,
        help="Optional experiment plan Markdown file to update with the postprocess result.",
    )
    return parser.parse_args(argv)


def postprocess_invp_only_result(
    *,
    plan_path: Path,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 1,
    reference_auc: float = 0.793897025948,
    plan_doc_path: Path | None = None,
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

    branch_report = gate_invp_only_result(
        results_path,
        reference_auc=reference_auc,
        expected_rows=expected_rows,
    )
    branch_path = output_dir / f"{run_id}_branch_gate.json"
    _write_json(branch_path, branch_report)

    status = "pass" if validation_report["status"] == "pass" and branch_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "branch_gate": str(branch_path),
        "validation_status": validation_report["status"],
        "branch_status": branch_report["status"],
        "decision": branch_report["decision"],
        "action": branch_report["action"],
        "reference_auc": branch_report["reference_auc"],
        "paligned_mcnd_1m_auc": branch_report["paligned_mcnd_1m_auc"],
        "auc": branch_report["auc"],
        "auc_delta": branch_report["auc_delta"],
        "auc_delta_vs_paligned_mcnd_1m": branch_report["auc_delta_vs_paligned_mcnd_1m"],
        "accuracy": branch_report["accuracy"],
        "calibrated_accuracy": branch_report["calibrated_accuracy"],
        "loss": branch_report["loss"],
        "claim_scope": branch_report["claim_scope"],
    }
    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    report["summary"] = str(summary_path)
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    report["summary_markdown"] = str(markdown_path)
    if plan_doc_path is not None:
        update_plan_doc_with_postprocess_result(plan_doc_path, report)
        report["plan_doc"] = str(plan_doc_path)
    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- branch_status: `{report['branch_status']}`",
            f"- auc: `{_format_value(report['auc'])}`",
            f"- accuracy: `{_format_value(report['accuracy'])}`",
            f"- calibrated_accuracy: `{_format_value(report['calibrated_accuracy'])}`",
            f"- loss: `{_format_value(report['loss'])}`",
            f"- auc_delta_vs_zhang_wang_1m: `{_format_value(report['auc_delta'])}`",
            f"- auc_delta_vs_paligned_mcnd_1m: `{_format_value(report['auc_delta_vs_paligned_mcnd_1m'])}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- claim_scope: {report['claim_scope']}",
            "",
            "Artifacts:",
            "",
            f"- results: `{report['results']}`",
            f"- validation_report: `{report['validation_report']}`",
            f"- branch_gate: `{report['branch_gate']}`",
            f"- curves: `{report['curves']}`",
            f"- history_csv: `{report['history_csv']}`",
            f"- summary: `{report['summary']}`",
            *( [f"- plan_doc: `{report['plan_doc']}`"] if "plan_doc" in report else [] ),
            "",
        ]
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.12f}"
    return str(value)


def update_plan_doc_with_postprocess_result(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    status = "completed / postprocessed / branch gated" if report["status"] == "pass" else "retrieved / postprocess failed"
    text = text.replace("**Status:** running remotely / tmux monitor active", f"**Status:** {status}", 1)
    section = _plan_doc_result_section(report)
    header = "## Retrieved Result Record"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- invp-postprocess:{report['run_id']}:start -->"
    end = f"<!-- invp-postprocess:{report['run_id']}:end -->"
    block = f"{start}\n{section}\n{end}"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    plan_doc_path.write_text(text, encoding="utf-8")


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Branch status", report["branch_status"]),
        ("AUC", _format_value(report["auc"])),
        ("Accuracy", _format_value(report["accuracy"])),
        ("Calibrated accuracy", _format_value(report["calibrated_accuracy"])),
        ("Loss", _format_value(report["loss"])),
        ("Delta vs Zhang/Wang 1M AUC", _format_value(report["auc_delta"])),
        ("Delta vs p-aligned MCND 1M AUC", _format_value(report["auc_delta_vs_paligned_mcnd_1m"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Branch gate", report["branch_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Postprocess Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_invp_only_result(
        plan_path=args.plan,
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_rows=args.expected_rows,
        reference_auc=args.reference_auc,
        plan_doc_path=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
