from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.ddt_graph_gate import gate_ddt_graph_result
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local post-retrieval validation, plotting, and gating for DDT graph results."
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan CSV path.")
    parser.add_argument("--results", required=True, type=Path, help="Retrieved DDT graph JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated reports.")
    parser.add_argument("--run-id", required=True, help="Run id used in output filenames and plot title.")
    parser.add_argument("--expected-rows", type=int, default=5)
    parser.add_argument(
        "--update-plan-doc",
        type=Path,
        action="append",
        default=[],
        help="Optional experiment plan Markdown file to update. Can be repeated.",
    )
    return parser.parse_args(argv)


def postprocess_ddt_graph_result(
    *,
    plan_path: Path,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 5,
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

    ddt_gate_report = gate_ddt_graph_result(
        results_path,
        expected_rows=expected_rows,
    )
    ddt_gate_path = output_dir / f"{run_id}_ddt_graph_gate.json"
    _write_json(ddt_gate_path, ddt_gate_report)

    status = "pass" if validation_report["status"] == "pass" and ddt_gate_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "ddt_graph_gate": str(ddt_gate_path),
        "validation_status": validation_report["status"],
        "ddt_graph_status": ddt_gate_report["status"],
        "decision": ddt_gate_report["decision"],
        "action": ddt_gate_report["action"],
        "interpretation": ddt_gate_report["interpretation"],
        "max_control_auc": ddt_gate_report["max_control_auc"],
        "margin_vs_best_control_auc": ddt_gate_report["margin_vs_best_control_auc"],
        "margin_vs_invp_auc": ddt_gate_report["margin_vs_invp_auc"],
        "margin_vs_transition_no_ddt_auc": ddt_gate_report["margin_vs_transition_no_ddt_auc"],
        "margin_vs_no_ddt_graph_auc": ddt_gate_report["margin_vs_no_ddt_graph_auc"],
        "margin_vs_no_ddt_auc": ddt_gate_report["margin_vs_no_ddt_auc"],
        "margin_vs_shuffled_auc": ddt_gate_report["margin_vs_shuffled_auc"],
        "calibrated_delta_vs_invp": ddt_gate_report["calibrated_delta_vs_invp"],
        "required_margin": ddt_gate_report["required_margin"],
        "models": ddt_gate_report["models"],
        "claim_scope": ddt_gate_report["claim_scope"],
    }
    report["next_action"] = _next_action(report)
    report["next_steps"] = _next_steps(report)
    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    markdown_path = output_dir / f"{run_id}_postprocess_summary.md"
    next_action_readiness_path = output_dir / f"{run_id}_next_action_readiness.json"
    report["summary"] = str(summary_path)
    report["summary_markdown"] = str(markdown_path)
    report["next_action_readiness"] = str(next_action_readiness_path)

    update_paths = _merge_plan_doc_paths(plan_doc_path, plan_doc_paths)
    if update_paths:
        for path in update_paths:
            update_plan_doc_with_ddt_graph_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    _write_json(next_action_readiness_path, _next_action_readiness_report(report, summary_path))
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_ddt_graph_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved DDT Graph Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- ddt-graph-postprocess:{report['run_id']}:start -->"
    end = f"<!-- ddt-graph-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_ddt_graph_route":
        launch_config = "configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json"
        return {
            "branch": "ddt_graph_seed1_confirmation",
            "should_launch_remote": True,
            "requires_implementation": False,
            "launch_remote_config": launch_config,
            "readiness_command": _readiness_command(launch_config),
            "run_id": "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630",
            "monitor_owner": "tmux watcher or sub-agent",
            "plan_doc": "docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md",
            "reason": decision,
        }
    if decision == "weak_ddt_graph_signal":
        launch_config = "configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json"
        return {
            "branch": "ddt_graph_seed1_variance_check",
            "should_launch_remote": True,
            "requires_implementation": False,
            "launch_remote_config": launch_config,
            "readiness_command": _readiness_command(launch_config),
            "run_id": "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630",
            "monitor_owner": "tmux watcher or sub-agent",
            "plan_doc": "docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md",
            "reason": decision,
        }
    if decision == "stop_ddt_graph_route":
        return {
            "branch": "candidate_trail_consistency",
            "should_launch_remote": False,
            "requires_implementation": True,
            "plan_doc": "docs/experiments/innovation1-candidate-trail-consistency-plan.md",
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
            "Do not branch yet; inspect validation_report and ddt_graph_gate errors.",
            "Fix retrieval, plan alignment, or metric availability before launching another run.",
        ]
    if branch == "ddt_graph_seed1_confirmation":
        next_action = report["next_action"]
        return [
            "Update the experiment plan with this positive medium-scale diagnostic result.",
            f"Run the remote readiness gate: {next_action['readiness_command']}",
            f"Launch {next_action['launch_remote_config']} from the pushed commit.",
            "Hand off seed1 monitoring and retrieval to a local tmux watcher or sub-agent.",
            "Do not make paper-scale or breakthrough claims from this single 262144/class seed.",
        ]
    if branch == "ddt_graph_seed1_variance_check":
        next_action = report["next_action"]
        return [
            "Update the experiment plan with this weak DDT graph signal.",
            f"Run the remote readiness gate: {next_action['readiness_command']}",
            f"Launch {next_action['launch_remote_config']} as a 262144/class seed1 variance check from the pushed commit.",
            "Hand off seed1 monitoring and retrieval to a local tmux watcher or sub-agent.",
            "Do not promote DDT graph as the main route yet.",
        ]
    if branch == "candidate_trail_consistency":
        return [
            "Update the experiment plan with tied or negative DDT graph evidence.",
            "Do not scale this DDT graph route to 1M.",
            "Switch to the candidate-trail consistency data representation plan before creating a medium remote config.",
            "Keep pair-set aggregation as a deferred attribution control unless the user explicitly selects it.",
        ]
    return ["Review the DDT graph gate manually before launching another experiment."]


def _next_action_readiness_report(report: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    next_action = report.get("next_action", {})
    if not isinstance(next_action, dict):
        next_action = {}

    readiness_reports: list[dict[str, Any]] = []
    for role, key in [
        ("stage_a", "stage_a_remote_config"),
        ("primary", "launch_remote_config"),
    ]:
        config = next_action.get(key)
        if not isinstance(config, str) or not config:
            continue
        readiness_reports.append(
            {
                "role": role,
                "config": config,
                "readiness": _readiness_report(Path(config)),
            }
        )

    should_launch_remote = bool(next_action.get("should_launch_remote"))
    readiness_statuses = [item["readiness"]["status"] for item in readiness_reports]
    readiness_pass = bool(readiness_reports) and all(status == "pass" for status in readiness_statuses)
    return {
        "status": "pass" if (not should_launch_remote or readiness_pass) else "fail",
        "summary": str(summary_path),
        "run_id": report.get("run_id"),
        "decision": report.get("decision"),
        "action": report.get("action"),
        "branch": next_action.get("branch"),
        "should_launch_remote": should_launch_remote,
        "requires_implementation": bool(next_action.get("requires_implementation")),
        "readiness_pass": readiness_pass,
        "readiness_reports": readiness_reports,
        "next_action": next_action,
        "claim_scope": report.get("claim_scope"),
        "errors": _next_action_readiness_errors(
            should_launch_remote=should_launch_remote,
            readiness_reports=readiness_reports,
        ),
    }


def _next_action_readiness_errors(
    *,
    should_launch_remote: bool,
    readiness_reports: list[dict[str, Any]],
) -> list[str]:
    if not should_launch_remote:
        return []
    if not readiness_reports:
        return ["next_action requests remote launch but no launch_remote_config was provided"]
    errors: list[str] = []
    for item in readiness_reports:
        readiness = item["readiness"]
        if readiness["status"] != "pass":
            errors.append(f"{item['role']} readiness failed: {readiness.get('errors', [])}")
    return errors


def _readiness_report(config_path: Path) -> dict[str, Any]:
    try:
        return remote_readiness_report(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "config": str(config_path),
            "errors": [str(exc)],
            "warnings": [],
        }


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- ddt_graph_status: `{report['ddt_graph_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- margin_vs_best_control_auc: `{_format_value(report['margin_vs_best_control_auc'])}`",
            f"- margin_vs_no_ddt_graph_auc: `{_format_value(report['margin_vs_no_ddt_graph_auc'])}`",
            f"- max_control_auc: `{_format_value(report['max_control_auc'])}`",
            f"- calibrated_delta_vs_invp: `{_format_value(report['calibrated_delta_vs_invp'])}`",
            f"- claim_scope: {report['claim_scope']}",
            "",
            "Models:",
            "",
            *_model_lines(report["models"]),
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
            f"- ddt_graph_gate: `{report['ddt_graph_gate']}`",
            f"- curves: `{report['curves']}`",
            f"- history_csv: `{report['history_csv']}`",
            f"- summary: `{report['summary']}`",
            f"- next_action_readiness: `{report['next_action_readiness']}`",
            *( [f"- plan_docs: `{'; '.join(report['plan_docs'])}`"] if "plan_docs" in report else [] ),
            "",
        ]
    )


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("DDT graph gate status", report["ddt_graph_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("Max control AUC", _format_value(report["max_control_auc"])),
        ("Margin vs best control AUC", _format_value(report["margin_vs_best_control_auc"])),
        ("Margin vs InvP AUC", _format_value(report["margin_vs_invp_auc"])),
        ("Margin vs transition no-DDT AUC", _format_value(report["margin_vs_transition_no_ddt_auc"])),
        ("Margin vs same-graph no-DDT AUC", _format_value(report["margin_vs_no_ddt_graph_auc"])),
        ("Margin vs shuffled AUC", _format_value(report["margin_vs_shuffled_auc"])),
        ("Calibrated delta vs InvP", _format_value(report["calibrated_delta_vs_invp"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next action should launch remote", report["next_action"]["should_launch_remote"]),
        ("Next action launch config", report["next_action"].get("launch_remote_config", "")),
        ("Next action readiness command", report["next_action"].get("readiness_command", "")),
        ("Next action run id", report["next_action"].get("run_id", "")),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("DDT graph gate", report["ddt_graph_gate"]),
        ("Curves", report["curves"]),
        ("History CSV", report["history_csv"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
        ("Next action readiness", report["next_action_readiness"]),
    ]
    lines = [f"### {report['run_id']} DDT Graph Result", "", "| Field | Value |", "|---|---|"]
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


def _readiness_command(config_path: str) -> str:
    return f"UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config {config_path}"


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
    report = postprocess_ddt_graph_result(
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
