from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.active_auxiliary_gate import (
    DEFAULT_ACTIVE_AUXILIARY_MARGIN,
    DEFAULT_ANCHOR_MODEL,
    DEFAULT_CANDIDATE_MODEL,
    DEFAULT_SHUFFLED_MODEL,
    gate_active_auxiliary_result,
)
from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.next_action_readiness import next_action_readiness_report
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess active-pattern auxiliary-head results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--shuffled-model", default=DEFAULT_SHUFFLED_MODEL)
    parser.add_argument("--allow-missing-shuffled-control", action="store_true")
    parser.add_argument("--anchor-auc", type=float, default=None)
    parser.add_argument("--anchor-calibrated-accuracy", type=float, default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_ACTIVE_AUXILIARY_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_active_auxiliary_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    shuffled_model: str = DEFAULT_SHUFFLED_MODEL,
    require_shuffled_control: bool = True,
    anchor_auc: float | None = None,
    anchor_calibrated_accuracy: float | None = None,
    margin: float = DEFAULT_ACTIVE_AUXILIARY_MARGIN,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_active_auxiliary_result(
        results_path,
        expected_rows=expected_rows,
        anchor_model=anchor_model,
        candidate_model=candidate_model,
        shuffled_model=shuffled_model,
        require_shuffled_control=require_shuffled_control,
        anchor_auc=anchor_auc,
        anchor_calibrated_accuracy=anchor_calibrated_accuracy,
        margin=margin,
    )
    gate_path = output_dir / f"{run_id}_active_auxiliary_gate.json"
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
        "active_auxiliary_gate": str(gate_path),
        "validation_status": validation_status,
        "active_auxiliary_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "candidate_model": gate_report["candidate_model"],
        "candidate_auc": gate_report["candidate_auc"],
        "anchor_auc": gate_report["anchor_auc"],
        "shuffled_auc": gate_report["shuffled_auc"],
        "require_shuffled_control": gate_report["require_shuffled_control"],
        "margin_vs_anchor_auc": gate_report["margin_vs_anchor_auc"],
        "margin_vs_shuffled_auc": gate_report["margin_vs_shuffled_auc"],
        "required_margin": gate_report["required_margin"],
        "candidate_calibrated_accuracy": gate_report["candidate_calibrated_accuracy"],
        "anchor_calibrated_accuracy": gate_report["anchor_calibrated_accuracy"],
        "calibrated_delta_vs_anchor": gate_report["calibrated_delta_vs_anchor"],
        "claim_scope": gate_report["claim_scope"],
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
            update_plan_doc_with_active_auxiliary_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    _write_json(next_action_readiness_path, _next_action_readiness_report(report, summary_path))
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def update_plan_doc_with_active_auxiliary_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Active-Auxiliary Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- active-auxiliary-postprocess:{report['run_id']}:start -->"
    end = f"<!-- active-auxiliary-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_active_auxiliary_route":
        return {
            "branch": "active_auxiliary_seed1_confirmation",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1.json"
            ),
            "launch_remote_config": (
                "configs/remote/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
            ),
            "suggested_remote_config": (
                "configs/remote/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
            ),
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "weak_active_auxiliary_signal":
        return {
            "branch": "active_auxiliary_variance_check",
            "should_launch_remote": True,
            "requires_implementation": False,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md",
            "suggested_plan_config": (
                "configs/experiment/innovation1/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1.json"
            ),
            "launch_remote_config": (
                "configs/remote/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
            ),
            "suggested_remote_config": (
                "configs/remote/"
                "innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
            ),
            "monitor_owner": "tmux watcher or sub-agent",
        }
    if decision == "stop_active_auxiliary_route":
        return {
            "branch": "stop_active_auxiliary_route",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_plan_options": [
                "docs/experiments/innovation1-pairset-aggregation-control-plan.md",
                "docs/research/spn_structured_nn_research_plan.md",
            ],
            "fallback_hypotheses": [
                "pair_set_evidence_pooling",
                "ddt_graph_or_topology_aware_route",
                "cross_cipher_gift_skinny_transfer",
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
            "Do not branch yet; inspect validation_report and active_auxiliary_gate errors.",
            "Fix result retrieval, plan alignment, or gate inputs before launching another run.",
        ]
    if branch == "active_auxiliary_seed1_confirmation":
        return [
            "Record this as positive medium diagnostic evidence only.",
            "Use the prepared seed1 active-auxiliary plan/config/launcher/monitor assets.",
            "Run local readiness, then commit and push any documentation updates before remote launch.",
            "Launch seed1 only from a pushed commit and hand off monitoring to a local tmux watcher or sub-agent.",
            "Do not make formal or breakthrough claims from a single diagnostic seed.",
        ]
    if branch == "active_auxiliary_variance_check":
        return [
            "Record this as weak active-auxiliary evidence.",
            "Use the prepared seed1 active-auxiliary variance-check assets only if this is the selected next hypothesis.",
            "Keep the InvP-only anchor and shuffled-target control in the next matrix.",
            "Do not jump to 1M/class before a seed1 medium confirmation.",
        ]
    if branch == "stop_active_auxiliary_route":
        return [
            "Record this as tied or negative active-auxiliary evidence.",
            "Do not scale active-pattern auxiliary supervision as a main route.",
            "Switch to pair-set evidence pooling, DDT/topology-aware features, or route consolidation planning.",
            "Write the next docs/experiments plan before implementing or launching a new route.",
        ]
    return ["Review the active-auxiliary gate manually before launching another experiment."]


def _next_action_readiness_report(report: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    next_action = report.get("next_action", {})
    requires_implementation = bool(next_action.get("requires_implementation")) if isinstance(next_action, dict) else False
    readiness = next_action_readiness_report(
        summary_path=summary_path,
        report=report,
        implementation_checklist=_implementation_checklist(next_action) if requires_implementation else [],
    )
    if requires_implementation and not next_action.get("should_launch_remote"):
        readiness["status"] = "needs_implementation"
    return readiness


def _implementation_checklist(next_action: dict[str, Any]) -> list[str]:
    branch = str(next_action.get("branch") or "<next-branch>")
    plan_doc = str(next_action.get("next_plan_doc") or "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md")
    suggested_plan = str(next_action.get("suggested_plan_config") or "<next active-auxiliary config>")
    suggested_remote = str(next_action.get("suggested_remote_config") or "<next active-auxiliary remote config>")
    return [
        f"Prepare `{branch}` before any remote launch.",
        f"Update the experiment plan in `{plan_doc}`.",
        f"Create and review `{suggested_plan}` with the same active-auxiliary hypothesis.",
        f"Create and review `{suggested_remote}` under G:\\lxy remote artifact rules.",
        "Run local smoke/readiness checks.",
        "Commit and push docs/config/code changes before launching from GitHub.",
    ]


def _markdown_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {report['run_id']}",
            "",
            f"- status: `{report['status']}`",
            f"- validation_status: `{report['validation_status']}`",
            f"- active_auxiliary_status: `{report['active_auxiliary_status']}`",
            f"- decision: `{report['decision']}`",
            f"- action: `{report['action']}`",
            f"- candidate_model: `{report['candidate_model']}`",
            f"- candidate_auc: `{_format_value(report['candidate_auc'])}`",
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
            f"- active_auxiliary_gate: `{report['active_auxiliary_gate']}`",
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
        ("Active-auxiliary gate status", report["active_auxiliary_status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Interpretation", report["interpretation"]),
        ("Candidate model", report["candidate_model"]),
        ("Candidate AUC", _format_value(report["candidate_auc"])),
        ("Anchor AUC", _format_value(report["anchor_auc"])),
        ("Shuffled-control AUC", _format_value(report["shuffled_auc"])),
        ("Require shuffled control", report["require_shuffled_control"]),
        ("Margin vs anchor AUC", _format_value(report["margin_vs_anchor_auc"])),
        ("Margin vs shuffled AUC", _format_value(report["margin_vs_shuffled_auc"])),
        ("Required margin", _format_value(report["required_margin"])),
        ("Claim scope", report["claim_scope"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Next steps", "; ".join(report["next_steps"])),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Active-auxiliary gate", report["active_auxiliary_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
        ("Next action readiness", report["next_action_readiness"]),
    ]
    lines = [f"### {report['run_id']} Active-Auxiliary Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_active_auxiliary_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        anchor_model=args.anchor_model,
        candidate_model=args.candidate_model,
        shuffled_model=args.shuffled_model,
        require_shuffled_control=not args.allow_missing_shuffled_control,
        anchor_auc=args.anchor_auc,
        anchor_calibrated_accuracy=args.anchor_calibrated_accuracy,
        margin=args.margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
