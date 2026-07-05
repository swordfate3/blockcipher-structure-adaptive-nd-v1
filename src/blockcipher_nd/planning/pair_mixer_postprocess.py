from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


DEFAULT_ANCHOR_MODEL = "present_nibble_invp_pair_consistency_spn_only"
DEFAULT_CANDIDATE_MODEL = "present_nibble_invp_pair_mixer_consistency_spn_only"
DEFAULT_SUPPORT_MARGIN = 0.003


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess PRESENT pair-mixer consistency results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--anchor-model", default=DEFAULT_ANCHOR_MODEL)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--support-margin", type=float, default=DEFAULT_SUPPORT_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_pair_mixer_consistency_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
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

    gate_report = gate_pair_mixer_consistency_result(
        results_path,
        expected_rows=expected_rows,
        anchor_model=anchor_model,
        candidate_model=candidate_model,
        support_margin=support_margin,
    )
    gate_path = output_dir / f"{run_id}_pair_mixer_gate.json"
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
        "pair_mixer_gate": str(gate_path),
        "validation_status": validation_status,
        "pair_mixer_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "anchor_model": gate_report["anchor_model"],
        "candidate_model": gate_report["candidate_model"],
        "anchor_auc": gate_report["anchor_auc"],
        "candidate_auc": gate_report["candidate_auc"],
        "delta_vs_anchor_auc": gate_report["delta_vs_anchor_auc"],
        "support_margin": gate_report["support_margin"],
        "models": gate_report["models"],
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
            update_plan_doc_with_pair_mixer_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_pair_mixer_consistency_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    anchor_model: str = DEFAULT_ANCHOR_MODEL,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    support_margin: float = DEFAULT_SUPPORT_MARGIN,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")
    models = {_model_name(row): _metrics(row) for row in rows}
    anchor_auc = models.get(anchor_model, {}).get("auc")
    candidate_auc = models.get(candidate_model, {}).get("auc")
    if not isinstance(anchor_auc, (int, float)):
        errors.append(f"missing anchor auc: {anchor_model}")
    if not isinstance(candidate_auc, (int, float)):
        errors.append(f"missing candidate auc: {candidate_model}")

    delta = None
    if isinstance(anchor_auc, (int, float)) and isinstance(candidate_auc, (int, float)):
        delta = float(candidate_auc) - float(anchor_auc)

    if errors:
        decision = "invalid_pair_mixer_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "Result rows are incomplete or missing required model metrics."
        status = "fail"
    elif delta is not None and delta >= support_margin:
        decision = "support_pair_mixer_consistency_route"
        action = "prepare_seed_or_r9_pair_mixer_diagnostic_when_gpu_free"
        interpretation = "Cross-pair mixer beats the pair-consistency anchor by the configured margin."
        status = "pass"
    elif delta is not None and delta > 0:
        decision = "weak_pair_mixer_positive_needs_seed_or_scale_check"
        action = "wait_for_r8_1m_and_r9_weak_probe_before_expanding_pair_mixer"
        interpretation = "Cross-pair mixer is positive but below the support margin."
        status = "pass"
    else:
        decision = "stop_pair_mixer_route_for_now"
        action = "do_not_expand_pair_mixer_without_new_evidence"
        interpretation = "Cross-pair mixer does not beat the current pair-consistency anchor."
        status = "pass"

    return {
        "status": status,
        "results": str(results_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "errors": errors,
        "anchor_model": anchor_model,
        "candidate_model": candidate_model,
        "anchor_auc": anchor_auc,
        "candidate_auc": candidate_auc,
        "delta_vs_anchor_auc": delta,
        "support_margin": support_margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "models": models,
        "claim_scope": (
            "PRESENT pair-mixer consistency medium diagnostic only; "
            "not paper-scale, formal, or breakthrough evidence"
        ),
    }


def update_plan_doc_with_pair_mixer_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Pair-Mixer Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- pair-mixer-postprocess:{report['run_id']}:start -->"
    end = f"<!-- pair-mixer-postprocess:{report['run_id']}:end -->"
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
    if decision == "support_pair_mixer_consistency_route":
        return {
            "branch": "pair_mixer_seed_or_r9_diagnostic",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-present-pair-mixer-consistency-plan.md",
            "implementation_checklist": [
                "Wait for active r8 1M and r9 weak-probe gates before choosing r8 seed1 or r9 pair-mixer.",
                "Keep the next matrix lean: current anchor plus pair-mixer candidate only.",
                "Do not change difference_profile, negative_mode, sample_structure, or validation key.",
            ],
        }
    if decision == "weak_pair_mixer_positive_needs_seed_or_scale_check":
        return {
            "branch": "pair_mixer_hold_for_active_gates",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_pair_mixer_route_for_now":
        return {
            "branch": "stop_pair_mixer_route_for_now",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": [
                "r9_difference_screen",
                "r8_to_r9_curriculum",
                "new_spn_data_representation",
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
            "Do not analyze pair-mixer metrics yet; inspect validation and gate errors.",
            "Fix result retrieval, plan alignment, or result-row completeness before branching.",
        ]
    if branch == "pair_mixer_seed_or_r9_diagnostic":
        return [
            "Record this as medium diagnostic evidence only.",
            "Wait for active r8 1M and r9 weak-probe gates before launching the next pair-mixer run.",
            "Choose r8 seed1, r9 262144/class, or no launch based on those gates.",
        ]
    if branch == "pair_mixer_hold_for_active_gates":
        return [
            "Treat the result as a weak architecture signal.",
            "Do not expand until active r8 1M and r9 weak-probe evidence clarifies the route.",
        ]
    if branch == "stop_pair_mixer_route_for_now":
        return [
            "Stop this pair-mixer candidate for now.",
            "Prefer difference search, curriculum, or a new SPN-aware data representation.",
        ]
    return ["Manual review required before the next experiment branch."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Anchor model", report["anchor_model"]),
        ("Anchor AUC", _format_value(report["anchor_auc"])),
        ("Candidate model", report["candidate_model"]),
        ("Candidate AUC", _format_value(report["candidate_auc"])),
        ("Delta vs anchor AUC", _format_value(report["delta_vs_anchor_auc"])),
        ("Support margin", _format_value(report["support_margin"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Pair-mixer gate", report["pair_mixer_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Pair-Mixer Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Pair-Mixer Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Model Metrics",
        "",
        "| Model | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, metrics in sorted(report["models"].items()):
        lines.append(
            "| "
            f"`{model}` | "
            f"{_format_value(metrics.get('auc'))} | "
            f"{_format_value(metrics.get('calibrated_accuracy'))} | "
            f"{_format_value(metrics.get('accuracy'))} | "
            f"{_format_value(metrics.get('loss'))} |"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _model_name(row: dict[str, Any]) -> str:
    return str(row.get("model") or row.get("selected_model") or row.get("model_key") or "")


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    source = metrics if isinstance(metrics, dict) else row
    return {
        "auc": source.get("auc"),
        "accuracy": source.get("accuracy"),
        "calibrated_accuracy": source.get("calibrated_accuracy"),
        "loss": source.get("loss"),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_pair_mixer_consistency_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        anchor_model=args.anchor_model,
        candidate_model=args.candidate_model,
        support_margin=args.support_margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
