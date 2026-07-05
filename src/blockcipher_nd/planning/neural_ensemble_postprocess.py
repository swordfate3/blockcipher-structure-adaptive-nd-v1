from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_postprocess import _format_value


DEFAULT_IMPROVEMENT_MARGIN = 0.001
DEFAULT_MAX_ERROR_JACCARD = 0.85


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a PRESENT neural ensemble score result.")
    parser.add_argument("--train-results", required=True, type=Path)
    parser.add_argument("--ensemble-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument("--improvement-margin", type=float, default=DEFAULT_IMPROVEMENT_MARGIN)
    parser.add_argument("--max-error-jaccard", type=float, default=DEFAULT_MAX_ERROR_JACCARD)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_neural_ensemble_result(
    *,
    train_results_path: Path,
    ensemble_summary_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 3,
    improvement_margin: float = DEFAULT_IMPROVEMENT_MARGIN,
    max_error_jaccard: float = DEFAULT_MAX_ERROR_JACCARD,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_report = gate_neural_ensemble_result(
        train_results_path=train_results_path,
        ensemble_summary_path=ensemble_summary_path,
        expected_rows=expected_rows,
        improvement_margin=improvement_margin,
        max_error_jaccard=max_error_jaccard,
    )
    gate_path = output_dir / f"{run_id}_neural_ensemble_gate.json"
    _write_json(gate_path, gate_report)

    status = "pass" if gate_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "train_results": str(train_results_path),
        "ensemble_summary": str(ensemble_summary_path),
        "output_dir": str(output_dir),
        "neural_ensemble_gate": str(gate_path),
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "best_single": gate_report["best_single"],
        "best_ensemble": gate_report["best_ensemble"],
        "delta_best_ensemble_vs_single_auc": gate_report["delta_best_ensemble_vs_single_auc"],
        "improvement_margin": gate_report["improvement_margin"],
        "max_error_jaccard_at_0_5": gate_report["max_error_jaccard_at_0_5"],
        "max_allowed_error_jaccard_at_0_5": gate_report["max_allowed_error_jaccard_at_0_5"],
        "max_double_fault_rate_at_0_5": gate_report["max_double_fault_rate_at_0_5"],
        "diverse_expert_pool": gate_report["diverse_expert_pool"],
        "models": gate_report["models"],
        "ensembles": gate_report["ensembles"],
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
            update_plan_doc_with_neural_ensemble_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_neural_ensemble_result(
    *,
    train_results_path: Path,
    ensemble_summary_path: Path,
    expected_rows: int = 3,
    improvement_margin: float = DEFAULT_IMPROVEMENT_MARGIN,
    max_error_jaccard: float = DEFAULT_MAX_ERROR_JACCARD,
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_results_path)
    summary = json.loads(ensemble_summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if len(train_rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(train_rows)}")
    if summary.get("status") != "pass":
        errors.append(f"ensemble_summary status is not pass: {summary.get('status')}")

    best_single = summary.get("best_single")
    best_ensemble = summary.get("best_ensemble")
    if not isinstance(best_single, dict):
        errors.append("missing best_single")
        best_single = {}
    if not isinstance(best_ensemble, dict):
        errors.append("missing best_ensemble")
        best_ensemble = {}

    best_single_auc = _metric_auc(best_single)
    best_ensemble_auc = _metric_auc(best_ensemble)
    if best_single_auc is None:
        errors.append("missing best_single metrics.auc")
    if best_ensemble_auc is None:
        errors.append("missing best_ensemble metrics.auc")

    delta = _float_or_none(summary.get("delta_best_ensemble_vs_single_auc"))
    if delta is None and best_single_auc is not None and best_ensemble_auc is not None:
        delta = best_ensemble_auc - best_single_auc

    pairwise = _pairwise_diversity(summary)
    if not pairwise:
        errors.append("missing pairwise diversity")
    max_observed_error_jaccard = _max_pairwise(pairwise, "error_jaccard_at_0_5")
    max_observed_double_fault = _max_pairwise(pairwise, "double_fault_rate_at_0_5")
    diverse_expert_pool = summary.get("diverse_expert_pool")
    if not isinstance(diverse_expert_pool, dict):
        diverse_expert_pool = {"status": "unknown", "decision": "diverse_expert_pool_not_reported", "errors": []}
    diverse_pool_failed = diverse_expert_pool.get("status") == "fail"

    if errors:
        status = "fail"
        decision = "invalid_neural_ensemble_result"
        action = "repair_result_retrieval_or_artifact_alignment_before_branching"
        interpretation = "The neural ensemble result is incomplete or missing required metrics."
    elif (
        diverse_pool_failed
        and delta is not None
        and delta >= improvement_margin
        and (max_observed_error_jaccard is None or max_observed_error_jaccard <= max_error_jaccard)
    ):
        status = "pass"
        decision = "keep_near_neighbor_ensemble_control_not_diverse_pool"
        action = "record_as_near_neighbor_control_and_seek_non_neighbor_expert"
        interpretation = (
            "Fixed score aggregation improves, but the diverse expert pool gate failed; "
            "this is a near-neighbor control, not a diverse multi-network route."
        )
    elif delta is not None and delta >= improvement_margin and (
        max_observed_error_jaccard is None or max_observed_error_jaccard <= max_error_jaccard
    ):
        status = "pass"
        decision = "keep_neural_ensemble_route_prepare_262k_confirmation"
        action = "prepare_same_protocol_262144_class_neural_ensemble_confirmation"
        interpretation = "Fixed score aggregation improves over the best single model and error overlap is acceptable."
    elif delta is not None and delta >= improvement_margin:
        status = "pass"
        decision = "hold_neural_ensemble_route_high_error_overlap"
        action = "inspect_diversity_before_scaling_or_change_candidate_pool"
        interpretation = "AUC improves, but pairwise error overlap is too high for a clean complementarity claim."
    elif delta is not None and delta > 0:
        status = "pass"
        decision = "weak_neural_ensemble_positive_below_gate"
        action = "treat_as_calibration_or_threshold_diagnostic_only"
        interpretation = "The ensemble is positive but below the predeclared AUC improvement margin."
    else:
        status = "pass"
        decision = "stop_neural_ensemble_candidate_pool"
        action = "return_to_spn_architecture_or_data_representation_changes"
        interpretation = "The ensemble does not beat the best single model at the tested diagnostic scale."

    return {
        "status": status,
        "train_results": str(train_results_path),
        "ensemble_summary": str(ensemble_summary_path),
        "expected_rows": expected_rows,
        "actual_rows": len(train_rows),
        "errors": errors,
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "delta_best_ensemble_vs_single_auc": delta,
        "improvement_margin": improvement_margin,
        "max_error_jaccard_at_0_5": max_observed_error_jaccard,
        "max_allowed_error_jaccard_at_0_5": max_error_jaccard,
        "max_double_fault_rate_at_0_5": max_observed_double_fault,
        "diverse_expert_pool": diverse_expert_pool,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "models": summary.get("models", []),
        "ensembles": summary.get("ensembles", []),
        "claim_scope": (
            "PRESENT neural ensemble application-level diagnostic only; "
            "not raw single-sample SOTA, not formal evidence, and not a breakthrough claim"
        ),
    }


def update_plan_doc_with_neural_ensemble_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Neural Ensemble Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- neural-ensemble-postprocess:{report['run_id']}:start -->"
    end = f"<!-- neural-ensemble-postprocess:{report['run_id']}:end -->"
    block = f"{start}\n{_plan_doc_result_section(report)}\n{end}"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    plan_doc_path.write_text(text, encoding="utf-8")


def _next_action(report: dict[str, Any]) -> dict[str, Any]:
    decision = str(report["decision"])
    if report["status"] != "pass":
        return {
            "branch": "invalid",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": "postprocess_status_failed",
        }
    if decision == "keep_neural_ensemble_route_prepare_262k_confirmation":
        return {
            "branch": "neural_ensemble_262k_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md",
            "implementation_checklist": [
                "Prepare a same-protocol 262144/class matrix with the same candidate pool.",
                "Keep negative_mode, sample_structure, difference_profile, and validation_key unchanged.",
                "Reuse frozen score artifact export and evaluate-neural-ensemble.",
            ],
        }
    if decision == "hold_neural_ensemble_route_high_error_overlap":
        return {
            "branch": "neural_ensemble_diversity_review",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "keep_near_neighbor_ensemble_control_not_diverse_pool":
        return {
            "branch": "neural_ensemble_near_neighbor_control",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "next_plan_doc": "docs/experiments/innovation1-present-diverse-expert-pool-plan.md",
        }
    if decision == "weak_neural_ensemble_positive_below_gate":
        return {
            "branch": "neural_ensemble_diagnostic_only",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_neural_ensemble_candidate_pool":
        return {
            "branch": "stop_neural_ensemble_candidate_pool",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": [
                "improve InvP/P-layer architecture",
                "try a new SPN-aware data representation",
                "change candidate pool only with a new hypothesis",
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
            "Do not interpret neural ensemble metrics yet.",
            "Repair retrieval, result completeness, or score artifact alignment first.",
        ]
    if branch == "neural_ensemble_262k_confirmation":
        return [
            "Record the 65536/class result as diagnostic application-level support only.",
            "Prepare a same-protocol 262144/class confirmation with the same three-model pool.",
            "Do not call the result formal until 1000000/class multi-seed evidence exists.",
        ]
    if branch == "neural_ensemble_diversity_review":
        return [
            "Inspect pairwise diversity before spending a larger run.",
            "Prefer candidate-pool redesign over simply adding more weak models.",
        ]
    if branch == "neural_ensemble_near_neighbor_control":
        return [
            "Record the ensemble as a near-neighbor control only.",
            "Do not prepare a larger confirmation until a non-neighbor expert passes the diverse pool gate.",
            "Return to SPN representation/data search for a compatible weak-positive expert.",
        ]
    if branch == "neural_ensemble_diagnostic_only":
        return [
            "Treat the result as a calibration or weak-threshold diagnostic.",
            "Return to SPN architecture/data-representation changes unless another candidate pool is justified.",
        ]
    if branch == "stop_neural_ensemble_candidate_pool":
        return [
            "Stop this ensemble candidate pool.",
            "Prefer improving the strongest InvP/P-layer route or designing a new SPN-aware representation.",
        ]
    return ["Manual review required before the next experiment branch."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    best_single = report["best_single"]
    best_ensemble = report["best_ensemble"]
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Best single", _single_label(best_single)),
        ("Best single AUC", _format_value(_metric_auc(best_single))),
        ("Best ensemble", str(best_ensemble.get("mode", ""))),
        ("Best ensemble AUC", _format_value(_metric_auc(best_ensemble))),
        ("Delta vs best single AUC", _format_value(report["delta_best_ensemble_vs_single_auc"])),
        ("Improvement margin", _format_value(report["improvement_margin"])),
        ("Max error Jaccard at 0.5", _format_value(report["max_error_jaccard_at_0_5"])),
        ("Max allowed error Jaccard", _format_value(report["max_allowed_error_jaccard_at_0_5"])),
        ("Max double-fault rate at 0.5", _format_value(report["max_double_fault_rate_at_0_5"])),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Train results", report["train_results"]),
        ("Ensemble summary", report["ensemble_summary"]),
        ("Gate JSON", report["neural_ensemble_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Neural Ensemble Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Neural Ensemble Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Models",
        "",
        "| Model | AUC | Calibrated accuracy | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for model in report["models"]:
        metrics = model.get("metrics", {}) if isinstance(model, dict) else {}
        lines.append(
            "| "
            f"`{model.get('model_key', '')}` | "
            f"{_format_value(metrics.get('auc'))} | "
            f"{_format_value(metrics.get('calibrated_accuracy'))} | "
            f"{_format_value(metrics.get('accuracy'))} |"
        )
    lines.extend(["", "## Ensembles", "", "| Mode | AUC | Calibrated accuracy |", "|---|---:|---:|"])
    for ensemble in report["ensembles"]:
        metrics = ensemble.get("metrics", {}) if isinstance(ensemble, dict) else {}
        lines.append(
            "| "
            f"`{ensemble.get('mode', '')}` | "
            f"{_format_value(metrics.get('auc'))} | "
            f"{_format_value(metrics.get('calibrated_accuracy'))} |"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_auc(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _float_or_none(metrics.get("auc"))


def _single_label(row: dict[str, Any]) -> str:
    return str(row.get("model_key") or row.get("model") or row.get("mode") or "")


def _pairwise_diversity(summary: dict[str, Any]) -> list[dict[str, Any]]:
    diversity = summary.get("diversity")
    if not isinstance(diversity, dict):
        return []
    pairwise = diversity.get("pairwise")
    if not isinstance(pairwise, list):
        return []
    return [item for item in pairwise if isinstance(item, dict)]


def _max_pairwise(pairwise: list[dict[str, Any]], field: str) -> float | None:
    values = [_float_or_none(item.get(field)) for item in pairwise]
    numeric = [value for value in values if value is not None]
    return max(numeric) if numeric else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_neural_ensemble_result(
        train_results_path=args.train_results,
        ensemble_summary_path=args.ensemble_summary,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_rows=args.expected_rows,
        improvement_margin=args.improvement_margin,
        max_error_jaccard=args.max_error_jaccard,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
