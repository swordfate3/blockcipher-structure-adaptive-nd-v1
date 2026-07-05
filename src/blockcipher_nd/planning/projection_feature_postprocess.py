from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


DEFAULT_PROMOTION_MARGIN = 0.01
DEFAULT_WEAK_POSITIVE_AUC = 0.505


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess PRESENT truncated/projection feature screen results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--promotion-margin", type=float, default=DEFAULT_PROMOTION_MARGIN)
    parser.add_argument("--weak-positive-auc", type=float, default=DEFAULT_WEAK_POSITIVE_AUC)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_projection_feature_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    weak_positive_auc: float = DEFAULT_WEAK_POSITIVE_AUC,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_projection_feature_result(
        results_path,
        expected_rows=expected_rows,
        promotion_margin=promotion_margin,
        weak_positive_auc=weak_positive_auc,
    )
    gate_path = output_dir / f"{run_id}_projection_feature_gate.json"
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
        "projection_feature_gate": str(gate_path),
        "validation_status": validation_status,
        "projection_feature_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "anchor": gate_report["anchor"],
        "best": gate_report["best"],
        "delta_vs_anchor_auc": gate_report["delta_vs_anchor_auc"],
        "promotion_margin": gate_report["promotion_margin"],
        "weak_positive_auc": gate_report["weak_positive_auc"],
        "weak_ensemble_candidates": gate_report["weak_ensemble_candidates"],
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
            update_plan_doc_with_projection_feature_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_projection_feature_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    weak_positive_auc: float = DEFAULT_WEAK_POSITIVE_AUC,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")
    entries = [_entry(row) for row in rows]
    anchor = next((entry for entry in entries if entry["architecture_rank"] == 0), None)
    candidates = [entry for entry in entries if entry is not anchor]
    if anchor is None:
        errors.append("missing architecture_rank=0 full raw-pair anchor row")
    if not candidates:
        errors.append("missing projection candidate rows")
    for entry in entries:
        if not isinstance(entry.get("auc"), (int, float)):
            errors.append(f"missing auc for {entry['id']}")

    ranking = sorted(entries, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    best = next((entry for entry in ranking if entry is not anchor), None)
    delta = _auc_delta(best, anchor)
    weak_candidates = _weak_ensemble_candidates(candidates, anchor, weak_positive_auc)

    if errors:
        status = "fail"
        decision = "invalid_projection_feature_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "Result rows are incomplete or missing required metrics."
    elif delta is not None and delta >= promotion_margin:
        status = "pass"
        decision = "promote_projection_to_262k_confirmation"
        action = "prepare_262k_confirmation_for_best_projection_when_gpu_free"
        interpretation = "A structure-defined projection beats the full raw-pair anchor by the screen margin."
    elif len(weak_candidates) >= 2:
        status = "pass"
        decision = "run_projection_ensemble_diagnostic"
        action = "run_evaluate_projection_ensemble_before_scaling_projection_rows"
        interpretation = "Multiple projection views show weak-positive signal, so fusion can test error complementarity."
    elif delta is not None and delta > 0:
        status = "pass"
        decision = "weak_projection_candidate_hold"
        action = "do_not_scale_yet; keep_best_projection_as_weak_candidate"
        interpretation = "A projection is positive but below the configured promotion margin and without enough ensemble candidates."
    else:
        status = "pass"
        decision = "stop_projection_rule_for_now"
        action = "do_not_expand_this_projection_rule_without_new_prior"
        interpretation = "No projection candidate beats the full raw-pair anchor."

    return {
        "status": status,
        "results": str(results_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "errors": errors,
        "anchor": anchor,
        "best": best,
        "delta_vs_anchor_auc": delta,
        "promotion_margin": promotion_margin,
        "weak_positive_auc": weak_positive_auc,
        "weak_ensemble_candidates": weak_candidates,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "claim_scope": (
            "PRESENT r8 truncated/projection feature screen only; 65536/class is "
            "diagnostic, not paper-scale, formal, ensemble-confirmed, or breakthrough evidence"
        ),
    }


def update_plan_doc_with_projection_feature_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Truncated / Projection Feature Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- projection-feature-postprocess:{report['run_id']}:start -->"
    end = f"<!-- projection-feature-postprocess:{report['run_id']}:end -->"
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
    if decision == "promote_projection_to_262k_confirmation":
        return {
            "branch": "projection_feature_262k_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "selected_projection": (report.get("best") or {}).get("id", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-truncated-projection-feature-plan.md",
        }
    if decision == "run_projection_ensemble_diagnostic":
        return {
            "branch": "projection_ensemble_diagnostic",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "command": (
                "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-projection-ensemble "
                "--plan configs/experiment/innovation1/innovation1_spn_present_truncated_projection_feature_screen_65k_seed0.csv "
                "--source-results outputs/remote_results/<run_id>/results/<run_id>.jsonl "
                "--device auto --epochs 1 --batch-size 2048 --hidden-bits 128 "
                "--output outputs/remote_results/<run_id>/<run_id>_projection_ensemble.json"
            ),
        }
    if decision == "weak_projection_candidate_hold":
        return {
            "branch": "projection_feature_hold",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_projection_rule_for_now":
        return {
            "branch": "stop_projection_rule_for_now",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": ["new_projection_rule", "SPN-aware architecture", "DDT/trail feature route"],
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
        return ["Inspect validation and gate errors before interpreting projection metrics."]
    if branch == "projection_feature_262k_confirmation":
        return [
            "Record this as a 65536/class projection screen signal only.",
            "Prepare a lean 262144/class confirmation for full raw anchor versus the selected projection.",
            "Keep sample structure, negatives, validation key, and metric fixed.",
        ]
    if branch == "projection_ensemble_diagnostic":
        return [
            "Run the projection ensemble evaluator on the same screen plan to test weak-model complementarity.",
            "Treat ensemble results as same-validation diagnostic only.",
            "If ensemble beats best single by a clear margin, plan an independent confirmation split or seed.",
        ]
    if branch == "projection_feature_hold":
        return ["Keep the best projection as a weak candidate but do not scale it yet."]
    if branch == "stop_projection_rule_for_now":
        return ["Stop this projection rule and design a new structure-defined projection before another screen."]
    return ["Manual review required before branching."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    anchor = report.get("anchor") or {}
    best = report.get("best") or {}
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Anchor", anchor.get("id", "")),
        ("Anchor AUC", _format_value(anchor.get("auc"))),
        ("Best projection", best.get("id", "")),
        ("Best AUC", _format_value(best.get("auc"))),
        ("Delta vs anchor AUC", _format_value(report["delta_vs_anchor_auc"])),
        ("Weak ensemble candidates", len(report["weak_ensemble_candidates"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Projection gate", report["projection_feature_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Projection Feature Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Projection Feature Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Ranking",
        "",
        "| Rank | Row | Feature | Selected bits | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            f"{index} | `{entry['id']}` | `{entry['feature_encoding']}` | "
            f"{entry['selected_bit_count']} | "
            f"{_format_value(entry.get('auc'))} | "
            f"{_format_value(entry.get('calibrated_accuracy'))} | "
            f"{_format_value(entry.get('accuracy'))} | "
            f"{_format_value(entry.get('loss'))} |"
        )
    lines.extend(["", "## Weak Ensemble Candidates", ""])
    if report["weak_ensemble_candidates"]:
        lines.extend(f"- `{entry['id']}` AUC `{_format_value(entry.get('auc'))}`" for entry in report["weak_ensemble_candidates"])
    else:
        lines.append("- None")
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    selected = training.get("selected_bit_indices") or row.get("selected_bit_indices") or []
    architecture = str(row.get("architecture") or row.get("family") or row.get("selected_model") or "")
    feature_encoding = str(row.get("feature_encoding") or training.get("feature_encoding") or "")
    architecture_rank = int(row.get("architecture_rank", -1))
    return {
        "id": f"{architecture}:{feature_encoding}:{len(selected)}",
        "architecture": architecture,
        "architecture_rank": architecture_rank,
        "model": row.get("selected_model") or row.get("model"),
        "feature_encoding": feature_encoding,
        "selected_bit_count": len(selected),
        "selected_bit_indices": list(selected),
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "calibrated_accuracy": metrics.get("calibrated_accuracy"),
        "loss": metrics.get("loss"),
    }


def _weak_ensemble_candidates(
    candidates: list[dict[str, Any]],
    anchor: dict[str, Any] | None,
    weak_positive_auc: float,
) -> list[dict[str, Any]]:
    if anchor is None or not isinstance(anchor.get("auc"), (int, float)):
        return []
    anchor_auc = float(anchor["auc"])
    return [
        entry
        for entry in candidates
        if isinstance(entry.get("auc"), (int, float))
        and float(entry["auc"]) > weak_positive_auc
        and float(entry["auc"]) >= anchor_auc
    ]


def _auc_delta(candidate: dict[str, Any] | None, anchor: dict[str, Any] | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    candidate_auc = candidate.get("auc")
    anchor_auc = anchor.get("auc")
    if not isinstance(candidate_auc, (int, float)) or not isinstance(anchor_auc, (int, float)):
        return None
    return float(candidate_auc) - float(anchor_auc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_projection_feature_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        promotion_margin=args.promotion_margin,
        weak_positive_auc=args.weak_positive_auc,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
