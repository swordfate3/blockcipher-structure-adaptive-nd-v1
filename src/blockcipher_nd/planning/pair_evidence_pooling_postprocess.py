from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


DEFAULT_PROMOTION_MARGIN = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess PRESENT pair-evidence pooling screen results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--promotion-margin", type=float, default=DEFAULT_PROMOTION_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_pair_evidence_pooling_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_pair_evidence_pooling_result(
        results_path,
        expected_rows=expected_rows,
        promotion_margin=promotion_margin,
    )
    gate_path = output_dir / f"{run_id}_pair_evidence_pooling_gate.json"
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
        "pair_evidence_pooling_gate": str(gate_path),
        "validation_status": validation_status,
        "pair_evidence_pooling_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "anchor": gate_report["anchor"],
        "best": gate_report["best"],
        "delta_vs_anchor_auc": gate_report["delta_vs_anchor_auc"],
        "promotion_margin": gate_report["promotion_margin"],
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
            update_plan_doc_with_pair_evidence_pooling_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_pair_evidence_pooling_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")
    entries = [_entry(row) for row in rows]
    anchor = next((entry for entry in entries if entry["architecture_rank"] == 0), None)
    candidates = [entry for entry in entries if entry is not anchor]
    if anchor is None:
        errors.append("missing architecture_rank=0 anchor row")
    if not candidates:
        errors.append("missing pooling candidate rows")
    for entry in entries:
        if not isinstance(entry.get("auc"), (int, float)):
            errors.append(f"missing auc for {entry['id']}")

    ranking = sorted(entries, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    best = next((entry for entry in ranking if entry is not anchor), None)
    delta = None
    if anchor is not None and best is not None:
        anchor_auc = anchor.get("auc")
        best_auc = best.get("auc")
        if isinstance(anchor_auc, (int, float)) and isinstance(best_auc, (int, float)):
            delta = float(best_auc) - float(anchor_auc)

    if errors:
        status = "fail"
        decision = "invalid_pair_evidence_pooling_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "Result rows are incomplete or missing required metrics."
    elif delta is not None and delta >= promotion_margin:
        status = "pass"
        decision = "promote_best_pooling_to_262k_confirmation"
        action = "prepare_262k_confirmation_for_best_pooling_when_gate_selects_this_route"
        interpretation = "A pair-mixer pooling candidate beats the anchor by the configured screen margin."
    elif delta is not None and delta > 0:
        status = "pass"
        decision = "weak_pooling_candidate_wait_for_active_gates"
        action = "do_not_launch_confirmation_until_r8_1m_or_r9_gate_selects_pooling"
        interpretation = "A pair-mixer pooling candidate is positive but below the screen margin."
    else:
        status = "pass"
        decision = "stop_pooling_screen_for_now"
        action = "do_not_expand_pooling_without_new_pairset_evidence"
        interpretation = "No pair-mixer pooling candidate beats the current anchor."

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
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "claim_scope": (
            "PRESENT pair-evidence pooling screen only; 65536/class is not paper-scale, "
            "formal, or breakthrough evidence"
        ),
    }


def update_plan_doc_with_pair_evidence_pooling_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Pair-Evidence Pooling Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- pair-evidence-pooling-postprocess:{report['run_id']}:start -->"
    end = f"<!-- pair-evidence-pooling-postprocess:{report['run_id']}:end -->"
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
    if decision == "promote_best_pooling_to_262k_confirmation":
        return {
            "branch": "pair_evidence_pooling_262k_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "selected_pooling": (report.get("best") or {}).get("pooling", ""),
            "next_plan_doc": "docs/experiments/innovation1-present-pair-evidence-pooling-screen-plan.md",
        }
    if decision == "weak_pooling_candidate_wait_for_active_gates":
        return {
            "branch": "pair_evidence_pooling_hold_for_active_gates",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_pooling_screen_for_now":
        return {
            "branch": "stop_pair_evidence_pooling_for_now",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": ["r9_difference_screen", "r8_to_r9_curriculum", "new_spn_data_representation"],
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
        return ["Inspect validation and gate errors before interpreting pooling metrics."]
    if branch == "pair_evidence_pooling_262k_confirmation":
        return [
            "Record this as a 65536/class screen signal only.",
            "Prepare a lean 262144/class confirmation for the selected pooling versus the anchor.",
            "Keep all benchmark fields fixed so the result is attributable to pooling mode.",
        ]
    if branch == "pair_evidence_pooling_hold_for_active_gates":
        return [
            "Treat the best pooling mode as a weak screen candidate.",
            "Wait for r8 1M and r9 weak-probe gates before confirmation.",
        ]
    if branch == "stop_pair_evidence_pooling_for_now":
        return ["Stop this pooling screen branch unless later pair-set evidence reopens it."]
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
        ("Best candidate", best.get("id", "")),
        ("Best pooling", best.get("pooling", "")),
        ("Best AUC", _format_value(best.get("auc"))),
        ("Delta vs anchor AUC", _format_value(report["delta_vs_anchor_auc"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Pooling gate", report["pair_evidence_pooling_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Pair-Evidence Pooling Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Pair-Evidence Pooling Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Ranking",
        "",
        "| Rank | Row | Pooling | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            f"{index} | `{entry['id']}` | `{entry['pooling']}` | "
            f"{_format_value(entry.get('auc'))} | "
            f"{_format_value(entry.get('calibrated_accuracy'))} | "
            f"{_format_value(entry.get('accuracy'))} | "
            f"{_format_value(entry.get('loss'))} |"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines) + "\n"


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    training = row.get("training") if isinstance(row.get("training"), dict) else {}
    model_options = training.get("model_options") if isinstance(training.get("model_options"), dict) else {}
    architecture = str(row.get("architecture") or row.get("family") or row.get("selected_model") or "")
    pooling = str(model_options.get("pooling") or "")
    architecture_rank = int(row.get("architecture_rank", -1))
    return {
        "id": f"{architecture}:{pooling}",
        "architecture": architecture,
        "architecture_rank": architecture_rank,
        "model": row.get("selected_model") or row.get("model"),
        "pooling": pooling,
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "calibrated_accuracy": metrics.get("calibrated_accuracy"),
        "loss": metrics.get("loss"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_pair_evidence_pooling_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        promotion_margin=args.promotion_margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
