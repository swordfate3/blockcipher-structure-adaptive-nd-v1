from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.invp_postprocess import _format_value
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


DEFAULT_RAW_MARGIN = 0.01
DEFAULT_INVP_MARGIN = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess PRESENT integral/inverse-round feature screen.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--raw-margin", type=float, default=DEFAULT_RAW_MARGIN)
    parser.add_argument("--invp-margin", type=float, default=DEFAULT_INVP_MARGIN)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def postprocess_integral_inverse_feature_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None = None,
    expected_rows: int | None = None,
    raw_margin: float = DEFAULT_RAW_MARGIN,
    invp_margin: float = DEFAULT_INVP_MARGIN,
    plan_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report: dict[str, Any] | None = None
    validation_path: Path | None = None
    if plan_path is not None:
        validation_report = validate_result_plan_alignment(plan_path, results_path, expected_rows=expected_rows)
        validation_path = output_dir / f"{run_id}_local_result_gate.json"
        _write_json(validation_path, validation_report)

    gate_report = gate_integral_inverse_feature_result(
        results_path,
        expected_rows=expected_rows,
        raw_margin=raw_margin,
        invp_margin=invp_margin,
    )
    gate_path = output_dir / f"{run_id}_integral_inverse_feature_gate.json"
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
        "integral_inverse_feature_gate": str(gate_path),
        "validation_status": validation_status,
        "integral_inverse_feature_status": gate_report["status"],
        "decision": gate_report["decision"],
        "action": gate_report["action"],
        "interpretation": gate_report["interpretation"],
        "raw_anchor": gate_report["raw_anchor"],
        "invp_anchor": gate_report["invp_anchor"],
        "sinv_candidate": gate_report["sinv_candidate"],
        "best": gate_report["best"],
        "delta_sinv_vs_raw_auc": gate_report["delta_sinv_vs_raw_auc"],
        "delta_sinv_vs_invp_auc": gate_report["delta_sinv_vs_invp_auc"],
        "raw_margin": gate_report["raw_margin"],
        "invp_margin": gate_report["invp_margin"],
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
            update_plan_doc_with_integral_inverse_feature_postprocess(path, report)
        report["plan_docs"] = [str(path) for path in update_paths]
        report["plan_doc"] = str(update_paths[0])

    _write_json(summary_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def gate_integral_inverse_feature_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    raw_margin: float = DEFAULT_RAW_MARGIN,
    invp_margin: float = DEFAULT_INVP_MARGIN,
) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected_rows={expected_rows} actual_rows={len(rows)}")
    entries = [_entry(row) for row in rows]
    raw_anchor = next((entry for entry in entries if entry["architecture_rank"] == 0), None)
    invp_anchor = next((entry for entry in entries if entry["feature_encoding"] == "present_pair_xor_paligned_cell_matrix_bits"), None)
    sinv_candidate = next(
        (entry for entry in entries if entry["feature_encoding"] == "present_pair_xor_paligned_sinv_cell_matrix_bits"),
        None,
    )
    if raw_anchor is None:
        errors.append("missing architecture_rank=0 raw anchor row")
    if invp_anchor is None:
        errors.append("missing InvP matrix candidate row")
    if sinv_candidate is None:
        errors.append("missing InvP+Sinv matrix candidate row")
    for entry in entries:
        if not isinstance(entry.get("auc"), (int, float)):
            errors.append(f"missing auc for {entry['id']}")

    ranking = sorted(entries, key=lambda item: float(item.get("auc") or -1.0), reverse=True)
    best = ranking[0] if ranking else None
    delta_sinv_vs_raw = _auc_delta(sinv_candidate, raw_anchor)
    delta_sinv_vs_invp = _auc_delta(sinv_candidate, invp_anchor)

    if errors:
        status = "fail"
        decision = "invalid_integral_inverse_feature_result"
        action = "repair_result_or_plan_alignment_before_branching"
        interpretation = "Result rows are incomplete or missing required metrics."
    elif (delta_sinv_vs_raw or 0.0) >= raw_margin and (delta_sinv_vs_invp or 0.0) >= invp_margin:
        status = "pass"
        decision = "promote_sinv_inverse_feature_to_262k_confirmation"
        action = "prepare_262k_confirmation_for_invp_sinv_feature_when_gpu_free"
        interpretation = "The InvP+Sinv previous-round feature beats both raw and InvP-only anchors by the screen margins."
    elif best is not None and best is not raw_anchor and isinstance(best.get("auc"), (int, float)):
        raw_auc = raw_anchor.get("auc") if raw_anchor else None
        best_delta = float(best["auc"]) - float(raw_auc) if isinstance(raw_auc, (int, float)) else None
        if best_delta is not None and best_delta > 0:
            status = "pass"
            decision = "weak_inverse_feature_candidate_wait_for_active_gates"
            action = "do_not_expand_until_r8_1m_or_r9_gate_selects_new_data_representation"
            interpretation = "An inverse-round feature candidate is positive but below the configured promotion margins."
        else:
            status = "pass"
            decision = "stop_integral_inverse_feature_screen_for_now"
            action = "return_to_r9_weak_probe_curriculum_or_difference_screen"
            interpretation = "No inverse-round candidate beats the raw integral-nibble anchor."
    else:
        status = "pass"
        decision = "stop_integral_inverse_feature_screen_for_now"
        action = "return_to_r9_weak_probe_curriculum_or_difference_screen"
        interpretation = "No inverse-round candidate beats the raw integral-nibble anchor."

    return {
        "status": status,
        "results": str(results_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "errors": errors,
        "raw_anchor": raw_anchor,
        "invp_anchor": invp_anchor,
        "sinv_candidate": sinv_candidate,
        "best": best,
        "delta_sinv_vs_raw_auc": delta_sinv_vs_raw,
        "delta_sinv_vs_invp_auc": delta_sinv_vs_invp,
        "raw_margin": raw_margin,
        "invp_margin": invp_margin,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "claim_scope": (
            "PRESENT r8 integral/inverse-round feature screen only; 65536/class is not "
            "Zhang/Wang same-protocol model evidence, paper-scale evidence, or breakthrough evidence"
        ),
    }


def update_plan_doc_with_integral_inverse_feature_postprocess(plan_doc_path: Path, report: dict[str, Any]) -> None:
    text = plan_doc_path.read_text(encoding="utf-8")
    header = "## Retrieved Integral / Inverse-Feature Result"
    if header not in text:
        text = text.rstrip() + "\n\n" + header + "\n\n"
    start = f"<!-- integral-inverse-feature-postprocess:{report['run_id']}:start -->"
    end = f"<!-- integral-inverse-feature-postprocess:{report['run_id']}:end -->"
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
    if decision == "promote_sinv_inverse_feature_to_262k_confirmation":
        return {
            "branch": "integral_inverse_feature_262k_confirmation",
            "should_launch_remote": False,
            "requires_implementation": True,
            "reason": decision,
            "selected_feature_encoding": "present_pair_xor_paligned_sinv_cell_matrix_bits",
            "next_plan_doc": "docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md",
        }
    if decision == "weak_inverse_feature_candidate_wait_for_active_gates":
        return {
            "branch": "integral_inverse_feature_hold_for_active_gates",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
        }
    if decision == "stop_integral_inverse_feature_screen_for_now":
        return {
            "branch": "stop_integral_inverse_feature_for_now",
            "should_launch_remote": False,
            "requires_implementation": False,
            "reason": decision,
            "fallback_hypotheses": ["r9_weak_probe", "r9_curriculum", "r9_difference_screen"],
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
        return ["Inspect validation and gate errors before interpreting inverse-feature metrics."]
    if branch == "integral_inverse_feature_262k_confirmation":
        return [
            "Record this as a 65536/class high-round data-representation screen signal only.",
            "Prepare a lean 262144/class confirmation for raw anchor versus InvP+Sinv.",
            "Keep sample_structure, negative_mode, validation key, and metric fixed.",
        ]
    if branch == "integral_inverse_feature_hold_for_active_gates":
        return [
            "Treat the best inverse-round representation as a weak screen candidate.",
            "Wait for r8 1M and r9 weak-probe gates before confirmation.",
        ]
    if branch == "stop_integral_inverse_feature_for_now":
        return ["Stop this integral/inverse feature screen unless later high-round evidence reopens it."]
    return ["Manual review required before branching."]


def _plan_doc_result_section(report: dict[str, Any]) -> str:
    raw_anchor = report.get("raw_anchor") or {}
    invp_anchor = report.get("invp_anchor") or {}
    sinv_candidate = report.get("sinv_candidate") or {}
    best = report.get("best") or {}
    rows = [
        ("Run ID", report["run_id"]),
        ("Postprocess status", report["status"]),
        ("Validation status", report["validation_status"]),
        ("Raw anchor AUC", _format_value(raw_anchor.get("auc"))),
        ("InvP matrix AUC", _format_value(invp_anchor.get("auc"))),
        ("InvP+Sinv AUC", _format_value(sinv_candidate.get("auc"))),
        ("Best row", best.get("id", "")),
        ("Best feature", best.get("feature_encoding", "")),
        ("Delta Sinv vs raw AUC", _format_value(report["delta_sinv_vs_raw_auc"])),
        ("Delta Sinv vs InvP AUC", _format_value(report["delta_sinv_vs_invp_auc"])),
        ("Decision", report["decision"]),
        ("Action", report["action"]),
        ("Next action branch", report["next_action"]["branch"]),
        ("Claim scope", report["claim_scope"]),
        ("Results JSONL", report["results"]),
        ("Validation report", report["validation_report"]),
        ("Integral/inverse gate", report["integral_inverse_feature_gate"]),
        ("Summary JSON", report["summary"]),
        ("Summary Markdown", report["summary_markdown"]),
    ]
    lines = [f"### {report['run_id']} Integral / Inverse-Feature Result", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {field} | `{value}` |" for field, value in rows)
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run_id']} Integral / Inverse-Feature Postprocess",
        "",
        _plan_doc_result_section(report),
        "",
        "## Ranking",
        "",
        "| Rank | Row | Feature | AUC | Calibrated accuracy | Accuracy | Loss |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(report["ranking"], start=1):
        lines.append(
            "| "
            f"{index} | `{entry['id']}` | `{entry['feature_encoding']}` | "
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
    feature_encoding = str(row.get("feature_encoding") or (row.get("training") or {}).get("feature_encoding") or "")
    architecture = str(row.get("architecture") or row.get("family") or row.get("selected_model") or "")
    architecture_rank = int(row.get("architecture_rank", -1))
    return {
        "id": architecture,
        "architecture": architecture,
        "architecture_rank": architecture_rank,
        "model": row.get("selected_model") or row.get("model"),
        "feature_encoding": feature_encoding,
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "calibrated_accuracy": metrics.get("calibrated_accuracy"),
        "loss": metrics.get("loss"),
    }


def _auc_delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float | None:
    if left is None or right is None:
        return None
    left_auc = left.get("auc")
    right_auc = right.get("auc")
    if not isinstance(left_auc, (int, float)) or not isinstance(right_auc, (int, float)):
        return None
    return float(left_auc) - float(right_auc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_integral_inverse_feature_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        raw_margin=args.raw_margin,
        invp_margin=args.invp_margin,
        plan_doc_paths=args.update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
