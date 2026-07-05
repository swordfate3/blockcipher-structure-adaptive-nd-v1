from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DIFFERENCE = "present_zhang_wang2022_mcnd:0"
DEFAULT_PROMOTION_MARGIN = 0.01
DEFAULT_RANDOM_AUC_CEILING = 0.505


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate a PRESENT input-difference screen.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--reference-difference", default=DEFAULT_REFERENCE_DIFFERENCE)
    parser.add_argument("--promotion-margin", type=float, default=DEFAULT_PROMOTION_MARGIN)
    parser.add_argument("--random-auc-ceiling", type=float, default=DEFAULT_RANDOM_AUC_CEILING)
    return parser.parse_args(argv)


def gate_difference_screen_result(
    results_path: Path,
    *,
    expected_rows: int | None = None,
    reference_difference: str = DEFAULT_REFERENCE_DIFFERENCE,
    promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
    random_auc_ceiling: float = DEFAULT_RANDOM_AUC_CEILING,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    errors: list[str] = []
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"result_rows={len(rows)} expected_rows={expected_rows}")

    ranking = [_difference_entry(row) for row in rows]
    for entry in ranking:
        if entry["auc"] is None:
            errors.append(f"missing_auc={entry['difference_id']}")
    duplicate_ids = _duplicates(entry["difference_id"] for entry in ranking)
    if duplicate_ids:
        errors.append(f"duplicate_difference_ids={','.join(duplicate_ids)}")

    ranking.sort(key=lambda item: item["auc"] if item["auc"] is not None else -1.0, reverse=True)
    reference = next((item for item in ranking if item["difference_id"] == reference_difference), None)
    best = ranking[0] if ranking else None
    if reference is None:
        errors.append(f"missing_reference_difference={reference_difference}")

    best_auc = best["auc"] if best else None
    reference_auc = reference["auc"] if reference else None
    delta_vs_reference = _delta(best_auc, reference_auc)
    decision = "invalid"
    action = "fix_difference_screen_gate_inputs_before_claim"
    interpretation = "difference screen gate cannot be evaluated"
    if not errors and best is not None and best_auc is not None:
        if best["difference_id"] == reference_difference:
            decision = "keep_current_difference_no_screen_winner"
            action = "do_not_prioritize_difference_search"
            interpretation = (
                "the Zhang/Wang reference input difference remains the best row in this screen; "
                "continue model/curriculum/aggregation routes before expanding difference search"
            )
        elif best_auc <= random_auc_ceiling:
            decision = "all_candidates_near_random_stop_difference_screen"
            action = "stop_this_difference_screen_and_return_to_curriculum_or_aggregation"
            interpretation = (
                "the best input-difference candidate is still near random at this screen scale; "
                "do not promote a high-round difference from this run"
            )
        elif delta_vs_reference is not None and delta_vs_reference >= promotion_margin:
            decision = "promote_best_difference_to_262k_confirmation"
            action = "prepare_262k_confirmation_for_best_difference"
            interpretation = (
                "a non-reference input difference beats the Zhang/Wang reference by the required screen margin; "
                "promote only that candidate to medium confirmation"
            )
        elif delta_vs_reference is not None and delta_vs_reference > 0.0:
            decision = "weak_difference_candidate_repeat_or_confirm_at_262k"
            action = "repeat_or_confirm_best_difference_before_scaling"
            interpretation = (
                "a non-reference input difference is best but below the promotion margin; "
                "treat as a weak screen signal, not route evidence"
            )
        else:
            decision = "manual_review_difference_screen"
            action = "inspect_difference_screen_metrics_before_branching"
            interpretation = "screen ranking did not match an automatic branch rule"

    return {
        "status": "pass" if not errors else "fail",
        "results_path": str(results_path),
        "result_rows": len(rows),
        "expected_rows": expected_rows,
        "reference_difference": reference_difference,
        "reference": reference,
        "best": best,
        "delta_vs_reference_auc": delta_vs_reference,
        "promotion_margin": promotion_margin,
        "random_auc_ceiling": random_auc_ceiling,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "ranking": ranking,
        "claim_scope": (
            "PRESENT input-difference/data-construction screen; not same-protocol model-improvement "
            "evidence, not paper-scale, formal, or breakthrough evidence"
        ),
        "errors": errors,
        "warnings": [],
    }


def _difference_entry(row: dict[str, Any]) -> dict[str, Any]:
    profile = str(row.get("difference_profile") or "")
    member = str(row.get("difference_member") if row.get("difference_member") is not None else "")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    return {
        "difference_id": f"{profile}:{member}",
        "difference_profile": profile,
        "difference_member": member,
        "input_difference": row.get("input_difference"),
        "model": row.get("model") or row.get("selected_model") or row.get("model_key") or "",
        "auc": _optional_float(metrics.get("auc") if isinstance(metrics, dict) else None),
        "accuracy": _optional_float(metrics.get("accuracy") if isinstance(metrics, dict) else None),
        "calibrated_accuracy": _optional_float(
            metrics.get("calibrated_accuracy") if isinstance(metrics, dict) else None
        ),
        "loss": _optional_float(metrics.get("loss") if isinstance(metrics, dict) else None),
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = gate_difference_screen_result(
        args.results,
        expected_rows=args.expected_rows,
        reference_difference=args.reference_difference,
        promotion_margin=args.promotion_margin,
        random_auc_ceiling=args.random_auc_ceiling,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
