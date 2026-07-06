from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_MODEL = "present_pairset_global_stats"
DEFAULT_CANDIDATE_MODEL = "present_pairset_histogram_hybrid"
DEFAULT_MIN_MEAN_MARGIN = 0.01
DEFAULT_MIN_CANDIDATE_AUC = 0.55


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate PRESENT r8 cell-value histogram local diagnostic results."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--min-mean-margin", type=float, default=DEFAULT_MIN_MEAN_MARGIN)
    parser.add_argument("--min-candidate-auc", type=float, default=DEFAULT_MIN_CANDIDATE_AUC)
    return parser.parse_args(argv)


def gate_cell_value_histogram_result(
    results_path: Path,
    *,
    baseline_model: str = DEFAULT_BASELINE_MODEL,
    candidate_model: str = DEFAULT_CANDIDATE_MODEL,
    min_mean_margin: float = DEFAULT_MIN_MEAN_MARGIN,
    min_candidate_auc: float = DEFAULT_MIN_CANDIDATE_AUC,
) -> dict[str, Any]:
    rows = _load_jsonl_rows(results_path)
    by_seed = _metrics_by_seed(rows)
    errors: list[str] = []
    incomplete_reasons: list[str] = []
    per_seed: list[dict[str, Any]] = []

    for seed in sorted(by_seed):
        seed_rows = by_seed[seed]
        baseline_auc = _metric(seed_rows.get(baseline_model), "auc")
        candidate_auc = _metric(seed_rows.get(candidate_model), "auc")
        if baseline_auc is None:
            incomplete_reasons.append(f"seed={seed} missing_baseline_model={baseline_model}")
        if candidate_auc is None:
            incomplete_reasons.append(f"seed={seed} missing_candidate_model={candidate_model}")
        per_seed.append(
            {
                "seed": seed,
                "baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "candidate_margin_vs_baseline_auc": _delta(candidate_auc, baseline_auc),
                "candidate_clears_baseline": _gt(candidate_auc, baseline_auc),
                "candidate_clears_min_auc": (
                    candidate_auc is not None and candidate_auc >= min_candidate_auc
                ),
            }
        )

    if not per_seed:
        incomplete_reasons.append("no_seed_results")

    baseline_aucs = [row["baseline_auc"] for row in per_seed if row["baseline_auc"] is not None]
    candidate_aucs = [row["candidate_auc"] for row in per_seed if row["candidate_auc"] is not None]
    mean_baseline_auc = _mean_or_none(baseline_aucs)
    mean_candidate_auc = _mean_or_none(candidate_aucs)
    mean_margin = _delta(mean_candidate_auc, mean_baseline_auc)
    min_candidate_auc_observed = min(candidate_aucs) if candidate_aucs else None

    decision = "invalid_cell_value_histogram_screen"
    action = "fix_cell_value_histogram_gate_inputs_before_claim"
    interpretation = "cell-value histogram screen cannot be evaluated"
    completion_status = "complete"
    if incomplete_reasons:
        completion_status = "incomplete"
        decision = "pending_cell_value_histogram_screen"
        action = "wait_for_all_planned_seed_pairs_before_claim"
        interpretation = (
            "cell-value histogram screen is still incomplete; do not treat this as "
            "a route failure or promotion signal"
        )
    elif not errors:
        candidate_beats_baseline_all = all(row["candidate_clears_baseline"] for row in per_seed)
        candidate_clears_auc_all = all(row["candidate_clears_min_auc"] for row in per_seed)
        mean_margin_clears = mean_margin is not None and mean_margin >= min_mean_margin
        if candidate_beats_baseline_all and candidate_clears_auc_all and mean_margin_clears:
            decision = "support_cell_value_histogram_local_weak_positive"
            action = "export_frozen_scores_and_test_diversity_only_after_controlled_expert_gate"
            interpretation = (
                "cell-value histogram candidate beats the same-input global-statistics control "
                "on every checked seed and clears the local weak-positive screen"
            )
        else:
            decision = "hold_cell_value_histogram_local_screen"
            action = "do_not_promote_to_diverse_expert_pool"
            interpretation = (
                "cell-value histogram candidate does not clear the same-input global-statistics "
                "control and weak-positive margins on every checked seed"
            )

    return {
        "status": "pass" if not errors and not incomplete_reasons else "fail",
        "completion_status": completion_status,
        "results_path": str(results_path),
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "required_min_mean_margin": min_mean_margin,
        "required_min_candidate_auc": min_candidate_auc,
        "result_rows": len(rows),
        "seeds": [row["seed"] for row in per_seed],
        "per_seed": per_seed,
        "mean_baseline_auc": mean_baseline_auc,
        "mean_candidate_auc": mean_candidate_auc,
        "mean_candidate_margin_vs_baseline_auc": mean_margin,
        "min_candidate_auc": min_candidate_auc_observed,
        "decision": decision,
        "action": action,
        "interpretation": interpretation,
        "claim_scope": (
            "PRESENT r8 cell-value histogram local diagnostic only; not remote evidence, "
            "not formal SPN/PRESENT evidence, not a breakthrough claim, and not diverse "
            "expert pool evidence by itself"
        ),
        "errors": errors,
        "incomplete_reasons": incomplete_reasons,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metrics_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, float | None]]]:
    by_seed: dict[int, dict[str, dict[str, float | None]]] = {}
    for row in rows:
        seed = _int_or_none(row.get("seed"))
        if seed is None:
            continue
        by_seed.setdefault(seed, {})[_model_key(row)] = _metrics(row)
    return by_seed


def _model_key(row: dict[str, Any]) -> str:
    for key in ("model", "selected_model", "model_key", "route"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    metrics = row["metrics"] if isinstance(row.get("metrics"), dict) else row
    return {
        "auc": _optional_float(metrics.get("auc") if "auc" in metrics else metrics.get("val_auc")),
        "accuracy": _optional_float(
            metrics.get("accuracy") if "accuracy" in metrics else metrics.get("val_accuracy")
        ),
        "loss": _optional_float(metrics.get("loss") if "loss" in metrics else metrics.get("val_loss")),
    }


def _metric(metrics: dict[str, float | None] | None, key: str) -> float | None:
    if metrics is None:
        return None
    return metrics.get(key)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _gt(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = gate_cell_value_histogram_result(
        args.results,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        min_mean_margin=args.min_mean_margin,
        min_candidate_auc=args.min_candidate_auc,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["status"] == "pass" else 1


__all__ = ["gate_cell_value_histogram_result", "main"]
