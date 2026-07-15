from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


OBSERVED_RATE_COLUMN = "observed_q1_rate_256key"
SOURCE_DECISIONS = {
    "innovation2_integral_rate_target_unstable",
    "innovation2_integral_calibration_insufficient",
    "innovation2_integral_calibration_advance_seed1_geometry",
}
RANKING_PASS_DECISIONS = {
    "innovation2_integral_ranking_utility_advance_independent_confirmation",
    "innovation2_integral_ranking_utility_independent_confirmation_passed",
}


@dataclass(frozen=True)
class RankingModel:
    role: str
    model: str
    score_column: str


MODELS = (
    RankingModel(
        role="anchor",
        model="linear_same_input",
        score_column="linear_same_input_calibrated_predicted_q1_rate",
    ),
    RankingModel(
        role="candidate",
        model="structure_mlp",
        score_column="structure_mlp_calibrated_predicted_q1_rate",
    ),
    RankingModel(
        role="control",
        model="structure_mlp_shuffled_labels",
        score_column=(
            "structure_mlp_shuffled_labels_calibrated_predicted_q1_rate"
        ),
    ),
)


@dataclass(frozen=True)
class IntegralRankingThresholds:
    minimum_candidate_linear_spearman_margin: float = 0.05
    minimum_candidate_global_topk_balance_advantage: float = 0.05
    minimum_candidate_linear_topk_balance_advantage: float = 0.03
    maximum_control_global_topk_balance_advantage: float = 0.02


def evaluate_integral_ranking(
    *,
    run_id: str,
    source_rows: list[dict[str, str]],
    source_gate: dict[str, Any],
    top_k: int = 16,
    thresholds: IntegralRankingThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or IntegralRankingThresholds()
    _validate_source(source_rows, source_gate, top_k=top_k)

    structure_ids = np.asarray(
        [str(row["structure_id"]) for row in source_rows], dtype=str
    )
    observed_q1_rates = _float_column(source_rows, OBSERVED_RATE_COLUMN)
    source_seed = _source_seed(str(source_gate["run_id"]))
    global_q1_rate = float(observed_q1_rates.mean())
    global_balance_rate = 1.0 - global_q1_rate

    model_rows: list[dict[str, Any]] = []
    model_details: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        predicted_q1_rates = _float_column(source_rows, model.score_column)
        selected_indices = _lowest_score_indices(
            predicted_q1_rates,
            structure_ids,
            top_k=top_k,
        )
        selected_q1_rate = float(observed_q1_rates[selected_indices].mean())
        selected_balance_rate = 1.0 - selected_q1_rate
        row = {
            "run_id": run_id,
            "task": "innovation2_integral_property_ranking",
            "cipher": "PRESENT-80",
            "rounds": 5,
            "seed": source_seed,
            "role": model.role,
            "model": model.model,
            "score_column": model.score_column,
            "structure_count": len(source_rows),
            "stability_keys_per_structure": 256,
            "top_k": top_k,
            "spearman_stable_q1_rate": spearman_correlation(
                predicted_q1_rates,
                observed_q1_rates,
            ),
            "global_observed_q1_rate": global_q1_rate,
            "global_observed_balance_rate": global_balance_rate,
            "topk_observed_q1_rate": selected_q1_rate,
            "topk_observed_balance_rate": selected_balance_rate,
            "topk_balance_advantage_vs_global": (
                selected_balance_rate - global_balance_rate
            ),
            "selected_structure_ids": structure_ids[selected_indices].tolist(),
            "source_run_id": str(source_gate["run_id"]),
            "source_decision": str(source_gate["decision"]),
            "claim_scope": (
                "read-only E2 utility adjudication on the frozen PRESENT-r5 E1 "
                "128-structure, 256-key stability set; not retraining, not an "
                "independent replication, and not a deterministic integral proof"
            ),
        }
        model_rows.append(row)
        model_details[model.role] = {
            "spec": model,
            "predicted_q1_rates": predicted_q1_rates,
            "selected_indices": selected_indices,
            "row": row,
        }

    gate = adjudicate_integral_ranking(
        run_id=run_id,
        model_rows=model_rows,
        source_gate=source_gate,
        thresholds=thresholds,
    )
    ranking_rows = _ranking_rows(
        source_rows=source_rows,
        observed_q1_rates=observed_q1_rates,
        model_details=model_details,
    )
    return {
        "rows": model_rows,
        "ranking_rows": ranking_rows,
        "gate": gate,
    }


def adjudicate_integral_ranking(
    *,
    run_id: str,
    model_rows: list[dict[str, Any]],
    source_gate: dict[str, Any],
    thresholds: IntegralRankingThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or IntegralRankingThresholds()
    by_role = {str(row["role"]): row for row in model_rows}
    required_roles = {"anchor", "candidate", "control"}
    if set(by_role) != required_roles:
        raise ValueError("ranking adjudication requires anchor, candidate, and control")

    linear = by_role["anchor"]
    candidate = by_role["candidate"]
    control = by_role["control"]
    candidate_linear_spearman_margin = float(
        candidate["spearman_stable_q1_rate"]
    ) - float(linear["spearman_stable_q1_rate"])
    candidate_global_topk_advantage = float(
        candidate["topk_balance_advantage_vs_global"]
    )
    candidate_linear_topk_advantage = float(
        candidate["topk_observed_balance_rate"]
    ) - float(linear["topk_observed_balance_rate"])
    control_global_topk_advantage = float(
        control["topk_balance_advantage_vs_global"]
    )
    checks = {
        "candidate_linear_spearman_margin_at_least_0_05": (
            candidate_linear_spearman_margin
            >= thresholds.minimum_candidate_linear_spearman_margin
        ),
        "candidate_global_top16_balance_advantage_at_least_0_05": (
            candidate_global_topk_advantage
            >= thresholds.minimum_candidate_global_topk_balance_advantage
        ),
        "candidate_linear_top16_balance_advantage_at_least_0_03": (
            candidate_linear_topk_advantage
            >= thresholds.minimum_candidate_linear_topk_balance_advantage
        ),
        "control_global_top16_balance_advantage_at_most_0_02": (
            control_global_topk_advantage
            <= thresholds.maximum_control_global_topk_balance_advantage
        ),
    }
    if all(checks.values()):
        status = "pass"
        if int(candidate["seed"]) == 0:
            decision = (
                "innovation2_integral_ranking_utility_advance_independent_confirmation"
            )
            next_action = (
                "Run a frozen local E3 seed1 confirmation with the same random-split "
                "protocol, E1 budgets, linear/shuffled controls, 256-key stability "
                "labels, and top-16 gate. Change only the seed; defer held-out geometry "
                "to the next phase and do not launch remote GPU work."
            )
        else:
            decision = (
                "innovation2_integral_ranking_utility_independent_confirmation_passed"
            )
            next_action = (
                "Jointly adjudicate the frozen seed0 and seed1 E2 gates. If both "
                "remain valid, freeze one local active/output/mask geometry-holdout "
                "experiment with the same budgets and controls; do not launch remote "
                "GPU work."
            )
    elif not checks[
        "control_global_top16_balance_advantage_at_most_0_02"
    ]:
        status = "hold"
        decision = "innovation2_integral_ranking_control_not_attributed"
        next_action = (
            "Do not claim top-k utility from this split. Freeze a new local "
            "seed1/geometry-held-out confirmation with the same three models and "
            "budgets; require the shuffled top-16 advantage to return to at most "
            "+0.02 before interpreting candidate selection."
        )
    elif checks["candidate_linear_spearman_margin_at_least_0_05"]:
        status = "hold"
        decision = "innovation2_integral_ranking_explanatory_only"
        next_action = (
            "Retain the correlation as explanatory evidence only. Add one "
            "PRESENT P-layer reachability feature group to the same 111-bit "
            "representation and rerun the local E0-E2 matrix without increasing "
            "structures, epochs, seeds, or remote budget."
        )
    else:
        status = "hold"
        decision = "innovation2_integral_ranking_redesign_representation"
        next_action = (
            "Stop the current 111-bit representation. Add only P-layer "
            "reachability features, keep the existing E0/E1 budgets and controls, "
            "and require the E2 correlation and top-16 gates before any scale-up."
        )

    return {
        "status": status,
        "decision": decision,
        "run_id": run_id,
        "gate_mode": "ranking-utility",
        "source_run_id": str(source_gate["run_id"]),
        "source_decision": str(source_gate["decision"]),
        "training_performed": False,
        "checks": checks,
        "thresholds": {
            "minimum_candidate_linear_spearman_margin": (
                thresholds.minimum_candidate_linear_spearman_margin
            ),
            "minimum_candidate_global_topk_balance_advantage": (
                thresholds.minimum_candidate_global_topk_balance_advantage
            ),
            "minimum_candidate_linear_topk_balance_advantage": (
                thresholds.minimum_candidate_linear_topk_balance_advantage
            ),
            "maximum_control_global_topk_balance_advantage": (
                thresholds.maximum_control_global_topk_balance_advantage
            ),
        },
        "metrics": {
            "candidate_spearman": float(candidate["spearman_stable_q1_rate"]),
            "linear_spearman": float(linear["spearman_stable_q1_rate"]),
            "control_spearman": float(control["spearman_stable_q1_rate"]),
            "candidate_linear_spearman_margin": (
                candidate_linear_spearman_margin
            ),
            "global_observed_balance_rate": float(
                candidate["global_observed_balance_rate"]
            ),
            "candidate_top16_observed_balance_rate": float(
                candidate["topk_observed_balance_rate"]
            ),
            "linear_top16_observed_balance_rate": float(
                linear["topk_observed_balance_rate"]
            ),
            "control_top16_observed_balance_rate": float(
                control["topk_observed_balance_rate"]
            ),
            "candidate_global_top16_balance_advantage": (
                candidate_global_topk_advantage
            ),
            "candidate_linear_top16_balance_advantage": (
                candidate_linear_topk_advantage
            ),
            "control_global_top16_balance_advantage": (
                control_global_topk_advantage
            ),
        },
        "next_action": next_action,
        "claim_scope": (
            "read-only E2 utility adjudication on the frozen PRESENT-r5 E1 "
            "128-structure, 256-key stability set; not retraining, not an "
            "independent replication, and not a deterministic integral proof"
        ),
    }


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = np.asarray(tuple(left), dtype=np.float64)
    right_values = np.asarray(tuple(right), dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if len(left_values) < 2:
        raise ValueError("Spearman correlation requires at least two values")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("Spearman inputs must be finite")
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return float(np.sum(left_centered * right_centered) / denominator)


def adjudicate_joint_integral_ranking(
    *,
    run_id: str,
    source_gates: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(source_gates) != 2:
        raise ValueError("joint ranking adjudication requires exactly two gates")
    rows: list[dict[str, Any]] = []
    threshold_payloads: list[dict[str, Any]] = []
    for gate in source_gates:
        source_run_id = str(gate.get("run_id", ""))
        seed = _source_seed(source_run_id)
        checks = gate.get("checks")
        metrics = gate.get("metrics")
        thresholds = gate.get("thresholds")
        if not isinstance(checks, dict) or not isinstance(metrics, dict):
            raise ValueError("source ranking gates must contain checks and metrics")
        if not isinstance(thresholds, dict):
            raise ValueError("source ranking gates must contain frozen thresholds")
        threshold_payloads.append(thresholds)
        rows.append(
            {
                "run_id": run_id,
                "task": "innovation2_integral_property_ranking_joint",
                "cipher": "PRESENT-80",
                "rounds": 5,
                "seed": seed,
                "source_run_id": source_run_id,
                "source_status": str(gate.get("status", "")),
                "source_decision": str(gate.get("decision", "")),
                "all_source_checks_passed": all(bool(value) for value in checks.values()),
                **{key: float(value) for key, value in metrics.items()},
            }
        )
    seeds = {int(row["seed"]) for row in rows}
    thresholds_match = threshold_payloads[0] == threshold_payloads[1]
    all_sources_pass = all(
        row["source_status"] == "pass"
        and row["source_decision"] in RANKING_PASS_DECISIONS
        and row["all_source_checks_passed"]
        for row in rows
    )
    checks = {
        "exact_seed0_seed1_pair": seeds == {0, 1},
        "frozen_thresholds_match": thresholds_match,
        "both_seed_gates_pass": all_sources_pass,
    }
    metric_names = (
        "candidate_spearman",
        "linear_spearman",
        "control_spearman",
        "candidate_linear_spearman_margin",
        "candidate_top16_observed_balance_rate",
        "linear_top16_observed_balance_rate",
        "control_top16_observed_balance_rate",
        "candidate_global_top16_balance_advantage",
        "candidate_linear_top16_balance_advantage",
        "control_global_top16_balance_advantage",
    )
    metrics = {
        f"{name}_{suffix}": statistic
        for name in metric_names
        for suffix, statistic in (
            ("min", min(float(row[name]) for row in rows)),
            ("mean", float(np.mean([float(row[name]) for row in rows]))),
            ("max", max(float(row[name]) for row in rows)),
        )
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_integral_ranking_utility_two_seed_confirmed"
        next_action = (
            "Freeze one local geometry-holdout experiment that withholds unseen "
            "active/output/mask combinations while keeping the E1 budgets, network, "
            "linear/shuffled controls, 256-key labels, and E2 gates unchanged. Do "
            "not launch remote GPU work before the geometry gate passes."
        )
    else:
        status = "hold"
        decision = "innovation2_integral_ranking_utility_two_seed_not_confirmed"
        next_action = (
            "Do not advance to geometry holdout or remote scale. Audit the missing "
            "seed, threshold mismatch, or failed per-seed control before any new run."
        )
    gate = {
        "status": status,
        "decision": decision,
        "run_id": run_id,
        "gate_mode": "ranking-utility-joint",
        "source_run_ids": [row["source_run_id"] for row in sorted(rows, key=lambda row: row["seed"])],
        "checks": checks,
        "thresholds": threshold_payloads[0] if thresholds_match else {},
        "metrics": metrics,
        "training_performed": False,
        "next_action": next_action,
        "claim_scope": (
            "two-seed local PRESENT-r5 ranking/top-16 confirmation over two "
            "independent 128-structure, 256-key stability sets; not geometry-held-out, "
            "not remote, not paper-scale, and not a deterministic integral proof"
        ),
    }
    return {"rows": sorted(rows, key=lambda row: row["seed"]), "gate": gate}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _validate_source(
    source_rows: list[dict[str, str]],
    source_gate: dict[str, Any],
    *,
    top_k: int,
) -> None:
    if str(source_gate.get("status")) not in {"hold", "pass"}:
        raise ValueError("ranking evaluation requires a valid E1 calibration result")
    if str(source_gate.get("decision")) not in SOURCE_DECISIONS:
        raise ValueError("ranking evaluation requires a recognized E1 decision")
    if not source_gate.get("run_id"):
        raise ValueError("source gate must identify its run_id")
    if len(source_rows) != 128:
        raise ValueError("E2 requires the frozen 128 E1 test structures")
    if top_k != 16:
        raise ValueError("the frozen E2 protocol requires top_k=16")
    required_columns = {
        "structure_id",
        "signature",
        OBSERVED_RATE_COLUMN,
        *(model.score_column for model in MODELS),
    }
    missing = required_columns - set(source_rows[0]) if source_rows else required_columns
    if missing:
        raise ValueError(f"source structure rates are missing columns: {sorted(missing)}")
    structure_ids = [str(row["structure_id"]) for row in source_rows]
    if len(set(structure_ids)) != len(structure_ids):
        raise ValueError("source structure ids must be unique")
    for column in (OBSERVED_RATE_COLUMN, *(model.score_column for model in MODELS)):
        values = _float_column(source_rows, column)
        if not np.all((0.0 <= values) & (values <= 1.0)):
            raise ValueError(f"{column} must contain probabilities in [0, 1]")


def _float_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric column: {column}") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{column} must contain finite values")
    return values


def _lowest_score_indices(
    scores: np.ndarray,
    structure_ids: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    order = np.lexsort((structure_ids, scores))
    return order[:top_k]


def _source_seed(run_id: str) -> int:
    match = re.search(r"_seed(?P<seed>\d+)$", run_id)
    if match is None:
        raise ValueError("source E1 run_id must end with _seedN")
    return int(match.group("seed"))


def _ranking_rows(
    *,
    source_rows: list[dict[str, str]],
    observed_q1_rates: np.ndarray,
    model_details: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranks_by_role: dict[str, np.ndarray] = {}
    selected_by_role: dict[str, set[int]] = {}
    for role, details in model_details.items():
        scores = details["predicted_q1_rates"]
        ranks_by_role[role] = _average_ranks(scores)
        selected_by_role[role] = set(details["selected_indices"].tolist())

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        row: dict[str, Any] = {
            "structure_id": source["structure_id"],
            "signature": source["signature"],
            "active_nibble": int(source["active_nibble"]),
            "output_nibble": int(source["output_nibble"]),
            "output_mask": source["output_mask"],
            "observed_q1_rate_256key": float(observed_q1_rates[index]),
            "observed_balance_rate_256key": float(1.0 - observed_q1_rates[index]),
        }
        for role, details in model_details.items():
            model = details["spec"]
            row[f"{role}_model"] = model.model
            row[f"{role}_predicted_q1_rate"] = float(
                details["predicted_q1_rates"][index]
            )
            row[f"{role}_rank"] = float(ranks_by_role[role][index])
            row[f"{role}_selected_top16"] = index in selected_by_role[role]
        rows.append(row)
    return rows
