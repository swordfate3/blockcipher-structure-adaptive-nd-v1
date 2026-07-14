from __future__ import annotations

import math
from typing import Any


EXPECTED_CELLS = {
    2: ("e4_r4", 0),
    3: ("e4_r4", 0),
    4: ("e4_r5", 1),
    5: ("e4_r5", 1),
}

COMPARISON_SPECS = {
    "scratch_margin": {
        "bootstrap_role": "typed_scratch",
        "threshold": 0.004,
        "label": "true transfer - typed scratch",
    },
    "source_topology_margin": {
        "bootstrap_role": "shuffled_to_true",
        "threshold": 0.005,
        "label": "true source - shuffled source",
    },
    "target_topology_margin": {
        "bootstrap_role": "true_to_shuffled",
        "threshold": 0.003,
        "label": "true target - shuffled target",
    },
}


def build_cross_spn_e4_synthesis(reports: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(reports) != 4:
        errors.append(f"expected exactly 4 gate reports, received {len(reports)}")

    cells: list[dict[str, Any]] = []
    seen_target_seeds: set[int] = set()
    for index, report in enumerate(reports):
        cell, cell_errors = _validated_cell(report, index=index)
        errors.extend(cell_errors)
        if cell is None:
            continue
        target_seed = int(cell["target_seed"])
        if target_seed in seen_target_seeds:
            errors.append(f"duplicate target seed {target_seed}")
        seen_target_seeds.add(target_seed)
        cells.append(cell)

    expected_targets = set(EXPECTED_CELLS)
    if seen_target_seeds != expected_targets:
        errors.append(
            "target seed set mismatch: "
            f"expected={sorted(expected_targets)} actual={sorted(seen_target_seeds)}"
        )
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_e4_cross_spn_synthesis",
            "errors": errors,
            "next_action": "repair_input_selection_without_interpreting_metrics",
        }

    cells.sort(key=lambda item: int(item["target_seed"]))
    comparisons = {
        key: _comparison_summary(cells, key, float(spec["threshold"]))
        for key, spec in COMPARISON_SPECS.items()
    }
    source_strata = {
        str(source_seed): _source_stratum_summary(cells, source_seed)
        for source_seed in (0, 1)
    }
    between_stratum_shift = {
        key: (
            source_strata["1"]["mean_margins"][key]
            - source_strata["0"]["mean_margins"][key]
        )
        for key in COMPARISON_SPECS
    }

    scratch_robust = comparisons["scratch_margin"]["all_cells_pass"]
    topology_robust = (
        comparisons["source_topology_margin"]["all_cells_pass"]
        and comparisons["target_topology_margin"]["all_cells_pass"]
    )
    if topology_robust and not scratch_robust:
        decision = (
            "e4_typed_topology_attribution_robust_scratch_efficiency_conditional"
        )
        next_action = "freeze_e4_result_stop_transfer_scale_prepare_method_reporting"
    elif topology_robust:
        decision = "e4_typed_topology_and_scratch_efficiency_robust"
        next_action = "audit_contradiction_with_frozen_e4_r5_joint_decision"
    else:
        decision = "e4_typed_topology_attribution_inconsistent"
        next_action = "downgrade_e4_to_inconsistent_representation_evidence"

    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "experiment_stage": "e4_final_synthesis",
        "protocol": {
            "cipher": "GIFT-64",
            "rounds": 6,
            "samples_per_class": 65536,
            "train_samples_total_per_target_seed": 131072,
            "validation_samples_per_class": 32768,
            "validation_samples_total_per_target_seed": 65536,
            "target_epochs": 1,
            "target_seeds": [2, 3, 4, 5],
            "source_seeds": [0, 1],
            "negative_mode": "encrypted_random_plaintexts",
            "bootstrap_replicates_per_cell": 10000,
            "new_training": False,
        },
        "comparisons": comparisons,
        "source_strata": source_strata,
        "between_source_strata_shift": between_stratum_shift,
        "cells": cells,
        "inference_boundary": {
            "pooled_confidence_interval": False,
            "causal_source_seed_effect": False,
            "reason": (
                "target cells share source checkpoints within strata and source/target "
                "seeds are not fully crossed"
            ),
        },
        "claim_scope": (
            "four-cell remote medium diagnostic synthesis: robust typed source/target "
            "topology attribution and conditional scratch efficiency; not formal, "
            "paper-scale, SOTA, breakthrough, persistent, or causal source-seed evidence"
        ),
        "next_action": next_action,
        "stopped_actions": [
            "e4_r6",
            "mechanical_262144_class_scale",
            "formal_1000000_class_scale",
            "extra_target_epochs_or_seeds_to_rescue_e4_r4",
        ],
    }


def _validated_cell(
    report: dict[str, Any],
    *,
    index: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    prefix = f"report[{index}]"
    if not isinstance(report, dict):
        return None, [f"{prefix} must be a JSON object"]
    if report.get("status") != "pass":
        errors.append(f"{prefix} status must be pass")
    if report.get("errors") != []:
        errors.append(f"{prefix} errors must be empty")
    alignment = report.get("alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "pass":
        errors.append(f"{prefix} alignment must pass")
    elif alignment.get("errors") != []:
        errors.append(f"{prefix} alignment errors must be empty")
    if report.get("score_rows") != 65536:
        errors.append(f"{prefix} score_rows must be 65536")
    if report.get("score_pairing") != (
        "same validation cache, identical sample_ids and labels"
    ):
        errors.append(f"{prefix} score pairing is not the frozen paired cache")
    if report.get("research_decision_applied") is not True:
        errors.append(f"{prefix} must contain an applied research decision")

    try:
        target_seed = int(report["expected_seed"])
        source_seed = int(report["source_pretraining_cost"]["seed"])
        expected_stage, expected_source_seed = EXPECTED_CELLS[target_seed]
        stage = str(report["experiment_stage"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"{prefix} missing protocol identity: {exc}"]

    if source_seed != expected_source_seed:
        errors.append(
            f"target seed {target_seed} requires source seed {expected_source_seed}, "
            f"found {source_seed}"
        )
    if stage != expected_stage:
        errors.append(
            f"target seed {target_seed} requires stage {expected_stage}, found {stage}"
        )
    if report.get("samples_per_class") != 65536:
        errors.append(f"target seed {target_seed} samples_per_class must be 65536")
    if report.get("epochs") != 1:
        errors.append(f"target seed {target_seed} epochs must be 1")

    bootstrap = report.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return None, [*errors, f"target seed {target_seed} bootstrap must be an object"]
    if bootstrap.get("replicates") != 10000:
        errors.append(f"target seed {target_seed} bootstrap replicates must be 10000")
    if bootstrap.get("seed") != 20260715:
        errors.append(f"target seed {target_seed} bootstrap seed must be 20260715")
    if not _close(bootstrap.get("confidence"), 0.95):
        errors.append(f"target seed {target_seed} bootstrap confidence must be 0.95")
    if bootstrap.get("method") != (
        "paired label-stratified fixed-size nonparametric bootstrap"
    ):
        errors.append(f"target seed {target_seed} bootstrap method mismatch")

    comparisons: dict[str, dict[str, Any]] = {}
    for margin_key, spec in COMPARISON_SPECS.items():
        comparison, comparison_errors = _validated_comparison(
            report,
            margin_key=margin_key,
            bootstrap_role=str(spec["bootstrap_role"]),
            threshold=float(spec["threshold"]),
            target_seed=target_seed,
        )
        errors.extend(comparison_errors)
        if comparison is not None:
            comparisons[margin_key] = comparison

    if errors:
        return None, errors
    aucs = report["aucs"]
    expected_decision = _expected_cell_decision(stage, comparisons)
    if report.get("decision") != expected_decision:
        return None, [
            f"target seed {target_seed} decision mismatch: "
            f"expected {expected_decision}, found {report.get('decision')}"
        ]
    return (
        {
            "source_seed": source_seed,
            "target_seed": target_seed,
            "experiment_stage": stage,
            "decision": str(report["decision"]),
            "samples_per_class": 65536,
            "epochs": 1,
            "true_auc": float(aucs["true_to_true"]),
            "scratch_auc": float(aucs["typed_scratch"]),
            "source_shuffled_auc": float(aucs["shuffled_to_true"]),
            "target_shuffled_auc": float(aucs["true_to_shuffled"]),
            "comparisons": comparisons,
        },
        [],
    )


def _validated_comparison(
    report: dict[str, Any],
    *,
    margin_key: str,
    bootstrap_role: str,
    threshold: float,
    target_seed: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        point = float(report["margins"][margin_key])
        declared_threshold = float(report["thresholds"][margin_key])
        bootstrap = report["bootstrap"]["comparisons"][bootstrap_role]
        bootstrap_point = float(bootstrap["point_difference"])
        ci_lower = float(bootstrap["ci_lower"])
        ci_upper = float(bootstrap["ci_upper"])
        auc_difference = float(report["aucs"]["true_to_true"]) - float(
            report["aucs"][bootstrap_role]
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"target seed {target_seed} invalid {margin_key}: {exc}"]

    errors = []
    values = (point, declared_threshold, bootstrap_point, ci_lower, ci_upper)
    if not all(math.isfinite(value) for value in values):
        errors.append(f"target seed {target_seed} {margin_key} values must be finite")
    if not _close(declared_threshold, threshold):
        errors.append(
            f"target seed {target_seed} {margin_key} threshold "
            f"must be {threshold}"
        )
    if not _close(point, bootstrap_point):
        errors.append(
            f"target seed {target_seed} {margin_key} point/bootstrap mismatch"
        )
    if not _close(point, auc_difference):
        errors.append(f"target seed {target_seed} {margin_key} AUC/margin mismatch")
    if not ci_lower <= point <= ci_upper:
        errors.append(f"target seed {target_seed} {margin_key} CI does not contain point")
    if errors:
        return None, errors

    point_pass = point >= threshold
    ci_positive = ci_lower > 0.0
    return (
        {
            "label": str(COMPARISON_SPECS[margin_key]["label"]),
            "delta_auc": point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "threshold": threshold,
            "point_pass": point_pass,
            "ci_positive": ci_positive,
            "gate_pass": point_pass and ci_positive,
        },
        [],
    )


def _comparison_summary(
    cells: list[dict[str, Any]],
    key: str,
    threshold: float,
) -> dict[str, Any]:
    comparisons = [cell["comparisons"][key] for cell in cells]
    values = [float(item["delta_auc"]) for item in comparisons]
    return {
        "label": str(COMPARISON_SPECS[key]["label"]),
        "threshold": threshold,
        "cell_count": len(cells),
        "pass_count": sum(bool(item["gate_pass"]) for item in comparisons),
        "point_pass_count": sum(bool(item["point_pass"]) for item in comparisons),
        "ci_positive_count": sum(bool(item["ci_positive"]) for item in comparisons),
        "all_cells_pass": all(bool(item["gate_pass"]) for item in comparisons),
        "unweighted_mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _expected_cell_decision(
    stage: str,
    comparisons: dict[str, dict[str, Any]],
) -> str:
    margins_pass = all(bool(item["point_pass"]) for item in comparisons.values())
    scratch_ci_positive = bool(comparisons["scratch_margin"]["ci_positive"])
    controls_positive = all(float(item["delta_auc"]) > 0.0 for item in comparisons.values())
    if margins_pass and scratch_ci_positive:
        return f"{stage}_target_adaptation_efficiency_confirmed"
    if controls_positive:
        return f"{stage}_target_adaptation_signal_unstable"
    return f"{stage}_target_adaptation_rejected"


def _source_stratum_summary(
    cells: list[dict[str, Any]],
    source_seed: int,
) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["source_seed"] == source_seed]
    return {
        "target_seeds": [int(cell["target_seed"]) for cell in selected],
        "cell_count": len(selected),
        "mean_margins": {
            key: sum(float(cell["comparisons"][key]["delta_auc"]) for cell in selected)
            / len(selected)
            for key in COMPARISON_SPECS
        },
    }


def _close(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False
