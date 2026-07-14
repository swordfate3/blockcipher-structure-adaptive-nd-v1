from __future__ import annotations

from typing import Any


OBJECTIVE_LABELS = {
    "e5": "E5 topology-identity BCE",
    "e6": "E6 functional topology margin",
}
CONTROL_ROLES = ("off_transfer", "placebo_transfer", "scratch")


def build_cross_spn_e4_e6_synthesis(
    e4: dict[str, Any],
    e5: dict[str, Any],
    e6: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    _validate_e4(e4, errors)
    _validate_objective_gate("e5", e5, errors)
    _validate_objective_gate("e6", e6, errors)
    if not errors:
        _validate_shared_anchors(e5, e6, errors)
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_e4_e6_synthesis",
            "errors": errors,
            "next_action": "repair_input_selection_without_interpreting_metrics",
        }

    cells = [
        _objective_cell(stage, gate, seed)
        for stage, gate in (("e5", e5), ("e6", e6))
        for seed in (2, 3)
    ]
    e4_summary = {
        key: {
            "pass_count": int(e4["comparisons"][key]["pass_count"]),
            "cell_count": int(e4["comparisons"][key]["cell_count"]),
            "minimum": float(e4["comparisons"][key]["minimum"]),
            "maximum": float(e4["comparisons"][key]["maximum"]),
            "all_cells_pass": bool(e4["comparisons"][key]["all_cells_pass"]),
        }
        for key in (
            "scratch_margin",
            "source_topology_margin",
            "target_topology_margin",
        )
    }
    objective_complete_passes = sum(cell["complete_gate_pass"] for cell in cells)
    off_passes = sum(cell["comparisons"]["off_transfer"]["gate_pass"] for cell in cells)
    placebo_passes = sum(
        cell["comparisons"]["placebo_transfer"]["gate_pass"] for cell in cells
    )
    scratch_passes = sum(cell["comparisons"]["scratch"]["gate_pass"] for cell in cells)
    return {
        "status": "pass",
        "decision": (
            "typed_topology_representation_retained_source_objectives_rejected"
        ),
        "errors": [],
        "protocol_separation": {
            "e4_samples_per_class": 65536,
            "e5_e6_samples_per_class": 8192,
            "target_epochs": 1,
            "pooled_analysis": False,
        },
        "e4_representation": e4_summary,
        "objective_cells": cells,
        "objective_summary": {
            "cell_count": len(cells),
            "complete_gate_pass_count": objective_complete_passes,
            "candidate_vs_off_pass_count": off_passes,
            "candidate_vs_placebo_pass_count": placebo_passes,
            "candidate_vs_scratch_pass_count": scratch_passes,
            "all_objectives_rejected": objective_complete_passes == 0,
            "ordinary_transfer_signal_retained": scratch_passes == len(cells),
        },
        "claim_scope": (
            "controlled synthesis of separate local objective and remote medium "
            "representation evidence; not formal, paper-scale, SOTA, or breakthrough evidence"
        ),
        "next_action": "freeze_innovation1_method_result_prepare_paper_reporting",
        "stopped_actions": [
            "e7_architecture_or_source_objective",
            "source_seed1_rescue",
            "65536_per_class_objective_scale",
            "262144_per_class",
            "1000000_per_class",
        ],
    }


def _validate_e4(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("status") != "pass" or report.get("errors") != []:
        errors.append("E4 synthesis is not valid pass evidence")
        return
    if report.get("decision") != (
        "e4_typed_topology_attribution_robust_scratch_efficiency_conditional"
    ):
        errors.append("E4 synthesis decision mismatch")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, dict):
        errors.append("E4 comparisons are missing")
        return
    for key in ("source_topology_margin", "target_topology_margin"):
        if (comparisons.get(key) or {}).get("pass_count") != 4:
            errors.append(f"E4 {key} must pass 4/4")


def _validate_objective_gate(
    stage: str,
    report: dict[str, Any],
    errors: list[str],
) -> None:
    expected_decision = {
        "e5": "e5_r0_source_objective_rejected",
        "e6": "e6_r0_functional_margin_rejected",
    }[stage]
    if report.get("status") != "pass" or report.get("errors") != []:
        errors.append(f"{stage.upper()} joint gate is not valid pass evidence")
    if report.get("decision") != expected_decision:
        errors.append(f"{stage.upper()} joint decision mismatch")
    if report.get("expected_source_seed") != 0:
        errors.append(f"{stage.upper()} source seed must equal 0")
    if report.get("expected_target_seeds") != [2, 3]:
        errors.append(f"{stage.upper()} target seeds must equal [2, 3]")
    if report.get("research_decision_applied") is not True:
        errors.append(f"{stage.upper()} research decision was not applied")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, dict) or set(per_seed) != {"2", "3"}:
        errors.append(f"{stage.upper()} per-seed gates are incomplete")


def _validate_shared_anchors(
    e5: dict[str, Any],
    e6: dict[str, Any],
    errors: list[str],
) -> None:
    for seed in (2, 3):
        e5_aucs = e5["per_seed"][str(seed)]["aucs"]
        e6_aucs = e6["per_seed"][str(seed)]["aucs"]
        for role in ("scratch", "off_transfer"):
            if float(e5_aucs[role]) != float(e6_aucs[role]):
                errors.append(
                    f"target seed {seed} shared anchor {role} differs between E5 and E6"
                )


def _objective_cell(
    stage: str,
    report: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    seed_report = report["per_seed"][str(seed)]
    comparisons = {
        role: {
            "delta_auc": float(seed_report["margins"][role]),
            "ci_lower": float(
                seed_report["bootstrap"]["comparisons"][role]["ci_lower"]
            ),
            "ci_upper": float(
                seed_report["bootstrap"]["comparisons"][role]["ci_upper"]
            ),
            "point_pass": bool(seed_report["point_pass"][role]),
            "ci_pass": bool(seed_report["ci_pass"][role]),
            "gate_pass": bool(
                seed_report["point_pass"][role]
                and seed_report["ci_pass"][role]
            ),
        }
        for role in CONTROL_ROLES
    }
    return {
        "experiment_stage": stage,
        "objective_label": OBJECTIVE_LABELS[stage],
        "source_seed": 0,
        "target_seed": seed,
        "samples_per_class": 8192,
        "target_epochs": 1,
        "candidate_auc": float(seed_report["aucs"]["candidate_transfer"]),
        "off_auc": float(seed_report["aucs"]["off_transfer"]),
        "placebo_auc": float(seed_report["aucs"]["placebo_transfer"]),
        "scratch_auc": float(seed_report["aucs"]["scratch"]),
        "comparisons": comparisons,
        "complete_gate_pass": bool(seed_report["gate_pass"]),
    }


__all__ = ["build_cross_spn_e4_e6_synthesis"]
