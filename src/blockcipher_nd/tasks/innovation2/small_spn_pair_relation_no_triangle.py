from __future__ import annotations

import math
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_expanded_neural_screen import (
    BASELINE_AUC,
)
from blockcipher_nd.tasks.innovation2.small_spn_pair_relation_reasoner import (
    PairRelationRowSpec,
    PairRelationTrainingConfig,
)


E39_PHASE_A_RUN_ID = "i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718"
E39_PHASE_B_RUN_ID = (
    "i2_small_spn_pair_relation_fair_control_seed0_seed1_20260718"
)


def no_triangle_training_matrix(
    config: PairRelationTrainingConfig,
) -> tuple[PairRelationRowSpec, ...]:
    if config.mode == "smoke":
        return (
            PairRelationRowSpec("true", "true", 0),
            PairRelationRowSpec("true", "shuffled", 0),
        )
    return (
        PairRelationRowSpec("true", "true", 0),
        PairRelationRowSpec("true", "true", 1),
        PairRelationRowSpec("true", "shuffled", 0),
    )


def adjudicate_no_triangle_ablation(
    config: PairRelationTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool | list[int] | str | None],
    rows: list[dict[str, Any]],
    source_gate: dict[str, Any],
    topology_gate: dict[str, Any],
    source_true_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    expected_rows = 2 if config.mode == "smoke" else 3
    protocol = {
        **readiness,
        "source_phase_a_is_candidate_screened": source_gate.get("decision")
        == "innovation2_small_spn_pair_relation_candidate_screened",
        "source_phase_a_run_id_matches": source_gate.get("run_id")
        == E39_PHASE_A_RUN_ID,
        "source_phase_b_is_topology_confirmed": topology_gate.get("decision")
        == "innovation2_small_spn_pair_relation_topology_confirmed",
        "source_phase_b_run_id_matches": topology_gate.get("run_id")
        == E39_PHASE_B_RUN_ID,
        "two_source_true_rows_present": len(source_true_rows) == 2,
        "source_true_rows_use_seeds_0_1": sorted(
            int(row["seed"]) for row in source_true_rows
        )
        == [0, 1],
        "expected_matrix_rows_present": len(rows) == expected_rows,
        "all_metrics_finite": all(
            math.isfinite(float(row[key])) for row in rows for key in metric_keys
        ),
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
        "all_rows_are_no_triangle": all(
            row.get("model_name") == "pair_relation_no_triangle"
            and row.get("processor_mode") == "local"
            for row in rows
        ),
        "initial_pair_shape_matches": bool(contract["initial_pair_shape_matches"]),
        "pair_count_is_256": int(contract["pair_count"]) == 256,
        "triangle_block_count_is_zero": int(contract["shared_triangle_block_count"])
        == 0,
        "one_shared_local_block": int(contract["shared_local_block_count"]) == 1,
        "processor_mode_is_local": contract["processor_mode"] == "local",
        "step_schedule_is_2_3_4_5": contract["step_schedule"] == [2, 3, 4, 5],
        "parameter_count_matches_triangle": int(contract["parameter_count"])
        == int(contract["counterpart_parameter_count"]),
        "source_parameter_count_matches": all(
            int(row["parameter_count"]) == int(contract["parameter_count"])
            for row in source_true_rows
        )
        if config.mode == "full"
        else True,
        "off_pair_influence_at_most_1e_7": float(
            contract["off_pair_influence_max_abs"]
        )
        <= 1e-7,
        "cell_relabeling_error_at_most_1e_6": float(
            contract["cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "true_corrupted_logit_difference_at_least_1e_5": float(
            contract["true_corrupted_max_abs_logit_difference"]
        )
        >= 1e-5,
        "absolute_ids_absent": bool(
            contract["absolute_bit_cell_or_variant_embedding_absent"]
        ),
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_pair_relation_no_triangle_protocol_invalid",
            protocol,
            {},
            contract,
            "repair E39 ownership, pair locality, parameter, invariance, or training protocol",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_pair_relation_no_triangle_readiness_passed",
            protocol,
            {},
            contract,
            "run the frozen E40 two-seed plus label-shuffle full ablation",
        )

    candidate_rows = [row for row in rows if row["label_mode"] == "true"]
    shuffle_rows = [row for row in rows if row["label_mode"] == "shuffled"]
    source_by_seed = {int(row["seed"]): row for row in source_true_rows}
    candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
    source_mean = {
        split: float(np.mean([row[f"{split}_auc"] for row in source_true_rows]))
        for split in BASELINE_AUC
    }
    candidate_mean = {
        split: float(np.mean([row[f"{split}_auc"] for row in candidate_rows]))
        for split in BASELINE_AUC
    }
    per_seed_dual_delta = {
        str(seed): source_by_seed[seed]["dual_unseen_auc"]
        - candidate_by_seed[seed]["dual_unseen_auc"]
        for seed in (0, 1)
    }
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": len(shuffle_rows) == 1
        and shuffle_rows[0]["dual_unseen_auc"] <= 0.60,
        "triangle_beats_no_triangle_each_seed_dual": all(
            per_seed_dual_delta[str(seed)] > 0.0 for seed in (0, 1)
        ),
        "triangle_mean_dual_beats_no_triangle_by_0p03": source_mean["dual_unseen"]
        >= candidate_mean["dual_unseen"] + 0.03,
    }
    metrics = {
        "baseline_auc": BASELINE_AUC,
        "mean_auc": {"triangle": source_mean, "no_triangle": candidate_mean},
        "per_seed_dual_delta": per_seed_dual_delta,
        "triangle_dual_delta_vs_no_triangle": source_mean["dual_unseen"]
        - candidate_mean["dual_unseen"],
        "no_triangle_dual_delta_vs_id_baseline": candidate_mean["dual_unseen"]
        - BASELINE_AUC["dual_unseen"],
    }
    if not checks["label_shuffle_dual_auc_at_most_0p60"]:
        status = "hold"
        decision = "innovation2_small_spn_pair_relation_no_triangle_not_attributed"
        action = "repair or explain the no-triangle label-shuffle control"
    elif all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_pair_relation_triangle_attributed"
        action = "design a real-cipher output-property transfer readiness audit without remote launch"
    else:
        status = "hold"
        decision = "innovation2_small_spn_pair_relation_triangle_not_isolated"
        action = "retain pair-state evidence, stop triangle-specific claims, and rank query-conditioned NBFNet versus structured P families"
    gate = _gate(config, status, decision, protocol, checks, contract, action)
    gate["metrics"] = metrics
    return gate


def _gate(
    config: PairRelationTrainingConfig,
    status: str,
    decision: str,
    protocol: dict[str, bool],
    checks: dict[str, bool],
    contract: dict[str, float | int | bool | list[int] | str | None],
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": protocol,
        "screen_checks": checks,
        "model_contract": contract,
        "claim_scope": (
            "same-budget no-triangle ablation of E39 on the expanded 16-bit synthetic "
            "SPN family; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }
