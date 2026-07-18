from __future__ import annotations

import math
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_expanded_neural_screen import (
    BASELINE_AUC,
)
from blockcipher_nd.tasks.innovation2.small_spn_pair_relation_reasoner import (
    PairRelationTrainingConfig,
)


E40_RUN_ID = "i2_small_spn_pair_relation_no_triangle_seed0_seed1_20260718"


def adjudicate_pair_state_topology_control(
    config: PairRelationTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool | list[int] | str | None],
    source_gate: dict[str, Any],
    source_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    true_rows = [
        row
        for row in source_rows
        if row.get("topology_mode") == "true" and row.get("label_mode") == "true"
    ]
    shuffle_rows = [row for row in source_rows if row.get("label_mode") == "shuffled"]
    protocol = {
        **readiness,
        "source_gate_is_triangle_not_isolated": source_gate.get("decision")
        == "innovation2_small_spn_pair_relation_triangle_not_isolated",
        "source_run_id_matches": source_gate.get("run_id") == E40_RUN_ID,
        "two_true_source_rows_present": len(true_rows) == 2,
        "one_source_shuffle_row_present": len(shuffle_rows) == 1,
        "source_shuffle_dual_at_most_0p60": len(shuffle_rows) == 1
        and shuffle_rows[0]["dual_unseen_auc"] <= 0.60,
        "two_control_rows_present": len(control_rows) == 2,
        "true_source_rows_use_seeds_0_1": sorted(int(row["seed"]) for row in true_rows)
        == [0, 1],
        "control_rows_use_seeds_0_1": sorted(
            int(row["seed"]) for row in control_rows
        )
        == [0, 1],
        "true_rows_are_local_pair_state": all(
            row.get("model_name") == "pair_relation_no_triangle"
            and row.get("processor_mode") == "local"
            for row in true_rows
        ),
        "control_rows_are_local_pair_state": all(
            row.get("model_name") == "pair_relation_no_triangle"
            and row.get("processor_mode") == "local"
            for row in control_rows
        ),
        "control_rows_use_corrupted_topology": all(
            row.get("topology_mode") == "corrupted" for row in control_rows
        ),
        "all_control_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in control_rows
            for key in metric_keys
        ),
        "all_control_rows_trained": all(
            row.get("training_performed") is True for row in control_rows
        ),
        "parameter_budget_matches_source": len(
            {int(row["parameter_count"]) for row in true_rows + control_rows}
        )
        == 1,
        "triangle_block_count_is_zero": int(contract["shared_triangle_block_count"])
        == 0,
        "one_shared_local_block": int(contract["shared_local_block_count"]) == 1,
        "processor_mode_is_local": contract["processor_mode"] == "local",
        "parameter_count_matches_triangle": int(contract["parameter_count"])
        == int(contract["counterpart_parameter_count"]),
        "off_pair_influence_is_zero": float(contract["off_pair_influence_max_abs"])
        == 0.0,
        "cell_relabeling_error_at_most_1e_6": float(
            contract["cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "fair_control_heldout_avoids_true_train": bool(
            contract["fair_control_heldout_avoids_true_train"]
        ),
        "fair_control_heldout_avoids_corrupted_train": bool(
            contract["fair_control_heldout_avoids_corrupted_train"]
        ),
        "all_corrupted_players_are_permutations": bool(
            contract["all_corrupted_players_are_permutations"]
        ),
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_pair_state_topology_control_protocol_invalid",
            protocol,
            {},
            contract,
            "repair E40 ownership, fair topology, seed, parameter, locality, or metric protocol",
        )

    true_by_seed = {int(row["seed"]): row for row in true_rows}
    control_by_seed = {int(row["seed"]): row for row in control_rows}
    mean_auc = {
        family: {
            split: float(np.mean([row[f"{split}_auc"] for row in rows]))
            for split in BASELINE_AUC
        }
        for family, rows in (("true", true_rows), ("fair_corrupted", control_rows))
    }
    per_seed_dual_delta = {
        str(seed): true_by_seed[seed]["dual_unseen_auc"]
        - control_by_seed[seed]["dual_unseen_auc"]
        for seed in (0, 1)
    }
    checks = {
        "true_each_seed_beats_corrupted_dual": all(
            per_seed_dual_delta[str(seed)] > 0.0 for seed in (0, 1)
        ),
        "true_mean_dual_beats_corrupted_by_0p03": mean_auc["true"]["dual_unseen"]
        >= mean_auc["fair_corrupted"]["dual_unseen"] + 0.03,
    }
    metrics = {
        "mean_auc": mean_auc,
        "per_seed_dual_delta": per_seed_dual_delta,
        "true_dual_delta_vs_fair_corrupted": mean_auc["true"]["dual_unseen"]
        - mean_auc["fair_corrupted"]["dual_unseen"],
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_pair_state_topology_confirmed"
        action = "design real-cipher output-property transfer readiness with triangle and local pair-state processors"
    else:
        status = "hold"
        decision = "innovation2_small_spn_pair_state_topology_not_attributed"
        action = "use pair-local only as a capacity control and design real-cipher readiness around E39 triangle plus ID baseline"
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
            "fair-topology attribution of the no-triangle directed pair-state model "
            "on the expanded 16-bit synthetic SPN family; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }
