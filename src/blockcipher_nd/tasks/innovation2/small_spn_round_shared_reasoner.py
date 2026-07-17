from __future__ import annotations

import math
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    BASELINE_AUC,
    TopologyTrainingConfig,
    TrainingRowSpec,
)


E33R_STATIC_MEAN_AUC = {
    "true_player": {
        "unseen_sbox": 0.8222691735763631,
        "unseen_player": 0.6842779542892283,
        "dual_unseen": 0.7115477923338185,
    },
    "wrong_player": {
        "unseen_sbox": 0.8256814596683877,
        "unseen_player": 0.7121816388234088,
        "dual_unseen": 0.7088306647258612,
    },
}


def round_shared_training_matrix(
    config: TopologyTrainingConfig,
) -> tuple[TrainingRowSpec, ...]:
    def row(topology: str, label: str, seed: int) -> TrainingRowSpec:
        return TrainingRowSpec(
            "graphgps",
            topology,
            label,
            seed,
            "cell_equivariant",
            "round_shared",
        )

    if config.mode == "smoke":
        return (
            row("true", "true", 0),
            row("shuffled", "true", 0),
            row("true", "shuffled", 0),
        )
    return tuple(
        [row("true", "true", seed) for seed in (0, 1)]
        + [row("shuffled", "true", seed) for seed in (0, 1)]
        + [row("true", "shuffled", 0)]
    )


def adjudicate_round_shared_reasoner(
    config: TopologyTrainingConfig,
    readiness: dict[str, bool],
    relabeling_error: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = 3 if config.mode == "smoke" else 5
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    protocol = {
        **readiness,
        "expected_matrix_rows_present": len(rows) == expected_rows,
        "all_metrics_finite": all(
            math.isfinite(float(row[key])) for row in rows for key in metric_keys
        ),
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
        "all_rows_use_cell_equivariant_positions": all(
            row.get("position_mode") == "cell_equivariant" for row in rows
        ),
        "all_rows_use_round_shared_processor": all(
            row.get("processor_mode") == "round_shared" for row in rows
        ),
        "cell_relabeling_error_at_most_1e_6": relabeling_error <= 1e-6,
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_round_shared_protocol_invalid",
            protocol,
            {},
            relabeling_error,
            "repair variable-step execution, relabeling, source, split, or metric protocol",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_round_shared_readiness_passed",
            protocol,
            {},
            relabeling_error,
            "run the frozen two-seed E34 round-shared attribution matrix",
        )

    groups = {
        "round_shared_true": [
            row
            for row in rows
            if row["topology_mode"] == "true" and row["label_mode"] == "true"
        ],
        "round_shared_wrong_player": [
            row
            for row in rows
            if row["topology_mode"] == "shuffled" and row["label_mode"] == "true"
        ],
        "label_shuffle": [row for row in rows if row["label_mode"] == "shuffled"],
    }
    means = {
        name: {
            split: float(np.mean([row[f"{split}_auc"] for row in group]))
            for split in ("unseen_sbox", "unseen_player", "dual_unseen")
        }
        for name, group in groups.items()
    }
    candidate = means["round_shared_true"]
    wrong = means["round_shared_wrong_player"]
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": means["label_shuffle"]
        ["dual_unseen"]
        <= 0.60,
        "candidate_each_seed_beats_dual_baseline": all(
            row["dual_unseen_auc"] > BASELINE_AUC["dual_unseen"]
            for row in groups["round_shared_true"]
        ),
        "candidate_mean_dual_beats_baseline_by_0p03": candidate["dual_unseen"]
        >= BASELINE_AUC["dual_unseen"] + 0.03,
        "candidate_mean_dual_beats_wrong_player_by_0p03": candidate["dual_unseen"]
        >= wrong["dual_unseen"] + 0.03,
        "candidate_unseen_sbox_not_below_baseline_minus_0p01": candidate[
            "unseen_sbox"
        ]
        >= BASELINE_AUC["unseen_sbox"] - 0.01,
        "candidate_unseen_player_not_below_baseline_minus_0p01": candidate[
            "unseen_player"
        ]
        >= BASELINE_AUC["unseen_player"] - 0.01,
    }
    metrics = {
        "baseline_auc": BASELINE_AUC,
        "e33r_static_mean_auc": E33R_STATIC_MEAN_AUC,
        "e34_mean_auc": means,
        "candidate_dual_delta_vs_baseline": candidate["dual_unseen"]
        - BASELINE_AUC["dual_unseen"],
        "candidate_dual_delta_vs_wrong_player": candidate["dual_unseen"]
        - wrong["dual_unseen"],
        "candidate_dual_delta_vs_static_anchor": candidate["dual_unseen"]
        - E33R_STATIC_MEAN_AUC["true_player"]["dual_unseen"],
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_round_shared_reasoner_confirmed"
        action = "prepare a round-shared SCGT basis audit; keep real-cipher scale closed"
    elif checks["candidate_mean_dual_beats_baseline_by_0p03"] and not checks[
        "candidate_mean_dual_beats_wrong_player_by_0p03"
    ]:
        status = "hold"
        decision = "innovation2_small_spn_round_shared_topology_not_attributed"
        action = "stop topology claims; do not enlarge the looped model"
    else:
        status = "hold"
        decision = "innovation2_small_spn_round_shared_reasoner_not_ready"
        action = "stop the synthetic GraphGPS/looped family without adding scale or capacity"
    gate = _gate(
        config,
        status,
        decision,
        protocol,
        checks,
        relabeling_error,
        action,
    )
    gate["metrics"] = metrics
    return gate


def _gate(
    config: TopologyTrainingConfig,
    status: str,
    decision: str,
    protocol: dict[str, bool],
    checks: dict[str, bool],
    relabeling_error: float,
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": protocol,
        "attribution_checks": checks,
        "cell_relabeling_max_abs_logit_error": relabeling_error,
        "claim_scope": (
            "single-variable round-shared neural-algorithmic-reasoner audit on the "
            "frozen 16-bit synthetic-SPN matched-contrast benchmark; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }

