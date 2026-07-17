from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    SmallSpnModelSpec,
    SmallSpnTopologyPredictor,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    BASELINE_AUC,
    TopologyTrainingConfig,
    TrainingRowSpec,
)


E33_ABSOLUTE_MEAN_AUC = {
    "graphgps_true": {
        "unseen_sbox": 0.8348848561266862,
        "unseen_player": 0.7263304550579072,
        "dual_unseen": 0.6826722464822901,
    },
    "graphgps_wrong_player": {
        "unseen_sbox": 0.8268027058876732,
        "unseen_player": 0.7133935892179973,
        "dual_unseen": 0.7524442018437652,
    },
}


def equivariance_training_matrix(
    config: TopologyTrainingConfig,
) -> tuple[TrainingRowSpec, ...]:
    if config.mode == "smoke":
        return (
            TrainingRowSpec("graphgps", "true", "true", 0, "cell_equivariant"),
            TrainingRowSpec("graphgps", "shuffled", "true", 0, "cell_equivariant"),
            TrainingRowSpec("graphgps", "true", "shuffled", 0, "cell_equivariant"),
        )
    return tuple(
        [
            TrainingRowSpec("graphgps", "true", "true", seed, "cell_equivariant")
            for seed in (0, 1)
        ]
        + [
            TrainingRowSpec(
                "graphgps", "shuffled", "true", seed, "cell_equivariant"
            )
            for seed in (0, 1)
        ]
        + [
            TrainingRowSpec(
                "graphgps", "true", "shuffled", 0, "cell_equivariant"
            )
        ]
    )


def measure_cell_relabeling_error(
    data: dict[str, Any], *, processor_mode: str = "stacked"
) -> float:
    cell_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(16)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled_players = node_permutation[data["players"][:, inverse]]
    relabeled_active = data["structure_active"][:, inverse]
    relabeled_basis = data["structure_basis"][..., inverse]
    relabeled_masks = data["output_mask_bits"][:, inverse]
    spec = SmallSpnModelSpec(
        model_name="graphgps",
        topology_mode="true",
        position_mode="cell_equivariant",
        processor_mode=processor_mode,
        hidden_dim=32,
        blocks=2,
        heads=4,
        dropout=0.0,
    )
    torch.manual_seed(7733)
    original = SmallSpnTopologyPredictor(
        spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        structure_basis=data["structure_basis"],
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    relabeled = SmallSpnTopologyPredictor(
        spec,
        sboxes=data["sboxes"],
        players=relabeled_players,
        structure_active_bits=relabeled_active,
        structure_basis=relabeled_basis,
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=relabeled_masks,
    ).eval()
    original_parameters = dict(original.named_parameters())
    with torch.no_grad():
        for name, parameter in relabeled.named_parameters():
            parameter.copy_(original_parameters[name])
        variants = torch.arange(16, dtype=torch.long)
        rounds = torch.arange(16, dtype=torch.long) % 4
        structures = torch.arange(16, dtype=torch.long) % len(data["structure_active"])
        masks = (torch.arange(16, dtype=torch.long) * 7) % len(
            data["output_mask_bits"]
        )
        expected = original(variants, rounds, structures, masks)
        actual = relabeled(variants, rounds, structures, masks)
    return float(torch.max(torch.abs(expected - actual)))


def adjudicate_cell_equivariance(
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
        "cell_relabeling_error_at_most_1e_6": relabeling_error <= 1e-6,
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_cell_equivariance_protocol_invalid",
            protocol,
            {},
            relabeling_error,
            "repair the representation, relabeling contract, source, split, or metric path",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_cell_equivariance_readiness_passed",
            protocol,
            {},
            relabeling_error,
            "run the frozen two-seed E33-R cell-equivariant attribution matrix",
        )

    groups = {
        "equivariant_true": [
            row
            for row in rows
            if row["topology_mode"] == "true" and row["label_mode"] == "true"
        ],
        "equivariant_wrong_player": [
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
    candidate = means["equivariant_true"]
    wrong = means["equivariant_wrong_player"]
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": means["label_shuffle"]
        ["dual_unseen"]
        <= 0.60,
        "candidate_each_seed_beats_dual_baseline": all(
            row["dual_unseen_auc"] > BASELINE_AUC["dual_unseen"]
            for row in groups["equivariant_true"]
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
        "e33_absolute_mean_auc": E33_ABSOLUTE_MEAN_AUC,
        "e33r_mean_auc": means,
        "candidate_dual_delta_vs_baseline": candidate["dual_unseen"]
        - BASELINE_AUC["dual_unseen"],
        "candidate_dual_delta_vs_wrong_player": candidate["dual_unseen"]
        - wrong["dual_unseen"],
        "candidate_dual_delta_vs_e33_absolute": candidate["dual_unseen"]
        - E33_ABSOLUTE_MEAN_AUC["graphgps_true"]["dual_unseen"],
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_cell_equivariance_repair_confirmed"
        action = "prepare an equivariant SCGT basis-branch audit; keep real-cipher scale closed"
    elif checks["candidate_mean_dual_beats_baseline_by_0p03"] and not checks[
        "candidate_mean_dual_beats_wrong_player_by_0p03"
    ]:
        status = "hold"
        decision = "innovation2_small_spn_cell_equivariance_topology_not_attributed"
        action = "stop topology claims; diagnose whether edge interactions need a different operator"
    else:
        status = "hold"
        decision = "innovation2_small_spn_cell_equivariance_repair_not_ready"
        action = "stop this GraphGPS representation route without adding scale or capacity"
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
            "single-variable cell-relabeling-equivariant representation audit on the frozen "
            "16-bit synthetic-SPN matched-contrast benchmark; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }
