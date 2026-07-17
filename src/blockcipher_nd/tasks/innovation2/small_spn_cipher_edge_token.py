from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.small_spn_edge_token_models import (
    SmallSpnCipherEdgeTokenTransformer,
    SmallSpnEdgeTokenSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    BASELINE_AUC,
    TopologyTrainingConfig,
    TrainingRowSpec,
)


E33R_BEST_NEURAL_ANCHOR = {
    "unseen_sbox": 0.8222691735763631,
    "unseen_player": 0.6842779542892283,
    "dual_unseen": 0.7115477923338185,
}


def cipher_edge_token_training_matrix(
    config: TopologyTrainingConfig,
) -> tuple[TrainingRowSpec, ...]:
    def row(topology: str, label: str, seed: int) -> TrainingRowSpec:
        return TrainingRowSpec(
            "cett",
            topology,
            label,
            seed,
            "cell_equivariant",
            "edge_token_transformer",
        )

    if config.mode == "smoke":
        return (
            row("true", "true", 0),
            row("corrupted", "true", 0),
            row("true", "shuffled", 0),
        )
    return tuple(
        [row("true", "true", seed) for seed in (0, 1)]
        + [row("corrupted", "true", seed) for seed in (0, 1)]
        + [row("true", "shuffled", 0)]
    )


def measure_cipher_edge_token_contract(
    data: dict[str, Any],
) -> dict[str, float | int | bool]:
    cell_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(16)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled_players = node_permutation[data["players"][:, inverse]]
    relabeled_active = data["structure_active"][:, inverse]
    relabeled_masks = data["output_mask_bits"][:, inverse]
    spec = SmallSpnEdgeTokenSpec(
        topology_mode="true",
        hidden_dim=32,
        layers=2,
        heads=4,
        dropout=0.0,
    )
    torch.manual_seed(8535)
    original = SmallSpnCipherEdgeTokenTransformer(
        spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    relabeled = SmallSpnCipherEdgeTokenTransformer(
        spec,
        sboxes=data["sboxes"],
        players=relabeled_players,
        structure_active_bits=relabeled_active,
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
        tokens = original.build_tokens(variants, rounds, structures, masks)
        expected = original(variants, rounds, structures, masks)
        actual = relabeled(variants, rounds, structures, masks)
    corrupted_players = topology_players(data["players"], "corrupted")
    train_players = data["players"][[0, 1, 2]]
    heldout_indices = (3, 7, 11, 15)
    return {
        "token_count": int(tokens.shape[1]),
        "cell_relabeling_max_abs_logit_error": float(
            torch.max(torch.abs(expected - actual))
        ),
        "fair_control_heldout_avoids_train_players": all(
            not any(
                np.array_equal(corrupted_players[index], train_player)
                for train_player in train_players
            )
            for index in heldout_indices
        ),
        "all_corrupted_players_are_permutations": all(
            np.array_equal(np.sort(row), np.arange(16))
            for row in corrupted_players
        ),
    }


def adjudicate_cipher_edge_token(
    config: TopologyTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = 3 if config.mode == "smoke" else 5
    relabeling_error = float(contract["cell_relabeling_max_abs_logit_error"])
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
        "all_rows_are_cett": all(row.get("model_name") == "cett" for row in rows),
        "all_rows_use_cell_equivariant_positions": all(
            row.get("position_mode") == "cell_equivariant" for row in rows
        ),
        "all_rows_use_edge_token_transformer": all(
            row.get("processor_mode") == "edge_token_transformer" for row in rows
        ),
        "token_count_is_37": int(contract["token_count"]) == 37,
        "cell_relabeling_error_at_most_1e_6": relabeling_error <= 1e-6,
        "fair_control_heldout_avoids_train_players": bool(
            contract["fair_control_heldout_avoids_train_players"]
        ),
        "all_corrupted_players_are_permutations": bool(
            contract["all_corrupted_players_are_permutations"]
        ),
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_cipher_edge_token_protocol_invalid",
            protocol,
            {},
            contract,
            "repair tokenization, relabeling, source, split, or metric protocol",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_cipher_edge_token_readiness_passed",
            protocol,
            {},
            contract,
            "run the frozen two-seed E35 edge-token attribution matrix",
        )

    groups = {
        "cett_true": [
            row
            for row in rows
            if row["topology_mode"] == "true" and row["label_mode"] == "true"
        ],
        "cett_wrong_player": [
            row
            for row in rows
            if row["topology_mode"] == "corrupted" and row["label_mode"] == "true"
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
    candidate = means["cett_true"]
    wrong = means["cett_wrong_player"]
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": means["label_shuffle"]
        ["dual_unseen"]
        <= 0.60,
        "candidate_each_seed_beats_dual_baseline": all(
            row["dual_unseen_auc"] > BASELINE_AUC["dual_unseen"]
            for row in groups["cett_true"]
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
        "e33r_best_neural_anchor": E33R_BEST_NEURAL_ANCHOR,
        "e35_mean_auc": means,
        "candidate_dual_delta_vs_baseline": candidate["dual_unseen"]
        - BASELINE_AUC["dual_unseen"],
        "candidate_dual_delta_vs_wrong_player": candidate["dual_unseen"]
        - wrong["dual_unseen"],
        "candidate_dual_delta_vs_neural_anchor": candidate["dual_unseen"]
        - E33R_BEST_NEURAL_ANCHOR["dual_unseen"],
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_cipher_edge_token_confirmed"
        action = "prepare a real-cipher edge-token transfer readiness audit; keep remote scale closed"
    elif checks["candidate_mean_dual_beats_baseline_by_0p03"] and not checks[
        "candidate_mean_dual_beats_wrong_player_by_0p03"
    ]:
        status = "hold"
        decision = "innovation2_small_spn_cipher_edge_token_not_attributed"
        action = "stop topology claims and close the synthetic neural architecture search"
    else:
        status = "hold"
        decision = "innovation2_small_spn_cipher_edge_token_not_ready"
        action = "close the synthetic neural architecture search and return to label/task design"
    gate = _gate(
        config,
        status,
        decision,
        protocol,
        checks,
        contract,
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
    contract: dict[str, float | int | bool],
    action: str,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": protocol,
        "attribution_checks": checks,
        "token_contract": contract,
        "claim_scope": (
            "single-architecture cipher edge-token Transformer audit on the frozen 16-bit "
            "synthetic-SPN matched-contrast benchmark; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }
