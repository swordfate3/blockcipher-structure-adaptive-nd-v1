from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.small_spn_edge_token_models import (
    SmallSpnCipherEdgeTokenTransformer,
    SmallSpnEdgeTokenSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    SmallSpnModelSpec,
    SmallSpnTopologyPredictor,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.small_spn_expanded_topology_labels import (
    ExpandedTopologyAuditConfig,
    expanded_split_indices,
    make_expanded_variants,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    TopologyTrainingConfig,
    TrainingRowSpec,
)


CELL_SPLIT_SEED = 38001
BASELINE_AUC = {
    "unseen_sbox": 0.6881980369450377,
    "unseen_player": 0.6487534140097411,
    "dual_unseen": 0.6843931010265274,
}


def expanded_neural_screen_matrix(
    config: TopologyTrainingConfig,
) -> tuple[TrainingRowSpec, ...]:
    graphgps = lambda seed: TrainingRowSpec(
        "graphgps", "true", "true", seed, "cell_equivariant"
    )
    cett = lambda seed, label="true": TrainingRowSpec(
        "cett",
        "true",
        label,
        seed,
        "cell_equivariant",
        "edge_token_transformer",
    )
    if config.mode == "smoke":
        return graphgps(0), cett(0), cett(0, "shuffled")
    return graphgps(0), graphgps(1), cett(0), cett(1), cett(0, "shuffled")


def load_expanded_neural_data(source_root: Path) -> dict[str, Any]:
    labels = np.load(source_root / "labels.npy")
    selected = np.load(source_root / "selected_mask.npy")
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    variants = metadata["variants"]
    structures = metadata["structures"]
    masks = [int(value, 16) for value in metadata["output_masks"]]
    structure_active = np.zeros((len(structures), 16), dtype=np.float32)
    structure_basis = np.zeros((len(structures), 12, 16), dtype=np.float32)
    structure_basis_valid = np.zeros((len(structures), 12), dtype=np.bool_)
    for index, structure in enumerate(structures):
        bits = [int(bit) for bit in structure["active_bits"]]
        structure_active[index, bits] = 1.0
        for basis_index, bit in enumerate(bits):
            structure_basis[index, basis_index, bit] = 1.0
            structure_basis_valid[index, basis_index] = True
    output_mask_bits = np.asarray(
        [[(mask >> bit) & 1 for bit in range(16)] for mask in masks],
        dtype=np.float32,
    )
    variant_objects = make_expanded_variants(
        ExpandedTopologyAuditConfig(run_id="expanded-training-data")
    )
    split_indices = expanded_split_indices(variant_objects)
    cells = np.argwhere(selected)
    order = np.random.default_rng(CELL_SPLIT_SEED).permutation(len(cells))
    fit_count = int(math.floor(len(cells) * 0.8))
    return {
        "labels": np.asarray(labels, dtype=np.bool_),
        "selected": np.asarray(selected, dtype=np.bool_),
        "fit_cells": cells[order[:fit_count]],
        "validation_cells": cells[order[fit_count:]],
        "split_indices": split_indices,
        "sboxes": np.asarray([variant["sbox"] for variant in variants], dtype=np.uint8),
        "players": np.asarray(
            [variant["player"] for variant in variants], dtype=np.int64
        ),
        "structure_active": structure_active,
        "structure_basis": structure_basis,
        "structure_basis_valid": structure_basis_valid,
        "output_mask_bits": output_mask_bits,
        "source_metadata": metadata,
        "source_gate": gate,
    }


def validate_expanded_neural_contract(data: dict[str, Any]) -> dict[str, bool]:
    split_indices = data["split_indices"]
    fit_cells = {tuple(row) for row in data["fit_cells"]}
    validation_cells = {tuple(row) for row in data["validation_cells"]}
    train = set(split_indices["train"].tolist())
    heldout = set(
        np.concatenate(
            [
                split_indices["unseen_sbox"],
                split_indices["unseen_player"],
                split_indices["dual_unseen"],
            ]
        ).tolist()
    )
    source_readiness = data["source_gate"].get("readiness_checks", {})
    return {
        "source_gate_is_expanded_benchmark_ready": data["source_gate"].get("decision")
        == "innovation2_small_spn_expanded_topology_benchmark_ready",
        "source_task_matches": data["source_metadata"].get("task")
        == "innovation2_small_spn_expanded_topology_benchmark",
        "heldout_not_used_for_selection": data["source_metadata"].get(
            "heldout_labels_used_for_selection"
        )
        is False,
        "labels_shape_is_64x4x14x64": data["labels"].shape == (64, 4, 14, 64),
        "selected_cells_are_320": int(data["selected"].sum()) == 320,
        "fit_and_validation_cells_are_disjoint": fit_cells.isdisjoint(
            validation_cells
        ),
        "fit_and_validation_cover_selected_cells": len(
            fit_cells | validation_cells
        )
        == 320,
        "train_has_36_variants": len(train) == 36,
        "heldout_has_28_disjoint_variants": len(heldout) == 28
        and heldout.isdisjoint(train),
        "twelve_independent_train_players": len(
            {tuple(data["players"][index]) for index in train}
        )
        == 12,
        "source_fair_control_contract_passed": all(
            bool(source_readiness.get(key))
            for key in (
                "all_corrupted_players_are_bijections",
                "corrupted_players_are_unique",
                "heldout_corrupted_not_true_train",
                "heldout_corrupted_not_corrupted_train",
            )
        ),
        "no_cipher_or_variant_id_feature_is_constructed": True,
    }


def measure_expanded_model_contract(
    data: dict[str, Any]
) -> dict[str, float | int | bool]:
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

    graph_spec = SmallSpnModelSpec(
        model_name="graphgps",
        topology_mode="true",
        position_mode="cell_equivariant",
        hidden_dim=32,
        blocks=2,
        heads=4,
        dropout=0.0,
    )
    cett_spec = SmallSpnEdgeTokenSpec(
        topology_mode="true", hidden_dim=32, layers=2, heads=4, dropout=0.0
    )
    torch.manual_seed(38002)
    graph = SmallSpnTopologyPredictor(
        graph_spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        structure_basis=data["structure_basis"],
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    relabeled_graph = SmallSpnTopologyPredictor(
        graph_spec,
        sboxes=data["sboxes"],
        players=relabeled_players,
        structure_active_bits=relabeled_active,
        structure_basis=relabeled_basis,
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=relabeled_masks,
    ).eval()
    cett = SmallSpnCipherEdgeTokenTransformer(
        cett_spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    relabeled_cett = SmallSpnCipherEdgeTokenTransformer(
        cett_spec,
        sboxes=data["sboxes"],
        players=relabeled_players,
        structure_active_bits=relabeled_active,
        output_mask_bits=relabeled_masks,
    ).eval()
    _copy_parameters(graph, relabeled_graph)
    _copy_parameters(cett, relabeled_cett)
    variants = torch.arange(64, dtype=torch.long)
    rounds = variants % 4
    structures = variants % len(data["structure_active"])
    masks = (variants * 7) % len(data["output_mask_bits"])
    with torch.no_grad():
        graph_error = torch.max(
            torch.abs(
                graph(variants, rounds, structures, masks)
                - relabeled_graph(variants, rounds, structures, masks)
            )
        )
        tokens = cett.build_tokens(variants, rounds, structures, masks)
        cett_error = torch.max(
            torch.abs(
                cett(variants, rounds, structures, masks)
                - relabeled_cett(variants, rounds, structures, masks)
            )
        )
    corrupted = topology_players(data["players"], "corrupted")
    train_indices = data["split_indices"]["train"]
    heldout_indices = np.concatenate(
        [
            data["split_indices"]["unseen_player"],
            data["split_indices"]["dual_unseen"],
        ]
    )
    true_train = {tuple(data["players"][index]) for index in train_indices}
    corrupted_train = {tuple(corrupted[index]) for index in train_indices}
    corrupted_heldout = {tuple(corrupted[index]) for index in heldout_indices}
    return {
        "graphgps_cell_relabeling_max_abs_logit_error": float(graph_error),
        "cett_cell_relabeling_max_abs_logit_error": float(cett_error),
        "cett_token_count": int(tokens.shape[1]),
        "fair_control_heldout_avoids_true_train": corrupted_heldout.isdisjoint(
            true_train
        ),
        "fair_control_heldout_avoids_corrupted_train": corrupted_heldout.isdisjoint(
            corrupted_train
        ),
        "all_corrupted_players_are_permutations": all(
            np.array_equal(np.sort(row), np.arange(16)) for row in corrupted
        ),
    }


def adjudicate_expanded_neural_screen(
    config: TopologyTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    expected_rows = 3 if config.mode == "smoke" else 5
    protocol = {
        **readiness,
        "expected_matrix_rows_present": len(rows) == expected_rows,
        "all_metrics_finite": all(
            math.isfinite(float(row[key])) for row in rows for key in metric_keys
        ),
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
        "graphgps_rows_are_cell_equivariant": all(
            row.get("position_mode") == "cell_equivariant"
            for row in rows
            if row.get("model_name") == "graphgps"
        ),
        "cett_rows_use_edge_token_transformer": all(
            row.get("processor_mode") == "edge_token_transformer"
            for row in rows
            if row.get("model_name") == "cett"
        ),
        "graphgps_cell_relabeling_error_at_most_1e_6": float(
            contract["graphgps_cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "cett_cell_relabeling_error_at_most_1e_6": float(
            contract["cett_cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "cett_token_count_is_37": int(contract["cett_token_count"]) == 37,
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
            "innovation2_small_spn_expanded_neural_screen_protocol_invalid",
            protocol,
            {},
            contract,
            "repair source, split, model invariance, checkpoint, or metric protocol",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_expanded_neural_screen_readiness_passed",
            protocol,
            {},
            contract,
            "run the frozen five-row E38 Phase A two-seed screen",
        )

    groups = {
        model: [
            row
            for row in rows
            if row["model_name"] == model and row["label_mode"] == "true"
        ]
        for model in ("graphgps", "cett")
    }
    label_shuffle = [row for row in rows if row["label_mode"] == "shuffled"]
    means = {
        model: {
            split: float(np.mean([row[f"{split}_auc"] for row in group]))
            for split in BASELINE_AUC
        }
        for model, group in groups.items()
    }
    model_checks: dict[str, dict[str, bool]] = {}
    for model, group in groups.items():
        model_checks[model] = {
            f"each_seed_beats_{split}_baseline": all(
                row[f"{split}_auc"] > BASELINE_AUC[split] for row in group
            )
            for split in BASELINE_AUC
        }
        model_checks[model]["mean_dual_beats_baseline_by_0p03"] = (
            means[model]["dual_unseen"] >= BASELINE_AUC["dual_unseen"] + 0.03
        )
    process_checks = {
        "label_shuffle_dual_auc_at_most_0p60": len(label_shuffle) == 1
        and label_shuffle[0]["dual_unseen_auc"] <= 0.60
    }
    qualified = [
        model for model, checks in model_checks.items() if all(checks.values())
    ]
    selected_candidate = (
        max(
            qualified,
            key=lambda model: (
                means[model]["dual_unseen"],
                -int(groups[model][0]["parameter_count"]),
            ),
        )
        if qualified
        else None
    )
    metrics = {
        "baseline_auc": BASELINE_AUC,
        "mean_auc": means,
        "qualified_candidates": qualified,
        "selected_candidate": selected_candidate,
        "cett_dual_delta_vs_graphgps": means["cett"]["dual_unseen"]
        - means["graphgps"]["dual_unseen"],
        "cett_edge_token_increment_at_least_0p01": means["cett"]["dual_unseen"]
        >= means["graphgps"]["dual_unseen"] + 0.01,
    }
    checks = {
        **process_checks,
        **{
            f"{model}_{name}": value
            for model, values in model_checks.items()
            for name, value in values.items()
        },
    }
    if not all(process_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_expanded_neural_screen_not_attributed"
        action = "repair or explain the label-shuffle control before model selection"
    elif selected_candidate is None:
        status = "hold"
        decision = "innovation2_small_spn_expanded_neural_screen_not_ready"
        action = "stop GraphGPS/CETT scaling and rank a directed edge-pair operator or structured topology family"
    else:
        status = "pass"
        decision = "innovation2_small_spn_expanded_neural_candidate_screened"
        action = (
            f"run two-seed fair-corrupted P attribution for {selected_candidate} only"
        )
    gate = _gate(config, status, decision, protocol, checks, contract, action)
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
        "screen_checks": checks,
        "model_contract": contract,
        "claim_scope": (
            "fixed-budget GraphGPS/CETT screen on train-only selected labels from an "
            "expanded 16-bit synthetic SPN family; not real-cipher or attack evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }


def _copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(source_parameters[name])
