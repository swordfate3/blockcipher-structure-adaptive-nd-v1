from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    PairLocalBlock,
    PairTriangleBlock,
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.small_spn_expanded_neural_screen import (
    BASELINE_AUC,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    _evaluate,
    _example_arrays,
    _seed_everything,
    _tensor_dataset,
)


GRAPHGPS_PARAMETER_ANCHOR = 297409


@dataclass(frozen=True)
class PairRelationTrainingConfig:
    run_id: str
    mode: str = "full"
    hidden_dim: int = 64
    path_rank: int = 8
    epochs: int = 40
    batch_size: int = 128
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be smoke or full")
        if min(self.hidden_dim, self.path_rank, self.epochs, self.batch_size) <= 0:
            raise ValueError("model and training dimensions must be positive")
        if self.mode == "full" and (
            self.hidden_dim != 64
            or self.path_rank != 8
            or self.epochs != 40
            or self.batch_size != 128
            or self.dropout != 0.10
        ):
            raise ValueError(
                "E39 full mode freezes hidden64, rank8, epochs40, batch128, dropout0.10"
            )


@dataclass(frozen=True)
class PairRelationRowSpec:
    topology_mode: str
    label_mode: str
    seed: int

    @property
    def row_id(self) -> str:
        suffix = (
            "label_shuffle" if self.label_mode == "shuffled" else self.topology_mode
        )
        return f"pair_relation_reasoner_{suffix}_seed{self.seed}"


def pair_relation_training_matrix(
    config: PairRelationTrainingConfig,
) -> tuple[PairRelationRowSpec, ...]:
    if config.mode == "smoke":
        return (
            PairRelationRowSpec("true", "true", 0),
            PairRelationRowSpec("corrupted", "true", 0),
            PairRelationRowSpec("true", "shuffled", 0),
        )
    return (
        PairRelationRowSpec("true", "true", 0),
        PairRelationRowSpec("true", "true", 1),
        PairRelationRowSpec("true", "shuffled", 0),
    )


def pair_relation_fair_control_matrix() -> tuple[PairRelationRowSpec, ...]:
    return (
        PairRelationRowSpec("corrupted", "true", 0),
        PairRelationRowSpec("corrupted", "true", 1),
    )


def train_pair_relation_row(
    config: PairRelationTrainingConfig,
    row_spec: PairRelationRowSpec,
    data: dict[str, Any],
    *,
    processor_mode: str = "triangle",
) -> dict[str, Any]:
    _seed_everything(row_spec.seed)
    model_name = (
        "pair_relation_reasoner"
        if processor_mode == "triangle"
        else "pair_relation_no_triangle"
    )
    row_id = (
        row_spec.row_id
        if processor_mode == "triangle"
        else row_spec.row_id.replace("pair_relation_reasoner", model_name)
    )
    device = torch.device(config.device)
    model = SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(
            topology_mode=row_spec.topology_mode,
            processor_mode=processor_mode,
            hidden_dim=config.hidden_dim,
            path_rank=config.path_rank,
            dropout=config.dropout,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).to(device)
    train_arrays = _example_arrays(
        data["labels"], data["split_indices"]["train"], data["fit_cells"]
    )
    validation_arrays = _example_arrays(
        data["labels"],
        data["split_indices"]["train"],
        data["validation_cells"],
    )
    if row_spec.label_mode == "shuffled":
        combined = np.concatenate((train_arrays[-1], validation_arrays[-1])).copy()
        rng = np.random.default_rng(49000 + row_spec.seed)
        combined = combined[rng.permutation(len(combined))]
        train_arrays = (*train_arrays[:-1], combined[: len(train_arrays[-1])])
        validation_arrays = (
            *validation_arrays[:-1],
            combined[len(train_arrays[-1]) :],
        )
    generator = torch.Generator().manual_seed(59000 + row_spec.seed)
    loader = DataLoader(
        _tensor_dataset(train_arrays),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_rows += len(target)
        validation = _evaluate(
            model, validation_arrays, device, config.batch_size
        )
        history.append(
            {
                "row_id": row_id,
                "epoch": epoch,
                "train_loss": total_loss / total_rows,
                "validation_loss": validation["loss"],
                "validation_auc": validation["auc"],
            }
        )
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    evaluations = {
        split: _evaluate(
            model,
            _example_arrays(
                data["labels"], indices, np.argwhere(data["selected"])
            ),
            device,
            config.batch_size,
        )
        for split, indices in data["split_indices"].items()
    }
    result = {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_pair_relation_reasoner",
        "row_id": row_id,
        "model_name": model_name,
        "processor_mode": processor_mode,
        "topology_mode": row_spec.topology_mode,
        "label_mode": row_spec.label_mode,
        "seed": row_spec.seed,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_auc": evaluations["train"]["auc"],
        "unseen_sbox_auc": evaluations["unseen_sbox"]["auc"],
        "unseen_player_auc": evaluations["unseen_player"]["auc"],
        "dual_unseen_auc": evaluations["dual_unseen"]["auc"],
        "training_performed": True,
    }
    return {"result": result, "history": history, "state_dict": best_state}


def measure_pair_relation_contract(
    data: dict[str, Any],
    *,
    hidden_dim: int = 32,
    path_rank: int = 4,
    processor_mode: str = "triangle",
) -> dict[str, float | int | bool | list[int]]:
    cell_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(16)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled_players = node_permutation[data["players"][:, inverse]]
    relabeled_active = data["structure_active"][:, inverse]
    relabeled_masks = data["output_mask_bits"][:, inverse]
    true_spec = SmallSpnPairRelationSpec(
        topology_mode="true",
        processor_mode=processor_mode,
        hidden_dim=hidden_dim,
        path_rank=path_rank,
        dropout=0.0,
    )
    corrupted_spec = SmallSpnPairRelationSpec(
        topology_mode="corrupted",
        processor_mode=processor_mode,
        hidden_dim=hidden_dim,
        path_rank=path_rank,
        dropout=0.0,
    )
    torch.manual_seed(39001)
    original = _make_model(true_spec, data).eval()
    relabeled = SmallSpnPairRelationReasoner(
        true_spec,
        sboxes=data["sboxes"],
        players=relabeled_players,
        structure_active_bits=relabeled_active,
        output_mask_bits=relabeled_masks,
    ).eval()
    corrupted = _make_model(corrupted_spec, data).eval()
    _copy_parameters(original, relabeled)
    _copy_parameters(original, corrupted)
    variants = torch.arange(64, dtype=torch.long)
    rounds = variants % 4
    structures = variants % len(data["structure_active"])
    masks = (variants * 7) % len(data["output_mask_bits"])
    with torch.no_grad():
        initial, _ = original.build_initial_relation(
            variants, rounds, structures, masks
        )
        expected = original(variants, rounds, structures, masks)
        relabeled_output = relabeled(variants, rounds, structures, masks)
        corrupted_output = corrupted(variants, rounds, structures, masks)
    off_pair_influence = None
    if processor_mode == "local":
        torch.manual_seed(39002)
        relation = torch.randn(2, 16, 16, hidden_dim)
        changed = relation.clone()
        changed[0, 3, 11] += torch.randn(hidden_dim)
        with torch.no_grad():
            baseline_pairs = original.local_block(relation)
            changed_pairs = original.local_block(changed)
        pair_delta = torch.abs(baseline_pairs - changed_pairs)
        pair_delta[0, 3, 11] = 0.0
        off_pair_influence = float(pair_delta.max())
    counterpart_spec = SmallSpnPairRelationSpec(
        topology_mode="true",
        processor_mode="triangle" if processor_mode == "local" else "local",
        hidden_dim=hidden_dim,
        path_rank=path_rank,
        dropout=0.0,
    )
    counterpart = _make_model(counterpart_spec, data)
    corrupted_players = topology_players(data["players"], "corrupted")
    train_indices = data["split_indices"]["train"]
    heldout_indices = np.concatenate(
        [
            data["split_indices"]["unseen_player"],
            data["split_indices"]["dual_unseen"],
        ]
    )
    true_train = {tuple(data["players"][index]) for index in train_indices}
    corrupted_train = {tuple(corrupted_players[index]) for index in train_indices}
    corrupted_heldout = {
        tuple(corrupted_players[index]) for index in heldout_indices
    }
    return {
        "initial_pair_shape_matches": list(initial.shape) == [64, 16, 16, hidden_dim],
        "pair_count": int(initial.shape[1] * initial.shape[2]),
        "shared_triangle_block_count": sum(
            isinstance(module, PairTriangleBlock) for module in original.modules()
        ),
        "shared_local_block_count": sum(
            isinstance(module, PairLocalBlock) for module in original.modules()
        ),
        "processor_mode": processor_mode,
        "off_pair_influence_max_abs": off_pair_influence,
        "step_schedule": original.step_counts(torch.arange(4)).tolist(),
        "parameter_count": sum(
            parameter.numel() for parameter in original.parameters()
        ),
        "counterpart_parameter_count": sum(
            parameter.numel() for parameter in counterpart.parameters()
        ),
        "cell_relabeling_max_abs_logit_error": float(
            torch.max(torch.abs(expected - relabeled_output))
        ),
        "true_corrupted_max_abs_logit_difference": float(
            torch.max(torch.abs(expected - corrupted_output))
        ),
        "fair_control_heldout_avoids_true_train": corrupted_heldout.isdisjoint(
            true_train
        ),
        "fair_control_heldout_avoids_corrupted_train": corrupted_heldout.isdisjoint(
            corrupted_train
        ),
        "all_corrupted_players_are_permutations": all(
            np.array_equal(np.sort(row), np.arange(16))
            for row in corrupted_players
        ),
        "absolute_bit_cell_or_variant_embedding_absent": not any(
            keyword in name
            for name, _ in original.named_modules()
            for keyword in ("bit_embedding", "cell_embedding", "variant_embedding")
        ),
    }


def adjudicate_pair_relation_reasoner(
    config: PairRelationTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool | list[int]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    expected_rows = 3
    protocol = {
        **readiness,
        "expected_matrix_rows_present": len(rows) == expected_rows,
        "all_metrics_finite": all(
            math.isfinite(float(row[key])) for row in rows for key in metric_keys
        ),
        "all_rows_trained": all(row.get("training_performed") is True for row in rows),
        "all_rows_are_pair_relation_reasoner": all(
            row.get("model_name") == "pair_relation_reasoner" for row in rows
        ),
        "initial_pair_shape_matches": bool(contract["initial_pair_shape_matches"]),
        "pair_count_is_256": int(contract["pair_count"]) == 256,
        "one_shared_triangle_block": int(contract["shared_triangle_block_count"]) == 1,
        "step_schedule_is_2_3_4_5": contract["step_schedule"] == [2, 3, 4, 5],
        "parameter_count_at_most_graphgps": int(contract["parameter_count"])
        <= GRAPHGPS_PARAMETER_ANCHOR,
        "cell_relabeling_error_at_most_1e_6": float(
            contract["cell_relabeling_max_abs_logit_error"]
        )
        <= 1e-6,
        "true_corrupted_logit_difference_at_least_1e_5": float(
            contract["true_corrupted_max_abs_logit_difference"]
        )
        >= 1e-5,
        "fair_control_heldout_avoids_true_train": bool(
            contract["fair_control_heldout_avoids_true_train"]
        ),
        "fair_control_heldout_avoids_corrupted_train": bool(
            contract["fair_control_heldout_avoids_corrupted_train"]
        ),
        "all_corrupted_players_are_permutations": bool(
            contract["all_corrupted_players_are_permutations"]
        ),
        "absolute_ids_absent": bool(
            contract["absolute_bit_cell_or_variant_embedding_absent"]
        ),
    }
    if not all(protocol.values()):
        return _gate(
            config,
            "fail",
            "innovation2_small_spn_pair_relation_protocol_invalid",
            protocol,
            {},
            contract,
            "repair pair initialization, triangle update, invariance, source, or metric protocol",
        )
    if config.mode == "smoke":
        return _gate(
            config,
            "pass",
            "innovation2_small_spn_pair_relation_readiness_passed",
            protocol,
            {},
            contract,
            "run the frozen E39 Phase A true-seed0/1 plus label-shuffle screen",
        )

    true_rows = [
        row
        for row in rows
        if row["topology_mode"] == "true" and row["label_mode"] == "true"
    ]
    shuffle_rows = [row for row in rows if row["label_mode"] == "shuffled"]
    mean_auc = {
        split: float(np.mean([row[f"{split}_auc"] for row in true_rows]))
        for split in BASELINE_AUC
    }
    checks = {
        "label_shuffle_dual_auc_at_most_0p60": len(shuffle_rows) == 1
        and shuffle_rows[0]["dual_unseen_auc"] <= 0.60,
        **{
            f"each_seed_beats_{split}_baseline": all(
                row[f"{split}_auc"] > baseline for row in true_rows
            )
            for split, baseline in BASELINE_AUC.items()
        },
        "mean_dual_beats_baseline_by_0p03": mean_auc["dual_unseen"]
        >= BASELINE_AUC["dual_unseen"] + 0.03,
    }
    metrics = {
        "baseline_auc": BASELINE_AUC,
        "mean_auc": mean_auc,
        "dual_delta_vs_id_baseline": mean_auc["dual_unseen"]
        - BASELINE_AUC["dual_unseen"],
        "dual_delta_vs_graphgps_mean": mean_auc["dual_unseen"]
        - 0.6416819230939694,
        "dual_delta_vs_cett_mean": mean_auc["dual_unseen"]
        - 0.6296566195848206,
    }
    if not checks["label_shuffle_dual_auc_at_most_0p60"]:
        status = "hold"
        decision = "innovation2_small_spn_pair_relation_not_attributed"
        action = "repair or explain label-shuffle behavior before candidate selection"
    elif all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_pair_relation_candidate_screened"
        action = "run two-seed fair-corrupted P attribution for SPN-PRR only"
    else:
        status = "hold"
        decision = "innovation2_small_spn_pair_relation_reasoner_not_ready"
        action = "stop pair-model scaling and compare query-conditioned NBFNet with structured P-layer data"
    gate = _gate(config, status, decision, protocol, checks, contract, action)
    gate["metrics"] = metrics
    return gate


def adjudicate_pair_relation_attribution(
    config: PairRelationTrainingConfig,
    readiness: dict[str, bool],
    contract: dict[str, float | int | bool | list[int]],
    source_gate: dict[str, Any],
    true_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = (
        "best_validation_auc",
        "train_auc",
        "unseen_sbox_auc",
        "unseen_player_auc",
        "dual_unseen_auc",
    )
    protocol = {
        **readiness,
        "source_gate_is_candidate_screened": source_gate.get("decision")
        == "innovation2_small_spn_pair_relation_candidate_screened",
        "source_run_id_matches": source_gate.get("run_id")
        == "i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718",
        "two_true_source_rows_present": len(true_rows) == 2,
        "two_control_rows_present": len(control_rows) == 2,
        "true_source_rows_use_seeds_0_1": sorted(row["seed"] for row in true_rows)
        == [0, 1],
        "control_rows_use_seeds_0_1": sorted(row["seed"] for row in control_rows)
        == [0, 1],
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
            "innovation2_small_spn_pair_relation_attribution_protocol_invalid",
            protocol,
            {},
            contract,
            "repair Phase A ownership, fair control, seed, parameter, or metric protocol",
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
    checks = {
        "true_each_seed_beats_corrupted_dual": all(
            true_by_seed[seed]["dual_unseen_auc"]
            > control_by_seed[seed]["dual_unseen_auc"]
            for seed in (0, 1)
        ),
        "true_mean_dual_beats_corrupted_by_0p03": mean_auc["true"]["dual_unseen"]
        >= mean_auc["fair_corrupted"]["dual_unseen"] + 0.03,
    }
    metrics = {
        "mean_auc": mean_auc,
        "true_dual_delta_vs_fair_corrupted": mean_auc["true"]["dual_unseen"]
        - mean_auc["fair_corrupted"]["dual_unseen"],
        "per_seed_dual_delta": {
            str(seed): true_by_seed[seed]["dual_unseen_auc"]
            - control_by_seed[seed]["dual_unseen_auc"]
            for seed in (0, 1)
        },
    }
    if all(checks.values()):
        status = "pass"
        decision = "innovation2_small_spn_pair_relation_topology_confirmed"
        action = "run a same-budget no-triangle pair-state ablation before real-cipher transfer"
    else:
        status = "hold"
        decision = "innovation2_small_spn_pair_relation_topology_not_attributed"
        action = "stop SPN topology claims without increasing capacity, epochs, or scale"
    gate = _gate(config, status, decision, protocol, checks, contract, action)
    gate["metrics"] = metrics
    return gate


def _make_model(
    spec: SmallSpnPairRelationSpec, data: dict[str, Any]
) -> SmallSpnPairRelationReasoner:
    return SmallSpnPairRelationReasoner(
        spec,
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(source_parameters[name])


def _gate(
    config: PairRelationTrainingConfig,
    status: str,
    decision: str,
    protocol: dict[str, bool],
    checks: dict[str, bool],
    contract: dict[str, float | int | bool | list[int]],
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
            "fixed-budget low-rank pair-relation path reasoning on train-only selected "
            "labels from an expanded 16-bit synthetic SPN family; not real-cipher evidence"
        ),
        "next_action": {"action": action, "remote_scale": False},
    }
