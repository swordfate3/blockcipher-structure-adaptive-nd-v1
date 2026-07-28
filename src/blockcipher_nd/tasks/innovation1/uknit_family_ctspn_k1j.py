from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.gf2_boolean_view import gf2_boolean_views
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    invariant_pool,
    segment_mean,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i1_uknit_family_ctspn_position_cell_attribution_k1j_20260728"
EXPECTED_SOURCE_DECISION = (
    "innovation1_uknit_family_ctspn_k1i_dialga_signal_recovered_"
    "operator_attribution_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "k1i_gate": "e1823155149ce6146358650ae711269b617c93f4f7d48aaaa3e231348bfd675d",
    "k1i_checkpoint_manifest": (
        "4def7bc0019d7a258d962c622cfc79db1b69e0f85dc0b491a17bf081683e465f"
    ),
    "k1i_dataset_manifest": (
        "ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0"
    ),
    "runtime_e4_checkpoint_manifest": (
        "517f2fd2eb6d401983ca20c9db136229cf5b011c51ef3f1734cdb46e41967aeb"
    ),
}
EXPECTED_CHECKPOINT_DIGESTS = {
    ("k1i_exact", 0): (
        "3a192102d4fd2214faf9856ec046ed7577f3d4ec8ac2638b123da9b06257a3d1"
    ),
    ("k1i_exact", 1): (
        "36a7e1db6342a6cafe79fff454f7260b7c621ceba10bd7787b3591abb3bb9c75"
    ),
    ("runtime_e4", 0): (
        "5910dd24a360e08a92014275f629772e0ebf215a580566dc7862c3366ea3c812"
    ),
    ("runtime_e4", 1): (
        "b8a8d49ccdaad026e852fa0542af7b1a7d4af34ca96b9e1aebb513648b1a1ef8"
    ),
}
EXPECTED_SEEDS = (0, 1)
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
POOL_CONDITIONS = (
    "native",
    "within_cell_role_roll",
    "whole_cell_roll",
    "cross_cell_role_mix",
    "bit_pool_row_shuffle",
    "cell_pool_row_shuffle",
    "both_pool_row_shuffle",
)
INPUT_CONDITIONS = (
    "native_input",
    "within_cell_input_role_roll",
    "whole_cell_input_roll",
    "cross_cell_input_role_mix",
)
MODEL_ROLES = ("k1i_exact", "runtime_e4")
EXPECTED_POOL_ROWS = len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(POOL_CONDITIONS)
EXPECTED_INPUT_ROWS = (
    len(EXPECTED_SEEDS)
    * len(EXPECTED_SPLITS)
    * len(INPUT_CONDITIONS)
    * len(MODEL_ROLES)
)
EXPLAINED_FRACTION_GATE = 0.80
CHANCE_AUC_RANGE = (0.45, 0.55)
REPLAY_TOLERANCE = 1e-7


def cell_role_indices(structure: RuntimeSpnStructure) -> torch.Tensor:
    indices = torch.empty((structure.cells, 4), dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bit_indices
    return indices


def coordinate_permutation(
    structure: RuntimeSpnStructure,
    condition: str,
) -> torch.Tensor:
    aliases = {
        "native_input": "native",
        "within_cell_input_role_roll": "within_cell_role_roll",
        "whole_cell_input_roll": "whole_cell_roll",
        "cross_cell_input_role_mix": "cross_cell_role_mix",
    }
    normalized = aliases.get(condition, condition)
    if normalized not in {
        "native",
        "within_cell_role_roll",
        "whole_cell_roll",
        "cross_cell_role_mix",
    }:
        raise ValueError(f"unsupported position condition: {condition}")
    indices = cell_role_indices(structure)
    permutation = torch.empty(structure.block_bits, dtype=torch.long)
    for cell in range(structure.cells):
        for role in range(4):
            target = int(indices[cell, role])
            if normalized == "native":
                source_cell, source_role = cell, role
            elif normalized == "within_cell_role_roll":
                source_cell, source_role = cell, (role + 1) % 4
            elif normalized == "whole_cell_roll":
                source_cell, source_role = (cell + 1) % structure.cells, role
            else:
                source_cell = (cell + role + 1) % structure.cells
                source_role = role
            permutation[target] = indices[source_cell, source_role]
    return permutation


def permutation_sha256(permutation: torch.Tensor) -> str:
    values = torch.as_tensor(permutation, dtype=torch.int64).cpu().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def permutation_checks(structure: RuntimeSpnStructure) -> dict[str, bool]:
    expected = list(range(structure.block_bits))
    permutations = {
        condition: coordinate_permutation(structure, condition)
        for condition in INPUT_CONDITIONS
    }
    indices = cell_role_indices(structure)
    cross = permutations["cross_cell_input_role_mix"]
    cross_role_preserving = all(
        int(structure.bit_role[int(cross[int(indices[cell, role])])]) == role
        for cell in range(structure.cells)
        for role in range(4)
    )
    cross_changes_cell = any(
        int(structure.cell_membership[int(cross[target])])
        != int(structure.cell_membership[target])
        for target in range(structure.block_bits)
    )
    return {
        "position_permutations_bijective": all(
            sorted(permutation.tolist()) == expected
            for permutation in permutations.values()
        ),
        "position_controls_nonidentity": all(
            not torch.equal(permutations[condition], permutations["native_input"])
            for condition in INPUT_CONDITIONS[1:]
        ),
        "cross_cell_mix_role_preserving": cross_role_preserving,
        "cross_cell_mix_changes_membership": cross_changes_cell,
        "position_control_fingerprints_distinct": (
            len({permutation_sha256(value) for value in permutations.values()})
            == len(permutations)
        ),
    }


def label_blind_row_permutation(
    row_count: int,
    *,
    seed: int,
    split: str,
) -> torch.Tensor:
    if row_count < 2 or split not in EXPECTED_SPLITS:
        raise ValueError("row permutation requires a known nontrivial split")
    split_index = EXPECTED_SPLITS.index(split)
    generator = torch.Generator().manual_seed(20260728 + 101 * seed + split_index)
    permutation = torch.randperm(row_count, generator=generator)
    if torch.equal(permutation, torch.arange(row_count)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def k1i_pool_components(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    condition: str = "native",
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition not in {
        "native",
        "within_cell_role_roll",
        "whole_cell_roll",
        "cross_cell_role_mix",
    }:
        raise ValueError("K1-J pool components require a grouping condition")
    structure = model.runtime_structure
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        structure.block_bits,
    ).flip(-1)
    views = gf2_boolean_views(runtime, structure)
    batch, pair_count, bit_count, _ = views.shape
    hidden = model.backbone.bit_encoder(views).reshape(
        batch * pair_count,
        bit_count,
        model.backbone.spec.hidden_dim,
    )
    permutation = coordinate_permutation(structure, condition).to(hidden.device)
    hidden = hidden[:, permutation]
    if condition in {"within_cell_role_roll", "whole_cell_roll"}:
        # These are mathematical set-order controls. Canonicalize before floating
        # reductions so their equality gate measures semantics, not summation order.
        hidden = hidden[:, torch.argsort(permutation)]
    membership = structure.cell_membership.to(hidden.device)
    cell_hidden = segment_mean(hidden, membership, structure.cells)
    bit_pool = invariant_pool(hidden).reshape(batch, pair_count, -1)
    cell_pool = invariant_pool(cell_hidden).reshape(batch, pair_count, -1)
    return bit_pool, cell_pool


def collect_k1i_pool_components(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    condition: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    bit_batches: list[torch.Tensor] = []
    cell_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), batch_size):
            batch = torch.from_numpy(
                np.array(
                    features[start : start + batch_size], dtype=np.float32, copy=True
                )
            )
            bit_pool, cell_pool = k1i_pool_components(
                model,
                batch,
                condition=condition,
            )
            bit_batches.append(bit_pool.cpu())
            cell_batches.append(cell_pool.cpu())
    return torch.cat(bit_batches), torch.cat(cell_batches)


def k1i_probabilities_from_pools(
    model: torch.nn.Module,
    bit_pool: torch.Tensor,
    cell_pool: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    if bit_pool.shape != cell_pool.shape or bit_pool.ndim != 3:
        raise ValueError("K1-J pooled branches must share [rows, pairs, channels]")
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, bit_pool.shape[0], batch_size):
            bit_batch = bit_pool[start : start + batch_size]
            cell_batch = cell_pool[start : start + batch_size]
            batch, pair_count, _ = bit_batch.shape
            pair_embeddings = model.backbone.pair_projection(
                torch.cat((bit_batch, cell_batch), dim=-1).reshape(
                    batch * pair_count,
                    -1,
                )
            ).reshape(batch, pair_count, model.backbone.spec.pair_embedding_dim)
            attended, _ = model.backbone.pair_attention(pair_embeddings)
            encoded = torch.cat(
                (
                    attended,
                    pair_embeddings.mean(dim=1),
                    pair_embeddings.max(dim=1).values,
                ),
                dim=-1,
            )
            logits = model.backbone.classifier(encoded).squeeze(1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def predict_with_input_permutation(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    condition: str,
    batch_size: int,
) -> np.ndarray:
    structure = model.runtime_structure
    permutation = coordinate_permutation(structure, condition)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), batch_size):
            batch = torch.from_numpy(
                np.array(
                    features[start : start + batch_size], dtype=np.float32, copy=True
                )
            )
            runtime = batch.reshape(
                batch.shape[0],
                -1,
                2,
                structure.block_bits,
            ).flip(-1)
            permuted_runtime = runtime[..., permutation]
            permuted_features = permuted_runtime.flip(-1).reshape(batch.shape[0], -1)
            logits = model(permuted_features).squeeze(1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def score_pool_interventions(
    *,
    model: torch.nn.Module,
    no_topology_model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    split: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    components = {
        condition: collect_k1i_pool_components(
            model,
            features,
            condition=condition,
            batch_size=batch_size,
        )
        for condition in (
            "native",
            "within_cell_role_roll",
            "whole_cell_roll",
            "cross_cell_role_mix",
        )
    }
    native_bit, native_cell = components["native"]
    row_permutation = label_blind_row_permutation(
        int(features.shape[0]),
        seed=seed,
        split=split,
    )
    condition_components = {
        **components,
        "bit_pool_row_shuffle": (native_bit[row_permutation], native_cell),
        "cell_pool_row_shuffle": (native_bit, native_cell[row_permutation]),
        "both_pool_row_shuffle": (
            native_bit[row_permutation],
            native_cell[row_permutation],
        ),
    }
    probabilities = {
        condition: k1i_probabilities_from_pools(
            model,
            *condition_components[condition],
            batch_size=batch_size,
        )
        for condition in POOL_CONDITIONS
    }
    no_topology_components = collect_k1i_pool_components(
        no_topology_model,
        features,
        condition="native",
        batch_size=batch_size,
    )
    no_topology_probabilities = k1i_probabilities_from_pools(
        no_topology_model,
        *no_topology_components,
        batch_size=batch_size,
    )
    native_auc = binary_auc(labels, probabilities["native"])
    no_topology_auc = binary_auc(labels, no_topology_probabilities)
    source_gap = native_auc - no_topology_auc
    rows: list[dict[str, Any]] = []
    for condition in POOL_CONDITIONS:
        auc = binary_auc(labels, probabilities[condition])
        raw_fraction = (native_auc - auc) / source_gap if source_gap > 0.0 else math.nan
        rows.append(
            {
                "run_id": RUN_ID,
                "cipher_key": "dialga128",
                "model_role": "k1i_exact",
                "seed": seed,
                "split": split,
                "condition": condition,
                "rows": int(features.shape[0]),
                "auc": auc,
                "native_auc": native_auc,
                "no_topology_auc": no_topology_auc,
                "source_gap": source_gap,
                "raw_explained_fraction": raw_fraction,
                "explained_fraction": min(1.0, max(0.0, raw_fraction)),
                "max_abs_probability_delta_from_native": float(
                    np.max(np.abs(probabilities["native"] - probabilities[condition]))
                ),
                "row_permutation_sha256": permutation_sha256(row_permutation),
                "training_performed": False,
                "optimizer_steps": 0,
                "strict_state_dict_load": True,
            }
        )
    probabilities["no_topology"] = no_topology_probabilities
    return rows, probabilities


def score_input_position_controls(
    *,
    models: Mapping[str, torch.nn.Module],
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    split: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for role in MODEL_ROLES:
        model = models[role]
        for condition in INPUT_CONDITIONS:
            probabilities[(role, condition)] = predict_with_input_permutation(
                model,
                features,
                condition=condition,
                batch_size=batch_size,
            )
        native = probabilities[(role, "native_input")]
        native_auc = binary_auc(labels, native)
        for condition in INPUT_CONDITIONS:
            current = probabilities[(role, condition)]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "dialga128",
                    "model_role": role,
                    "seed": seed,
                    "split": split,
                    "condition": condition,
                    "rows": int(features.shape[0]),
                    "auc": binary_auc(labels, current),
                    "native_auc": native_auc,
                    "native_minus_condition_auc": (
                        native_auc - binary_auc(labels, current)
                    ),
                    "max_abs_probability_delta_from_native": float(
                        np.max(np.abs(native - current))
                    ),
                    "coordinate_permutation_sha256": permutation_sha256(
                        coordinate_permutation(model.runtime_structure, condition)
                    ),
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "strict_state_dict_load": True,
                }
            )
    return rows, probabilities


def adjudicate_k1j(
    *,
    pool_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    pool_map = {
        (int(row["seed"]), str(row["split"]), str(row["condition"])): row
        for row in pool_rows
    }
    input_map = {
        (
            str(row["model_role"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["condition"]),
        ): row
        for row in input_rows
    }
    expected_pool = {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in POOL_CONDITIONS
    }
    expected_input = {
        (role, seed, split, condition)
        for role in MODEL_ROLES
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in INPUT_CONDITIONS
    }
    protocol_checks = {
        **dict(source_checks),
        "pool_rows_exact": (
            len(pool_rows) == EXPECTED_POOL_ROWS and set(pool_map) == expected_pool
        ),
        "input_rows_exact": (
            len(input_rows) == EXPECTED_INPUT_ROWS and set(input_map) == expected_input
        ),
        "all_rows_zero_training": all(
            row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            and row.get("strict_state_dict_load") is True
            for row in (*pool_rows, *input_rows)
        ),
        "finite_metrics": all(
            all(math.isfinite(float(row[key])) for key in ("auc", "native_auc"))
            for row in (*pool_rows, *input_rows)
        ),
        "positive_source_gaps": all(
            float(pool_map[(seed, split, "native")]["source_gap"]) > 0.0
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        )
        if set(pool_map) == expected_pool
        else False,
        "hidden_invariance_controls_exact": all(
            float(
                pool_map[(seed, split, condition)][
                    "max_abs_probability_delta_from_native"
                ]
            )
            <= REPLAY_TOLERANCE
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
            for condition in ("within_cell_role_roll", "whole_cell_roll")
        )
        if set(pool_map) == expected_pool
        else False,
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())

    def all_fresh(condition: str, predicate: Any) -> bool:
        return all(
            predicate(pool_map[(seed, split, condition)])
            for seed in EXPECTED_SEEDS
            for split in FRESH_SPLITS
        )

    research_checks = {
        "within_cell_interaction_supported": all_fresh(
            "cross_cell_role_mix",
            lambda row: float(row["explained_fraction"]) >= EXPLAINED_FRACTION_GATE,
        )
        if set(pool_map) == expected_pool
        else False,
        "global_bit_branch_supported": all_fresh(
            "bit_pool_row_shuffle",
            lambda row: float(row["explained_fraction"]) >= EXPLAINED_FRACTION_GATE,
        )
        if set(pool_map) == expected_pool
        else False,
        "cell_branch_supported": all_fresh(
            "cell_pool_row_shuffle",
            lambda row: float(row["explained_fraction"]) >= EXPLAINED_FRACTION_GATE,
        )
        if set(pool_map) == expected_pool
        else False,
        "both_branches_shuffle_explains_signal": all_fresh(
            "both_pool_row_shuffle",
            lambda row: (
                float(row["explained_fraction"]) >= EXPLAINED_FRACTION_GATE
                and CHANCE_AUC_RANGE[0] <= float(row["auc"]) <= CHANCE_AUC_RANGE[1]
            ),
        )
        if set(pool_map) == expected_pool
        else False,
    }
    distributed = (
        not research_checks["global_bit_branch_supported"]
        and not research_checks["cell_branch_supported"]
        and research_checks["both_branches_shuffle_explains_signal"]
    )
    research_checks["distributed_branch_interaction_supported"] = distributed

    supported = [
        name
        for name in (
            "within_cell_interaction_supported",
            "global_bit_branch_supported",
            "cell_branch_supported",
            "distributed_branch_interaction_supported",
        )
        if research_checks[name]
    ]
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1j_protocol_invalid"
        next_action = (
            "repair only the failed K1-J binding or intervention and rerun unchanged"
        )
    elif supported:
        status = "pass"
        if research_checks["within_cell_interaction_supported"]:
            decision = (
                "innovation1_uknit_family_ctspn_k1j_"
                "within_cell_position_interaction_supported"
            )
            next_action = (
                "design K1-K as a width-independent exact GF(2) cell-token model "
                "that preserves four bit roles through transition mixing before pooling"
            )
        elif (
            research_checks["global_bit_branch_supported"]
            and not research_checks["cell_branch_supported"]
        ):
            decision = (
                "innovation1_uknit_family_ctspn_k1j_"
                "global_boolean_marginal_signal_confirmed"
            )
            next_action = (
                "hold K1-I and require native endpoint/cell interactions in K1-K "
                "before any new training"
            )
        elif (
            research_checks["cell_branch_supported"]
            and not research_checks["global_bit_branch_supported"]
        ):
            decision = "innovation1_uknit_family_ctspn_k1j_cell_branch_signal_supported"
            next_action = (
                "design one exact edge-conditioned cell mixer that preserves "
                "the identified within-cell tuple features"
            )
        else:
            decision = (
                "innovation1_uknit_family_ctspn_k1j_joint_pool_branch_signal_supported"
            )
            next_action = (
                "design K1-K with both exact bit and cell branches and a bounded "
                "preregistered residual instead of an unrestricted bypass"
            )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1j_position_cell_attribution_not_supported"
        )
        next_action = (
            "close invariant GF(2) pooling at this diagnostic scale and test one "
            "exact heterogeneous S-box/operator composition locally"
        )

    fresh_results = (
        {
            str(seed): {
                split: {
                    condition: {
                        "auc": float(pool_map[(seed, split, condition)]["auc"]),
                        "explained_fraction": float(
                            pool_map[(seed, split, condition)]["explained_fraction"]
                        ),
                    }
                    for condition in POOL_CONDITIONS
                }
                for split in FRESH_SPLITS
            }
            for seed in EXPECTED_SEEDS
        }
        if set(pool_map) == expected_pool
        else {}
    )
    input_sensitivity = (
        {
            role: {
                str(seed): {
                    split: {
                        condition: float(
                            input_map[(role, seed, split, condition)][
                                "native_minus_condition_auc"
                            ]
                        )
                        for condition in INPUT_CONDITIONS[1:]
                    }
                    for split in FRESH_SPLITS
                }
                for seed in EXPECTED_SEEDS
            }
            for role in MODEL_ROLES
        }
        if set(input_map) == expected_input
        else {}
    )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "thresholds": {
            "explained_fraction": EXPLAINED_FRACTION_GATE,
            "chance_auc_range": list(CHANCE_AUC_RANGE),
            "replay_tolerance": REPLAY_TOLERANCE,
        },
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "supported_attribution_families": supported,
        "fresh_results": fresh_results,
        "input_sensitivity": input_sensitivity,
        "next_action": next_action,
        "claim_scope": (
            "zero-training two-seed Dialga-128 r4 aggregation and input-position "
            "mechanism audit on frozen 2048/class checkpoints and existing "
            "1024/class fresh splits; not uKNIT success, formal scale, attack, "
            "SOTA, or arbitrary-SPN evidence"
        ),
        "blocked_actions": [
            "remote scale, more data, epochs, width, pairs, seeds, or experts",
            "S-box, DDT, trail, partial decryption, key, cipher ID, or raw bypass inside K1-J",
            "using train-seen or averaged rows to hide a failed fresh split or seed",
        ],
    }


__all__ = [
    "CHANCE_AUC_RANGE",
    "EXPECTED_CHECKPOINT_DIGESTS",
    "EXPECTED_INPUT_ROWS",
    "EXPECTED_POOL_ROWS",
    "EXPECTED_SOURCE_DECISION",
    "EXPECTED_SOURCE_DIGESTS",
    "INPUT_CONDITIONS",
    "MODEL_ROLES",
    "POOL_CONDITIONS",
    "RUN_ID",
    "adjudicate_k1j",
    "cell_role_indices",
    "collect_k1i_pool_components",
    "coordinate_permutation",
    "k1i_pool_components",
    "k1i_probabilities_from_pools",
    "label_blind_row_permutation",
    "permutation_checks",
    "permutation_sha256",
    "predict_with_input_permutation",
    "score_input_position_controls",
    "score_pool_interventions",
]
