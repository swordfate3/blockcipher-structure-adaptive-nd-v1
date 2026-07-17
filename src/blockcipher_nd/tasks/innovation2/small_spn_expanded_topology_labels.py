from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    AUDIT_ROUNDS,
    KEY_EXPANSION_SEED,
    MASK_COUNT,
    MASK_SEED,
    MASTER_KEYS,
    PLAYER_SEED,
    SBOX_SEED,
    STATE_BITS,
    CoordinateStructure,
    SmallSpnVariant,
    _binary_auc,
    _parity16,
    encrypt_scalar,
    encrypt_words,
    enumerate_structure_points,
    make_output_masks,
    make_player_family,
    make_players,
    make_round_keys,
    make_sboxes,
    make_structures,
)


EXPANDED_PLAYERS = 16
TRAIN_SBOXES = 3
TRAIN_PLAYERS = 12
SELECTED_MIN_POSITIVES = 9
SELECTED_MAX_POSITIVES = 27
EXPECTED_E32_PLAYERS = (
    (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15),
    (6, 11, 5, 14, 12, 8, 10, 0, 1, 7, 15, 3, 2, 4, 13, 9),
    (3, 7, 6, 14, 8, 13, 5, 15, 0, 2, 11, 4, 1, 9, 12, 10),
    (3, 0, 11, 9, 8, 13, 1, 5, 10, 14, 7, 12, 2, 15, 6, 4),
)
MINIMUM_WIDTHS = {
    "selected_base_cells": 256,
    "train_positive": 3000,
    "train_negative": 3000,
    "unseen_sbox_positive": 768,
    "unseen_sbox_negative": 768,
    "unseen_player_positive": 768,
    "unseen_player_negative": 768,
    "dual_unseen_positive": 192,
    "dual_unseen_negative": 192,
    "distinct_topology_patterns": 128,
    "supported_rounds": 2,
}
MINIMUM_TOPOLOGY_FRACTIONS = {
    "train_p_sensitive_any_s": 0.75,
    "train_p_sensitive_all_s": 0.20,
    "heldout_p_novel_any_s": 0.50,
    "dual_p_effect_cells": 0.40,
    "train_interaction_cells": 0.50,
    "full_interaction_cells": 0.70,
}
MINIMUM_DUAL_EFFECT_ROWS = {
    "dual_p_effect_positive_rows": 192,
    "dual_p_effect_negative_rows": 192,
}
MAXIMUM_MARGINAL_AUC = {
    "unseen_sbox": 0.80,
    "unseen_player": 0.80,
    "dual_unseen": 0.75,
}
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ExpandedTopologyAuditConfig:
    run_id: str
    mode: str = "audit"
    sbox_variants: int = 4
    player_variants: int = EXPANDED_PLAYERS
    rounds: tuple[int, ...] = AUDIT_ROUNDS
    keys: int = MASTER_KEYS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.sbox_variants <= 0 or self.sbox_variants > 4:
            raise ValueError("sbox_variants must be between 1 and 4")
        if self.player_variants <= 0 or self.player_variants > EXPANDED_PLAYERS:
            raise ValueError("player_variants must be between 1 and 16")
        if not self.rounds or any(rounds <= 0 for rounds in self.rounds):
            raise ValueError("rounds must be positive")
        if self.keys <= 0 or self.keys > MASTER_KEYS:
            raise ValueError("keys must be between 1 and 256")
        if self.mode == "audit" and (
            self.sbox_variants != 4
            or self.player_variants != EXPANDED_PLAYERS
            or self.rounds != AUDIT_ROUNDS
            or self.keys != MASTER_KEYS
        ):
            raise ValueError(
                "E37 audit freezes 4x16 variants, rounds 2..5, and all 256 keys"
            )


def make_expanded_variants(
    config: ExpandedTopologyAuditConfig,
) -> tuple[SmallSpnVariant, ...]:
    sboxes = make_sboxes()[: config.sbox_variants]
    players = make_player_family(config.player_variants)
    return tuple(
        SmallSpnVariant(
            variant_id=f"s{sbox_id}_p{player_id}",
            sbox_id=sbox_id,
            player_id=player_id,
            sbox=sbox,
            player=player,
        )
        for sbox_id, sbox in enumerate(sboxes)
        for player_id, player in enumerate(players)
    )


def expanded_split_indices(
    variants: tuple[SmallSpnVariant, ...],
) -> dict[str, np.ndarray]:
    groups = {
        "train": [
            index
            for index, variant in enumerate(variants)
            if variant.sbox_id < TRAIN_SBOXES
            and variant.player_id < TRAIN_PLAYERS
        ],
        "unseen_sbox": [
            index
            for index, variant in enumerate(variants)
            if variant.sbox_id == TRAIN_SBOXES
            and variant.player_id < TRAIN_PLAYERS
        ],
        "unseen_player": [
            index
            for index, variant in enumerate(variants)
            if variant.sbox_id < TRAIN_SBOXES
            and variant.player_id >= TRAIN_PLAYERS
        ],
        "dual_unseen": [
            index
            for index, variant in enumerate(variants)
            if variant.sbox_id == TRAIN_SBOXES
            and variant.player_id >= TRAIN_PLAYERS
        ],
    }
    return {
        name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()
    }


def select_expanded_train_cells(labels: np.ndarray) -> np.ndarray:
    cube = _label_cube(labels)
    positive_count = cube[:TRAIN_SBOXES, :TRAIN_PLAYERS].sum(axis=(0, 1))
    return (positive_count >= SELECTED_MIN_POSITIVES) & (
        positive_count <= SELECTED_MAX_POSITIVES
    )


def fair_corrupt_player(player: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((((target // 4 + 1) % 4) * 4 + target % 4) for target in player)


def fair_control_contract(players: tuple[tuple[int, ...], ...]) -> dict[str, bool]:
    corrupted = tuple(fair_corrupt_player(player) for player in players)
    train_true = set(players[:TRAIN_PLAYERS])
    train_corrupted = set(corrupted[:TRAIN_PLAYERS])
    heldout_corrupted = set(corrupted[TRAIN_PLAYERS:])
    return {
        "all_corrupted_players_are_bijections": all(
            sorted(player) == list(range(STATE_BITS)) for player in corrupted
        ),
        "corrupted_players_are_unique": len(set(corrupted)) == len(corrupted),
        "heldout_corrupted_not_true_train": heldout_corrupted.isdisjoint(train_true),
        "heldout_corrupted_not_corrupted_train": heldout_corrupted.isdisjoint(
            train_corrupted
        ),
        "destination_lane_is_preserved": all(
            corrupted_target % 4 == true_target % 4
            for player, corrupted_player in zip(players, corrupted, strict=True)
            for true_target, corrupted_target in zip(
                player, corrupted_player, strict=True
            )
        ),
    }


def run_cached_expanded_labels(
    config: ExpandedTopologyAuditConfig,
    *,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    variants = make_expanded_variants(config)
    structures = make_structures()
    masks = make_output_masks()
    metadata = _metadata(config, variants, structures, masks)
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / "cache_metadata.json"
    parity_path = cache_root / "parity_words.npy"
    labels_path = cache_root / "labels.npy"
    completed_path = cache_root / "completed.npy"
    required = (metadata_path, parity_path, labels_path, completed_path)
    expected_shape = (len(variants), len(config.rounds), len(structures))

    if any(path.exists() for path in required):
        if not all(path.exists() for path in required):
            raise ValueError("partial E37 cache is missing required files")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("E37 cache metadata does not match config")
        parity = np.load(parity_path, mmap_mode="r+")
        labels = np.load(labels_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if parity.shape != (*expected_shape, config.keys) or parity.dtype != np.uint16:
            raise ValueError("E37 parity cache shape or dtype mismatch")
        if labels.shape != (*expected_shape, len(masks)) or labels.dtype != np.bool_:
            raise ValueError("E37 label cache shape or dtype mismatch")
        if completed.shape != expected_shape or completed.dtype != np.bool_:
            raise ValueError("E37 completion cache shape or dtype mismatch")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        parity = np.lib.format.open_memmap(
            parity_path,
            mode="w+",
            dtype=np.uint16,
            shape=(*expected_shape, config.keys),
        )
        labels = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.bool_,
            shape=(*expected_shape, len(masks)),
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=expected_shape
        )
        parity[:] = 0
        labels[:] = False
        completed[:] = False
        parity.flush()
        labels.flush()
        completed.flush()

    generated_blocks = 0
    masks_array = np.asarray(masks, dtype=np.uint16)
    for variant_index, variant in enumerate(variants):
        for round_index, rounds in enumerate(config.rounds):
            round_keys = make_round_keys(rounds=rounds, key_count=config.keys)
            for structure_index, structure in enumerate(structures):
                if completed[variant_index, round_index, structure_index]:
                    continue
                _emit(
                    progress_callback,
                    "expanded_label_block_start",
                    {
                        "variant_id": variant.variant_id,
                        "rounds": rounds,
                        "structure_id": structure.structure_id,
                    },
                )
                ciphertexts = encrypt_words(
                    enumerate_structure_points(structure), round_keys, variant
                )
                output_xor = np.bitwise_xor.reduce(ciphertexts, axis=1)
                block_labels = np.all(
                    _parity16(output_xor[:, None] & masks_array[None, :]) == 0,
                    axis=0,
                )
                parity[variant_index, round_index, structure_index] = output_xor
                labels[variant_index, round_index, structure_index] = block_labels
                parity.flush()
                labels.flush()
                completed[variant_index, round_index, structure_index] = True
                completed.flush()
                generated_blocks += 1
                _emit(
                    progress_callback,
                    "expanded_label_block_done",
                    {
                        "variant_id": variant.variant_id,
                        "rounds": rounds,
                        "structure_id": structure.structure_id,
                        "positive_masks": int(block_labels.sum()),
                    },
                )
    return {
        "variants": variants,
        "structures": structures,
        "masks": masks,
        "parity_words": np.asarray(parity).copy(),
        "labels": np.asarray(labels).copy(),
        "completed": np.asarray(completed).copy(),
        "generated_blocks": generated_blocks,
        "metadata": metadata,
    }


def evaluate_expanded_labels(
    config: ExpandedTopologyAuditConfig,
    cache: dict[str, Any],
    *,
    resume_generated_blocks: int,
) -> dict[str, Any]:
    variants: tuple[SmallSpnVariant, ...] = cache["variants"]
    labels = np.asarray(cache["labels"], dtype=np.bool_)
    parity = np.asarray(cache["parity_words"], dtype=np.uint16)
    masks = np.asarray(cache["masks"], dtype=np.uint16)
    selected = select_expanded_train_cells(labels)
    cube = _label_cube(labels)
    split_arrays = _split_arrays(cube)
    split_metrics = {
        name: _label_counts(values[:, selected])
        for name, values in split_arrays.items()
    }
    marginal_baselines = _marginal_baselines(cube, selected)
    topology_metrics = _topology_metrics(cube, selected)
    train_positive_count = cube[:TRAIN_SBOXES, :TRAIN_PLAYERS].sum(axis=(0, 1))
    patterns = cube[:, :, selected].transpose(2, 0, 1).reshape(int(selected.sum()), -1)
    distinct_patterns = len(
        {np.packbits(pattern, bitorder="little").tobytes() for pattern in patterns}
    )
    per_round_selected = [int(selected[index].sum()) for index in range(len(config.rounds))]
    supported_rounds = sum(value >= 32 for value in per_round_selected)
    recomputed = np.all(
        _parity16(parity[..., None] & masks[None, None, None, None, :]) == 0,
        axis=3,
    )
    players = make_player_family(EXPANDED_PLAYERS)
    control_checks = fair_control_contract(players)
    split_indices = expanded_split_indices(variants)
    readiness = {
        "sixty_four_unique_bijective_variants": len(variants) == 64
        and len({(variant.sbox, variant.player) for variant in variants}) == 64
        and all(sorted(variant.sbox) == list(range(16)) for variant in variants)
        and all(sorted(variant.player) == list(range(16)) for variant in variants),
        "first_four_players_match_e32": players[:4] == EXPECTED_E32_PLAYERS
        and make_players() == EXPECTED_E32_PLAYERS,
        "variant_order_is_sbox_then_player": all(
            variant.sbox_id == index // EXPANDED_PLAYERS
            and variant.player_id == index % EXPANDED_PLAYERS
            for index, variant in enumerate(variants)
        ),
        "split_sizes_are_36_12_12_4": [
            len(split_indices[name])
            for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
        ]
        == [36, 12, 12, 4],
        "twelve_independent_train_players": len(
            {variants[index].player for index in split_indices["train"]}
        )
        == TRAIN_PLAYERS,
        "fourteen_structures_and_sixty_four_masks": len(cache["structures"]) == 14
        and len(masks) == MASK_COUNT
        and len(set(int(mask) for mask in masks)) == MASK_COUNT,
        "all_master_keys_covered": config.keys == MASTER_KEYS,
        "all_blocks_completed": bool(np.asarray(cache["completed"]).all()),
        "resume_generates_zero_blocks": resume_generated_blocks == 0,
        "labels_recompute_from_parity_words": bool(np.array_equal(labels, recomputed)),
        "scalar_and_vectorized_cipher_match": expanded_scalar_vectorized_matches(),
        "label_shape_is_64x4x14x64": labels.shape == (64, 4, 14, 64),
        "selection_matches_frozen_train_counts": bool(
            np.array_equal(
                selected,
                (train_positive_count >= SELECTED_MIN_POSITIVES)
                & (train_positive_count <= SELECTED_MAX_POSITIVES),
            )
        ),
        **control_checks,
    }
    metrics = {
        "selected_base_cells": int(selected.sum()),
        "selected_total_label_rows": int(
            sum(values["total"] for values in split_metrics.values())
        ),
        "distinct_topology_patterns": distinct_patterns,
        "supported_rounds": supported_rounds,
        "per_round_selected_cells": per_round_selected,
        "split_metrics": split_metrics,
        "marginal_baselines": marginal_baselines,
        "topology": topology_metrics,
    }
    gate = adjudicate_expanded_topology(
        run_id=config.run_id, readiness=readiness, metrics=metrics
    )
    selected_rows = _selected_rows(
        selected,
        train_positive_count,
        config.rounds,
        cache["structures"],
        cache["masks"],
    )
    rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_small_spn_expanded_topology_benchmark",
            "split": split,
            **split_metrics[split],
            "strongest_marginal_auc": marginal_baselines.get(split, {}).get(
                "strongest_auc"
            ),
            "selected_base_cells": int(selected.sum()),
            "training_performed": False,
        }
        for split in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ]
    return {
        "rows": rows,
        "selected_rows": selected_rows,
        "selected_mask": selected,
        "gate": gate,
        "metadata": {
            **cache["metadata"],
            "selection_rule": "9 <= positives <= 27 over 36 train topologies",
            "heldout_labels_used_for_selection": False,
            "heldout_labels_used_for_audit": True,
            "fair_control": "fixed destination-cell rotation composed per P-layer",
            "claim_scope": gate["claim_scope"],
        },
        "summary": {
            "run_id": config.run_id,
            "gate": gate,
            "split_metrics": split_metrics,
            "marginal_baselines": marginal_baselines,
        },
    }


def adjudicate_expanded_topology(
    *, run_id: str, readiness: dict[str, bool], metrics: dict[str, Any]
) -> dict[str, Any]:
    splits = metrics["split_metrics"]
    topology = metrics["topology"]
    width_values = {
        "selected_base_cells": metrics["selected_base_cells"],
        "train_positive": splits["train"]["positive"],
        "train_negative": splits["train"]["negative"],
        "unseen_sbox_positive": splits["unseen_sbox"]["positive"],
        "unseen_sbox_negative": splits["unseen_sbox"]["negative"],
        "unseen_player_positive": splits["unseen_player"]["positive"],
        "unseen_player_negative": splits["unseen_player"]["negative"],
        "dual_unseen_positive": splits["dual_unseen"]["positive"],
        "dual_unseen_negative": splits["dual_unseen"]["negative"],
        "distinct_topology_patterns": metrics["distinct_topology_patterns"],
        "supported_rounds": metrics["supported_rounds"],
    }
    width_checks = {
        f"{key}_at_least_{minimum}": int(width_values[key]) >= minimum
        for key, minimum in MINIMUM_WIDTHS.items()
    }
    topology_checks = {
        f"{key}_fraction_at_least_{minimum:g}": float(
            topology["fractions"][key]
        )
        >= minimum
        for key, minimum in MINIMUM_TOPOLOGY_FRACTIONS.items()
    }
    topology_checks.update(
        {
            f"{key}_at_least_{minimum}": int(topology["counts"][key]) >= minimum
            for key, minimum in MINIMUM_DUAL_EFFECT_ROWS.items()
        }
    )
    shortcut_checks = {
        f"{split}_strongest_marginal_auc_at_most_{limit:g}": float(
            metrics["marginal_baselines"][split]["strongest_auc"]
        )
        <= limit
        for split, limit in MAXIMUM_MARGINAL_AUC.items()
    }
    if not readiness or not all(readiness.values()):
        status = "fail"
        decision = "innovation2_small_spn_expanded_topology_protocol_invalid"
        action = "repair exact-label cache, frozen split, train-only selection, or fair topology control"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_expanded_topology_benchmark_not_ready"
        action = "stop this random P-layer benchmark and redesign the topology or target family"
    elif not all(topology_checks.values()) or not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_expanded_topology_benchmark_not_ready"
        action = "stop neural scaling and redesign labels, structured P-layers, or group splits"
    else:
        status = "pass"
        decision = "innovation2_small_spn_expanded_topology_benchmark_ready"
        action = "run one same-budget ID baseline vs equivariant GraphGPS vs CETT attribution matrix"
    return {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "width_checks": width_checks,
        "topology_checks": topology_checks,
        "shortcut_checks": shortcut_checks,
        "thresholds": {
            "width": MINIMUM_WIDTHS,
            "topology_fractions": MINIMUM_TOPOLOGY_FRACTIONS,
            "dual_effect_rows": MINIMUM_DUAL_EFFECT_ROWS,
            "maximum_marginal_auc": MAXIMUM_MARGINAL_AUC,
        },
        "metrics": metrics,
        "claim_scope": (
            "exact all-256-key labels for an expanded 16-bit synthetic SPN topology "
            "benchmark; no neural training, real-cipher result, or attack claim"
        ),
        "next_action": {"action": action, "training": False, "remote_scale": False},
    }


def expanded_scalar_vectorized_matches() -> bool:
    config = ExpandedTopologyAuditConfig(
        run_id="fixture",
        mode="smoke",
        sbox_variants=4,
        player_variants=16,
        rounds=(2,),
        keys=16,
    )
    variants = make_expanded_variants(config)
    points = np.asarray([0, 1, 0x1234, 0xFFFF], dtype=np.uint16)
    for variant_index in (0, 15, 48, 63):
        variant = variants[variant_index]
        keys = make_round_keys(rounds=2, key_count=config.keys)
        vectorized = encrypt_words(points, keys, variant)
        for key_index in (0, 7, 15):
            scalar = np.asarray(
                [
                    encrypt_scalar(
                        int(point),
                        tuple(int(value) for value in keys[:, key_index]),
                        variant,
                    )
                    for point in points
                ],
                dtype=np.uint16,
            )
            if not np.array_equal(vectorized[key_index], scalar):
                return False
    return True


def _label_cube(labels: np.ndarray) -> np.ndarray:
    matrix = np.asarray(labels, dtype=np.bool_)
    if matrix.shape[0] != 4 * EXPANDED_PLAYERS:
        raise ValueError("labels must contain 64 ordered S-box/P-layer variants")
    return matrix.reshape(4, EXPANDED_PLAYERS, *matrix.shape[1:])


def _split_arrays(cube: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "train": cube[:TRAIN_SBOXES, :TRAIN_PLAYERS].reshape(-1, *cube.shape[2:]),
        "unseen_sbox": cube[TRAIN_SBOXES:, :TRAIN_PLAYERS].reshape(
            -1, *cube.shape[2:]
        ),
        "unseen_player": cube[:TRAIN_SBOXES, TRAIN_PLAYERS:].reshape(
            -1, *cube.shape[2:]
        ),
        "dual_unseen": cube[TRAIN_SBOXES:, TRAIN_PLAYERS:].reshape(
            -1, *cube.shape[2:]
        ),
    }


def _label_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "total": int(values.size),
        "positive": int(values.sum()),
        "negative": int(values.size - values.sum()),
    }


def _topology_metrics(cube: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    selected_cube = cube[:, :, selected]
    selected_cells = int(selected.sum())
    train = selected_cube[:TRAIN_SBOXES, :TRAIN_PLAYERS]
    train_p_var_by_s = np.any(train != train[:, :1], axis=1)
    heldout = selected_cube[:TRAIN_SBOXES, TRAIN_PLAYERS:]
    heldout_diff = heldout[:, :, None, :] != train[:, None, :, :]
    heldout_novel = np.any(heldout_diff, axis=(0, 1, 2))
    dual_train = selected_cube[TRAIN_SBOXES, :TRAIN_PLAYERS]
    dual_heldout = selected_cube[TRAIN_SBOXES, TRAIN_PLAYERS:]
    dual_effect_by_p = np.any(
        dual_heldout[:, None, :] != dual_train[None, :, :], axis=1
    )
    dual_effect_cells = np.any(dual_effect_by_p, axis=0)
    dual_effect_targets = dual_heldout[dual_effect_by_p]
    train_interaction = _interaction_mask(train)
    full_interaction = _interaction_mask(selected_cube)
    counts = {
        "train_p_sensitive_any_s": int(np.any(train_p_var_by_s, axis=0).sum()),
        "train_p_sensitive_all_s": int(np.all(train_p_var_by_s, axis=0).sum()),
        "heldout_p_novel_any_s": int(heldout_novel.sum()),
        "dual_p_effect_cells": int(dual_effect_cells.sum()),
        "dual_p_effect_positive_rows": int(dual_effect_targets.sum()),
        "dual_p_effect_negative_rows": int(
            len(dual_effect_targets) - dual_effect_targets.sum()
        ),
        "train_interaction_cells": int(train_interaction.sum()),
        "full_interaction_cells": int(full_interaction.sum()),
    }
    fractions = {
        key: float(value / selected_cells) if selected_cells else 0.0
        for key, value in counts.items()
        if key
        in {
            "train_p_sensitive_any_s",
            "train_p_sensitive_all_s",
            "heldout_p_novel_any_s",
            "dual_p_effect_cells",
            "train_interaction_cells",
            "full_interaction_cells",
        }
    }
    return {"counts": counts, "fractions": fractions}


def _interaction_mask(cube: np.ndarray) -> np.ndarray:
    base = cube[0, 0]
    mixed = cube ^ cube[:, :1] ^ cube[:1, :] ^ base
    return np.any(mixed, axis=(0, 1))


def _marginal_baselines(
    cube: np.ndarray, selected: np.ndarray
) -> dict[str, dict[str, float]]:
    split_arrays = _split_arrays(cube)
    train = split_arrays["train"].astype(np.float64)
    cell_shape = cube.shape[2:]
    predictors = {
        "global": np.full(cell_shape, float(train.mean())),
        "mask_only": np.broadcast_to(
            train.mean(axis=(0, 1, 2))[None, None, :], cell_shape
        ),
        "round_mask": np.broadcast_to(
            train.mean(axis=(0, 2))[:, None, :], cell_shape
        ),
        "structure_mask": np.broadcast_to(
            train.mean(axis=(0, 1))[None, :, :], cell_shape
        ),
        "round_structure_mask": train.mean(axis=0),
    }
    output: dict[str, dict[str, float]] = {}
    for split in ("unseen_sbox", "unseen_player", "dual_unseen"):
        target = split_arrays[split][:, selected]
        scores = {
            name: _binary_auc(
                target.reshape(-1),
                np.broadcast_to(score[selected], target.shape).reshape(-1),
            )
            for name, score in predictors.items()
        }
        output[split] = {**scores, "strongest_auc": max(scores.values())}
    return output


def _selected_rows(
    selected: np.ndarray,
    train_positive_count: np.ndarray,
    rounds: tuple[int, ...],
    structures: tuple[CoordinateStructure, ...],
    masks: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index, structure_index, mask_index in np.argwhere(selected):
        rows.append(
            {
                "round_index": int(round_index),
                "rounds": int(rounds[int(round_index)]),
                "structure_index": int(structure_index),
                "structure_id": structures[int(structure_index)].structure_id,
                "mask_index": int(mask_index),
                "mask_hex": f"0x{int(masks[int(mask_index)]):04X}",
                "train_positive_count": int(
                    train_positive_count[
                        int(round_index), int(structure_index), int(mask_index)
                    ]
                ),
            }
        )
    return rows


def _metadata(
    config: ExpandedTopologyAuditConfig,
    variants: tuple[SmallSpnVariant, ...],
    structures: tuple[CoordinateStructure, ...],
    masks: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_expanded_topology_benchmark",
        "mode": config.mode,
        "state_bits": STATE_BITS,
        "master_key_bits": 8,
        "master_keys": config.keys,
        "rounds": list(config.rounds),
        "sbox_seed": SBOX_SEED,
        "player_seed": PLAYER_SEED,
        "mask_seed": MASK_SEED,
        "key_expansion_seed": KEY_EXPANSION_SEED,
        "variants": [
            {
                "variant_id": variant.variant_id,
                "sbox_id": variant.sbox_id,
                "player_id": variant.player_id,
                "sbox": list(variant.sbox),
                "player": list(variant.player),
                "fair_corrupted_player": list(fair_corrupt_player(variant.player)),
            }
            for variant in variants
        ],
        "structures": [
            {
                "structure_id": structure.structure_id,
                "active_nibbles": list(structure.active_nibbles),
                "active_bits": list(structure.active_bits),
                "dimension": structure.dimension,
            }
            for structure in structures
        ],
        "output_masks": [f"0x{mask:04X}" for mask in masks],
        "label_definition": (
            "1 iff every frozen master key has zero linear-mask XOR over the "
            "complete input set"
        ),
        "training_performed": False,
    }


def _emit(
    callback: ProgressCallback | None, event: str, payload: dict[str, Any]
) -> None:
    if callback is not None:
        callback(event, payload)
