from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    kernel_basis_valid,
)


STATE_BITS = 16
MASTER_KEYS = 256
AUDIT_SBOXES = 4
AUDIT_PLAYERS = 4
MAX_PLAYER_FAMILY = 16
AUDIT_ROUNDS = (2, 3, 4, 5)
MASK_COUNT = 64
SBOX_SEED = 32001
PLAYER_SEED = 32002
MASK_SEED = 32003
KEY_EXPANSION_SEED = 32004
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SmallSpnVariant:
    variant_id: str
    sbox_id: int
    player_id: int
    sbox: tuple[int, ...]
    player: tuple[int, ...]


@dataclass(frozen=True)
class CoordinateStructure:
    structure_id: str
    active_nibbles: tuple[int, ...]
    active_bits: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.active_bits)


@dataclass(frozen=True)
class SmallSpnAuditConfig:
    run_id: str
    mode: str = "audit"
    sbox_variants: int = AUDIT_SBOXES
    player_variants: int = AUDIT_PLAYERS
    rounds: tuple[int, ...] = AUDIT_ROUNDS
    keys: int = MASTER_KEYS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.sbox_variants <= 0 or self.sbox_variants > AUDIT_SBOXES:
            raise ValueError("sbox_variants must be between 1 and 4")
        if self.player_variants <= 0 or self.player_variants > AUDIT_PLAYERS:
            raise ValueError("player_variants must be between 1 and 4")
        if not self.rounds or any(rounds <= 0 for rounds in self.rounds):
            raise ValueError("rounds must be positive")
        if self.keys <= 0 or self.keys > MASTER_KEYS:
            raise ValueError("keys must be between 1 and 256")
        if self.mode == "audit" and (
            self.sbox_variants != AUDIT_SBOXES
            or self.player_variants != AUDIT_PLAYERS
            or self.rounds != AUDIT_ROUNDS
            or self.keys != MASTER_KEYS
        ):
            raise ValueError("E32 audit freezes 4x4 variants, rounds 2..5, and all 256 keys")


def make_sboxes() -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(SBOX_SEED)
    values = [tuple(int(value) for value in PRESENT_SBOX)]
    while len(values) < AUDIT_SBOXES:
        candidate = tuple(int(value) for value in rng.permutation(16))
        if candidate not in values:
            values.append(candidate)
    return tuple(values)


def make_player_family(count: int) -> tuple[tuple[int, ...], ...]:
    if count <= 0 or count > MAX_PLAYER_FAMILY:
        raise ValueError("player family count must be between 1 and 16")
    rng = np.random.default_rng(PLAYER_SEED)
    base = tuple((4 * bit) % 15 if bit < 15 else 15 for bit in range(16))
    values = [base]
    while len(values) < count:
        candidate = tuple(int(value) for value in rng.permutation(16))
        if candidate not in values:
            values.append(candidate)
    return tuple(values)


def make_players() -> tuple[tuple[int, ...], ...]:
    return make_player_family(AUDIT_PLAYERS)


def make_variants(config: SmallSpnAuditConfig) -> tuple[SmallSpnVariant, ...]:
    sboxes = make_sboxes()[: config.sbox_variants]
    players = make_players()[: config.player_variants]
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


def make_structures() -> tuple[CoordinateStructure, ...]:
    structures: list[CoordinateStructure] = []
    for active_count in (1, 2, 3):
        for active_nibbles in combinations(range(4), active_count):
            active_bits = tuple(
                bit
                for nibble in active_nibbles
                for bit in range(4 * nibble, 4 * nibble + 4)
            )
            structures.append(
                CoordinateStructure(
                    structure_id="n" + "".join(str(value) for value in active_nibbles),
                    active_nibbles=active_nibbles,
                    active_bits=active_bits,
                )
            )
    return tuple(structures)


def make_output_masks() -> tuple[int, ...]:
    masks: list[int] = [1 << bit for bit in range(STATE_BITS)]
    masks.extend(0xF << (4 * nibble) for nibble in range(4))
    rng = np.random.default_rng(MASK_SEED)
    weight_two = [sum(1 << bit for bit in pair) for pair in combinations(range(16), 2)]
    for index in rng.choice(len(weight_two), size=16, replace=False):
        masks.append(weight_two[int(index)])
    while len(masks) < MASK_COUNT:
        weight = int(rng.integers(3, 9))
        bits = rng.choice(STATE_BITS, size=weight, replace=False)
        mask = sum(1 << int(bit) for bit in bits)
        if mask not in masks:
            masks.append(mask)
    return tuple(masks)


def enumerate_structure_points(structure: CoordinateStructure) -> np.ndarray:
    points = np.zeros(1, dtype=np.uint16)
    for bit in structure.active_bits:
        points = np.concatenate((points, points ^ np.uint16(1 << bit)))
    return points


def make_round_keys(*, rounds: int, key_count: int = MASTER_KEYS) -> np.ndarray:
    rng = np.random.default_rng(KEY_EXPANSION_SEED)
    key_sbox = np.asarray(rng.permutation(256), dtype=np.uint16)
    master = np.arange(key_count, dtype=np.uint16)
    keys = np.empty((rounds + 1, key_count), dtype=np.uint16)
    for round_index in range(rounds + 1):
        shift = round_index % 8
        low = (
            ((master << shift) | (master >> ((8 - shift) % 8))) & np.uint16(0xFF)
        ) ^ np.uint16((0xA7 * round_index) & 0xFF)
        high = key_sbox[
            (master ^ np.uint16((0x3D * round_index) & 0xFF)).astype(np.uint8)
        ]
        keys[round_index] = low | (high << np.uint16(8))
    return keys


def encrypt_words(
    plaintexts: np.ndarray,
    round_keys: np.ndarray,
    variant: SmallSpnVariant,
) -> np.ndarray:
    points = np.asarray(plaintexts, dtype=np.uint16).reshape(-1)
    keys = np.asarray(round_keys, dtype=np.uint16)
    states = np.broadcast_to(points, (keys.shape[1], points.size)).copy()
    sbox = np.asarray(variant.sbox, dtype=np.uint16)
    for round_index in range(keys.shape[0] - 1):
        states ^= keys[round_index, :, None]
        states = _sbox_layer(states, sbox)
        states = _permutation_layer(states, variant.player)
    states ^= keys[-1, :, None]
    return states


def encrypt_scalar(
    plaintext: int,
    round_keys: tuple[int, ...],
    variant: SmallSpnVariant,
) -> int:
    state = plaintext & 0xFFFF
    for round_key in round_keys[:-1]:
        state ^= round_key
        state = sum(variant.sbox[(state >> (4 * nibble)) & 0xF] << (4 * nibble) for nibble in range(4))
        permuted = 0
        for source, target in enumerate(variant.player):
            permuted |= ((state >> source) & 1) << target
        state = permuted
    return (state ^ round_keys[-1]) & 0xFFFF


def run_cached_exact_labels(
    config: SmallSpnAuditConfig,
    *,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    variants = make_variants(config)
    structures = make_structures()
    masks = make_output_masks()
    metadata = _metadata(config, variants, structures, masks)
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / "cache_metadata.json"
    parity_path = cache_root / "parity_words.npy"
    labels_path = cache_root / "labels.npy"
    completed_path = cache_root / "completed.npy"
    expected_shape = (len(variants), len(config.rounds), len(structures))
    if any(path.exists() for path in (metadata_path, parity_path, labels_path, completed_path)):
        if not all(path.exists() for path in (metadata_path, parity_path, labels_path, completed_path)):
            raise ValueError("partial E32 cache is missing required files")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("E32 cache metadata does not match config")
        parity = np.load(parity_path, mmap_mode="r+")
        labels = np.load(labels_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if parity.shape != (*expected_shape, config.keys) or parity.dtype != np.uint16:
            raise ValueError("E32 parity cache shape or dtype mismatch")
        if labels.shape != (*expected_shape, len(masks)) or labels.dtype != np.bool_:
            raise ValueError("E32 label cache shape or dtype mismatch")
        if completed.shape != expected_shape or completed.dtype != np.bool_:
            raise ValueError("E32 completion cache shape or dtype mismatch")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        parity = np.lib.format.open_memmap(
            parity_path, mode="w+", dtype=np.uint16, shape=(*expected_shape, config.keys)
        )
        labels = np.lib.format.open_memmap(
            labels_path, mode="w+", dtype=np.bool_, shape=(*expected_shape, len(masks))
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
                    "exact_label_block_start",
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
                    "exact_label_block_done",
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


def evaluate_exact_labels(
    config: SmallSpnAuditConfig,
    cache: dict[str, Any],
    *,
    resume_generated_blocks: int,
) -> dict[str, Any]:
    variants: tuple[SmallSpnVariant, ...] = cache["variants"]
    structures: tuple[CoordinateStructure, ...] = cache["structures"]
    labels = np.asarray(cache["labels"], dtype=np.bool_)
    parity = np.asarray(cache["parity_words"], dtype=np.uint16)
    masks = np.asarray(cache["masks"], dtype=np.uint16)
    recomputed = np.all(
        _parity16(parity[..., None] & masks[None, None, None, None, :]) == 0,
        axis=3,
    )
    rows: list[dict[str, Any]] = []
    signatures: set[str] = set()
    all_kernel_valid = True
    for variant_index, variant in enumerate(variants):
        for round_index, rounds in enumerate(config.rounds):
            for structure_index, structure in enumerate(structures):
                block = labels[variant_index, round_index, structure_index]
                signature = np.packbits(block, bitorder="little").tobytes().hex()
                signatures.add(signature)
                basis = gf2_kernel_basis(
                    parity[variant_index, round_index, structure_index], width=STATE_BITS
                )
                valid = kernel_basis_valid(
                    parity[variant_index, round_index, structure_index], basis
                )
                all_kernel_valid = all_kernel_valid and valid
                rows.append(
                    {
                        "run_id": config.run_id,
                        "task": "innovation2_small_spn_exact_label_width",
                        "variant_id": variant.variant_id,
                        "sbox_id": variant.sbox_id,
                        "player_id": variant.player_id,
                        "rounds": rounds,
                        "structure_id": structure.structure_id,
                        "dimension": structure.dimension,
                        "positive_masks": int(block.sum()),
                        "negative_masks": int(len(block) - block.sum()),
                        "kernel_dimension": len(basis),
                        "kernel_basis_valid": valid,
                        "label_signature": signature,
                        "training_performed": False,
                    }
                )

    split_indices = _split_indices(variants)
    split_metrics = {
        name: _label_counts(labels[indices]) for name, indices in split_indices.items()
    }
    baselines = _marginal_baselines(labels, split_indices)
    variable_cells = _variable_cipher_cells(labels)
    total_cells = int(labels.size)
    readiness = {
        "sixteen_unique_bijective_cipher_variants": len(variants)
        == config.sbox_variants * config.player_variants
        and len({(variant.sbox, variant.player) for variant in variants}) == len(variants)
        and all(sorted(variant.sbox) == list(range(16)) for variant in variants)
        and all(sorted(variant.player) == list(range(16)) for variant in variants),
        "fourteen_unique_coordinate_structures": len(structures) == 14
        and len({structure.active_bits for structure in structures}) == 14,
        "sixty_four_unique_nonzero_masks": len(masks) == MASK_COUNT
        and len(set(int(mask) for mask in masks)) == MASK_COUNT
        and bool(np.all(masks != 0)),
        "all_master_keys_covered": config.mode == "smoke" or config.keys == MASTER_KEYS,
        "all_blocks_completed": bool(np.asarray(cache["completed"]).all()),
        "resume_generates_zero_blocks": resume_generated_blocks == 0,
        "labels_recompute_from_parity_words": bool(np.array_equal(labels, recomputed)),
        "all_kernel_bases_validate": all_kernel_valid,
        "scalar_and_vectorized_cipher_match": scalar_vectorized_fixture_matches(),
        "label_shape_matches_frozen_grid": labels.shape
        == (len(variants), len(config.rounds), len(structures), MASK_COUNT),
    }
    metrics = {
        "total_labels": total_cells,
        "positive_labels": int(labels.sum()),
        "negative_labels": int(total_cells - labels.sum()),
        "distinct_label_signatures": len(signatures),
        "cipher_variable_cells": variable_cells,
        "cipher_variable_cell_fraction": variable_cells
        / int(np.prod(labels.shape[1:])),
        "split_metrics": split_metrics,
        "marginal_baselines": baselines,
    }
    gate = adjudicate_exact_labels(config, readiness, metrics)
    return {
        "rows": rows,
        "gate": gate,
        "metadata": {**cache["metadata"], "claim_scope": gate["claim_scope"]},
        "summary": {
            "run_id": config.run_id,
            "gate": gate,
            "metadata": cache["metadata"],
            "split_metrics": split_metrics,
            "marginal_baselines": baselines,
        },
    }


def adjudicate_exact_labels(
    config: SmallSpnAuditConfig,
    readiness: dict[str, bool],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    split_metrics = metrics["split_metrics"]
    baselines = metrics["marginal_baselines"]
    width_checks = {
        "positive_labels_at_least_5000": metrics["positive_labels"] >= 5000,
        "negative_labels_at_least_5000": metrics["negative_labels"] >= 5000,
        "distinct_signatures_at_least_64": metrics["distinct_label_signatures"] >= 64,
        "each_split_has_at_least_256_per_class": all(
            values["positive"] >= 256 and values["negative"] >= 256
            for values in split_metrics.values()
        ),
        "cipher_variable_cells_at_least_10_percent": metrics[
            "cipher_variable_cell_fraction"
        ]
        >= 0.10,
    }
    shortcut_checks = {
        "unseen_sbox_strongest_marginal_auc_at_most_0p80": baselines["unseen_sbox"][
            "strongest_auc"
        ]
        <= 0.80,
        "unseen_player_strongest_marginal_auc_at_most_0p80": baselines[
            "unseen_player"
        ]["strongest_auc"]
        <= 0.80,
        "dual_unseen_strongest_marginal_auc_at_most_0p75": baselines["dual_unseen"][
            "strongest_auc"
        ]
        <= 0.75,
    }
    if not readiness or not all(readiness.values()):
        status = "fail"
        decision = "innovation2_small_spn_exact_label_protocol_invalid"
        action = "repair bijections, full-key coverage, cache, parity labels, or scalar/vector agreement"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_small_spn_exact_label_readiness_passed"
        action = "run the frozen 16-cipher, 256-key E32 exact-label audit"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_exact_label_too_narrow"
        action = "stop this frozen toy family without changing seeds to select labels"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_exact_label_shortcut_dominated"
        action = "redesign the synthetic cipher split or structure/mask family before neural training"
    else:
        status = "pass"
        decision = "innovation2_small_spn_exact_label_family_ready"
        action = "prepare E33 deterministic baseline vs small GraphGPS vs SCGT training matrix"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "claim_scope": (
            "exact all-256-key labels for a frozen 16-bit synthetic SPN family; not a "
            "PRESENT/GIFT/SKINNY result, a real-key-schedule claim, or neural training"
        ),
        "next_action": {"action": action, "training": False, "remote_scale": False},
    }


def scalar_vectorized_fixture_matches() -> bool:
    config = SmallSpnAuditConfig(
        run_id="fixture",
        mode="smoke",
        sbox_variants=2,
        player_variants=2,
        rounds=(2, 3),
        keys=16,
    )
    points = np.asarray([0, 1, 0x1234, 0xFFFF], dtype=np.uint16)
    for variant in make_variants(config):
        for rounds in config.rounds:
            keys = make_round_keys(rounds=rounds, key_count=config.keys)
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


def _metadata(
    config: SmallSpnAuditConfig,
    variants: tuple[SmallSpnVariant, ...],
    structures: tuple[CoordinateStructure, ...],
    masks: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "task": "innovation2_small_spn_exact_label_width",
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
        "label_definition": "1 iff every frozen master key has zero linear-mask XOR over the complete input set",
        "training_performed": False,
    }


def _sbox_layer(states: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    output = np.zeros_like(states)
    for nibble in range(4):
        shift = np.uint16(4 * nibble)
        values = ((states >> shift) & np.uint16(0xF)).astype(np.uint8)
        output |= sbox[values] << shift
    return output


def _permutation_layer(states: np.ndarray, player: tuple[int, ...]) -> np.ndarray:
    output = np.zeros_like(states)
    for source, target in enumerate(player):
        output |= ((states >> np.uint16(source)) & np.uint16(1)) << np.uint16(target)
    return output


def _parity16(values: np.ndarray) -> np.ndarray:
    folded = np.asarray(values, dtype=np.uint16).copy()
    folded ^= folded >> np.uint16(8)
    folded ^= folded >> np.uint16(4)
    folded ^= folded >> np.uint16(2)
    folded ^= folded >> np.uint16(1)
    return folded & np.uint16(1)


def _split_indices(variants: tuple[SmallSpnVariant, ...]) -> dict[str, np.ndarray]:
    groups = {
        "train": [i for i, v in enumerate(variants) if v.sbox_id < 3 and v.player_id < 3],
        "unseen_sbox": [i for i, v in enumerate(variants) if v.sbox_id == 3 and v.player_id < 3],
        "unseen_player": [i for i, v in enumerate(variants) if v.sbox_id < 3 and v.player_id == 3],
        "dual_unseen": [i for i, v in enumerate(variants) if v.sbox_id == 3 and v.player_id == 3],
    }
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "total": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
    }


def _variable_cipher_cells(labels: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(labels != labels[:1], axis=0)))


def _marginal_baselines(
    labels: np.ndarray, split_indices: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    train = labels[split_indices["train"]].astype(np.float64)
    predictors = {
        "global": np.full(labels.shape[1:], float(train.mean())),
        "mask_only": np.broadcast_to(train.mean(axis=(0, 1, 2))[None, None, :], labels.shape[1:]),
        "round_mask": np.broadcast_to(train.mean(axis=(0, 2))[:, None, :], labels.shape[1:]),
        "structure_mask": np.broadcast_to(train.mean(axis=(0, 1))[None, :, :], labels.shape[1:]),
        "round_structure_mask": train.mean(axis=0),
    }
    output: dict[str, dict[str, float]] = {}
    for split_name in ("unseen_sbox", "unseen_player", "dual_unseen"):
        target = labels[split_indices[split_name]].reshape(-1)
        scores = {
            name: _binary_auc(
                target,
                np.broadcast_to(score, labels[split_indices[split_name]].shape).reshape(-1),
            )
            for name, score in predictors.items()
        }
        output[split_name] = {**scores, "strongest_auc": max(scores.values())}
    return output


def select_train_contrast_cells(
    labels: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    train_positive_count = np.asarray(labels, dtype=np.bool_)[train_indices].sum(axis=0)
    return (train_positive_count >= 1) & (train_positive_count <= len(train_indices) - 1)


def evaluate_matched_contrast(
    *,
    run_id: str,
    labels: np.ndarray,
    source_metadata: dict[str, Any],
    source_gate: dict[str, Any],
) -> dict[str, Any]:
    config = SmallSpnAuditConfig(run_id="matched-source")
    variants = make_variants(config)
    structures = make_structures()
    masks = make_output_masks()
    matrix = np.asarray(labels, dtype=np.bool_)
    expected_shape = (16, 4, 14, 64)
    split_indices = _split_indices(variants)
    selected = select_train_contrast_cells(matrix, split_indices["train"])
    train_counts = matrix[split_indices["train"]].sum(axis=0)
    split_metrics = {
        name: _label_counts(matrix[indices][:, selected])
        for name, indices in split_indices.items()
    }
    baselines = _matched_marginal_baselines(matrix, split_indices, selected)
    patterns = matrix[:, selected].T
    distinct_patterns = len(
        {np.packbits(pattern, bitorder="little").tobytes() for pattern in patterns}
    )
    selected_rows: list[dict[str, Any]] = []
    for round_index, structure_index, mask_index in np.argwhere(selected):
        selected_rows.append(
            {
                "round_index": int(round_index),
                "rounds": AUDIT_ROUNDS[int(round_index)],
                "structure_index": int(structure_index),
                "structure_id": structures[int(structure_index)].structure_id,
                "mask_index": int(mask_index),
                "mask_hex": f"0x{masks[int(mask_index)]:04X}",
                "train_positive_count": int(
                    train_counts[int(round_index), int(structure_index), int(mask_index)]
                ),
            }
        )
    readiness = {
        "source_run_id_matches": source_gate.get("run_id")
        == "i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718",
        "source_task_matches": source_metadata.get("task")
        == "innovation2_small_spn_exact_label_width",
        "source_decision_is_shortcut_dominated": source_gate.get("decision")
        == "innovation2_small_spn_exact_label_shortcut_dominated",
        "source_readiness_checks_all_pass": bool(source_gate.get("readiness_checks"))
        and all(source_gate["readiness_checks"].values()),
        "source_label_shape_matches": matrix.shape == expected_shape,
        "selection_uses_nine_train_variants_only": len(split_indices["train"]) == 9,
        "selection_excludes_unanimous_train_cells": bool(
            np.all((train_counts[selected] >= 1) & (train_counts[selected] <= 8))
        )
        and not bool(np.any(selected & ((train_counts == 0) | (train_counts == 9)))),
        "selected_rows_match_mask": len(selected_rows) == int(selected.sum()),
    }
    raw_baselines = source_gate["metrics"]["marginal_baselines"]
    metrics = {
        "selected_base_cells": int(selected.sum()),
        "selected_total_label_rows": int(sum(item["total"] for item in split_metrics.values())),
        "distinct_topology_label_patterns": distinct_patterns,
        "split_metrics": split_metrics,
        "marginal_baselines": baselines,
        "raw_strongest_marginal_auc": {
            split: float(raw_baselines[split]["strongest_auc"])
            for split in ("unseen_sbox", "unseen_player", "dual_unseen")
        },
    }
    width_checks = {
        "selected_base_cells_at_least_512": metrics["selected_base_cells"] >= 512,
        "train_each_class_at_least_2000": split_metrics["train"]["positive"] >= 2000
        and split_metrics["train"]["negative"] >= 2000,
        "unseen_sbox_each_class_at_least_500": split_metrics["unseen_sbox"][
            "positive"
        ]
        >= 500
        and split_metrics["unseen_sbox"]["negative"] >= 500,
        "unseen_player_each_class_at_least_500": split_metrics["unseen_player"][
            "positive"
        ]
        >= 500
        and split_metrics["unseen_player"]["negative"] >= 500,
        "dual_unseen_each_class_at_least_200": split_metrics["dual_unseen"][
            "positive"
        ]
        >= 200
        and split_metrics["dual_unseen"]["negative"] >= 200,
        "distinct_topology_patterns_at_least_128": distinct_patterns >= 128,
    }
    shortcut_checks = {
        "unseen_sbox_strongest_marginal_auc_at_most_0p80": baselines["unseen_sbox"][
            "strongest_auc"
        ]
        <= 0.80,
        "unseen_player_strongest_marginal_auc_at_most_0p80": baselines[
            "unseen_player"
        ]["strongest_auc"]
        <= 0.80,
        "dual_unseen_strongest_marginal_auc_at_most_0p75": baselines["dual_unseen"][
            "strongest_auc"
        ]
        <= 0.75,
    }
    if not all(readiness.values()):
        status = "fail"
        decision = "innovation2_small_spn_matched_contrast_protocol_invalid"
        action = "repair source ownership, train-only selection, shape, or selected-cell indexing"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_matched_contrast_too_narrow"
        action = "stop without reading heldout labels to relax the selection rule"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_matched_contrast_still_shortcut_dominated"
        action = "stop the current synthetic benchmark without neural training"
    else:
        status = "pass"
        decision = "innovation2_small_spn_matched_contrast_ready"
        action = "prepare E33 deterministic baseline vs small GraphGPS vs SCGT training matrix"
    gate = {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "claim_scope": (
            "train-only matched-contrast readjudication of frozen exact labels from a "
            "16-bit synthetic SPN family; not a real-cipher result or neural training"
        ),
        "next_action": {"action": action, "training": False, "remote_scale": False},
    }
    result_rows = [
        {
            "run_id": run_id,
            "task": "innovation2_small_spn_matched_contrast_readjudication",
            "split": split,
            **split_metrics[split],
            "strongest_marginal_auc": baselines.get(split, {}).get("strongest_auc"),
            "training_performed": False,
        }
        for split in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ]
    return {
        "rows": result_rows,
        "selected_rows": selected_rows,
        "selected_mask": selected,
        "gate": gate,
        "summary": {
            "run_id": run_id,
            "gate": gate,
            "source_run_id": source_gate.get("run_id"),
            "split_metrics": split_metrics,
            "marginal_baselines": baselines,
        },
        "metadata": {
            "run_id": run_id,
            "task": "innovation2_small_spn_matched_contrast_readjudication",
            "source_run_id": source_gate.get("run_id"),
            "selection_rule": "1 <= train_positive_count <= 8 over nine train topologies",
            "heldout_labels_used_for_selection": False,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def _matched_marginal_baselines(
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, dict[str, float]]:
    train = labels[split_indices["train"]].astype(np.float64)
    full_predictors = {
        "global": np.full(labels.shape[1:], float(train.mean())),
        "mask_only": np.broadcast_to(
            train.mean(axis=(0, 1, 2))[None, None, :], labels.shape[1:]
        ),
        "round_mask": np.broadcast_to(
            train.mean(axis=(0, 2))[:, None, :], labels.shape[1:]
        ),
        "structure_mask": np.broadcast_to(
            train.mean(axis=(0, 1))[None, :, :], labels.shape[1:]
        ),
        "round_structure_mask": train.mean(axis=0),
    }
    output: dict[str, dict[str, float]] = {}
    for split_name in ("unseen_sbox", "unseen_player", "dual_unseen"):
        target_matrix = labels[split_indices[split_name]][:, selected]
        scores = {
            name: _binary_auc(
                target_matrix.reshape(-1),
                np.broadcast_to(score[selected], target_matrix.shape).reshape(-1),
            )
            for name, score in full_predictors.items()
        }
        output[split_name] = {**scores, "strongest_auc": max(scores.values())}
    return output


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.bool_)
    scores = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[y].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _emit(
    callback: ProgressCallback | None, event: str, payload: dict[str, Any]
) -> None:
    if callback is not None:
        callback(event, payload)
