from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    _binary_auc,
    _parity16,
)


VARIANTS = 64
ROUNDS = 4
STRUCTURES = 14
MASKS = 64
KEYS = 256
COORDINATES = STRUCTURES * MASKS
TRAIN_SBOXES = 3
TRAIN_PLAYERS = 12
CANDIDATE_SEED = 62001
CANDIDATE_PAIRS = 65536
MIN_TRAIN_POSITIVES = 9
MAX_TRAIN_POSITIVES = 27
MAX_SELECTED_PER_ROUND = 1024


@dataclass(frozen=True)
class SmallSpnRelationConfig:
    run_id: str
    candidate_seed: int = CANDIDATE_SEED
    candidate_pairs: int = CANDIDATE_PAIRS
    maximum_selected_per_round: int = MAX_SELECTED_PER_ROUND

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.candidate_seed != CANDIDATE_SEED:
            raise ValueError("E62 candidate seed is frozen")
        if self.candidate_pairs != CANDIDATE_PAIRS:
            raise ValueError("E62 candidate pair count is frozen")
        if self.maximum_selected_per_round != MAX_SELECTED_PER_ROUND:
            raise ValueError("E62 selected-per-round cap is frozen")


def generate_candidate_pairs(
    *, seed: int = CANDIDATE_SEED, count: int = CANDIDATE_PAIRS
) -> np.ndarray:
    maximum = COORDINATES * (COORDINATES - 1) // 2
    if count <= 0 or count > maximum:
        raise ValueError("candidate pair count is out of range")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < count:
        values = rng.integers(0, COORDINATES, size=(min(8192, count), 2))
        for raw_left, raw_right in values:
            left = int(raw_left)
            right = int(raw_right)
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) == count:
                break
    return np.asarray(pairs, dtype=np.uint16)


def coordinate_parity_bits(parity_words: np.ndarray, masks: np.ndarray) -> np.ndarray:
    parity = np.asarray(parity_words)
    mask_values = np.asarray(masks, dtype=np.uint16)
    expected_shape = (VARIANTS, ROUNDS, STRUCTURES, KEYS)
    if parity.shape != expected_shape or parity.dtype != np.uint16:
        raise ValueError("E62 parity_words shape or dtype mismatch")
    if mask_values.shape != (MASKS,):
        raise ValueError("E62 masks shape mismatch")
    bits = np.empty(
        (VARIANTS, ROUNDS, STRUCTURES, MASKS, KEYS), dtype=np.bool_
    )
    for mask_index, mask in enumerate(mask_values):
        bits[:, :, :, mask_index, :] = _parity16(parity & mask).astype(np.bool_)
    return bits.reshape(VARIANTS, ROUNDS, COORDINATES, KEYS)


def label_candidate_pairs(
    coordinate_bits: np.ndarray,
    candidate_pairs: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    bits = np.asarray(coordinate_bits, dtype=np.bool_)
    pairs = np.asarray(candidate_pairs, dtype=np.uint16)
    if bits.shape != (VARIANTS, ROUNDS, COORDINATES, KEYS):
        raise ValueError("E62 coordinate parity shape mismatch")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("candidate_pairs must have shape [N, 2]")
    labels = np.empty((VARIANTS, ROUNDS, len(pairs)), dtype=np.bool_)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        left = pairs[start:stop, 0]
        right = pairs[start:stop, 1]
        labels[:, :, start:stop] = np.all(
            bits[:, :, left, :] == bits[:, :, right, :], axis=-1
        )
    return labels


def variant_split_indices() -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {
        "train": [],
        "unseen_sbox": [],
        "unseen_player": [],
        "dual_unseen": [],
    }
    for sbox_id in range(4):
        for player_id in range(16):
            index = sbox_id * 16 + player_id
            if sbox_id < TRAIN_SBOXES and player_id < TRAIN_PLAYERS:
                groups["train"].append(index)
            elif sbox_id == TRAIN_SBOXES and player_id < TRAIN_PLAYERS:
                groups["unseen_sbox"].append(index)
            elif sbox_id < TRAIN_SBOXES and player_id >= TRAIN_PLAYERS:
                groups["unseen_player"].append(index)
            else:
                groups["dual_unseen"].append(index)
    return {
        name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()
    }


def select_relation_templates(
    labels: np.ndarray,
    *,
    maximum_per_round: int = MAX_SELECTED_PER_ROUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(labels, dtype=np.bool_)
    if matrix.shape[0:2] != (VARIANTS, ROUNDS):
        raise ValueError("E62 label matrix shape mismatch")
    train = variant_split_indices()["train"]
    train_positive_count = matrix[train].sum(axis=0)
    selected_rounds: list[int] = []
    selected_candidates: list[int] = []
    selected_counts: list[int] = []
    for round_index in range(ROUNDS):
        eligible = np.flatnonzero(
            (train_positive_count[round_index] >= MIN_TRAIN_POSITIVES)
            & (train_positive_count[round_index] <= MAX_TRAIN_POSITIVES)
        )[:maximum_per_round]
        selected_rounds.extend([round_index] * len(eligible))
        selected_candidates.extend(int(value) for value in eligible)
        selected_counts.extend(
            int(train_positive_count[round_index, value]) for value in eligible
        )
    return (
        np.asarray(selected_rounds, dtype=np.uint8),
        np.asarray(selected_candidates, dtype=np.int32),
        np.asarray(selected_counts, dtype=np.uint8),
    )


def selected_relation_labels_and_witnesses(
    coordinate_bits: np.ndarray,
    candidate_pairs: np.ndarray,
    selected_rounds: np.ndarray,
    selected_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_values = np.asarray(candidate_pairs, dtype=np.uint16)[selected_candidates]
    relation_count = len(selected_candidates)
    labels = np.empty((VARIANTS, relation_count), dtype=np.bool_)
    witnesses = np.full((VARIANTS, relation_count), -1, dtype=np.int16)
    all_certificates_valid = np.ones(relation_count, dtype=np.bool_)
    for relation_index, (round_index, pair) in enumerate(
        zip(selected_rounds, pair_values, strict=True)
    ):
        difference = coordinate_bits[:, int(round_index), int(pair[0]), :] ^ coordinate_bits[
            :, int(round_index), int(pair[1]), :
        ]
        relation_labels = ~np.any(difference, axis=1)
        labels[:, relation_index] = relation_labels
        negative = np.flatnonzero(~relation_labels)
        if len(negative):
            witnesses[negative, relation_index] = np.argmax(
                difference[negative], axis=1
            ).astype(np.int16)
        positives_valid = bool(np.all(~difference[relation_labels]))
        negatives_valid = bool(
            np.all(
                difference[
                    negative,
                    witnesses[negative, relation_index],
                ]
            )
        ) if len(negative) else True
        all_certificates_valid[relation_index] = positives_valid and negatives_valid
    return labels, witnesses, all_certificates_valid


def evaluate_relation_benchmark(
    config: SmallSpnRelationConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    completed: np.ndarray,
    candidate_pairs: np.ndarray,
    selected_rounds: np.ndarray,
    selected_candidates: np.ndarray,
    selected_train_counts: np.ndarray,
    relation_labels: np.ndarray,
    witness_key_indices: np.ndarray,
    certificates_valid: np.ndarray,
) -> dict[str, Any]:
    split_indices = variant_split_indices()
    split_metrics = {
        name: _label_counts(relation_labels[indices])
        for name, indices in split_indices.items()
    }
    baselines = _relation_baselines(
        relation_labels,
        candidate_pairs[selected_candidates],
        selected_rounds,
        source_metadata,
    )
    topology = _topology_metrics(relation_labels)
    patterns = relation_labels.T
    distinct_patterns = len(
        {np.packbits(pattern, bitorder="little").tobytes() for pattern in patterns}
    )
    per_round_selected = [
        int(np.sum(selected_rounds == round_index)) for round_index in range(ROUNDS)
    ]
    supported_rounds = sum(value >= 64 for value in per_round_selected)
    all_positive_certificates = bool(np.all(certificates_valid))
    negative_locations = np.argwhere(~relation_labels)
    all_negative_witnesses = bool(
        len(negative_locations)
        and np.all(
            witness_key_indices[
                negative_locations[:, 0], negative_locations[:, 1]
            ]
            >= 0
        )
    )
    source_checks = {
        "e37_source_gate_passed": source_gate.get("status") == "pass"
        and source_gate.get("decision")
        == "innovation2_small_spn_expanded_topology_benchmark_ready",
        "source_has_all_256_master_keys": source_metadata.get("master_keys") == KEYS
        and source_metadata.get("master_key_bits") == 8,
        "source_shape_contract_is_64x4x14x64": source_gate.get(
            "readiness_checks", {}
        ).get("label_shape_is_64x4x14x64")
        is True,
        "source_cache_is_complete": np.asarray(completed).shape
        == (VARIANTS, ROUNDS, STRUCTURES)
        and bool(np.asarray(completed).all()),
        "candidate_pairs_are_unique_distinct_and_ordered": len(candidate_pairs)
        == config.candidate_pairs
        and len({tuple(int(value) for value in pair) for pair in candidate_pairs})
        == config.candidate_pairs
        and bool(np.all(candidate_pairs[:, 0] < candidate_pairs[:, 1])),
        "candidate_seed_and_count_are_frozen": config.candidate_seed
        == CANDIDATE_SEED
        and config.candidate_pairs == CANDIDATE_PAIRS,
        "split_sizes_are_36_12_12_4": [
            len(split_indices[name])
            for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
        ]
        == [36, 12, 12, 4],
        "selection_uses_only_train_positive_counts": bool(
            np.all(
                (selected_train_counts >= MIN_TRAIN_POSITIVES)
                & (selected_train_counts <= MAX_TRAIN_POSITIVES)
            )
        ),
        "all_positive_all_key_certificates_validate": all_positive_certificates,
        "all_negative_witness_key_indices_are_present": all_negative_witnesses,
    }
    width_checks = {
        "selected_relation_templates_at_least_256": len(selected_rounds) >= 256,
        "supported_rounds_at_least_2": supported_rounds >= 2,
        "train_positive_at_least_3000": split_metrics["train"]["positive"] >= 3000,
        "train_negative_at_least_3000": split_metrics["train"]["negative"] >= 3000,
        "unseen_sbox_positive_at_least_768": split_metrics["unseen_sbox"][
            "positive"
        ]
        >= 768,
        "unseen_sbox_negative_at_least_768": split_metrics["unseen_sbox"][
            "negative"
        ]
        >= 768,
        "unseen_player_positive_at_least_768": split_metrics["unseen_player"][
            "positive"
        ]
        >= 768,
        "unseen_player_negative_at_least_768": split_metrics["unseen_player"][
            "negative"
        ]
        >= 768,
        "dual_unseen_positive_at_least_192": split_metrics["dual_unseen"][
            "positive"
        ]
        >= 192,
        "dual_unseen_negative_at_least_192": split_metrics["dual_unseen"][
            "negative"
        ]
        >= 192,
        "distinct_topology_patterns_at_least_128": distinct_patterns >= 128,
    }
    topology_checks = {
        "train_p_sensitive_any_s_fraction_at_least_0p75": topology["fractions"][
            "train_p_sensitive_any_s"
        ]
        >= 0.75,
        "dual_p_effect_relations_fraction_at_least_0p40": topology["fractions"][
            "dual_p_effect_relations"
        ]
        >= 0.40,
        "train_interaction_relations_fraction_at_least_0p50": topology[
            "fractions"
        ]["train_interaction_relations"]
        >= 0.50,
        "full_interaction_relations_fraction_at_least_0p70": topology[
            "fractions"
        ]["full_interaction_relations"]
        >= 0.70,
    }
    shortcut_checks = {
        "unseen_sbox_strongest_marginal_auc_at_most_0p80": baselines[
            "unseen_sbox"
        ]["strongest_auc"]
        <= 0.80,
        "unseen_player_strongest_marginal_auc_at_most_0p80": baselines[
            "unseen_player"
        ]["strongest_auc"]
        <= 0.80,
        "dual_unseen_strongest_marginal_auc_at_most_0p75": baselines[
            "dual_unseen"
        ]["strongest_auc"]
        <= 0.75,
    }
    metrics = {
        "candidate_relation_templates": len(candidate_pairs),
        "selected_relation_templates": len(selected_rounds),
        "per_round_selected_templates": per_round_selected,
        "supported_rounds": supported_rounds,
        "distinct_topology_patterns": distinct_patterns,
        "split_metrics": split_metrics,
        "marginal_baselines": baselines,
        "topology": topology,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_small_spn_multicoordinate_relation_protocol_invalid"
        action = "repair E37 cache, train-only selection, or strict all-key certificates"
    elif not all(width_checks.values()) or not all(topology_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_multicoordinate_relation_width_not_ready"
        action = "stop RCCA; the exact relation benchmark lacks label or topology width"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_multicoordinate_relation_shortcut_dominated"
        action = "stop RCCA; topology-free relation marginals explain the benchmark"
    else:
        status = "pass"
        decision = "innovation2_small_spn_multicoordinate_relation_training_ready"
        action = "run fixed-budget DeepSets versus RCCA with wrong-P and shuffle controls"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "width_checks": width_checks,
        "topology_checks": topology_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "thresholds": {
            "candidate_seed": CANDIDATE_SEED,
            "candidate_pairs": CANDIDATE_PAIRS,
            "train_positive_range": [MIN_TRAIN_POSITIVES, MAX_TRAIN_POSITIVES],
            "maximum_selected_per_round": MAX_SELECTED_PER_ROUND,
        },
        "claim_scope": (
            "exact all-256-master-key two-coordinate GF(2) relation labels for "
            "16-bit synthetic SPNs; no neural training, real-cipher result, or attack"
        ),
        "next_action": {
            "action": action,
            "training": status == "pass",
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_small_spn_multicoordinate_relation_readiness",
            "split": split,
            **split_metrics[split],
            "strongest_marginal_auc": baselines.get(split, {}).get("strongest_auc"),
            "selected_relation_templates": len(selected_rounds),
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for split in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ]
    return {"gate": gate, "result_rows": result_rows}


def load_source_metadata(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    return gate, metadata


def _label_counts(values: np.ndarray) -> dict[str, int]:
    matrix = np.asarray(values, dtype=np.bool_)
    positives = int(matrix.sum())
    return {
        "total": int(matrix.size),
        "positive": positives,
        "negative": int(matrix.size - positives),
    }


def _relation_baselines(
    labels: np.ndarray,
    selected_pairs: np.ndarray,
    selected_rounds: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, dict[str, float]]:
    split_indices = variant_split_indices()
    train = labels[split_indices["train"]].astype(np.float64)
    relation_rate = train.mean(axis=0)
    global_rate = np.full(len(selected_pairs), float(train.mean()))
    structures = metadata["structures"]
    masks = [int(value, 16) for value in metadata["output_masks"]]
    groups: list[tuple[int, ...]] = []
    for round_index, (left, right) in zip(
        selected_rounds, selected_pairs, strict=True
    ):
        left_structure, left_mask = divmod(int(left), MASKS)
        right_structure, right_mask = divmod(int(right), MASKS)
        descriptors = sorted(
            (
                (
                    int(structures[left_structure]["dimension"]),
                    masks[left_mask].bit_count(),
                ),
                (
                    int(structures[right_structure]["dimension"]),
                    masks[right_mask].bit_count(),
                ),
            )
        )
        groups.append(
            (
                int(round_index),
                *descriptors[0],
                *descriptors[1],
                int(left_structure == right_structure),
                int(left_mask == right_mask),
                (masks[left_mask] & masks[right_mask]).bit_count(),
            )
        )
    group_totals: dict[tuple[int, ...], list[float]] = {}
    for relation_index, group in enumerate(groups):
        group_totals.setdefault(group, []).extend(train[:, relation_index].tolist())
    group_rate = np.asarray(
        [float(np.mean(group_totals[group])) for group in groups], dtype=np.float64
    )
    coordinate_sums = np.zeros(COORDINATES, dtype=np.float64)
    coordinate_counts = np.zeros(COORDINATES, dtype=np.float64)
    relation_sums = train.sum(axis=0)
    for relation_sum, pair in zip(relation_sums, selected_pairs, strict=True):
        for coordinate in pair:
            coordinate_sums[int(coordinate)] += relation_sum
            coordinate_counts[int(coordinate)] += len(train)
    coordinate_rate = np.divide(
        coordinate_sums,
        coordinate_counts,
        out=np.full_like(coordinate_sums, float(train.mean())),
        where=coordinate_counts > 0,
    )
    pair_coordinate_rate = coordinate_rate[selected_pairs].mean(axis=1)
    predictors = {
        "global": global_rate,
        "relation_id_train_rate": relation_rate,
        "structural_feature_group": group_rate,
        "coordinate_marginal": pair_coordinate_rate,
    }
    output: dict[str, dict[str, float]] = {}
    for split in ("unseen_sbox", "unseen_player", "dual_unseen"):
        target = labels[split_indices[split]]
        scores = {
            name: _binary_auc(
                target.reshape(-1),
                np.broadcast_to(score, target.shape).reshape(-1),
            )
            for name, score in predictors.items()
        }
        output[split] = {**scores, "strongest_auc": max(scores.values())}
    return output


def _topology_metrics(labels: np.ndarray) -> dict[str, Any]:
    cube = np.asarray(labels, dtype=np.bool_).reshape(4, 16, -1)
    selected = cube.shape[-1]
    train = cube[:3, :12]
    train_p_sensitive = np.any(train != train[:, :1], axis=1)
    dual_train = cube[3, :12]
    dual_heldout = cube[3, 12:]
    dual_effect = np.any(
        np.any(dual_heldout[:, None, :] != dual_train[None, :, :], axis=1), axis=0
    )
    train_interaction = _interaction_mask(train)
    full_interaction = _interaction_mask(cube)
    counts = {
        "train_p_sensitive_any_s": int(np.any(train_p_sensitive, axis=0).sum()),
        "dual_p_effect_relations": int(dual_effect.sum()),
        "train_interaction_relations": int(train_interaction.sum()),
        "full_interaction_relations": int(full_interaction.sum()),
    }
    fractions = {
        key: float(value / selected) if selected else 0.0
        for key, value in counts.items()
    }
    return {"counts": counts, "fractions": fractions}


def _interaction_mask(cube: np.ndarray) -> np.ndarray:
    base = cube[0, 0]
    mixed = cube ^ cube[:, :1] ^ cube[:1, :] ^ base
    return np.any(mixed, axis=(0, 1))
