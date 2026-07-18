from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import _binary_auc
from blockcipher_nd.tasks.innovation2.small_spn_multicoordinate_relations import (
    variant_split_indices,
)


TRIVIAL_POSITIVE = 0
NONTRIVIAL_POSITIVE = 1
ONE_ZERO_NEGATIVE = 2
NONTRIVIAL_NEGATIVE = 3


@dataclass(frozen=True)
class RelationDecompositionConfig:
    run_id: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")


def decompose_relation_labels(
    coordinate_bits: np.ndarray,
    relation_pairs: np.ndarray,
    relation_rounds: np.ndarray,
    relation_labels: np.ndarray,
    witness_key_indices: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    bits = np.asarray(coordinate_bits, dtype=np.bool_)
    pairs = np.asarray(relation_pairs, dtype=np.int64)
    rounds = np.asarray(relation_rounds, dtype=np.int64)
    labels = np.asarray(relation_labels, dtype=np.bool_)
    witnesses = np.asarray(witness_key_indices, dtype=np.int64)
    variants, _, _, keys = bits.shape
    if labels.shape != (variants, len(pairs)) or witnesses.shape != labels.shape:
        raise ValueError("relation labels/witnesses do not match pairs")
    singleton_zero = np.empty((variants, len(pairs), 2), dtype=np.bool_)
    categories = np.empty(labels.shape, dtype=np.uint8)
    all_labels_match = True
    all_witnesses_match = True
    for relation_index, (round_index, pair) in enumerate(
        zip(rounds, pairs, strict=True)
    ):
        left = bits[:, int(round_index), int(pair[0]), :]
        right = bits[:, int(round_index), int(pair[1]), :]
        left_zero = ~np.any(left, axis=1)
        right_zero = ~np.any(right, axis=1)
        actual = ~np.any(left ^ right, axis=1)
        all_labels_match = all_labels_match and bool(
            np.array_equal(actual, labels[:, relation_index])
        )
        singleton_zero[:, relation_index, 0] = left_zero
        singleton_zero[:, relation_index, 1] = right_zero
        categories[:, relation_index] = np.where(
            actual,
            np.where(left_zero & right_zero, TRIVIAL_POSITIVE, NONTRIVIAL_POSITIVE),
            np.where(left_zero ^ right_zero, ONE_ZERO_NEGATIVE, NONTRIVIAL_NEGATIVE),
        )
        negative = np.flatnonzero(~actual)
        if len(negative):
            witness = witnesses[negative, relation_index]
            valid_range = (witness >= 0) & (witness < keys)
            if not bool(np.all(valid_range)):
                all_witnesses_match = False
            else:
                all_witnesses_match = all_witnesses_match and bool(
                    np.all((left ^ right)[negative, witness])
                )
    return {
        "singleton_zero": singleton_zero,
        "categories": categories,
        "all_relation_labels_recompute": all_labels_match,
        "all_negative_witnesses_replay_odd": all_witnesses_match,
    }


def evaluate_relation_decomposition(
    config: RelationDecompositionConfig,
    *,
    e62_gate: dict[str, Any],
    relation_labels: np.ndarray,
    singleton_zero: np.ndarray,
    categories: np.ndarray,
    all_relation_labels_recompute: bool,
    all_negative_witnesses_replay_odd: bool,
) -> dict[str, Any]:
    labels = np.asarray(relation_labels, dtype=np.bool_)
    zero = np.asarray(singleton_zero, dtype=np.bool_)
    category = np.asarray(categories, dtype=np.uint8)
    splits = variant_split_indices()
    split_metrics: dict[str, dict[str, float | int]] = {}
    baselines: dict[str, dict[str, float]] = {}
    for name, indices in splits.items():
        split_labels = labels[indices]
        split_categories = category[indices]
        trivial_positive = int(np.sum(split_categories == TRIVIAL_POSITIVE))
        nontrivial_positive = int(np.sum(split_categories == NONTRIVIAL_POSITIVE))
        one_zero_negative = int(np.sum(split_categories == ONE_ZERO_NEGATIVE))
        nontrivial_negative = int(np.sum(split_categories == NONTRIVIAL_NEGATIVE))
        total_positive = trivial_positive + nontrivial_positive
        split_metrics[name] = {
            "trivial_positive": trivial_positive,
            "nontrivial_positive": nontrivial_positive,
            "one_zero_negative": one_zero_negative,
            "nontrivial_negative": nontrivial_negative,
            "nontrivial_positive_fraction": (
                nontrivial_positive / total_positive if total_positive else 0.0
            ),
        }
        if name == "train":
            continue
        split_zero = zero[indices]
        scores = {
            "both_balanced": _binary_auc(
                split_labels.reshape(-1),
                np.all(split_zero, axis=-1).astype(np.float64).reshape(-1),
            ),
            "same_singleton_status": _binary_auc(
                split_labels.reshape(-1),
                (split_zero[..., 0] == split_zero[..., 1])
                .astype(np.float64)
                .reshape(-1),
            ),
            "either_balanced": _binary_auc(
                split_labels.reshape(-1),
                np.any(split_zero, axis=-1).astype(np.float64).reshape(-1),
            ),
        }
        baselines[name] = {**scores, "strongest_auc": max(scores.values())}
    source_checks = {
        "e62_training_gate_passed": e62_gate.get("status") == "pass"
        and e62_gate.get("decision")
        == "innovation2_small_spn_multicoordinate_relation_training_ready",
        "relation_shape_is_64x2048": labels.shape == (64, 2048),
        "singleton_status_shape_is_64x2048x2": zero.shape == (64, 2048, 2),
        "all_relation_labels_recompute": all_relation_labels_recompute,
        "all_negative_witnesses_replay_odd": all_negative_witnesses_replay_odd,
        "all_rows_have_one_of_four_categories": bool(
            np.all(np.isin(category, np.arange(4, dtype=np.uint8)))
        ),
    }
    width_checks = {
        "train_nontrivial_positive_at_least_3000": split_metrics["train"][
            "nontrivial_positive"
        ]
        >= 3000,
        "train_nontrivial_negative_at_least_3000": split_metrics["train"][
            "nontrivial_negative"
        ]
        >= 3000,
        "unseen_sbox_nontrivial_positive_at_least_768": split_metrics[
            "unseen_sbox"
        ]["nontrivial_positive"]
        >= 768,
        "unseen_sbox_nontrivial_negative_at_least_768": split_metrics[
            "unseen_sbox"
        ]["nontrivial_negative"]
        >= 768,
        "unseen_player_nontrivial_positive_at_least_768": split_metrics[
            "unseen_player"
        ]["nontrivial_positive"]
        >= 768,
        "unseen_player_nontrivial_negative_at_least_768": split_metrics[
            "unseen_player"
        ]["nontrivial_negative"]
        >= 768,
        "dual_nontrivial_positive_at_least_192": split_metrics["dual_unseen"][
            "nontrivial_positive"
        ]
        >= 192,
        "dual_nontrivial_negative_at_least_192": split_metrics["dual_unseen"][
            "nontrivial_negative"
        ]
        >= 192,
        "each_split_nontrivial_positive_fraction_at_least_0p25": all(
            float(metrics["nontrivial_positive_fraction"]) >= 0.25
            for metrics in split_metrics.values()
        ),
    }
    shortcut_checks = {
        "unseen_sbox_singleton_baseline_at_most_0p80": baselines["unseen_sbox"][
            "strongest_auc"
        ]
        <= 0.80,
        "unseen_player_singleton_baseline_at_most_0p80": baselines[
            "unseen_player"
        ]["strongest_auc"]
        <= 0.80,
        "dual_singleton_baseline_at_most_0p75": baselines["dual_unseen"][
            "strongest_auc"
        ]
        <= 0.75,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_small_spn_relation_decomposition_protocol_invalid"
        action = "repair E62 ownership, parity recomputation, or witness replay"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_relation_nontrivial_width_not_ready"
        action = "stop multicoordinate neural search; nontrivial cancellation is too narrow"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_small_spn_relation_singleton_shortcut_dominated"
        action = "stop multicoordinate neural search; singleton status explains the labels"
    else:
        status = "pass"
        decision = "innovation2_small_spn_relation_nontrivial_residual_ready"
        action = "audit the proven SPN pair-path reasoner on the nontrivial relation subset"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": {
            "split_metrics": split_metrics,
            "singleton_status_baselines": baselines,
        },
        "claim_scope": (
            "exact decomposition of all-256-key two-coordinate relation labels for "
            "16-bit synthetic SPNs; no neural training or real-cipher claim"
        ),
        "next_action": {
            "action": action,
            "new_relation_model": status == "pass",
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_small_spn_relation_decomposition",
            "split": split,
            **split_metrics[split],
            "singleton_status_auc": baselines.get(split, {}).get("strongest_auc"),
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for split in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ]
    return {"gate": gate, "result_rows": result_rows}
