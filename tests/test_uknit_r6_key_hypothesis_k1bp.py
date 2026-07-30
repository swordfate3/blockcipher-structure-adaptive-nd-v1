from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.uknit import UknitBc, uknit_round_keys
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    fit_diagonal_fisher,
)
from blockcipher_nd.tasks.innovation1.uknit_r6_key_hypothesis_k1bp import (
    EXPECTED_CANDIDATES,
    adjudicate_k1bp,
    build_cell_linear_lookup,
    dependency_cones,
    inverse_sbox_table,
    masked_r5_cell_values_from_r6,
    rank_sparse_hypotheses,
    round_key_runtime_bits,
    sparse_histogram_from_r5,
    strip_last_round,
    true_effective_hypothesis,
)


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/uknit64.json"


def test_k1bp_correct_full_round_key_strips_r6_to_exact_r5_ciphertexts() -> None:
    key = 0x0123456789ABCDEFFEDCBA9876543210
    plaintexts = [0, 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210] * 2
    r5 = [UknitBc(rounds=5, key=key).encrypt(value) for value in plaintexts]
    r6 = [UknitBc(rounds=6, key=key).encrypt(value) for value in plaintexts]
    features = _blocks_to_features(r6)
    suffix = load_runtime_spn_descriptor(DESCRIPTOR, rounds=1, round_start=5).structure

    stripped = strip_last_round(
        features,
        last_transition=suffix,
        round_key_bits=round_key_runtime_bits(key),
    )

    assert np.array_equal(stripped, _blocks_to_features(r5))
    assert uknit_round_keys(key)[5] != 0


def test_k1bp_every_one_cell_dependency_cone_is_exactly_twelve_bits() -> None:
    transition = load_runtime_spn_descriptor(
        DESCRIPTOR, rounds=1, round_start=4
    ).structure

    cones = dependency_cones(transition)

    assert len(cones) == 16
    assert {cone.source_key_bits for cone in cones} == {12}
    assert {cone.effective_key_bits for cone in cones} == {4}
    assert {cone.candidate_count for cone in cones} == {16}
    assert {cone.source_key_equivalence_size for cone in cones} == {256}
    assert all(len(cone.target_bits) == 4 for cone in cones)
    assert all(len(set(cone.source_bits)) == 12 for cone in cones)


def test_k1bp_sparse_lookup_matches_full_public_r5_composition() -> None:
    generator = np.random.default_rng(23)
    dataset = DifferentialDataset(
        features=generator.integers(0, 2, size=(32, 512), dtype=np.uint8),
        labels=np.tile(np.asarray((0, 1), dtype=np.uint8), 16),
        metadata={"pairs_per_sample": 4},
    )
    transition = load_runtime_spn_descriptor(
        DESCRIPTOR, rounds=1, round_start=4
    ).structure
    cone = dependency_cones(transition)[7]
    linear_lookup = build_cell_linear_lookup(transition, cone)
    inverse_table = inverse_sbox_table(transition, cone.target_cell)

    sparse = sparse_histogram_from_r5(
        dataset,
        cone=cone,
        linear_lookup=linear_lookup,
        inverse_table=inverse_table,
    )

    assert sparse.shape == (32, 16)
    assert np.allclose(sparse.sum(axis=1), 1.0)
    assert np.all(np.isin(sparse, (0.0, 0.25, 0.5, 0.75, 1.0)))


def test_k1bp_exhaustive_rank_recovers_synthetic_true_hypothesis() -> None:
    rows = 256
    pairs = 4
    true_guess = 0xA
    generator = np.random.default_rng(31)
    left = generator.integers(0, EXPECTED_CANDIDATES, size=(rows, pairs), dtype=np.uint8)
    right = left.copy()
    labels = np.tile(np.asarray((0, 1), dtype=np.uint8), rows // 2)
    inverse_table = np.asarray((0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10), dtype=np.uint8)
    right[labels == 1] ^= np.uint8(true_guess)
    train_features = np.zeros((rows, 16), dtype=np.float32)
    train_features[labels == 0, 0] = 1.0
    train_features[labels == 1, int(true_guess & 0xF)] = 1.0
    scorer = fit_diagonal_fisher(train_features, labels)

    result = rank_sparse_hypotheses(
        left_values=left,
        right_values=right,
        labels=labels,
        inverse_table=inverse_table,
        scorer=scorer,
        true_hypothesis=true_guess,
    )

    assert result.candidate_count == EXPECTED_CANDIDATES
    assert result.true_rank >= 1
    assert result.query_pairs == int((labels == 1).sum()) * 4
    assert np.isfinite(result.correct_auc)


def test_k1bp_true_cone_guess_matches_masked_r6_codes() -> None:
    key = 0x44444444444444444444444444444444
    transition = load_runtime_spn_descriptor(
        DESCRIPTOR, rounds=1, round_start=4
    ).structure
    suffix = load_runtime_spn_descriptor(DESCRIPTOR, rounds=1, round_start=5).structure
    cone = dependency_cones(transition)[0]
    plaintexts = list(range(8))
    r6 = [UknitBc(rounds=6, key=key).encrypt(value) for value in plaintexts]
    r5 = [UknitBc(rounds=5, key=key).encrypt(value) for value in plaintexts]
    r6_dataset = DifferentialDataset(
        features=_blocks_to_features(r6),
        labels=np.asarray((0,), dtype=np.uint8),
        metadata={"pairs_per_sample": 4},
    )
    r5_dataset = DifferentialDataset(
        features=_blocks_to_features(r5),
        labels=r6_dataset.labels,
        metadata={"pairs_per_sample": 4},
    )

    linear_lookup = build_cell_linear_lookup(transition, cone)
    left, right = masked_r5_cell_values_from_r6(
        r6_dataset,
        last_transition=suffix,
        cone=cone,
        linear_lookup=linear_lookup,
    )
    true_guess = true_effective_hypothesis(key, cone, linear_lookup)
    runtime_r5 = torch.as_tensor(r5_dataset.features, dtype=torch.float32).reshape(
        1, 4, 2, 64
    ).flip(-1)
    weights = 1 << np.arange(12, dtype=np.uint16)
    expected_left_codes = (
        runtime_r5[:, :, 0, list(cone.source_bits)].numpy().astype(np.uint16) * weights
    ).sum(axis=-1)
    expected_right_codes = (
        runtime_r5[:, :, 1, list(cone.source_bits)].numpy().astype(np.uint16) * weights
    ).sum(axis=-1)
    expected_left = linear_lookup[expected_left_codes]
    expected_right = linear_lookup[expected_right_codes]

    assert np.array_equal(left ^ np.uint16(true_guess), expected_left)
    assert np.array_equal(right ^ np.uint16(true_guess), expected_right)


def test_k1bp_reports_weak_discovery_without_relaxing_pass_gate() -> None:
    discovery_rows = [{"target_cell": 0, "minimum_fresh_auc": 0.514}]
    oracle_rows = [
        {
            "seed": seed,
            "split": split,
            "correct_key_auc": 0.97,
            "correct_minus_best_wrong_auc": 0.45,
        }
        for seed in (3, 4)
        for split in ("same_key_fresh", "cross_key_validation")
    ]
    sparse_rows = [
        {
            "seed": seed,
            "split": split,
            "r5_sparse_auc": 0.501,
            "correct_auc": 0.499,
            "true_rank": 1,
            "wrong_sbox_true_rank": 2,
            "label_shuffle_true_rank": 3,
        }
        for seed in (3, 4)
        for split in ("same_key_fresh", "cross_key_validation")
    ]

    gate = adjudicate_k1bp(
        protocol_checks={"frozen": True},
        discovery_rows=discovery_rows,
        full_oracle_rows=oracle_rows,
        sparse_rank_rows=sparse_rows,
        selected_cell=0,
        thresholds={
            "weak_signal_auc_floor": 0.51,
            "discovery_sparse_auc_floor": 0.55,
            "confirmation_sparse_auc_floor": 0.55,
            "full_oracle_auc_floor": 0.9,
            "full_oracle_wrong_key_margin": 0.01,
            "required_true_key_rank": 1,
        },
    )

    assert gate["status"] == "hold"
    assert gate["bounded_route_pass"] is False
    assert gate["weak_signal_observed"] is True
    assert gate["weak_signal_confirmed"] is False
    assert gate["evidence_tier"] == "weak_discovery_only_unconfirmed"


def _blocks_to_features(blocks: list[int]) -> np.ndarray:
    if len(blocks) % 8:
        raise ValueError("fixture blocks must contain complete four-pair samples")
    bits = np.asarray(
        [[(value >> bit) & 1 for bit in range(63, -1, -1)] for value in blocks],
        dtype=np.uint8,
    )
    return bits.reshape(-1, 8, 64).reshape(-1, 512)
