from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_window_depth_k1bo import (
    EXPECTED_FEATURE_DIMS,
    R5_EXACT2_POSITION,
    R6_EXACT2_POSITION,
    R6_EXACT3_INVARIANT,
    R6_EXACT3_POSITION,
    R6_RAW,
    R6_SHUFFLE3_INVARIANT,
    R6_SHUFFLE3_POSITION,
    R6_VIEWS,
    R6_WRONG3_INVARIANT,
    R6_WRONG3_POSITION,
    extract_window_depth_views,
)


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/uknit64.json"


def test_k1bo_three_round_views_extend_two_round_views_without_changing_data() -> None:
    two = load_runtime_spn_descriptor(DESCRIPTOR, rounds=2, round_start=4).structure
    three = load_runtime_spn_descriptor(DESCRIPTOR, rounds=3, round_start=3).structure
    wrong = three.shuffled_sbox_assignments(20260728)
    dataset = _dataset(rows=32, seed=7)

    views, manifests, prefix_equal = extract_window_depth_views(
        dataset,
        exact_two=two,
        exact_three=three,
        wrong_three=wrong,
        rounds=6,
        batch_size=11,
    )

    assert set(views) == set(R6_VIEWS)
    assert prefix_equal is True
    assert all(values.shape == (32, EXPECTED_FEATURE_DIMS[name]) for name, values in views.items())
    assert np.array_equal(views[R6_EXACT3_POSITION], views[R6_SHUFFLE3_POSITION])
    assert np.array_equal(views[R6_EXACT3_INVARIANT], views[R6_SHUFFLE3_INVARIANT])
    assert not np.array_equal(views[R6_EXACT3_POSITION], views[R6_WRONG3_POSITION])
    assert not np.array_equal(views[R6_EXACT3_INVARIANT], views[R6_WRONG3_INVARIANT])
    assert np.array_equal(
        views[R6_EXACT3_POSITION].reshape(32, 7, 16, 16)[:, :5].reshape(32, -1),
        views[R6_EXACT2_POSITION],
    )
    assert np.array_equal(
        views[R6_EXACT3_POSITION].reshape(32, 7, 16, 16)[:, :1].reshape(32, -1),
        views[R6_RAW],
    )
    assert all(row["normalized"] is True for row in manifests.values())
    assert all(row["finite"] is True for row in manifests.values())


def test_k1bo_r5_anchor_retains_five_stage_geometry() -> None:
    two = load_runtime_spn_descriptor(DESCRIPTOR, rounds=2, round_start=3).structure
    dataset = _dataset(rows=24, seed=9)

    views, manifests, prefix_equal = extract_window_depth_views(
        dataset,
        exact_two=two,
        rounds=5,
        batch_size=7,
    )

    assert set(views) == {R5_EXACT2_POSITION}
    assert views[R5_EXACT2_POSITION].shape == (
        24,
        EXPECTED_FEATURE_DIMS[R5_EXACT2_POSITION],
    )
    assert manifests[R5_EXACT2_POSITION]["normalized"] is True
    assert prefix_equal is True


def _dataset(*, rows: int, seed: int) -> DifferentialDataset:
    generator = np.random.default_rng(seed)
    features = generator.integers(0, 2, size=(rows, 512), dtype=np.uint8)
    labels = np.tile(np.asarray((0, 1), dtype=np.uint8), rows // 2)
    return DifferentialDataset(
        features=features,
        labels=labels,
        metadata={
            "rounds": 6,
            "pairs_per_sample": 4,
            "negative_mode": "encrypted_random_plaintexts",
        },
    )
