from __future__ import annotations

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_multicoordinate_relations import (
    COORDINATES,
    KEYS,
    ROUNDS,
    VARIANTS,
    generate_candidate_pairs,
    label_candidate_pairs,
    select_relation_templates,
    selected_relation_labels_and_witnesses,
    variant_split_indices,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_multicoordinate_relations import (
    render_small_spn_multicoordinate_relations,
)


def test_candidate_pairs_are_deterministic_unique_and_distinct() -> None:
    first = generate_candidate_pairs(count=256)
    second = generate_candidate_pairs(count=256)

    assert np.array_equal(first, second)
    assert first.shape == (256, 2)
    assert np.all(first[:, 0] < first[:, 1])
    assert len({tuple(int(value) for value in row) for row in first}) == 256


def test_pair_labels_and_witnesses_use_all_key_parities() -> None:
    bits = np.zeros((VARIANTS, ROUNDS, COORDINATES, KEYS), dtype=np.bool_)
    bits[0, 0, 1, 7] = True
    pairs = np.asarray([[0, 1]], dtype=np.uint16)

    labels = label_candidate_pairs(bits, pairs)
    selected_labels, witnesses, valid = selected_relation_labels_and_witnesses(
        bits,
        pairs,
        np.asarray([0], dtype=np.uint8),
        np.asarray([0], dtype=np.int32),
    )

    assert labels.shape == (VARIANTS, ROUNDS, 1)
    assert labels[0, 0, 0] is np.False_
    assert labels[1, 0, 0] is np.True_
    assert np.array_equal(selected_labels[:, 0], labels[:, 0, 0])
    assert witnesses[0, 0] == 7
    assert np.all(witnesses[1:, 0] == -1)
    assert valid.tolist() == [True]


def test_selection_uses_only_frozen_train_variants() -> None:
    labels = np.zeros((VARIANTS, ROUNDS, 10), dtype=np.bool_)
    train = variant_split_indices()["train"]
    labels[train[:18], :, :] = True
    labels[variant_split_indices()["dual_unseen"], :, :] = True

    rounds, candidates, counts = select_relation_templates(labels)

    assert len(rounds) == 40
    assert len(candidates) == 40
    assert np.all(counts == 18)
    assert [int(np.sum(rounds == index)) for index in range(ROUNDS)] == [10] * 4


def test_variant_splits_remain_36_12_12_4() -> None:
    splits = variant_split_indices()

    assert [
        len(splits[name])
        for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ] == [36, 12, 12, 4]
    assert len(set(np.concatenate(tuple(splits.values())).tolist())) == VARIANTS


def test_readiness_plot_explains_strict_width_and_next_training(tmp_path) -> None:
    summary = {
        "gate": {
            "metrics": {
                "per_round_selected_templates": [0, 1024, 1024, 0],
                "split_metrics": {
                    "train": {"positive": 41531, "negative": 32197},
                    "unseen_sbox": {"positive": 16376, "negative": 8200},
                    "unseen_player": {"positive": 13723, "negative": 10853},
                    "dual_unseen": {"positive": 6158, "negative": 2034},
                },
                "marginal_baselines": {
                    "unseen_sbox": {"strongest_auc": 0.668},
                    "unseen_player": {"strongest_auc": 0.653},
                    "dual_unseen": {"strongest_auc": 0.686},
                },
            }
        }
    }
    output = tmp_path / "curves.svg"

    render_small_spn_multicoordinate_relations(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E62" in svg
    assert "所有split都有宽正负类" in svg
    assert "可以训练DeepSets与RCCA" in svg
