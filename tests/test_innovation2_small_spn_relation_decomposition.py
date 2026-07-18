from __future__ import annotations

import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_relation_decomposition import (
    NONTRIVIAL_NEGATIVE,
    NONTRIVIAL_POSITIVE,
    ONE_ZERO_NEGATIVE,
    TRIVIAL_POSITIVE,
    decompose_relation_labels,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_relation_decomposition import (
    render_relation_decomposition,
)


def test_relation_decomposition_distinguishes_four_exact_cases() -> None:
    bits = np.zeros((2, 1, 4, 4), dtype=np.bool_)
    bits[1, 0, 0, 1] = True
    bits[1, 0, 1, 1] = True
    bits[0, 0, 3, 2] = True
    bits[1, 0, 2, 1] = True
    bits[1, 0, 3, 2] = True
    pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    rounds = np.asarray([0, 0], dtype=np.int64)
    labels = np.asarray([[True, False], [True, False]], dtype=np.bool_)
    witnesses = np.asarray([[-1, 2], [-1, 1]], dtype=np.int16)

    result = decompose_relation_labels(bits, pairs, rounds, labels, witnesses)

    assert result["all_relation_labels_recompute"] is True
    assert result["all_negative_witnesses_replay_odd"] is True
    assert result["categories"].tolist() == [
        [TRIVIAL_POSITIVE, ONE_ZERO_NEGATIVE],
        [NONTRIVIAL_POSITIVE, NONTRIVIAL_NEGATIVE],
    ]
    assert result["singleton_zero"].shape == (2, 2, 2)


def test_relation_decomposition_rejects_wrong_negative_witness() -> None:
    bits = np.zeros((1, 1, 2, 2), dtype=np.bool_)
    bits[0, 0, 1, 1] = True

    result = decompose_relation_labels(
        bits,
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([[False]], dtype=np.bool_),
        np.asarray([[0]], dtype=np.int16),
    )

    assert result["all_relation_labels_recompute"] is True
    assert result["all_negative_witnesses_replay_odd"] is False


def test_decomposition_plot_exposes_singleton_shortcut(tmp_path) -> None:
    split_metrics = {
        name: {
            "trivial_positive": 1000,
            "nontrivial_positive": 6,
            "one_zero_negative": 500,
            "nontrivial_negative": 50,
        }
        for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    }
    summary = {
        "gate": {
            "metrics": {
                "split_metrics": split_metrics,
                "singleton_status_baselines": {
                    name: {"strongest_auc": 0.999}
                    for name in ("unseen_sbox", "unseen_player", "dual_unseen")
                },
            }
        }
    }
    output = tmp_path / "curves.svg"

    render_relation_decomposition(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E64" in svg
    assert "正类几乎全是both-balanced" in svg
    assert "停止多坐标网络路线" in svg
