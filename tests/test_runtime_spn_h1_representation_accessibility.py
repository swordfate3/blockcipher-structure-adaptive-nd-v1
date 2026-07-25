from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_representation_accessibility import (
    adjudicate_h1_representation_accessibility,
    load_and_validate_h1_representation_accessibility_config,
    stratified_two_fold_indices,
    stratified_two_fold_ridge_scores,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_h1_representation_accessibility_a4_seed0_seed1.json"
)


def test_frozen_h1_a4_config_is_valid() -> None:
    config = load_and_validate_h1_representation_accessibility_config(
        CONFIG,
        project_root=ROOT,
    )

    assert config["audit"]["probe"] == "stratified_two_fold_closed_form_ridge"
    assert config["audit"]["split"] == "validation"


def test_stratified_folds_are_disjoint_balanced_and_complete() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    first, second = stratified_two_fold_indices(labels)

    assert set(first).isdisjoint(second)
    assert sorted(np.concatenate((first, second)).tolist()) == list(range(8))
    assert np.bincount(labels[first], minlength=2).tolist() == [2, 2]
    assert np.bincount(labels[second], minlength=2).tolist() == [2, 2]


def test_closed_form_probe_predicts_unseen_fold_on_separable_data() -> None:
    labels = np.asarray([0, 1] * 20, dtype=np.uint8)
    representations = np.column_stack(
        (
            labels.astype(np.float64) * 2.0 - 1.0,
            np.linspace(-0.1, 0.1, len(labels)),
        )
    )

    scores = stratified_two_fold_ridge_scores(
        representations,
        labels,
        ridge_lambda=0.01,
    )

    assert np.all(scores[labels == 1] > scores[labels == 0].max())


def test_gate_supports_shared_classifier_bottleneck() -> None:
    gate = adjudicate_h1_representation_accessibility(
        _payload(probe=(0.62, 0.60), shared=(0.49, 0.48), controls=(0.50, 0.51))
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("shared_classifier_bottleneck_supported")


def test_gate_calls_weak_representation_when_both_probes_are_low() -> None:
    gate = adjudicate_h1_representation_accessibility(
        _payload(probe=(0.52, 0.53), shared=(0.50, 0.51), controls=(0.49, 0.50))
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("shared_representation_weak")


def test_gate_fails_closed_on_invalid_protocol() -> None:
    payload = _payload(
        probe=(0.62, 0.60),
        shared=(0.49, 0.48),
        controls=(0.50, 0.51),
    )
    payload["validation"] = {"status": "fail"}

    gate = adjudicate_h1_representation_accessibility(payload)

    assert gate["status"] == "invalid"


def _payload(
    *,
    probe: tuple[float, float],
    shared: tuple[float, float],
    controls: tuple[float, float],
) -> dict[str, object]:
    config = load_and_validate_h1_representation_accessibility_config(
        CONFIG,
        project_root=ROOT,
    )
    return {
        "config": config,
        "metrics": [
            {
                "checkpoint_role": "a3",
                "seed": seed,
                "task": "skinny64",
                "shared_classifier_auc": shared[seed],
                "closed_form_probe_auc": probe[seed],
                "probe_gain": probe[seed] - shared[seed],
                "centroid_separation_ratio": 0.1,
            }
            for seed in (0, 1)
        ],
        "controls": [
            {"seed": seed, "closed_form_probe_auc": controls[seed]} for seed in (0, 1)
        ],
        "validation": {"status": "pass"},
    }
