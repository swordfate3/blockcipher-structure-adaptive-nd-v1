from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_exact_gf2_operator_response_k1bh import (
    render_k1bh_svg,
)
from blockcipher_nd.models.structure.spn.exact_gf2_operator_response import (
    apply_numpy_gf2_operator,
    exact_gf2_operator_response,
    extract_exact_gf2_operator_features,
)
from blockcipher_nd.models.structure.spn.gf2_boolean_view import (
    apply_gf2_operator,
    gf2_boolean_views,
)
from blockcipher_nd.tasks.innovation1.uknit_family_exact_gf2_operator_response_k1bh import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    OPERATOR_CONDITIONS,
    RESULT_CONDITIONS,
    RUN_ID,
    SCORER_CONDITIONS,
    adjudicate_k1bh,
    deterministic_label_shuffle,
    load_and_validate_config,
    load_authority,
)


def test_k1bh_numpy_response_matches_existing_torch_gf2_on_real_structures() -> None:
    config = load_and_validate_config()
    dataset_rows, datasets, structures, corrupted, cross, checks = load_authority(
        config
    )

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert set(structures) == set(corrupted) == set(cross)
    for cipher, seed in (("uknit64", 3), ("midori64", 6), ("dialga128", 0)):
        structure = structures[cipher]
        flat = np.asarray(
            datasets[(cipher, seed, "same_key_fresh")].features[:3],
            dtype=np.uint8,
        )
        runtime = flat.reshape(3, 4, 2, structure.block_bits)[..., ::-1].copy()
        observed = exact_gf2_operator_response(runtime, structure)
        expected = gf2_boolean_views(
            torch.as_tensor(runtime, dtype=torch.float32),
            structure,
        ).numpy()
        assert np.array_equal(observed, expected)

        raw = observed[..., :3]
        numpy_single = apply_numpy_gf2_operator(
            raw,
            structure.inverse_linear_matrices[0].numpy(),
        )
        torch_single = apply_gf2_operator(
            torch.as_tensor(raw, dtype=torch.float32),
            structure.inverse_linear_matrices[0],
        ).numpy()
        assert np.array_equal(numpy_single, torch_single)

        pooled = extract_exact_gf2_operator_features(flat, structure)
        assert pooled.shape == (3, structure.block_bits * 12)
        assert np.array_equal(pooled, observed.mean(axis=1).reshape(3, -1))


def test_k1bh_label_shuffle_is_exactly_seeded_and_count_preserving() -> None:
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 64)

    first, first_sha = deterministic_label_shuffle(labels, seed=73100)
    repeated, repeated_sha = deterministic_label_shuffle(labels, seed=73100)
    other, other_sha = deterministic_label_shuffle(labels, seed=73200)

    assert np.array_equal(first, repeated)
    assert first_sha == repeated_sha
    assert not np.array_equal(first, labels)
    assert not np.array_equal(first, other)
    assert first_sha != other_sha
    assert np.array_equal(np.sort(first), np.sort(labels))


def test_k1bh_gate_requires_every_panel_and_correct_scorer_reuse() -> None:
    features, scorers, results = _synthetic_artifacts()
    config = load_and_validate_config()

    passed = adjudicate_k1bh(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"source": True},
    )
    assert len(features) == EXPECTED_FEATURE_ROWS
    assert len(scorers) == EXPECTED_SCORER_ROWS
    assert len(results) == EXPECTED_RESULT_ROWS
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("exact_operator_topology_signal_supported")
    assert all(passed["protocol_checks"].values())
    assert all(passed["research_checks"].values())

    weak = deepcopy(results)
    row = next(
        item
        for item in weak
        if item["replica"] == 1
        and item["cipher_key"] == "uknit64"
        and item["split"] == "cross_key_validation"
        and item["condition"] == "correct_operator"
    )
    row["auc"] = 0.54
    held = adjudicate_k1bh(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=weak,
        source_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("exact_operator_signal_unstable")

    refitted = deepcopy(results)
    row = next(
        item
        for item in refitted
        if item["condition"] == "same_summary_corrupted_operator"
    )
    row["scorer_sha256"] = "wrong-specific-refit"
    invalid = adjudicate_k1bh(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=refitted,
        source_checks={"source": True},
    )
    assert invalid["status"] == "invalid"
    assert not invalid["protocol_checks"][
        "correct_fit_scorer_reused_for_all_operator_controls"
    ]


def test_k1bh_plot_uses_clear_chinese_margin_panels(tmp_path: Path) -> None:
    features, scorers, results = _synthetic_artifacts()
    gate = adjudicate_k1bh(
        config=load_and_validate_config(),
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"source": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1bh_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_panels"] == 12
    assert "正确的 GF(2) 扩散算子是否留下独有的标签信号" in svg
    assert "错误算子不得重新拟合" in svg
    assert "核心裁决：正确拓扑是否在每个面板都独有" in svg
    assert "不是神经网络准确率" in svg


def _synthetic_artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    feature_rows: list[dict[str, object]] = []
    scorer_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    seeds = {
        0: {"uknit64": 3, "midori64": 6, "dialga128": 0},
        1: {"uknit64": 4, "midori64": 7, "dialga128": 1},
    }
    rounds = {"uknit64": 5, "midori64": 4, "dialga128": 4}
    dimensions = {"uknit64": 768, "midori64": 768, "dialga128": 1536}
    aucs = {
        "correct_operator": 0.65,
        "same_summary_corrupted_operator": 0.60,
        "cross_cipher_operator": 0.59,
        "identity_operator": 0.60,
        "label_shuffled_correct_operator": 0.50,
    }
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            seed = seeds[replica][cipher]
            correct_scorer = f"correct-scorer-{replica}-{cipher}"
            shuffled_scorer = f"shuffle-scorer-{replica}-{cipher}"
            for condition in SCORER_CONDITIONS:
                scorer_rows.append(
                    {
                        "run_id": RUN_ID,
                        "replica": replica,
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "fit_condition": condition,
                        "fit_split": "train_seen",
                        "fit_rows": 4096,
                        "feature_dim": dimensions[cipher],
                        "variance_floor": 1e-6,
                        "class_counts": [2048, 2048],
                        "scorer_sha256": (
                            correct_scorer
                            if condition == "correct_operator"
                            else shuffled_scorer
                        ),
                        "label_permutation_sha256": (
                            None
                            if condition == "correct_operator"
                            else f"permutation-{replica}-{cipher}"
                        ),
                        "training_performed": False,
                        "neural_parameter_count": 0,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
            for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
                rows = 4096 if split == "train_seen" else 2048
                dataset_sha = f"dataset-{replica}-{cipher}-{split}"
                correct_feature_sha = f"feature-{replica}-{cipher}-{split}-correct"
                for condition in OPERATOR_CONDITIONS:
                    feature_rows.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "rounds": rounds[cipher],
                            "split": split,
                            "condition": condition,
                            "rows": rows,
                            "feature_dim": dimensions[cipher],
                            "expected_feature_dim": dimensions[cipher],
                            "feature_sha256": (
                                correct_feature_sha
                                if condition == "correct_operator"
                                else f"feature-{replica}-{cipher}-{split}-{condition}"
                            ),
                            "dataset_sha256": dataset_sha,
                            "operator_sha256": f"operator-{cipher}-{condition}",
                            "response_rms_from_correct": (
                                0.0 if condition == "correct_operator" else 0.1
                            ),
                            "finite": True,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "pairs_per_sample": 4,
                            "data_generation_performed": False,
                        }
                    )
                if split == "train_seen":
                    continue
                for condition in RESULT_CONDITIONS:
                    label_shuffle = condition == "label_shuffled_correct_operator"
                    feature_sha = (
                        correct_feature_sha
                        if condition in {
                            "correct_operator",
                            "label_shuffled_correct_operator",
                        }
                        else f"feature-{replica}-{cipher}-{split}-{condition}"
                    )
                    result_rows.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "rounds": rounds[cipher],
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows": rows,
                            "auc": aucs[condition],
                            "zero_threshold_accuracy": 0.60,
                            "score_mean": 0.0,
                            "score_std": 1.0,
                            "score_min": -1.0,
                            "score_max": 1.0,
                            "feature_dim": dimensions[cipher],
                            "feature_sha256": feature_sha,
                            "dataset_sha256": dataset_sha,
                            "scorer_sha256": (
                                shuffled_scorer if label_shuffle else correct_scorer
                            ),
                            "fit_condition": (
                                "label_shuffled_correct_operator"
                                if label_shuffle
                                else "correct_operator"
                            ),
                            "fit_split": "train_seen",
                            "fit_rows": 4096,
                            "pairs_per_sample": 4,
                            "negative_mode": "encrypted_random_plaintexts",
                            "variance_floor": 1e-6,
                            "training_performed": False,
                            "neural_parameter_count": 0,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return feature_rows, scorer_rows, result_rows
