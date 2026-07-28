from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1o import render_k1o_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1n import read_tasks
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    build_k1n_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    LABEL_SHUFFLE_VIEW,
    RUN_ID,
    SOURCE_RUN_ID,
    VIEW_NAMES,
    WRONG_SBOX_VIEW,
    adjudicate_k1o,
    deterministic_label_shuffle,
    extract_k1o_feature_views,
    fit_diagonal_fisher,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_exact_operator_composition_k1n_2048_seed0_seed1.csv"
)


def test_k1o_histograms_preserve_declared_geometry_and_normalization() -> None:
    tasks = candidate_task_map(read_tasks(PLAN))
    task = tasks[("uknit64", 0)]
    exact = build_k1n_control(
        task=task,
        condition="exact_composition",
        input_bits=512,
    ).runtime_structure
    wrong = build_k1n_control(
        task=task,
        condition="wrong_sbox_semantics",
        input_bits=512,
    ).runtime_structure
    generator = np.random.default_rng(20260728)
    dataset = DifferentialDataset(
        features=generator.integers(0, 2, size=(16, 512), dtype=np.uint8),
        labels=np.tile(np.array([0, 1], dtype=np.uint8), 8),
        metadata={},
    )

    views, manifests = extract_k1o_feature_views(
        dataset,
        exact_structure=exact,
        wrong_sbox_structure=wrong,
        batch_size=8,
    )

    assert set(views) == set(VIEW_NAMES)
    assert all(
        values.shape == (16, EXPECTED_FEATURE_DIMS[name])
        for name, values in views.items()
    )
    assert all(row["finite"] for row in manifests.values())
    assert all(row["nonnegative"] for row in manifests.values())
    assert all(row["normalized"] for row in manifests.values())
    assert views[CANDIDATE_VIEW] is views[LABEL_SHUFFLE_VIEW]
    assert (
        manifests[CANDIDATE_VIEW]["feature_sha256"]
        != manifests[WRONG_SBOX_VIEW]["feature_sha256"]
    )


def test_k1o_diagonal_fisher_has_fixed_positive_orientation() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 1.0],
            [0.2, 0.0],
            [0.8, 1.0],
            [0.9, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    scorer = fit_diagonal_fisher(features, labels)
    scores = scorer.score(features)

    assert binary_auc(labels, scores) == 1.0
    assert scorer.class_counts == (3, 3)
    assert scorer.weights[0] > 0.0
    assert scorer.variance_floor == 1e-6
    assert len(scorer.sha256) == 64


def test_k1o_label_shuffle_is_deterministic_nonidentity_and_seed_bound() -> None:
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 64)

    first, first_sha = deterministic_label_shuffle(labels, seed=0)
    repeated, repeated_sha = deterministic_label_shuffle(labels, seed=0)
    other, other_sha = deterministic_label_shuffle(labels, seed=1)

    assert np.array_equal(first, repeated)
    assert first_sha == repeated_sha
    assert not np.array_equal(first, labels)
    assert not np.array_equal(first, other)
    assert first_sha != other_sha
    assert np.array_equal(np.sort(first), np.sort(labels))


def test_k1o_gate_requires_every_fresh_split_and_semantic_control() -> None:
    result_rows, feature_rows, scorer_rows = synthetic_artifacts()

    gate = adjudicate_k1o(
        result_rows=result_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        source_checks={"source_binding": True},
    )

    assert len(result_rows) == EXPECTED_RESULT_ROWS
    assert len(feature_rows) == EXPECTED_FEATURE_ROWS
    assert len(scorer_rows) == EXPECTED_SCORER_ROWS
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("position_preserving_signal_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = deepcopy(result_rows)
    for row in failed:
        if (
            row["seed"] == 1
            and row["split"] == "cross_key_validation"
            and row["view"] == CANDIDATE_VIEW
        ):
            row["auc"] = 0.54
    held = adjudicate_k1o(
        result_rows=failed,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        source_checks={"source_binding": True},
    )

    assert held["status"] == "hold"
    assert held["decision"].endswith("partial_state_signal_unstable")
    assert not held["research_checks"][
        "seed1_cross_key_validation_exact_auc_floor"
    ]


def test_k1o_plot_explains_signal_audit_in_chinese(tmp_path: Path) -> None:
    result_rows, feature_rows, scorer_rows = synthetic_artifacts()
    gate = adjudicate_k1o(
        result_rows=result_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        source_checks={"source_binding": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1o_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "uKNIT 五轮的精确局部状态里是否真的存在可学习信号" in svg
    assert "同一密钥的新样本" in svg
    assert "正确局部状态的净优势" in svg
    assert "打乱训练标签" in svg
    assert report["uses_point_comparison_instead_of_overlapping_curves"] is True


def synthetic_artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    aucs = {
        "raw_position_histogram": 0.60,
        "exact_five_stage_position_histogram": 0.66,
        "no_sbox_five_stage_position_histogram": 0.63,
        "wrong_sbox_five_stage_position_histogram": 0.62,
        "exact_five_stage_invariant_histogram": 0.63,
        "label_shuffled_exact_position_histogram": 0.50,
    }
    result_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    scorer_rows: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        label_permutation = f"permutation-{seed}"
        for view in VIEW_NAMES:
            scorer_rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "uknit64",
                    "seed": seed,
                    "view": view,
                    "fit_split": "train_seen",
                    "fit_rows": 4096,
                    "feature_dim": EXPECTED_FEATURE_DIMS[view],
                    "variance_floor": 1e-6,
                    "class0_rows": 2048,
                    "class1_rows": 2048,
                    "label_permutation_sha256": (
                        label_permutation if view == LABEL_SHUFFLE_VIEW else None
                    ),
                    "training_performed": False,
                    "neural_parameter_count": 0,
                    "optimizer_steps": 0,
                    "epochs": 0,
                }
            )
        for split in EXPECTED_SPLITS:
            rows = 4096 if split == "train_seen" else 2048
            dataset_sha = f"dataset-{seed}-{split}"
            exact_sha = f"exact-{seed}-{split}"
            for view in VIEW_NAMES:
                feature_sha = (
                    exact_sha
                    if view in {CANDIDATE_VIEW, LABEL_SHUFFLE_VIEW}
                    else f"{view}-{seed}-{split}"
                )
                feature_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": "uknit64",
                        "seed": seed,
                        "split": split,
                        "view": view,
                        "rows": rows,
                        "feature_dim": EXPECTED_FEATURE_DIMS[view],
                        "dataset_sha256": dataset_sha,
                        "feature_sha256": feature_sha,
                        "finite": True,
                        "nonnegative": True,
                        "normalized": True,
                    }
                )
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "source_run_id": SOURCE_RUN_ID,
                        "cipher_key": "uknit64",
                        "rounds": 5,
                        "seed": seed,
                        "split": split,
                        "view": view,
                        "rows": rows,
                        "auc": aucs[view],
                        "zero_threshold_accuracy": 0.60,
                        "score_mean": 0.0,
                        "score_std": 1.0,
                        "score_min": -1.0,
                        "score_max": 1.0,
                        "feature_dim": EXPECTED_FEATURE_DIMS[view],
                        "dataset_sha256": dataset_sha,
                        "feature_sha256": feature_sha,
                        "scorer_sha256": f"scorer-{seed}-{view}",
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
    return result_rows, feature_rows, scorer_rows
