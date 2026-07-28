from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_uknit_family_ctspn_k1p import read_tasks
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1p import render_k1p_svg
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    build_k1n_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1p import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    RUN_ID,
    VIEW_NAMES,
    adjudicate_k1p,
    evaluate_lower_round,
    reuse_k1o_anchor,
    validate_k1p_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_partial_state_round_calibration_"
    "k1p_r3_r4_r5_seed0_seed1.csv"
)


def test_k1p_plan_changes_only_round_and_aligned_runtime_window() -> None:
    tasks = read_tasks(PLAN)
    checks = validate_k1p_tasks(tasks)

    assert len(tasks) == 6
    assert all(checks.values())
    assert {
        (
            task["rounds"],
            task["seed"],
            task["model_options"]["runtime_round_start"],
        )
        for task in tasks
    } == {
        (rounds, seed, rounds - 2)
        for rounds in (3, 4, 5)
        for seed in (0, 1)
    }


def test_k1p_lower_round_reuses_k1o_feature_and_scorer_contract() -> None:
    task = next(
        task
        for task in read_tasks(PLAN)
        if task["rounds"] == 3 and task["seed"] == 0
    )
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
    datasets = {
        (seed, split): random_dataset(seed * 10 + split_index)
        for seed in (0, 1)
        for split_index, split in enumerate(
            ("train_seen", "same_key_fresh", "cross_key_validation")
        )
    }

    features, scorers, results = evaluate_lower_round(
        rounds=3,
        datasets=datasets,
        exact_structures={0: exact, 1: exact},
        wrong_sbox_structures={0: wrong, 1: wrong},
        batch_size=16,
    )

    assert len(features) == 18
    assert len(scorers) == 6
    assert len(results) == 18
    assert {row["view"] for row in results} == set(VIEW_NAMES)
    assert all(row["run_id"] == RUN_ID and row["rounds"] == 3 for row in results)
    assert all(row["training_performed"] is False for row in results)


def test_k1p_gate_identifies_r5_boundary_and_lower_round_protocol_failure() -> None:
    tasks = read_tasks(PLAN)
    results, features, scorers, source_results, source_features, source_scorers = (
        synthetic_artifacts()
    )

    gate = adjudicate_k1p(
        tasks=tasks,
        result_rows=results,
        feature_rows=features,
        scorer_rows=scorers,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks={"source_binding": True},
    )

    assert len(results) == EXPECTED_RESULT_ROWS
    assert len(features) == EXPECTED_FEATURE_ROWS
    assert len(scorers) == EXPECTED_SCORER_ROWS
    assert gate["status"] == "pass"
    assert gate["round_pass"] == {"3": True, "4": True}
    assert gate["decision"].endswith("lower_round_signal_supported_r5_loss_boundary")
    assert all(gate["protocol_checks"].values())

    r3_control_degenerate = deepcopy(results)
    for row in r3_control_degenerate:
        if (
            row["rounds"] == 3
            and row["seed"] == 0
            and row["split"] != "train_seen"
            and row["view"] == LABEL_SHUFFLE_VIEW
        ):
            row["auc"] = 0.695
    r4_boundary = adjudicate_k1p(
        tasks=tasks,
        result_rows=r3_control_degenerate,
        feature_rows=features,
        scorer_rows=scorers,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks={"source_binding": True},
    )
    assert r4_boundary["round_pass"] == {"3": False, "4": True}
    assert r4_boundary["status"] == "pass"
    assert r4_boundary["decision"].endswith(
        "lower_round_signal_supported_r5_loss_boundary"
    )

    failed = deepcopy(results)
    for row in failed:
        if row["rounds"] in {3, 4} and row["split"] != "train_seen":
            row["auc"] = {
                CANDIDATE_VIEW: 0.52,
                RAW_VIEW: 0.51,
                LABEL_SHUFFLE_VIEW: 0.50,
            }[row["view"]]
    held = adjudicate_k1p(
        tasks=tasks,
        result_rows=failed,
        feature_rows=features,
        scorer_rows=scorers,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks={"source_binding": True},
    )

    assert held["status"] == "hold"
    assert held["round_pass"] == {"3": False, "4": False}
    assert held["decision"].endswith("current_difference_unresolved_from_r3")


def test_k1p_gate_fails_closed_when_one_fresh_split_is_weak() -> None:
    tasks = read_tasks(PLAN)
    results, features, scorers, source_results, source_features, source_scorers = (
        synthetic_artifacts()
    )
    for row in results:
        if (
            row["rounds"] == 4
            and row["seed"] == 1
            and row["split"] == "cross_key_validation"
            and row["view"] == CANDIDATE_VIEW
        ):
            row["auc"] = 0.54

    gate = adjudicate_k1p(
        tasks=tasks,
        result_rows=results,
        feature_rows=features,
        scorer_rows=scorers,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks={"source_binding": True},
    )

    assert gate["round_pass"] == {"3": True, "4": False}
    assert not gate["research_checks"][
        "r4_seed1_cross_key_validation_exact_auc_floor"
    ]
    assert gate["decision"].endswith("r3_signal_supported_boundary_before_r4")


def test_k1p_plot_explains_round_and_difference_in_chinese(tmp_path: Path) -> None:
    tasks = read_tasks(PLAN)
    results, features, scorers, source_results, source_features, source_scorers = (
        synthetic_artifacts()
    )
    gate = adjudicate_k1p(
        tasks=tasks,
        result_rows=results,
        feature_rows=features,
        scorer_rows=scorers,
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
        source_checks={"source_binding": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1p_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "uKNIT 的 0x40 输入差分从第几轮开始失去稳定信号" in svg
    assert "同一密钥的新样本" in svg
    assert "更换密钥的新样本" in svg
    assert "3轮" in svg and "4轮" in svg and "5轮" in svg
    assert report["seed_labels_offset_separately"] is True


def random_dataset(seed: int) -> DifferentialDataset:
    generator = np.random.default_rng(20260728 + seed)
    features = generator.integers(0, 2, size=(32, 512), dtype=np.uint8)
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 16)
    return DifferentialDataset(features=features, labels=labels, metadata={})


def synthetic_artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    source_results, source_features, source_scorers = synthetic_round(5, source=True)
    r5_features, r5_scorers, r5_results = reuse_k1o_anchor(
        source_results=source_results,
        source_features=source_features,
        source_scorers=source_scorers,
    )
    results: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    scorers: list[dict[str, object]] = []
    for rounds in (3, 4):
        round_results, round_features, round_scorers = synthetic_round(rounds)
        results.extend(round_results)
        features.extend(round_features)
        scorers.extend(round_scorers)
    results.extend(r5_results)
    features.extend(r5_features)
    scorers.extend(r5_scorers)
    return (
        results,
        features,
        scorers,
        source_results,
        source_features,
        source_scorers,
    )


def synthetic_round(
    rounds: int,
    *,
    source: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    result_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    scorer_rows: list[dict[str, object]] = []
    run_id = "synthetic-k1o" if source else RUN_ID
    aucs = {
        CANDIDATE_VIEW: 0.70 if rounds < 5 else 0.52,
        RAW_VIEW: 0.60 if rounds < 5 else 0.50,
        LABEL_SHUFFLE_VIEW: 0.50,
    }
    for seed in (0, 1):
        for view in VIEW_NAMES:
            scorer_rows.append(
                {
                    "run_id": run_id,
                    "rounds": rounds,
                    "cipher_key": "uknit64",
                    "seed": seed,
                    "view": view,
                    "fit_split": "train_seen",
                    "fit_rows": 4096,
                    "feature_dim": 256 if view == RAW_VIEW else 1280,
                    "variance_floor": 1e-6,
                    "class0_rows": 2048,
                    "class1_rows": 2048,
                    "label_permutation_sha256": (
                        f"permutation-{seed}"
                        if view == LABEL_SHUFFLE_VIEW
                        else None
                    ),
                    "training_performed": False,
                    "neural_parameter_count": 0,
                    "optimizer_steps": 0,
                    "epochs": 0,
                }
            )
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            rows = 4096 if split == "train_seen" else 2048
            dataset_sha = f"dataset-r{rounds}-s{seed}-{split}"
            exact_sha = f"exact-r{rounds}-s{seed}-{split}"
            for view in VIEW_NAMES:
                feature_sha = (
                    exact_sha
                    if view in {CANDIDATE_VIEW, LABEL_SHUFFLE_VIEW}
                    else f"raw-r{rounds}-s{seed}-{split}"
                )
                feature_rows.append(
                    {
                        "run_id": run_id,
                        "rounds": rounds,
                        "cipher_key": "uknit64",
                        "seed": seed,
                        "split": split,
                        "view": view,
                        "rows": rows,
                        "feature_dim": 256 if view == RAW_VIEW else 1280,
                        "dataset_sha256": dataset_sha,
                        "feature_sha256": feature_sha,
                        "finite": True,
                        "nonnegative": True,
                        "normalized": True,
                    }
                )
                result_rows.append(
                    {
                        "run_id": run_id,
                        "source_run_id": None,
                        "rounds": rounds,
                        "cipher_key": "uknit64",
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
                        "feature_dim": 256 if view == RAW_VIEW else 1280,
                        "dataset_sha256": dataset_sha,
                        "feature_sha256": feature_sha,
                        "scorer_sha256": f"scorer-r{rounds}-s{seed}-{view}",
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
