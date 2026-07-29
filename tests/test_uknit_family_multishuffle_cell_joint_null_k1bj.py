from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_multishuffle_cell_joint_null_k1bj import (
    render_k1bj_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multishuffle_cell_joint_null_k1bj import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    PERMUTATIONS,
    RUN_ID,
    adjudicate_k1bj,
    load_and_validate_config,
    load_authority,
    permutation_seed,
)


def test_k1bj_config_binds_distinct_frozen_permutation_seeds() -> None:
    config = load_and_validate_config()
    seeds = {
        permutation_seed(replica, cipher, permutation_index)
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
        for permutation_index in PERMUTATIONS
    }

    assert len(PERMUTATIONS) == 31
    assert len(seeds) == 2 * 3 * 31
    assert config["null"]["statistic"] == "abs(auc_minus_0.5)"


def test_k1bj_source_binds_completed_k1bi_visual_result() -> None:
    config = load_and_validate_config()
    *_, checks = load_authority(config)

    assert all(checks.values()), checks


def test_k1bj_gate_uses_orientation_invariant_empirical_null() -> None:
    config = load_and_validate_config()
    features, scorers, results, source_results = _synthetic_artifacts()
    source_gate = {"panels": []}

    passed = adjudicate_k1bj(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_gate=source_gate,
        source_results=source_results,
        source_checks={"source": True},
    )

    assert len(features) == EXPECTED_FEATURE_ROWS
    assert len(scorers) == EXPECTED_SCORER_ROWS
    assert len(results) == EXPECTED_RESULT_ROWS
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("linear_transport_boundary_confirmed")
    assert all(passed["protocol_checks"].values())
    assert all(passed["research_checks"].values())
    assert all(
        panel["empirical_p"] == 1 / 32
        for panel in passed["panels"]
        if panel["cipher_key"] in {"midori64", "dialga128"}
    )

    null_overlap = deepcopy(results)
    for row in null_overlap:
        if (
            row["replica"] == 0
            and row["cipher_key"] == "midori64"
            and row["split"] == "same_key_fresh"
            and row["condition"] == "shuffled_labels"
        ):
            row["auc"] = 0.94
            row["orientation_invariant_strength"] = 0.44
    held = adjudicate_k1bj(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=null_overlap,
        source_gate=source_gate,
        source_results=source_results,
        source_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("null_attribution_not_supported")

    replay_drift = deepcopy(source_results)
    row = next(
        item
        for item in replay_drift
        if item["replica"] == 0
        and item["cipher_key"] == "uknit64"
        and item["split"] == "same_key_fresh"
    )
    row["auc"] = float(row["auc"]) + 0.01
    invalid = adjudicate_k1bj(
        config=config,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_gate=source_gate,
        source_results=replay_drift,
        source_checks={"source": True},
    )
    assert invalid["status"] == "invalid"
    assert not invalid["protocol_checks"]["correct_auc_replay_exact"]


def test_k1bj_plot_explains_multishuffle_null(tmp_path: Path) -> None:
    features, scorers, results, source_results = _synthetic_artifacts()
    gate = adjudicate_k1bj(
        config=load_and_validate_config(),
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_gate={"panels": []},
        source_results=source_results,
        source_checks={"source": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1bj_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert report["null_permutations_per_panel"] == 31
    assert "31 次标签打乱零分布" in svg
    assert "|AUC-0.5|" in svg
    assert "结构信号领先随机方向效应多少" in svg
    assert "停止纯线性路线" in svg


def _synthetic_artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    feature_rows: list[dict[str, object]] = []
    scorer_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    source_results: list[dict[str, object]] = []
    seeds = {
        0: {"uknit64": 3, "midori64": 6, "dialga128": 0},
        1: {"uknit64": 4, "midori64": 7, "dialga128": 1},
    }
    dimensions = {"uknit64": 3072, "midori64": 3072, "dialga128": 6144}
    correct_aucs = {"uknit64": 0.50, "midori64": 0.93, "dialga128": 0.997}
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            seed = seeds[replica][cipher]
            true_scorer_sha = f"true-scorer-{replica}-{cipher}"
            for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
                rows = 4096 if split == "train_seen" else 2048
                feature_sha = f"feature-{replica}-{cipher}-{split}"
                dataset_sha = f"dataset-{replica}-{cipher}-{split}"
                feature_rows.append(
                    {
                        "run_id": RUN_ID,
                        "replica": replica,
                        "cipher_key": cipher,
                        "seed": seed,
                        "split": split,
                        "rows": rows,
                        "feature_dim": dimensions[cipher],
                        "expected_feature_dim": dimensions[cipher],
                        "feature_sha256": feature_sha,
                        "dataset_sha256": dataset_sha,
                        "representation": "runtime_cell_joint_16_value_histogram",
                        "finite": True,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "pairs_per_sample": 4,
                        "data_generation_performed": False,
                    }
                )
            scorer_rows.append(
                _scorer_row(
                    replica,
                    cipher,
                    seed,
                    "true_labels",
                    -1,
                    None,
                    None,
                    true_scorer_sha,
                    dimensions[cipher],
                )
            )
            for split in ("same_key_fresh", "cross_key_validation"):
                correct_auc = correct_aucs[cipher]
                source_results.append(
                    {
                        "replica": replica,
                        "cipher_key": cipher,
                        "split": split,
                        "condition": "correct_operator",
                        "auc": correct_auc,
                    }
                )
                result_rows.append(
                    _result_row(
                        replica,
                        cipher,
                        seed,
                        split,
                        "true_labels",
                        -1,
                        correct_auc,
                        f"feature-{replica}-{cipher}-{split}",
                        f"dataset-{replica}-{cipher}-{split}",
                        true_scorer_sha,
                    )
                )
            for permutation_index in PERMUTATIONS:
                scorer_sha = f"shuffle-scorer-{replica}-{cipher}-{permutation_index}"
                permutation_sha = (
                    f"permutation-{replica}-{cipher}-{permutation_index}"
                )
                scorer_rows.append(
                    _scorer_row(
                        replica,
                        cipher,
                        seed,
                        "shuffled_labels",
                        permutation_index,
                        permutation_seed(replica, cipher, permutation_index),
                        permutation_sha,
                        scorer_sha,
                        dimensions[cipher],
                    )
                )
                null_auc = 0.5 + ((permutation_index % 7) - 3) * 0.003
                for split in ("same_key_fresh", "cross_key_validation"):
                    result_rows.append(
                        _result_row(
                            replica,
                            cipher,
                            seed,
                            split,
                            "shuffled_labels",
                            permutation_index,
                            null_auc,
                            f"feature-{replica}-{cipher}-{split}",
                            f"dataset-{replica}-{cipher}-{split}",
                            scorer_sha,
                        )
                    )
    return feature_rows, scorer_rows, result_rows, source_results


def _scorer_row(
    replica: int,
    cipher: str,
    seed: int,
    condition: str,
    permutation_index: int,
    permutation_seed_value: int | None,
    permutation_sha: str | None,
    scorer_sha: str,
    feature_dim: int,
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "seed": seed,
        "condition": condition,
        "permutation_index": permutation_index,
        "permutation_seed": permutation_seed_value,
        "fit_split": "train_seen",
        "fit_rows": 4096,
        "feature_dim": feature_dim,
        "variance_floor": 1e-6,
        "class_counts": [2048, 2048],
        "scorer_sha256": scorer_sha,
        "label_permutation_sha256": permutation_sha,
        "training_performed": False,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }


def _result_row(
    replica: int,
    cipher: str,
    seed: int,
    split: str,
    condition: str,
    permutation_index: int,
    auc: float,
    feature_sha: str,
    dataset_sha: str,
    scorer_sha: str,
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "replica": replica,
        "cipher_key": cipher,
        "rounds": 5 if cipher == "uknit64" else 4,
        "seed": seed,
        "split": split,
        "condition": condition,
        "permutation_index": permutation_index,
        "rows": 2048,
        "auc": auc,
        "orientation_invariant_strength": abs(auc - 0.5),
        "feature_sha256": feature_sha,
        "dataset_sha256": dataset_sha,
        "scorer_sha256": scorer_sha,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "training_performed": False,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
