from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_uknit_family_midori64_k1ah import (
    build_structures,
    read_tasks,
)
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1q import render_k1q_svg
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    CONFIRMATION_PHASE,
    CONFIRMATION_VIEWS,
    DISCOVERY_PHASE,
    DISCOVERY_VIEWS,
    EXPECTED_CELLS,
    EXPECTED_SPLITS,
    candidate_bit_index,
    candidate_difference,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_difference_position_k1ah import (
    ANCHOR_CELL,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEED,
    RUN_ID,
    adjudicate_k1ah,
    build_confirmation_tasks,
    evaluate_position,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_uknit_family_midori64_difference_position_k1ah_seed5.csv"
)


def test_k1ah_plan_freezes_midori64_r4_same_role_scan() -> None:
    tasks = read_tasks(PLAN)

    assert len(tasks) == 16
    assert all(validate_discovery_tasks(tasks).values())
    assert {
        (
            int(task["model_options"]["active_cell"]),
            int(task["model_options"]["active_bit_role"]),
            int(task["input_difference"]),
            int(task["model_options"]["runtime_round_start"]),
            int(task["model_options"]["cipher_round_window_start"]),
        )
        for task in tasks
    } == {(cell, 1, 1 << (4 * cell + 2), 0, 2) for cell in EXPECTED_CELLS}


def test_k1ah_confirmation_uses_untouched_seeds_keys_and_anchor() -> None:
    confirmation = build_confirmation_tasks(read_tasks(PLAN), [5, 7])

    assert len(confirmation) == 6
    assert all(validate_confirmation_tasks(confirmation, [5, 7]).values())
    assert {
        (int(task["model_options"]["active_cell"]), int(task["seed"]))
        for task in confirmation
    } == {(cell, seed) for cell in (ANCHOR_CELL, 5, 7) for seed in CONFIRMATION_SEEDS}
    assert all(int(task["seed"]) != DISCOVERY_SEED for task in confirmation)


def test_k1ah_reuses_exact_feature_math_without_neural_training() -> None:
    task = read_tasks(PLAN)[0]
    datasets = {
        split: random_dataset(index) for index, split in enumerate(EXPECTED_SPLITS)
    }
    exact, wrong = build_structures(task, datasets)

    features, scorers, results = evaluate_position(
        phase=DISCOVERY_PHASE,
        cell=0,
        seed=DISCOVERY_SEED,
        datasets=datasets,
        exact_structure=exact,
        wrong_sbox_structure=wrong,
        batch_size=16,
    )

    assert len(features) == 6
    assert len(scorers) == 2
    assert len(results) == 6
    assert {row["view"] for row in results} == set(DISCOVERY_VIEWS)
    assert all(row["run_id"] == RUN_ID for row in results)
    assert all(row["cipher_key"] == "midori64" for row in results)
    assert all(row["rounds"] == 4 for row in results)
    assert all(row["training_performed"] is False for row in results)
    assert all(row["neural_parameter_count"] == 0 for row in results)


def test_k1ah_gate_requires_every_untouched_confirmation_scope() -> None:
    tasks = read_tasks(PLAN)
    results = discovery_result_rows(signal_cells={5})
    selection = select_discovery_candidates(results)
    add_confirmation_results(results, cells=(ANCHOR_CELL, 5), passing_cells={5})
    features, scorers, datasets = companion_rows(results)

    gate = adjudicate_k1ah(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1ag_source": True},
    )

    assert gate["status"] == "pass"
    assert gate["confirmed_cells"] == [5]
    assert gate["decision"].endswith("confirmed_r4_position_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in results]
    for row in failed:
        if (
            row["phase"] == CONFIRMATION_PHASE
            and row["cell"] == 5
            and row["seed"] == 7
            and row["split"] == "cross_key_validation"
            and row["view"] == CANDIDATE_VIEW
        ):
            row["auc"] = 0.54
    failed_features, failed_scorers, failed_datasets = companion_rows(failed)
    held = adjudicate_k1ah(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=failed_datasets,
        feature_rows=failed_features,
        scorer_rows=failed_scorers,
        result_rows=failed,
        source_checks={"k1ag_source": True},
    )

    assert held["status"] == "hold"
    assert held["confirmed_cells"] == []
    assert held["decision"].endswith("discovery_not_confirmed")


def test_k1ah_plot_names_midori64_r4_and_new_confirmation_seeds(
    tmp_path: Path,
) -> None:
    tasks = read_tasks(PLAN)
    results = discovery_result_rows(signal_cells={5})
    selection = select_discovery_candidates(results)
    add_confirmation_results(results, cells=(ANCHOR_CELL, 5), passing_cells={5})
    features, scorers, datasets = companion_rows(results)
    gate = adjudicate_k1ah(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1ag_source": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1q_svg(
        gate,
        output,
        cipher_label="Midori64",
        rounds=4,
        confirmation_seeds=CONFIRMATION_SEEDS,
        anchor_cell=ANCHOR_CELL,
    )

    svg = output.read_text(encoding="utf-8")
    assert "移动 Midori64 输入差分的位置，能否恢复第 4 轮信号" in svg
    assert "seed6" in svg
    assert "seed7" in svg
    assert report["all_sixteen_positions_visible"] is True


def random_dataset(seed: int) -> DifferentialDataset:
    generator = np.random.default_rng(20260729 + seed)
    return DifferentialDataset(
        features=generator.integers(0, 2, size=(32, 512), dtype=np.uint8),
        labels=np.tile(np.array([0, 1], dtype=np.uint8), 16),
        metadata={},
    )


def discovery_result_rows(signal_cells: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in EXPECTED_CELLS:
        exact = 0.68 if cell == 5 else (0.62 if cell in signal_cells else 0.52)
        raw = exact - (0.04 if cell in signal_cells else 0.005)
        for split in EXPECTED_SPLITS:
            for view in DISCOVERY_VIEWS:
                rows.append(
                    result_row(
                        phase=DISCOVERY_PHASE,
                        cell=cell,
                        seed=DISCOVERY_SEED,
                        split=split,
                        view=view,
                        auc=exact if view == CANDIDATE_VIEW else raw,
                    )
                )
    return rows


def add_confirmation_results(
    rows: list[dict[str, object]],
    *,
    cells: tuple[int, ...],
    passing_cells: set[int],
) -> None:
    for cell in cells:
        for seed in CONFIRMATION_SEEDS:
            for split in EXPECTED_SPLITS:
                exact = 0.64 if cell in passing_cells else 0.52
                aucs = {
                    CANDIDATE_VIEW: exact,
                    RAW_VIEW: exact - (0.04 if cell in passing_cells else 0.005),
                    LABEL_SHUFFLE_VIEW: (
                        exact - 0.08 if cell in passing_cells else 0.50
                    ),
                }
                for view in CONFIRMATION_VIEWS:
                    rows.append(
                        result_row(
                            phase=CONFIRMATION_PHASE,
                            cell=cell,
                            seed=seed,
                            split=split,
                            view=view,
                            auc=aucs[view],
                        )
                    )


def result_row(
    *,
    phase: str,
    cell: int,
    seed: int,
    split: str,
    view: str,
    auc: float,
) -> dict[str, object]:
    train_rows = 2048 if phase == DISCOVERY_PHASE else 4096
    holdout_rows = 1024 if phase == DISCOVERY_PHASE else 2048
    rows = train_rows if split == "train_seen" else holdout_rows
    return {
        "run_id": RUN_ID,
        "phase": phase,
        "cipher_key": "midori64",
        "rounds": 4,
        "cell": cell,
        "bit_index": candidate_bit_index(cell),
        "active_bit_role": 1,
        "input_difference": candidate_difference(cell),
        "input_difference_hex": f"0x{candidate_difference(cell):016x}",
        "seed": seed,
        "split": split,
        "view": view,
        "rows": rows,
        "auc": auc,
        "zero_threshold_accuracy": 0.6,
        "score_mean": 0.0,
        "score_std": 1.0,
        "score_min": -1.0,
        "score_max": 1.0,
        "feature_dim": EXPECTED_FEATURE_DIMS[view],
        "feature_sha256": f"feature-{phase}-{cell}-{seed}-{split}-{view}",
        "dataset_sha256": f"dataset-{phase}-{cell}-{seed}-{split}",
        "scorer_sha256": f"scorer-{phase}-{cell}-{seed}-{view}",
        "fit_split": "train_seen",
        "fit_rows": train_rows,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "variance_floor": 1e-6,
        "training_performed": False,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }


def companion_rows(
    result_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    feature_rows = []
    for row in result_rows:
        feature_sha = str(row["feature_sha256"])
        if row["view"] == LABEL_SHUFFLE_VIEW:
            feature_sha = feature_sha.replace(LABEL_SHUFFLE_VIEW, CANDIDATE_VIEW)
        feature_rows.append(
            {
                key: row[key]
                for key in (
                    "run_id",
                    "phase",
                    "cipher_key",
                    "rounds",
                    "cell",
                    "bit_index",
                    "active_bit_role",
                    "input_difference",
                    "input_difference_hex",
                    "seed",
                    "split",
                    "view",
                    "rows",
                    "feature_dim",
                    "dataset_sha256",
                )
            }
            | {
                "feature_sha256": feature_sha,
                "finite": True,
                "nonnegative": True,
                "normalized": True,
            }
        )
    scorer_rows = []
    seen = set()
    for row in result_rows:
        key = (row["phase"], row["cell"], row["seed"], row["view"])
        if key in seen:
            continue
        seen.add(key)
        scorer_rows.append(
            {
                field: row[field]
                for field in (
                    "run_id",
                    "phase",
                    "cipher_key",
                    "rounds",
                    "cell",
                    "bit_index",
                    "active_bit_role",
                    "input_difference",
                    "input_difference_hex",
                    "seed",
                    "view",
                    "feature_dim",
                    "scorer_sha256",
                    "fit_split",
                    "fit_rows",
                    "variance_floor",
                    "training_performed",
                    "neural_parameter_count",
                    "optimizer_steps",
                    "epochs",
                )
            }
        )
    dataset_rows = []
    dataset_seen = set()
    for row in result_rows:
        key = (row["phase"], row["cell"], row["seed"], row["split"])
        if key in dataset_seen:
            continue
        dataset_seen.add(key)
        dataset_rows.append(
            {
                "run_id": RUN_ID,
                "phase": row["phase"],
                "cell": row["cell"],
                "seed": row["seed"],
                "split": row["split"],
                "rows": row["rows"],
                "dataset_sha256": row["dataset_sha256"],
                "cache_payloads_present": True,
            }
        )
    return feature_rows, scorer_rows, dataset_rows
