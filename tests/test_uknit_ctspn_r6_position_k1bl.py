from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.audit_uknit_ctspn_r6_position_k1bl import (
    evaluate_k1bl_position,
    read_tasks,
)
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1q import render_k1q_svg
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_position_k1bl import (
    ANCHOR_CELL,
    CONFIRMATION_PHASE,
    CONFIRMATION_SEEDS,
    CONFIRMATION_VIEWS,
    DISCOVERY_PHASE,
    DISCOVERY_SEED,
    DISCOVERY_VIEWS,
    EXPECTED_CELLS,
    EXPECTED_SPLITS,
    ROUNDS,
    RUN_ID,
    adjudicate_k1bl,
    build_confirmation_tasks,
    candidate_bit_index,
    candidate_difference,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    build_k1n_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_ctspn_r6_role1_position_k1bl_seed2.csv"
)


def test_k1bl_plan_changes_only_r6_window_over_k1q_position_protocol() -> None:
    tasks = read_tasks(PLAN)

    assert len(tasks) == 16
    assert all(validate_discovery_tasks(tasks).values())
    assert {
        (
            int(task["model_options"]["active_cell"]),
            int(task["rounds"]),
            int(task["model_options"]["runtime_round_start"]),
            int(task["input_difference"]),
        )
        for task in tasks
    } == {
        (cell, 6, 4, 1 << (4 * cell + 2)) for cell in EXPECTED_CELLS
    }


def test_k1bl_always_confirms_r5_anchor_and_two_new_candidates() -> None:
    tasks = read_tasks(PLAN)
    results = discovery_result_rows(signal_cells={ANCHOR_CELL, 5, 7})

    selection = select_discovery_candidates(results)
    confirmation = build_confirmation_tasks(tasks, selection["selected_cells"])

    assert selection["anchor_passes_discovery"] is True
    assert selection["selected_cells"] == [5, 7]
    assert all(
        validate_confirmation_tasks(confirmation, selection["selected_cells"]).values()
    )
    assert {
        (int(task["model_options"]["active_cell"]), int(task["seed"]))
        for task in confirmation
    } == {
        (cell, seed)
        for cell in (ANCHOR_CELL, 5, 7)
        for seed in CONFIRMATION_SEEDS
    }


def test_k1bl_evaluator_records_r6_without_neural_training() -> None:
    task = read_tasks(PLAN)[0]
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
        split: random_dataset(index)
        for index, split in enumerate(EXPECTED_SPLITS)
    }

    features, scorers, results = evaluate_k1bl_position(
        phase=DISCOVERY_PHASE,
        task=task,
        datasets=datasets,
        exact=exact,
        wrong=wrong,
        batch_size=16,
    )

    assert len(features) == 6
    assert len(scorers) == 2
    assert len(results) == 6
    assert {row["rounds"] for row in results} == {6}
    assert {row["run_id"] for row in results} == {RUN_ID}
    assert {row["input_difference"] for row in results} == {0x4}
    assert all(row["training_performed"] is False for row in results)


def test_k1bl_gate_advances_only_a_confirmed_r6_position() -> None:
    tasks = read_tasks(PLAN)
    results = discovery_result_rows(signal_cells={ANCHOR_CELL, 5})
    selection = select_discovery_candidates(results)
    add_confirmation_results(
        results,
        cells=(ANCHOR_CELL, *selection["selected_cells"]),
        passing_cells={ANCHOR_CELL},
    )
    features, scorers, datasets = companion_rows(results)

    gate = adjudicate_k1bl(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"source_binding": True},
    )

    assert gate["status"] == "pass"
    assert gate["confirmed_cells"] == [ANCHOR_CELL]
    assert gate["decision"].endswith("confirmed_r6_role1_difference")
    assert all(gate["protocol_checks"].values())

    failed = [dict(row) for row in results]
    for row in failed:
        if row["phase"] == CONFIRMATION_PHASE and row["view"] == CANDIDATE_VIEW:
            row["auc"] = 0.52
    features, scorers, datasets = companion_rows(failed)
    held = adjudicate_k1bl(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=failed,
        source_checks={"source_binding": True},
    )

    assert held["status"] == "hold"
    assert held["confirmed_cells"] == []
    assert "remaining three single-bit roles" in held["next_action"]


def test_k1bl_plot_names_r6_and_the_cell11_anchor(tmp_path: Path) -> None:
    tasks = read_tasks(PLAN)
    results = discovery_result_rows(signal_cells={ANCHOR_CELL, 5})
    selection = select_discovery_candidates(results)
    add_confirmation_results(
        results,
        cells=(ANCHOR_CELL, *selection["selected_cells"]),
        passing_cells={ANCHOR_CELL},
    )
    features, scorers, datasets = companion_rows(results)
    gate = adjudicate_k1bl(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"source_binding": True},
    )
    output = tmp_path / "curves.svg"

    render_k1q_svg(
        gate,
        output,
        rounds=6,
        anchor_cell=ANCHOR_CELL,
        anchor_label="r5最强位置",
        always_show_anchor=True,
        left_margin=0.13,
    )

    svg = output.read_text(encoding="utf-8")
    assert "第 6 轮信号" in svg
    assert "r5最强位置" in svg


def random_dataset(seed: int) -> DifferentialDataset:
    generator = np.random.default_rng(20260729 + seed)
    features = generator.integers(0, 2, size=(32, 512), dtype=np.uint8)
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 16)
    return DifferentialDataset(features=features, labels=labels, metadata={})


def discovery_result_rows(signal_cells: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in EXPECTED_CELLS:
        exact = 0.62 if cell in signal_cells else 0.52
        if cell == 5:
            exact = 0.68
        elif cell == 7:
            exact = 0.65
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
        "cipher_key": "uknit64",
        "rounds": ROUNDS,
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
    seen = set()
    for row in result_rows:
        key = (row["phase"], row["cell"], row["seed"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        dataset_rows.append(
            {
                "run_id": RUN_ID,
                "phase": row["phase"],
                "cell": row["cell"],
                "seed": row["seed"],
                "split": row["split"],
                "cache_payloads_present": True,
                "row_overlap_with_train": 0,
            }
        )
    return feature_rows, scorer_rows, dataset_rows
