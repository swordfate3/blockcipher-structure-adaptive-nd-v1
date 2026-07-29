from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.audit_uknit_ctspn_r6_position_k1bl import read_tasks
from blockcipher_nd.cli.plot_uknit_ctspn_r6_remaining_roles_k1bm import (
    render_k1bm_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_remaining_roles_k1bm import (
    ACTIVE_BIT_ROLES,
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
    adjudicate_k1bm,
    build_confirmation_tasks,
    build_discovery_tasks,
    candidate_bit_index,
    candidate_difference,
    select_discovery_candidates,
    validate_confirmation_tasks,
    validate_discovery_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    CANDIDATE_VIEW,
    EXPECTED_FEATURE_DIMS,
    LABEL_SHUFFLE_VIEW,
    RAW_VIEW,
)


ROOT = Path(__file__).resolve().parents[1]
K1BL_PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_ctspn_r6_role1_position_k1bl_seed2.csv"
)


def test_k1bm_builds_all_48_remaining_single_bit_candidates() -> None:
    tasks = build_discovery_tasks(read_tasks(K1BL_PLAN))

    assert len(tasks) == 48
    assert all(validate_discovery_tasks(tasks).values())
    assert {
        (
            int(task["model_options"]["active_bit_role"]),
            int(task["model_options"]["active_cell"]),
            int(task["input_difference"]),
        )
        for task in tasks
    } == {
        (role, cell, 1 << (4 * cell + (3 - role)))
        for role in ACTIVE_BIT_ROLES
        for cell in EXPECTED_CELLS
    }


def test_k1bm_freezes_at_most_one_candidate_per_role() -> None:
    tasks = build_discovery_tasks(read_tasks(K1BL_PLAN))
    passing = {
        candidate_bit_index(2, 0): 0.64,
        candidate_bit_index(4, 0): 0.68,
        candidate_bit_index(5, 2): 0.66,
    }
    results = discovery_result_rows(passing)

    selection = select_discovery_candidates(results)
    confirmation = build_confirmation_tasks(tasks, selection["selected_bit_indices"])

    expected = [candidate_bit_index(4, 0), candidate_bit_index(5, 2)]
    assert selection["selected_bit_indices"] == expected
    assert selection["selected_by_role"] == {
        "0": candidate_bit_index(4, 0),
        "2": candidate_bit_index(5, 2),
    }
    assert len(confirmation) == len(expected) * len(CONFIRMATION_SEEDS)
    assert all(
        validate_confirmation_tasks(
            confirmation, selection["selected_bit_indices"]
        ).values()
    )
    assert {
        (task_bit_index(task), int(task["seed"])) for task in confirmation
    } == {
        (bit, seed) for bit in expected for seed in CONFIRMATION_SEEDS
    }


def test_k1bm_no_candidate_closes_only_the_single_bit_family() -> None:
    tasks = build_discovery_tasks(read_tasks(K1BL_PLAN))
    results = discovery_result_rows({})
    selection = select_discovery_candidates(results)
    features, scorers, datasets = companion_rows(results)

    gate = adjudicate_k1bm(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bl_source_gate": True},
    )

    assert selection["selected_bit_indices"] == []
    assert gate["status"] == "hold"
    assert gate["decision"].endswith("no_r6_single_bit_candidate")
    assert gate["confirmed_bit_indices"] == []
    assert "DDT/trail-guided multi-bit" in gate["next_action"]
    assert all(gate["protocol_checks"].values())


def test_k1bm_advances_only_an_untouched_confirmed_candidate() -> None:
    tasks = build_discovery_tasks(read_tasks(K1BL_PLAN))
    passing_bit = candidate_bit_index(4, 0)
    results = discovery_result_rows({passing_bit: 0.68})
    selection = select_discovery_candidates(results)
    add_confirmation_results(results, selection["selected_bit_indices"], {passing_bit})
    features, scorers, datasets = companion_rows(results)

    gate = adjudicate_k1bm(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bl_source_gate": True},
    )

    assert gate["status"] == "pass"
    assert gate["confirmed_bit_indices"] == [passing_bit]
    assert gate["decision"].endswith("confirmed_r6_single_bit_difference")
    assert "16-pair" in gate["next_action"]
    assert all(gate["protocol_checks"].values())


def test_k1bm_plot_explains_all_48_candidates_in_chinese(tmp_path: Path) -> None:
    tasks = build_discovery_tasks(read_tasks(K1BL_PLAN))
    results = discovery_result_rows({})
    selection = select_discovery_candidates(results)
    features, scorers, datasets = companion_rows(results)
    gate = adjudicate_k1bm(
        discovery_tasks=tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bl_source_gate": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1bm_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "uKNIT 第6轮剩余48个单 bit 差分扫描" in svg
    assert "64个单 bit 位置均无候选" in svg
    assert report["all_48_candidates_visible"] is True


def discovery_result_rows(
    exact_by_bit: dict[int, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ACTIVE_BIT_ROLES:
        for cell in EXPECTED_CELLS:
            bit = candidate_bit_index(cell, role)
            exact = exact_by_bit.get(bit, 0.52)
            raw = exact - (0.04 if bit in exact_by_bit else 0.005)
            for split in EXPECTED_SPLITS:
                for view in DISCOVERY_VIEWS:
                    rows.append(
                        result_row(
                            phase=DISCOVERY_PHASE,
                            cell=cell,
                            role=role,
                            seed=DISCOVERY_SEED,
                            split=split,
                            view=view,
                            auc=exact if view == CANDIDATE_VIEW else raw,
                        )
                    )
    return rows


def add_confirmation_results(
    rows: list[dict[str, object]],
    selected_bits: list[int],
    passing_bits: set[int],
) -> None:
    for bit in selected_bits:
        cell, role = coordinate_for_bit(bit)
        for seed in CONFIRMATION_SEEDS:
            for split in EXPECTED_SPLITS:
                exact = 0.64 if bit in passing_bits else 0.52
                aucs = {
                    CANDIDATE_VIEW: exact,
                    RAW_VIEW: exact - (0.04 if bit in passing_bits else 0.005),
                    LABEL_SHUFFLE_VIEW: exact - 0.08 if bit in passing_bits else 0.50,
                }
                for view in CONFIRMATION_VIEWS:
                    rows.append(
                        result_row(
                            phase=CONFIRMATION_PHASE,
                            cell=cell,
                            role=role,
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
    role: int,
    seed: int,
    split: str,
    view: str,
    auc: float,
) -> dict[str, object]:
    train_rows = 2048 if phase == DISCOVERY_PHASE else 4096
    holdout_rows = 1024 if phase == DISCOVERY_PHASE else 2048
    rows = train_rows if split == "train_seen" else holdout_rows
    bit = candidate_bit_index(cell, role)
    difference = candidate_difference(cell, role)
    return {
        "run_id": RUN_ID,
        "phase": phase,
        "cipher_key": "uknit64",
        "rounds": ROUNDS,
        "cell": cell,
        "bit_index": bit,
        "active_bit_role": role,
        "input_difference": difference,
        "input_difference_hex": f"0x{difference:016x}",
        "seed": seed,
        "split": split,
        "view": view,
        "rows": rows,
        "auc": auc,
        "feature_dim": EXPECTED_FEATURE_DIMS[view],
        "feature_sha256": f"feature-{phase}-{bit}-{seed}-{split}-{view}",
        "dataset_sha256": f"dataset-{phase}-{bit}-{seed}-{split}",
        "scorer_sha256": f"scorer-{phase}-{bit}-{seed}-{view}",
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
        key = (row["phase"], row["bit_index"], row["seed"], row["view"])
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
        key = (row["phase"], row["bit_index"], row["seed"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        dataset_rows.append(
            {
                "run_id": RUN_ID,
                "phase": row["phase"],
                "bit_index": row["bit_index"],
                "seed": row["seed"],
                "split": row["split"],
                "cache_payloads_present": True,
                "row_overlap_with_train": 0,
            }
        )
    return feature_rows, scorer_rows, dataset_rows


def coordinate_for_bit(bit_index: int) -> tuple[int, int]:
    for role in ACTIVE_BIT_ROLES:
        for cell in EXPECTED_CELLS:
            if candidate_bit_index(cell, role) == bit_index:
                return cell, role
    raise AssertionError(f"unknown K1-BM bit index: {bit_index}")


def task_bit_index(task: dict[str, object]) -> int:
    options = task["model_options"]
    assert isinstance(options, dict)
    return candidate_bit_index(
        int(options["active_cell"]), int(options["active_bit_role"])
    )
