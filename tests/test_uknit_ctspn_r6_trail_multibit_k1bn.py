from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from blockcipher_nd.cli.audit_uknit_ctspn_r6_position_k1bl import read_tasks
from blockcipher_nd.cli.plot_uknit_ctspn_r6_trail_multibit_k1bn import (
    render_k1bn_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_trail_multibit_k1bn import (
    CANDIDATE_FAMILIES,
    CONFIRMATION_PHASE,
    CONFIRMATION_SEEDS,
    CONFIRMATION_VIEWS,
    DISCOVERY_PHASE,
    DISCOVERY_SEED,
    DISCOVERY_VIEWS,
    EXPECTED_DISCOVERY_CANDIDATES,
    EXPECTED_SPLITS,
    FAMILY_CELL_LOCAL,
    FAMILY_TWO_CELL,
    RUN_ID,
    SELECTED_PER_FAMILY,
    adjudicate_k1bn,
    build_candidate_manifest,
    build_confirmation_tasks,
    build_discovery_tasks,
    select_discovery_candidates,
    validate_candidate_manifest,
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
EXPECTED_SOURCE_HASHES = {
    "uknit_sbox_tables_sha256": (
        "49020d0e12d2af45a725ccae4e536aa5c6d7d6e58de9658899fbeabd8fd74f25"
    ),
    "uknit_linear_target_sources_sha256": (
        "90ef4a38d81472688a582a7725e91de2d3e43807e8a1d6b13d85de0c79c0441e"
    ),
}


@pytest.fixture(scope="module")
def candidate_manifest() -> dict[str, Any]:
    return build_candidate_manifest()


@pytest.fixture(scope="module")
def candidates(candidate_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in candidate_manifest["selected_candidates"]]


@pytest.fixture(scope="module")
def discovery_tasks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_discovery_tasks(read_tasks(K1BL_PLAN)[0], candidates)


def test_k1bn_freezes_exact_sources_pools_and_48_multibit_candidates(
    candidate_manifest: Mapping[str, Any],
) -> None:
    checks = validate_candidate_manifest(candidate_manifest)
    selected = candidate_manifest["selected_candidates"]

    assert candidate_manifest["source_hashes"] == EXPECTED_SOURCE_HASHES
    assert candidate_manifest["pool_counts"] == {
        FAMILY_CELL_LOCAL: 176,
        FAMILY_TWO_CELL: 256,
    }
    assert len(selected) == EXPECTED_DISCOVERY_CANDIDATES
    assert all(checks.values())
    assert all(int(row["input_difference"]).bit_count() >= 2 for row in selected)
    assert {
        family: sum(row["family"] == family for row in selected)
        for family in CANDIDATE_FAMILIES
    } == {family: SELECTED_PER_FAMILY for family in CANDIDATE_FAMILIES}
    assert [row["candidate_index"] for row in selected] == list(
        range(EXPECTED_DISCOVERY_CANDIDATES)
    )


def test_k1bn_candidate_generation_replays_deterministically(
    candidate_manifest: Mapping[str, Any],
) -> None:
    replay = build_candidate_manifest()

    assert replay == candidate_manifest
    assert [
        (
            row["candidate_id"],
            row["input_difference_hex"],
            row["trail_log2_probability"],
            row["trail_total_active_sboxes"],
        )
        for row in replay["selected_candidates"][:3]
    ] == [
        ("cm_c04_da", "0x00000000000a0000", -96.0, 46),
        ("cm_c04_dc", "0x00000000000c0000", -96.0, 46),
        ("cm_c09_d3", "0x0000003000000000", -97.0, 47),
    ]


def test_k1bn_discovery_protocol_changes_only_the_input_difference(
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    template = read_tasks(K1BL_PLAN)[0]

    assert all(validate_discovery_tasks(discovery_tasks, candidates).values())
    assert len(discovery_tasks) == 48
    assert {
        (task["candidate_id"], task["input_difference"])
        for task in discovery_tasks
    } == {
        (candidate["candidate_id"], candidate["input_difference"])
        for candidate in candidates
    }
    for task in discovery_tasks:
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "independent_pairs"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["rounds"] == 6
        assert task["seed"] == DISCOVERY_SEED
        assert task["pairs_per_sample"] == 4
        assert set(task["model_options"]) == set(template["model_options"])
        assert all(
            token not in key.lower()
            for key in task["model_options"]
            for token in ("ddt", "trail", "probability", "candidate")
        )
        assert task["trail_log2_probability"] <= 0.0


def test_k1bn_freezes_at_most_one_confirmation_candidate_per_family(
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    preferred = {
        FAMILY_CELL_LOCAL: candidates[1]["candidate_id"],
        FAMILY_TWO_CELL: next(
            row["candidate_id"]
            for row in candidates
            if row["family"] == FAMILY_TWO_CELL
        ),
    }
    results = discovery_result_rows(candidates, passing=set(preferred.values()))
    selection = select_discovery_candidates(results, candidates)
    confirmation = build_confirmation_tasks(
        discovery_tasks, selection["selected_candidate_ids"]
    )

    assert selection["selected_by_family"] == preferred
    assert selection["selected_candidate_ids"] == [
        preferred[family] for family in CANDIDATE_FAMILIES
    ]
    assert len(confirmation) == len(preferred) * len(CONFIRMATION_SEEDS)
    assert all(
        validate_confirmation_tasks(
            confirmation, selection["selected_candidate_ids"]
        ).values()
    )
    assert {
        (task["candidate_id"], task["seed"]) for task in confirmation
    } == {
        (candidate_id, seed)
        for candidate_id in preferred.values()
        for seed in CONFIRMATION_SEEDS
    }


def test_k1bn_no_candidate_closes_only_the_frozen_difference_families(
    candidate_manifest: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    results = discovery_result_rows(candidates, passing=set())
    selection = select_discovery_candidates(results, candidates)
    features, scorers, datasets = companion_rows(results, candidates)

    gate = adjudicate_k1bn(
        candidate_manifest=candidate_manifest,
        discovery_tasks=discovery_tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bm_single_bit_source_completed_hold": True},
    )

    assert selection["selected_candidate_ids"] == []
    assert gate["status"] == "hold"
    assert gate["decision"].endswith("no_r6_multibit_candidate")
    assert gate["confirmed_candidate_ids"] == []
    assert "all 64 single-bit" in gate["next_action"]
    assert "universal" in gate["claim_scope"]
    assert all(gate["protocol_checks"].values())


def test_k1bn_advances_only_confirmed_unseen_seed_and_key_results(
    candidate_manifest: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    passing_discovery = {
        next(
            row["candidate_id"]
            for row in candidates
            if row["family"] == FAMILY_CELL_LOCAL
        ),
        next(
            row["candidate_id"]
            for row in candidates
            if row["family"] == FAMILY_TWO_CELL
        ),
    }
    results = discovery_result_rows(candidates, passing=passing_discovery)
    selection = select_discovery_candidates(results, candidates)
    passing_confirmation = {selection["selected_candidate_ids"][0]}
    add_confirmation_results(
        results,
        candidates,
        selection["selected_candidate_ids"],
        passing=passing_confirmation,
    )
    features, scorers, datasets = companion_rows(results, candidates)

    gate = adjudicate_k1bn(
        candidate_manifest=candidate_manifest,
        discovery_tasks=discovery_tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bm_single_bit_source_completed_hold": True},
    )

    assert gate["status"] == "pass"
    assert gate["confirmed_candidate_ids"] == list(passing_confirmation)
    assert gate["decision"].endswith("confirmed_r6_multibit_difference")
    assert "uKNIT-only r6 16-pair" in gate["next_action"]
    assert all(gate["protocol_checks"].values())


def test_k1bn_gate_fails_closed_without_the_completed_k1bm_source(
    candidate_manifest: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    results = discovery_result_rows(candidates, passing=set())
    selection = select_discovery_candidates(results, candidates)
    features, scorers, datasets = companion_rows(results, candidates)

    gate = adjudicate_k1bn(
        candidate_manifest=candidate_manifest,
        discovery_tasks=discovery_tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bm_single_bit_source_completed_hold": False},
    )

    assert gate["status"] == "invalid"
    assert gate["failed_protocol_checks"] == [
        "k1bm_single_bit_source_completed_hold"
    ]


def test_k1bn_plot_explains_all_48_multibit_candidates_in_chinese(
    tmp_path: Path,
    candidate_manifest: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    discovery_tasks: list[dict[str, Any]],
) -> None:
    results = discovery_result_rows(candidates, passing=set())
    selection = select_discovery_candidates(results, candidates)
    features, scorers, datasets = companion_rows(results, candidates)
    gate = adjudicate_k1bn(
        candidate_manifest=candidate_manifest,
        discovery_tasks=discovery_tasks,
        selection=selection,
        dataset_rows=datasets,
        feature_rows=features,
        scorer_rows=scorers,
        result_rows=results,
        source_checks={"k1bm_single_bit_source_completed_hold": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1bn_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "uKNIT 第6轮 DDT/轨迹引导多 bit 差分审判" in svg
    assert "轨迹信息不进入网络" in svg
    assert "48个DDT/轨迹优先多 bit 差分均无发现候选" in svg
    assert report["all_48_candidates_visible"] is True


def discovery_result_rows(
    candidates: list[dict[str, Any]],
    *,
    passing: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        exact = 0.66 if candidate_id in passing else 0.52
        raw = exact - (0.04 if candidate_id in passing else 0.005)
        for split in EXPECTED_SPLITS:
            for view in DISCOVERY_VIEWS:
                rows.append(
                    result_row(
                        phase=DISCOVERY_PHASE,
                        candidate=candidate,
                        seed=DISCOVERY_SEED,
                        split=split,
                        view=view,
                        auc=exact if view == CANDIDATE_VIEW else raw,
                    )
                )
    return rows


def add_confirmation_results(
    rows: list[dict[str, object]],
    candidates: list[dict[str, Any]],
    selected_ids: list[str],
    *,
    passing: set[str],
) -> None:
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for candidate_id in selected_ids:
        candidate = by_id[candidate_id]
        for seed in CONFIRMATION_SEEDS:
            for split in EXPECTED_SPLITS:
                exact = 0.64 if candidate_id in passing else 0.52
                aucs = {
                    CANDIDATE_VIEW: exact,
                    RAW_VIEW: exact - (0.04 if candidate_id in passing else 0.005),
                    LABEL_SHUFFLE_VIEW: (
                        exact - 0.08 if candidate_id in passing else 0.50
                    ),
                }
                for view in CONFIRMATION_VIEWS:
                    rows.append(
                        result_row(
                            phase=CONFIRMATION_PHASE,
                            candidate=candidate,
                            seed=seed,
                            split=split,
                            view=view,
                            auc=aucs[view],
                        )
                    )


def result_row(
    *,
    phase: str,
    candidate: Mapping[str, Any],
    seed: int,
    split: str,
    view: str,
    auc: float,
) -> dict[str, object]:
    train_rows = 2048 if phase == DISCOVERY_PHASE else 4096
    holdout_rows = 1024 if phase == DISCOVERY_PHASE else 2048
    rows = train_rows if split == "train_seen" else holdout_rows
    candidate_id = str(candidate["candidate_id"])
    return {
        "run_id": RUN_ID,
        "phase": phase,
        "cipher_key": "uknit64",
        "rounds": 6,
        "candidate_id": candidate_id,
        "candidate_index": int(candidate["candidate_index"]),
        "candidate_family": str(candidate["family"]),
        "candidate_family_rank": int(candidate["family_rank"]),
        "candidate_source_cells": list(candidate["source_cells"]),
        "candidate_source_nibbles": list(candidate["source_nibbles"]),
        "input_difference": int(candidate["input_difference"]),
        "input_difference_hex": str(candidate["input_difference_hex"]),
        "input_weight": int(candidate["input_weight"]),
        "trail_log2_probability": float(candidate["trail_log2_probability"]),
        "trail_total_active_sboxes": int(candidate["trail_total_active_sboxes"]),
        "seed": seed,
        "split": split,
        "view": view,
        "rows": rows,
        "auc": auc,
        "feature_dim": EXPECTED_FEATURE_DIMS[view],
        "feature_sha256": f"feature-{phase}-{candidate_id}-{seed}-{split}-{view}",
        "dataset_sha256": f"dataset-{phase}-{candidate_id}-{seed}-{split}",
        "scorer_sha256": f"scorer-{phase}-{candidate_id}-{seed}-{view}",
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
    candidates: list[dict[str, Any]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    candidate_fields = (
        "candidate_id",
        "candidate_index",
        "candidate_family",
        "candidate_family_rank",
        "candidate_source_cells",
        "candidate_source_nibbles",
        "input_difference",
        "input_difference_hex",
        "input_weight",
        "trail_log2_probability",
        "trail_total_active_sboxes",
    )
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
                    *candidate_fields,
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
        key = (row["phase"], row["candidate_id"], row["seed"], row["view"])
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
                    *candidate_fields,
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
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for row in result_rows:
        key = (row["phase"], row["candidate_id"], row["seed"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        candidate = by_id[str(row["candidate_id"])]
        dataset_rows.append(
            {
                "run_id": RUN_ID,
                "phase": row["phase"],
                "candidate_id": row["candidate_id"],
                "candidate_index": candidate["candidate_index"],
                "candidate_family": candidate["family"],
                "seed": row["seed"],
                "split": row["split"],
                "cache_payloads_present": True,
                "row_overlap_with_train": 0,
            }
        )
    return feature_rows, scorer_rows, dataset_rows
