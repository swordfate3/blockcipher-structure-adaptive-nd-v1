from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1j import render_k1j_svg
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1i import (
    build_k1i_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1j import (
    INPUT_CONDITIONS,
    MODEL_ROLES,
    POOL_CONDITIONS,
    adjudicate_k1j,
    cell_role_indices,
    coordinate_permutation,
    k1i_pool_components,
    k1i_probabilities_from_pools,
    label_blind_row_permutation,
    permutation_checks,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1.csv"
)


def dialga_task() -> dict[str, object]:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    return next(
        task
        for task in tasks
        if task["cipher_key"] == "dialga128" and task["seed"] == 0
    )


def model() -> torch.nn.Module:
    result = build_k1i_control(
        task=dialga_task(),
        condition="exact_ordered",
        input_bits=1024,
    )
    result.eval()
    return result


def test_k1j_position_controls_are_bijective_and_cross_cell_role_preserving() -> None:
    candidate = model()
    structure = candidate.runtime_structure
    checks = permutation_checks(structure)

    assert all(checks.values())
    assert len(
        {
            tuple(coordinate_permutation(structure, condition).tolist())
            for condition in INPUT_CONDITIONS
        }
    ) == len(INPUT_CONDITIONS)

    indices = cell_role_indices(structure)
    cross = coordinate_permutation(structure, "cross_cell_role_mix")
    source_cells_by_target = [
        {
            int(structure.cell_membership[int(cross[int(indices[cell, role])])])
            for role in range(4)
        }
        for cell in range(structure.cells)
    ]
    assert all(len(source_cells) == 4 for source_cells in source_cells_by_target)


def test_k1j_exposed_pool_path_matches_forward_and_declared_invariances() -> None:
    candidate = model()
    features = torch.randint(
        0,
        2,
        (24, 1024),
        generator=torch.Generator().manual_seed(20260728),
    ).float()
    with torch.inference_mode():
        direct = torch.sigmoid(candidate(features).squeeze(1)).numpy()

    condition_probabilities: dict[str, np.ndarray] = {}
    condition_components: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for condition in (
        "native",
        "within_cell_role_roll",
        "whole_cell_roll",
        "cross_cell_role_mix",
    ):
        components = k1i_pool_components(candidate, features, condition=condition)
        condition_components[condition] = components
        condition_probabilities[condition] = k1i_probabilities_from_pools(
            candidate,
            *components,
            batch_size=8,
        )

    np.testing.assert_allclose(condition_probabilities["native"], direct, atol=1e-7)
    np.testing.assert_allclose(
        condition_probabilities["within_cell_role_roll"],
        direct,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        condition_probabilities["whole_cell_roll"],
        direct,
        atol=1e-7,
    )
    assert np.array_equal(
        condition_probabilities["within_cell_role_roll"],
        condition_probabilities["native"],
    )
    assert np.array_equal(
        condition_probabilities["whole_cell_roll"],
        condition_probabilities["native"],
    )
    torch.testing.assert_close(
        condition_components["cross_cell_role_mix"][0],
        condition_components["native"][0],
        atol=1e-6,
        rtol=0.0,
    )
    assert not torch.allclose(
        condition_components["cross_cell_role_mix"][1],
        condition_components["native"][1],
    )


def test_k1j_row_permutation_is_deterministic_label_blind_and_nonidentity() -> None:
    first = label_blind_row_permutation(
        2048,
        seed=1,
        split="same_key_fresh",
    )
    repeated = label_blind_row_permutation(
        2048,
        seed=1,
        split="same_key_fresh",
    )
    other_split = label_blind_row_permutation(
        2048,
        seed=1,
        split="cross_key_validation",
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, torch.arange(2048))
    assert not torch.equal(first, other_split)
    assert sorted(first.tolist()) == list(range(2048))


def test_k1j_gate_requires_consistent_fresh_split_attribution() -> None:
    pool_rows = synthetic_pool_rows()
    input_rows = synthetic_input_rows()
    gate = adjudicate_k1j(
        pool_rows=pool_rows,
        input_rows=input_rows,
        source_checks={"source_binding": True, "state_unchanged": True},
    )

    assert gate["status"] == "pass"
    assert gate["research_checks"]["within_cell_interaction_supported"] is True
    assert gate["decision"].endswith("within_cell_position_interaction_supported")

    failed = [dict(row) for row in pool_rows]
    target = next(
        row
        for row in failed
        if row["seed"] == 1
        and row["split"] == "cross_key_validation"
        and row["condition"] == "cross_cell_role_mix"
    )
    target["auc"] = 0.94
    target["explained_fraction"] = 0.05
    held = adjudicate_k1j(
        pool_rows=failed,
        input_rows=input_rows,
        source_checks={"source_binding": True, "state_unchanged": True},
    )

    assert held["research_checks"]["within_cell_interaction_supported"] is False
    assert held["status"] == "pass"
    assert held["research_checks"]["global_bit_branch_supported"] is True

    invalid = adjudicate_k1j(
        pool_rows=pool_rows,
        input_rows=input_rows,
        source_checks={"source_binding": False, "state_unchanged": True},
    )
    assert invalid["status"] == "invalid"


def test_k1j_plot_explains_joint_pool_result_in_chinese(tmp_path: Path) -> None:
    pool_rows = synthetic_pool_rows()
    input_rows = synthetic_input_rows()
    gate = adjudicate_k1j(
        pool_rows=pool_rows,
        input_rows=input_rows,
        source_checks={"source_binding": True, "state_unchanged": True},
    )
    output = tmp_path / "curves.svg"

    render_k1j_svg(gate, pool_rows, input_rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "Dialga 强信号究竟来自位置、cell 还是联合统计" in svg
    assert "哪种干预能解释原始信号" in svg
    assert "通过门槛 80%" in svg
    assert "精确 GF(2)" in svg


def synthetic_pool_rows() -> list[dict[str, object]]:
    intervention_aucs = {
        "native": 0.96,
        "within_cell_role_roll": 0.96,
        "whole_cell_roll": 0.96,
        "cross_cell_role_mix": 0.54,
        "bit_pool_row_shuffle": 0.53,
        "cell_pool_row_shuffle": 0.94,
        "both_pool_row_shuffle": 0.50,
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            for condition in POOL_CONDITIONS:
                auc = intervention_aucs[condition]
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "auc": auc,
                        "native_auc": 0.96,
                        "no_topology_auc": 0.52,
                        "source_gap": 0.44,
                        "explained_fraction": max(0.0, min(1.0, (0.96 - auc) / 0.44)),
                        "max_abs_probability_delta_from_native": (
                            0.0
                            if condition
                            in {
                                "native",
                                "within_cell_role_roll",
                                "whole_cell_roll",
                            }
                            else 0.5
                        ),
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "strict_state_dict_load": True,
                    }
                )
    return rows


def synthetic_input_rows() -> list[dict[str, object]]:
    return [
        {
            "model_role": role,
            "seed": seed,
            "split": split,
            "condition": condition,
            "auc": 0.96 if condition == "native_input" else 0.60,
            "native_auc": 0.96,
            "native_minus_condition_auc": (
                0.0 if condition == "native_input" else 0.36
            ),
            "training_performed": False,
            "optimizer_steps": 0,
            "strict_state_dict_load": True,
        }
        for role in MODEL_ROLES
        for seed in (0, 1)
        for split in ("train_seen", "same_key_fresh", "cross_key_validation")
        for condition in INPUT_CONDITIONS
    ]
