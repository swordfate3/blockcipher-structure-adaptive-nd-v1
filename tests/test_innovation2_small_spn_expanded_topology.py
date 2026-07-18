from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_small_spn_expanded_topology import (
    render_expanded_topology_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_expanded_neural_screen import (
    render_expanded_neural_screen_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_pair_relation_reasoner import (
    render_pair_relation_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_pair_relation_fair_control import (
    render_pair_relation_fair_control_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_pair_relation_no_triangle import (
    render_no_triangle_ablation_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_pair_state_topology_control import (
    render_pair_state_topology_control_svg,
)
from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_expanded_topology_labels as expanded,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_expanded_neural_screen as neural_screen,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_pair_relation_reasoner as pair_reasoner,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_pair_relation_no_triangle as no_triangle,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_pair_state_topology_control as pair_state_control,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import (
    TopologyTrainingConfig,
)


def test_expanded_player_family_preserves_e32_prefix() -> None:
    players = expanded.make_player_family(16)
    assert len(players) == len(set(players)) == 16
    assert players[:4] == expanded.EXPECTED_E32_PLAYERS
    assert all(sorted(player) == list(range(16)) for player in players)


def test_expanded_audit_config_is_frozen() -> None:
    config = expanded.ExpandedTopologyAuditConfig(run_id="audit")
    variants = expanded.make_expanded_variants(config)
    split = expanded.expanded_split_indices(variants)
    assert len(variants) == 64
    assert [
        len(split[name])
        for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    ] == [36, 12, 12, 4]
    with pytest.raises(ValueError, match="E37 audit freezes"):
        expanded.ExpandedTopologyAuditConfig(run_id="bad", player_variants=15)


def test_expanded_cache_resumes_and_last_player_matches_scalar(tmp_path: Path) -> None:
    config = expanded.ExpandedTopologyAuditConfig(
        run_id="cache",
        mode="smoke",
        sbox_variants=1,
        player_variants=2,
        rounds=(1,),
        keys=4,
    )
    first = expanded.run_cached_expanded_labels(config, cache_root=tmp_path)
    resumed = expanded.run_cached_expanded_labels(config, cache_root=tmp_path)
    assert first["generated_blocks"] == 28
    assert resumed["generated_blocks"] == 0
    assert first["completed"].all()
    assert expanded.expanded_scalar_vectorized_matches() is True


def test_expanded_selection_uses_only_train_topologies() -> None:
    cube = np.zeros((4, 16, 1, 1, 2), dtype=np.bool_)
    cube[:3, :3, 0, 0, 0] = True
    cube[:3, :2, 0, 0, 1] = True
    labels = cube.reshape(64, 1, 1, 2)
    selected = expanded.select_expanded_train_cells(labels)
    assert selected.tolist() == [[[True, False]]]

    modified = cube.copy()
    modified[3] = True
    modified[:3, 12:] = True
    assert np.array_equal(
        selected,
        expanded.select_expanded_train_cells(modified.reshape(64, 1, 1, 2)),
    )


def test_fair_control_keeps_heldout_family_identity() -> None:
    checks = expanded.fair_control_contract(expanded.make_player_family(16))
    assert all(checks.values())


def test_expanded_gate_separates_ready_hold_and_invalid() -> None:
    split_metrics = {
        "train": {"total": 8000, "positive": 4000, "negative": 4000},
        "unseen_sbox": {"total": 2400, "positive": 1200, "negative": 1200},
        "unseen_player": {"total": 2400, "positive": 1200, "negative": 1200},
        "dual_unseen": {"total": 800, "positive": 400, "negative": 400},
    }
    topology = {
        "counts": {
            "dual_p_effect_positive_rows": 300,
            "dual_p_effect_negative_rows": 300,
        },
        "fractions": {
            key: minimum + 0.01
            for key, minimum in expanded.MINIMUM_TOPOLOGY_FRACTIONS.items()
        },
    }
    metrics = {
        "selected_base_cells": 300,
        "distinct_topology_patterns": 150,
        "supported_rounds": 2,
        "split_metrics": split_metrics,
        "topology": topology,
        "marginal_baselines": {
            split: {"strongest_auc": limit - 0.01}
            for split, limit in expanded.MAXIMUM_MARGINAL_AUC.items()
        },
    }
    ready = expanded.adjudicate_expanded_topology(
        run_id="ready", readiness={"protocol": True}, metrics=metrics
    )
    assert ready["decision"] == (
        "innovation2_small_spn_expanded_topology_benchmark_ready"
    )

    shortcut_metrics = {
        **metrics,
        "marginal_baselines": {
            **metrics["marginal_baselines"],
            "dual_unseen": {"strongest_auc": 0.9},
        },
    }
    hold = expanded.adjudicate_expanded_topology(
        run_id="hold", readiness={"protocol": True}, metrics=shortcut_metrics
    )
    assert hold["decision"] == (
        "innovation2_small_spn_expanded_topology_benchmark_not_ready"
    )
    invalid = expanded.adjudicate_expanded_topology(
        run_id="invalid", readiness={"protocol": False}, metrics=metrics
    )
    assert invalid["decision"] == (
        "innovation2_small_spn_expanded_topology_protocol_invalid"
    )


def test_expanded_plot_explains_scope_and_thresholds(tmp_path: Path) -> None:
    fractions = {
        key: minimum + 0.01
        for key, minimum in expanded.MINIMUM_TOPOLOGY_FRACTIONS.items()
    }
    split_metrics = {
        "train": {"positive": 4000, "negative": 4000},
        "unseen_sbox": {"positive": 1200, "negative": 1200},
        "unseen_player": {"positive": 1200, "negative": 1200},
        "dual_unseen": {"positive": 400, "negative": 400},
    }
    gate = {
        "decision": "innovation2_small_spn_expanded_topology_benchmark_ready",
        "thresholds": {
            "topology_fractions": expanded.MINIMUM_TOPOLOGY_FRACTIONS,
            "maximum_marginal_auc": expanded.MAXIMUM_MARGINAL_AUC,
        },
        "metrics": {
            "selected_base_cells": 300,
            "per_round_selected_cells": [0, 100, 180, 20],
            "split_metrics": split_metrics,
            "topology": {"fractions": fractions},
            "marginal_baselines": {
                split: {"strongest_auc": limit - 0.05}
                for split, limit in expanded.MAXIMUM_MARGINAL_AUC.items()
            },
        },
    }
    output = tmp_path / "curves.svg"
    render_expanded_topology_svg({"gate": gate}, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E37" in svg
    assert "训练含12个独立P-layer" in svg
    assert "不是神经训练" in svg


def _expanded_model_data() -> dict[str, np.ndarray]:
    config = expanded.ExpandedTopologyAuditConfig(run_id="model")
    variants = expanded.make_expanded_variants(config)
    structures = expanded.make_structures()
    masks = expanded.make_output_masks()
    active = np.zeros((14, 16), dtype=np.float32)
    basis = np.zeros((14, 12, 16), dtype=np.float32)
    valid = np.zeros((14, 12), dtype=np.bool_)
    for structure_index, structure in enumerate(structures):
        active[structure_index, list(structure.active_bits)] = 1
        for row, bit in enumerate(structure.active_bits):
            basis[structure_index, row, bit] = 1
            valid[structure_index, row] = True
    mask_bits = np.asarray(
        [[(mask >> bit) & 1 for bit in range(16)] for mask in masks],
        dtype=np.float32,
    )
    return {
        "sboxes": np.asarray([variant.sbox for variant in variants]),
        "players": np.asarray([variant.player for variant in variants]),
        "structure_active": active,
        "structure_basis": basis,
        "structure_basis_valid": valid,
        "output_mask_bits": mask_bits,
        "split_indices": expanded.expanded_split_indices(variants),
    }


def test_expanded_model_contract_covers_all_64_variants() -> None:
    contract = neural_screen.measure_expanded_model_contract(_expanded_model_data())
    assert contract["graphgps_cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["cett_cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["cett_token_count"] == 37
    assert contract["fair_control_heldout_avoids_true_train"] is True
    assert contract["fair_control_heldout_avoids_corrupted_train"] is True
    assert contract["all_corrupted_players_are_permutations"] is True


def test_expanded_neural_matrix_and_gate_are_staged() -> None:
    full = neural_screen.expanded_neural_screen_matrix(
        TopologyTrainingConfig(run_id="full")
    )
    assert len(full) == 5
    assert sum(row.model_name == "graphgps" for row in full) == 2
    assert sum(row.model_name == "cett" for row in full) == 3
    assert sum(row.label_mode == "shuffled" for row in full) == 1

    def row(model: str, seed: int, dual: float, label: str = "true") -> dict:
        return {
            "model_name": model,
            "topology_mode": "true",
            "position_mode": "cell_equivariant",
            "processor_mode": (
                "edge_token_transformer" if model == "cett" else "stacked"
            ),
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": 0.75,
            "unseen_player_auc": 0.72,
            "dual_unseen_auc": dual,
            "parameter_count": 100 if model == "graphgps" else 200,
            "training_performed": True,
        }

    rows = [
        row("graphgps", 0, 0.75),
        row("graphgps", 1, 0.76),
        row("cett", 0, 0.78),
        row("cett", 1, 0.79),
        row("cett", 0, 0.5, "shuffled"),
    ]
    contract = {
        "graphgps_cell_relabeling_max_abs_logit_error": 1e-8,
        "cett_cell_relabeling_max_abs_logit_error": 1e-8,
        "cett_token_count": 37,
        "fair_control_heldout_avoids_true_train": True,
        "fair_control_heldout_avoids_corrupted_train": True,
        "all_corrupted_players_are_permutations": True,
    }
    gate = neural_screen.adjudicate_expanded_neural_screen(
        TopologyTrainingConfig(run_id="full"),
        {"source": True},
        contract,
        rows,
    )
    assert gate["decision"] == (
        "innovation2_small_spn_expanded_neural_candidate_screened"
    )
    assert gate["metrics"]["selected_candidate"] == "cett"

    weak = [
        {**item, "dual_unseen_auc": 0.65}
        if item["label_mode"] == "true"
        else item
        for item in rows
    ]
    hold = neural_screen.adjudicate_expanded_neural_screen(
        TopologyTrainingConfig(run_id="weak"),
        {"source": True},
        contract,
        weak,
    )
    assert hold["decision"] == (
        "innovation2_small_spn_expanded_neural_screen_not_ready"
    )


def test_expanded_neural_plot_explains_models_and_scope(tmp_path: Path) -> None:
    rows = []
    for model, base in (("graphgps", 0.72), ("cett", 0.76)):
        for seed in (0, 1):
            rows.append(
                {
                    "model_name": model,
                    "label_mode": "true",
                    "unseen_sbox_auc": base + seed * 0.01,
                    "unseen_player_auc": base - 0.02 + seed * 0.01,
                    "dual_unseen_auc": base - 0.01 + seed * 0.01,
                }
            )
    rows.append(
        {
            "model_name": "cett",
            "label_mode": "shuffled",
            "unseen_sbox_auc": 0.5,
            "unseen_player_auc": 0.5,
            "dual_unseen_auc": 0.5,
        }
    )
    summary = {
        "rows": rows,
        "gate": {
            "decision": "innovation2_small_spn_expanded_neural_candidate_screened",
            "metrics": {
                "baseline_auc": neural_screen.BASELINE_AUC,
                "selected_candidate": "cett",
            },
        },
    }
    output = tmp_path / "curves.svg"
    render_expanded_neural_screen_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E38" in svg
    assert "GraphGPS与边Token网络" in svg
    assert "不是实际密码高轮结果" in svg


def test_pair_relation_reasoner_builds_pairs_and_uses_round_steps() -> None:
    data = _expanded_model_data()
    model = SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(hidden_dim=32, path_rank=4, dropout=0.0),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    variants = torch.tensor([0, 15, 48, 63])
    rounds = torch.arange(4)
    structures = torch.tensor([0, 3, 7, 13])
    masks = torch.tensor([0, 17, 39, 63])
    relation, _ = model.build_initial_relation(variants, rounds, structures, masks)
    output = model(variants, rounds, structures, masks)
    assert relation.shape == (4, 16, 16, 32)
    assert output.shape == (4,)
    assert torch.isfinite(output).all()
    assert model.step_counts(rounds).tolist() == [2, 3, 4, 5]


def test_pair_relation_contract_is_equivariant_sensitive_and_bounded() -> None:
    contract = pair_reasoner.measure_pair_relation_contract(_expanded_model_data())
    assert contract["initial_pair_shape_matches"] is True
    assert contract["pair_count"] == 256
    assert contract["shared_triangle_block_count"] == 1
    assert contract["step_schedule"] == [2, 3, 4, 5]
    assert contract["parameter_count"] <= pair_reasoner.GRAPHGPS_PARAMETER_ANCHOR
    assert contract["cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["true_corrupted_max_abs_logit_difference"] >= 1e-5
    assert contract["fair_control_heldout_avoids_true_train"] is True
    assert contract["fair_control_heldout_avoids_corrupted_train"] is True
    assert contract["absolute_bit_cell_or_variant_embedding_absent"] is True


def test_pair_relation_matrix_and_gate_are_frozen() -> None:
    full_config = pair_reasoner.PairRelationTrainingConfig(run_id="full")
    matrix = pair_reasoner.pair_relation_training_matrix(full_config)
    assert len(matrix) == 3
    assert [row.seed for row in matrix if row.label_mode == "true"] == [0, 1]
    assert sum(row.label_mode == "shuffled" for row in matrix) == 1

    def row(seed: int, dual: float, label: str = "true") -> dict:
        return {
            "model_name": "pair_relation_reasoner",
            "topology_mode": "true",
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": 0.75,
            "unseen_player_auc": 0.72,
            "dual_unseen_auc": dual,
            "parameter_count": 150000,
            "training_performed": True,
        }

    rows = [row(0, 0.75), row(1, 0.76), row(0, 0.5, "shuffled")]
    contract = {
        "initial_pair_shape_matches": True,
        "pair_count": 256,
        "shared_triangle_block_count": 1,
        "step_schedule": [2, 3, 4, 5],
        "parameter_count": 150000,
        "cell_relabeling_max_abs_logit_error": 1e-8,
        "true_corrupted_max_abs_logit_difference": 0.1,
        "fair_control_heldout_avoids_true_train": True,
        "fair_control_heldout_avoids_corrupted_train": True,
        "all_corrupted_players_are_permutations": True,
        "absolute_bit_cell_or_variant_embedding_absent": True,
    }
    ready = pair_reasoner.adjudicate_pair_relation_reasoner(
        full_config, {"source": True}, contract, rows
    )
    assert ready["decision"] == (
        "innovation2_small_spn_pair_relation_candidate_screened"
    )

    weak = [
        {**item, "dual_unseen_auc": 0.65}
        if item["label_mode"] == "true"
        else item
        for item in rows
    ]
    hold = pair_reasoner.adjudicate_pair_relation_reasoner(
        full_config, {"source": True}, contract, weak
    )
    assert hold["decision"] == (
        "innovation2_small_spn_pair_relation_reasoner_not_ready"
    )


def test_pair_relation_plot_has_method_baselines_and_scope(tmp_path: Path) -> None:
    rows = [
        {
            "model_name": "pair_relation_reasoner",
            "topology_mode": "true",
            "label_mode": "true",
            "seed": seed,
            "unseen_sbox_auc": 0.78 + seed * 0.01,
            "unseen_player_auc": 0.74 + seed * 0.01,
            "dual_unseen_auc": 0.75 + seed * 0.01,
            "parameter_count": 150000,
        }
        for seed in (0, 1)
    ]
    rows.append(
        {
            "model_name": "pair_relation_reasoner",
            "topology_mode": "true",
            "label_mode": "shuffled",
            "seed": 0,
            "unseen_sbox_auc": 0.5,
            "unseen_player_auc": 0.5,
            "dual_unseen_auc": 0.5,
            "parameter_count": 150000,
        }
    )
    summary = {
        "rows": rows,
        "gate": {
            "decision": "innovation2_small_spn_pair_relation_candidate_screened"
        },
    }
    output = tmp_path / "curves.svg"
    render_pair_relation_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E39" in svg
    assert "16×16关系状态" in svg
    assert "不是实际密码高轮结果" in svg


def test_pair_relation_fair_control_gate_requires_per_seed_and_mean_margin() -> None:
    config = pair_reasoner.PairRelationTrainingConfig(run_id="control")

    def row(seed: int, dual: float, topology: str) -> dict:
        return {
            "model_name": "pair_relation_reasoner",
            "topology_mode": topology,
            "label_mode": "true",
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
            "parameter_count": 111825,
            "training_performed": True,
        }

    true_rows = [row(0, 0.75, "true"), row(1, 0.76, "true")]
    control_rows = [row(0, 0.68, "corrupted"), row(1, 0.69, "corrupted")]
    source_gate = {
        "run_id": "i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718",
        "decision": "innovation2_small_spn_pair_relation_candidate_screened",
    }
    contract = {
        "fair_control_heldout_avoids_true_train": True,
        "fair_control_heldout_avoids_corrupted_train": True,
        "all_corrupted_players_are_permutations": True,
    }
    confirmed = pair_reasoner.adjudicate_pair_relation_attribution(
        config,
        {"source": True},
        contract,
        source_gate,
        true_rows,
        control_rows,
    )
    assert confirmed["decision"] == (
        "innovation2_small_spn_pair_relation_topology_confirmed"
    )

    close_control = [row(0, 0.74, "corrupted"), row(1, 0.75, "corrupted")]
    hold = pair_reasoner.adjudicate_pair_relation_attribution(
        config,
        {"source": True},
        contract,
        source_gate,
        true_rows,
        close_control,
    )
    assert hold["decision"] == (
        "innovation2_small_spn_pair_relation_topology_not_attributed"
    )


def test_pair_relation_fair_control_plot_explains_attribution(tmp_path: Path) -> None:
    def row(seed: int, dual: float, topology: str) -> dict:
        return {
            "seed": seed,
            "topology_mode": topology,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
        }

    summary = {
        "true_rows": [row(0, 0.75, "true"), row(1, 0.76, "true")],
        "control_rows": [
            row(0, 0.68, "corrupted"),
            row(1, 0.69, "corrupted"),
        ],
        "gate": {
            "decision": "innovation2_small_spn_pair_relation_topology_confirmed"
        },
    }
    output = tmp_path / "control.svg"
    render_pair_relation_fair_control_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "E39 Phase B" in svg
    assert "公平错误P-layer" in svg
    assert "不是实际密码高轮结果" in svg


def test_no_triangle_block_is_parameter_matched_and_pair_local() -> None:
    data = _expanded_model_data()
    triangle = SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(
            processor_mode="triangle", hidden_dim=32, path_rank=4, dropout=0.0
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    local = SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(
            processor_mode="local", hidden_dim=32, path_rank=4, dropout=0.0
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    assert sum(parameter.numel() for parameter in triangle.parameters()) == sum(
        parameter.numel() for parameter in local.parameters()
    )

    relation = torch.randn(2, 16, 16, 32)
    changed = relation.clone()
    changed[0, 3, 11] += torch.randn(32)
    with torch.no_grad():
        baseline = local.local_block(relation)
        perturbed = local.local_block(changed)
    delta = torch.abs(baseline - perturbed)
    delta[0, 3, 11] = 0.0
    assert float(delta.max()) == 0.0


def test_no_triangle_contract_and_gate_are_frozen() -> None:
    contract = pair_reasoner.measure_pair_relation_contract(
        _expanded_model_data(), processor_mode="local"
    )
    assert contract["shared_triangle_block_count"] == 0
    assert contract["shared_local_block_count"] == 1
    assert contract["processor_mode"] == "local"
    assert contract["parameter_count"] == contract["counterpart_parameter_count"]
    assert contract["off_pair_influence_max_abs"] == 0.0
    assert contract["cell_relabeling_max_abs_logit_error"] <= 1e-6

    config = pair_reasoner.PairRelationTrainingConfig(run_id="e40")

    def row(
        seed: int, dual: float, *, model: str, label: str = "true"
    ) -> dict:
        return {
            "model_name": model,
            "processor_mode": "local" if model.endswith("no_triangle") else "triangle",
            "topology_mode": "true",
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
            "parameter_count": 111825,
            "training_performed": True,
        }

    source_rows = [
        row(0, 0.70, model="pair_relation_reasoner"),
        row(1, 0.74, model="pair_relation_reasoner"),
    ]
    candidate_rows = [
        row(0, 0.64, model="pair_relation_no_triangle"),
        row(1, 0.67, model="pair_relation_no_triangle"),
        row(0, 0.50, model="pair_relation_no_triangle", label="shuffled"),
    ]
    source_gate = {
        "run_id": no_triangle.E39_PHASE_A_RUN_ID,
        "decision": "innovation2_small_spn_pair_relation_candidate_screened",
    }
    topology_gate = {
        "run_id": no_triangle.E39_PHASE_B_RUN_ID,
        "decision": "innovation2_small_spn_pair_relation_topology_confirmed",
    }
    gate_contract = {
        **contract,
        "parameter_count": 111825,
        "counterpart_parameter_count": 111825,
        "true_corrupted_max_abs_logit_difference": 0.01,
    }
    attributed = no_triangle.adjudicate_no_triangle_ablation(
        config,
        {"source": True},
        gate_contract,
        candidate_rows,
        source_gate,
        topology_gate,
        source_rows,
    )
    assert attributed["decision"] == (
        "innovation2_small_spn_pair_relation_triangle_attributed"
    )

    close_rows = [
        {**candidate_rows[0], "dual_unseen_auc": 0.69},
        {**candidate_rows[1], "dual_unseen_auc": 0.73},
        candidate_rows[2],
    ]
    not_isolated = no_triangle.adjudicate_no_triangle_ablation(
        config,
        {"source": True},
        gate_contract,
        close_rows,
        source_gate,
        topology_gate,
        source_rows,
    )
    assert not_isolated["decision"] == (
        "innovation2_small_spn_pair_relation_triangle_not_isolated"
    )


def test_no_triangle_plot_explains_single_variable_ablation(tmp_path: Path) -> None:
    def row(seed: int, dual: float, label: str = "true") -> dict:
        return {
            "seed": seed,
            "label_mode": label,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
        }

    summary = {
        "source_rows": [row(0, 0.72), row(1, 0.74)],
        "rows": [row(0, 0.66), row(1, 0.68), row(0, 0.50, "shuffled")],
        "gate": {
            "decision": "innovation2_small_spn_pair_relation_triangle_attributed"
        },
    }
    output = tmp_path / "e40.svg"
    render_no_triangle_ablation_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E40" in svg
    assert "逐pair局部更新" in svg
    assert "不是实际密码高轮结果" in svg


def test_pair_state_topology_control_requires_per_seed_and_mean_margin() -> None:
    config = pair_reasoner.PairRelationTrainingConfig(run_id="e41")

    def row(seed: int, dual: float, topology: str, label: str = "true") -> dict:
        return {
            "model_name": "pair_relation_no_triangle",
            "processor_mode": "local",
            "topology_mode": topology,
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
            "parameter_count": 111825,
            "training_performed": True,
        }

    source_rows = [
        row(0, 0.72, "true"),
        row(1, 0.74, "true"),
        row(0, 0.50, "true", "shuffled"),
    ]
    controls = [row(0, 0.66, "corrupted"), row(1, 0.68, "corrupted")]
    source_gate = {
        "run_id": pair_state_control.E40_RUN_ID,
        "decision": "innovation2_small_spn_pair_relation_triangle_not_isolated",
    }
    contract = {
        "shared_triangle_block_count": 0,
        "shared_local_block_count": 1,
        "processor_mode": "local",
        "parameter_count": 111825,
        "counterpart_parameter_count": 111825,
        "off_pair_influence_max_abs": 0.0,
        "cell_relabeling_max_abs_logit_error": 1e-8,
        "fair_control_heldout_avoids_true_train": True,
        "fair_control_heldout_avoids_corrupted_train": True,
        "all_corrupted_players_are_permutations": True,
    }
    confirmed = pair_state_control.adjudicate_pair_state_topology_control(
        config,
        {"source": True},
        contract,
        source_gate,
        source_rows,
        controls,
    )
    assert confirmed["decision"] == (
        "innovation2_small_spn_pair_state_topology_confirmed"
    )

    close_controls = [row(0, 0.71, "corrupted"), row(1, 0.73, "corrupted")]
    hold = pair_state_control.adjudicate_pair_state_topology_control(
        config,
        {"source": True},
        contract,
        source_gate,
        source_rows,
        close_controls,
    )
    assert hold["decision"] == (
        "innovation2_small_spn_pair_state_topology_not_attributed"
    )


def test_pair_state_topology_control_plot_explains_scope(tmp_path: Path) -> None:
    def row(seed: int, dual: float, topology: str) -> dict:
        return {
            "seed": seed,
            "topology_mode": topology,
            "unseen_sbox_auc": dual + 0.1,
            "unseen_player_auc": dual + 0.05,
            "dual_unseen_auc": dual,
        }

    summary = {
        "true_rows": [row(0, 0.72, "true"), row(1, 0.74, "true")],
        "control_rows": [
            row(0, 0.66, "corrupted"),
            row(1, 0.68, "corrupted"),
        ],
        "gate": {
            "decision": "innovation2_small_spn_pair_state_topology_confirmed"
        },
    }
    output = tmp_path / "e41.svg"
    render_pair_state_topology_control_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E41" in svg
    assert "局部pair-state" in svg
    assert "不是实际密码高轮结果" in svg
