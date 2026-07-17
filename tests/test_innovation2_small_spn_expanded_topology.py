from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli.plot_innovation2_small_spn_expanded_topology import (
    render_expanded_topology_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_expanded_neural_screen import (
    render_expanded_neural_screen_svg,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_expanded_topology_labels as expanded,
)
from blockcipher_nd.tasks.innovation2 import (
    small_spn_expanded_neural_screen as neural_screen,
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
