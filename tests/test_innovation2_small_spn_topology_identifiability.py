from __future__ import annotations

import numpy as np

from blockcipher_nd.cli.plot_innovation2_small_spn_topology_identifiability import (
    render_topology_identifiability_svg,
)

from blockcipher_nd.tasks.innovation2.small_spn_topology_identifiability import (
    MINIMUM_COUNTS,
    adjudicate_topology_identifiability,
    topology_identifiability_metrics,
)


def test_topology_identifiability_separates_main_effects_and_interaction() -> None:
    cube = np.zeros((4, 4, 1, 1, 4), dtype=np.bool_)
    for sbox in range(4):
        for player in range(4):
            cube[sbox, player, 0, 0, 0] = bool(player % 2)
            cube[sbox, player, 0, 0, 1] = bool((sbox & player) & 1)
            cube[sbox, player, 0, 0, 2] = bool(sbox % 2)
    labels = cube.reshape(16, 1, 1, 4)
    selected = np.ones((1, 1, 4), dtype=np.bool_)

    metrics = topology_identifiability_metrics(labels, selected)

    assert metrics["counts"]["selected_cells"] == 4
    assert metrics["counts"]["train_p_sensitive_any_s"] == 2
    assert metrics["counts"]["train_p_sensitive_all_s"] == 1
    assert metrics["counts"]["train_s_sensitive_any_p"] == 2
    assert metrics["counts"]["heldout_p3_novel_any_s"] == 2
    assert metrics["counts"]["heldout_p3_novel_all_s"] == 1
    assert metrics["counts"]["dual_p_effect_cells"] == 2
    assert metrics["counts"]["dual_p_effect_positive"] == 2
    assert metrics["counts"]["dual_p_effect_negative"] == 0
    assert metrics["counts"]["train_interaction_cells"] == 1
    assert metrics["counts"]["full_interaction_cells"] == 1


def test_topology_identifiability_gate_is_frozen() -> None:
    passing_counts = {key: value for key, value in MINIMUM_COUNTS.items()}
    passing_metrics = {"counts": passing_counts, "fractions": {}, "per_round": []}
    gate = adjudicate_topology_identifiability(
        run_id="pass", metrics=passing_metrics, readiness={"source": True}
    )
    assert gate["decision"] == "innovation2_small_spn_topology_labels_identifiable"

    failing_counts = {**passing_counts, "dual_p_effect_negative": 31}
    failing = adjudicate_topology_identifiability(
        run_id="hold",
        metrics={"counts": failing_counts, "fractions": {}, "per_round": []},
        readiness={"source": True},
    )
    assert failing["decision"] == (
        "innovation2_small_spn_topology_labels_not_identifiable"
    )


def test_topology_identifiability_plot_has_scope_and_thresholds(tmp_path) -> None:
    counts = {
        **MINIMUM_COUNTS,
        "train_s_sensitive_any_p": 300,
        "train_s_sensitive_all_p": 50,
    }
    metrics = {
        "counts": counts,
        "fractions": {},
        "per_round": [
            {
                "round_index": index,
                "selected_cells": 150,
                "train_p_sensitive_any_s": 120,
                "dual_p_effect_cells": 80,
                "full_interaction_cells": 100,
            }
            for index in range(4)
        ],
    }
    gate = adjudicate_topology_identifiability(
        run_id="plot", metrics=metrics, readiness={"source": True}
    )
    output = tmp_path / "curves.svg"
    render_topology_identifiability_svg({"gate": gate}, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E36" in svg
    assert "最低宽度门" in svg
    assert "不是神经结果" in svg
