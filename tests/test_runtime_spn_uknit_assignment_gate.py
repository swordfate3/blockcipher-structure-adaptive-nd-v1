from __future__ import annotations

from copy import deepcopy

from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_assignment import (
    adjudicate_uknit_sbox_assignment,
)


def _row(seed: int, role: str, auc: float) -> dict[str, object]:
    context = "late_pair" if role == "anchor" else "late_cell"
    mode = "sbox_shuffled" if role == "shuffled" else "true"
    model = (
        "runtime_spn_e4_equivariant_sbox_shuffled"
        if role == "shuffled"
        else "runtime_spn_e4_equivariant_true"
    )
    options = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": context,
    }
    if role == "shuffled":
        options["sbox_assignment_shuffle_seed"] = 20260724
    return {
        "cipher": "uKNIT-BC",
        "cipher_key": "uknit64",
        "structure": "SPN",
        "model": model,
        "runtime_structure_mode": mode,
        "runtime_structure_descriptor_sha256": "a" * 64,
        "runtime_structure_round_start": 2,
        "runtime_structure_available_rounds": 11,
        "runtime_structure_loaded_rounds": 2,
        "rounds": 4,
        "seed": seed,
        "train_key": 0,
        "validation_key": int("11" * 16, 16),
        "input_difference": 0x40,
        "samples_per_class": 2048,
        "target_epochs": 10,
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "parameter_count": 442466,
        "trainable_parameter_count": 442466,
        "metrics": {"auc": auc},
        "validation": {"samples_per_class": 1024},
        "training": {
            "seed": seed,
            "batch_size": 256,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "optimizer_state_transition": "reset_each_stage",
            "weight_decay": 0.00001,
            "loss": "mse",
            "lr_scheduler": "none",
            "checkpoint_metric": "val_auc",
            "selected_checkpoint": "best",
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
            "dataset_cache_root": "outputs/local_cache/u1",
            "model_options": options,
        },
    }


def _passing_rows() -> list[dict[str, object]]:
    return [
        _row(0, "candidate", 0.550),
        _row(0, "anchor", 0.540),
        _row(0, "shuffled", 0.530),
        _row(1, "candidate", 0.545),
        _row(1, "anchor", 0.535),
        _row(1, "shuffled", 0.525),
    ]


def test_uknit_assignment_gate_passes_only_two_seed_margin_panel() -> None:
    gate = adjudicate_uknit_sbox_assignment(run_id="u1", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_uknit_sbox_assignment_two_seed_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["seed_results"]["0"]["candidate_minus_anchor"] == 0.010000000000000009


def test_uknit_assignment_gate_holds_when_one_seed_misses_margin() -> None:
    rows = _passing_rows()
    rows[-1]["metrics"]["auc"] = 0.541  # type: ignore[index]

    gate = adjudicate_uknit_sbox_assignment(run_id="u1", rows=rows)

    assert gate["status"] == "hold"
    assert gate["protocol_checks"]["three_unique_roles_per_seed"] is True
    assert gate["research_checks"]["seed1_candidate_beats_shuffle_by_0p005"] is False
    assert "activation/gradient attribution" in gate["next_action"]


def test_uknit_assignment_gate_rejects_protocol_drift() -> None:
    rows = deepcopy(_passing_rows())
    rows[2]["negative_mode"] = "random_ciphertext"

    gate = adjudicate_uknit_sbox_assignment(run_id="u1", rows=rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_uknit_sbox_assignment_protocol_invalid"
    assert (
        gate["protocol_checks"]["strict_encrypted_random_plaintext_negatives"] is False
    )


def test_uknit_edge_gate_uses_the_same_two_seed_protocol() -> None:
    rows = _passing_rows()
    for row in rows:
        options = row["training"]["model_options"]
        if options["sbox_context_mode"] == "late_cell":
            options["sbox_context_mode"] = "edge_gate"

    gate = adjudicate_uknit_sbox_assignment(
        run_id="u2b",
        rows=rows,
        candidate_context="edge_gate",
    )

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_sbox_edge_gate_u2b"
    assert gate["decision"] == "innovation1_uknit_sbox_edge_gate_two_seed_supported"
    assert all(gate["protocol_checks"].values())


def test_uknit_state_triplet_gate_uses_difference_only_edge_anchor() -> None:
    rows = _passing_rows()
    for index, row in enumerate(rows):
        options = row["training"]["model_options"]
        options["sbox_context_mode"] = "edge_gate"
        options["cell_input_mode"] = (
            "difference_only" if index % 3 == 1 else "state_triplet"
        )

    gate = adjudicate_uknit_sbox_assignment(
        run_id="u2c",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="state_triplet",
        anchor_context="edge_gate",
        anchor_cell_input_mode="difference_only",
    )

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_state_triplet_u2c"
    assert gate["decision"] == "innovation1_uknit_state_triplet_two_seed_supported"
    assert all(gate["protocol_checks"].values())


def _inverse_sbox_triplet_rows() -> list[dict[str, object]]:
    rows = _passing_rows()
    for index, row in enumerate(rows):
        options = row["training"]["model_options"]
        options["sbox_context_mode"] = "edge_gate"
        options["cell_input_mode"] = (
            "state_triplet" if index % 3 == 1 else "inverse_sbox_triplet"
        )
    return rows


def _inverse_sbox_triplet_gate(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return adjudicate_uknit_sbox_assignment(
        run_id="u2d",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="inverse_sbox_triplet",
        anchor_context="edge_gate",
        anchor_cell_input_mode="state_triplet",
    )


def test_uknit_inverse_sbox_triplet_gate_passes_two_seed_panel() -> None:
    gate = _inverse_sbox_triplet_gate(_inverse_sbox_triplet_rows())

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_inverse_sbox_triplet_u2d"
    assert (
        gate["decision"] == "innovation1_uknit_inverse_sbox_triplet_two_seed_supported"
    )
    assert all(gate["protocol_checks"].values())


def test_uknit_inverse_sbox_triplet_gate_holds_failed_margin() -> None:
    rows = _inverse_sbox_triplet_rows()
    rows[3]["metrics"]["auc"] = 0.538  # type: ignore[index]

    gate = _inverse_sbox_triplet_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_inverse_sbox_triplet_hold"
    assert gate["research_checks"]["seed1_candidate_beats_anchor_by_0p005"] is False


def test_uknit_inverse_sbox_triplet_gate_rejects_protocol_drift() -> None:
    rows = _inverse_sbox_triplet_rows()
    rows[2]["negative_mode"] = "random_ciphertext"

    gate = _inverse_sbox_triplet_gate(rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_uknit_inverse_sbox_triplet_protocol_invalid"


def _dual_view_triplet_rows() -> list[dict[str, object]]:
    rows = _passing_rows()
    for index, row in enumerate(rows):
        options = row["training"]["model_options"]
        options["sbox_context_mode"] = "edge_gate"
        options["cell_input_mode"] = (
            "state_triplet" if index % 3 == 1 else "dual_view_triplet"
        )
    return rows


def _dual_view_triplet_gate(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return adjudicate_uknit_sbox_assignment(
        run_id="u2e",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="dual_view_triplet",
        anchor_context="edge_gate",
        anchor_cell_input_mode="state_triplet",
    )


def test_uknit_dual_view_triplet_gate_passes_two_seed_panel() -> None:
    gate = _dual_view_triplet_gate(_dual_view_triplet_rows())

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_dual_view_triplet_u2e"
    assert gate["decision"] == "innovation1_uknit_dual_view_triplet_two_seed_supported"
    assert all(gate["protocol_checks"].values())


def test_uknit_dual_view_triplet_gate_holds_failed_margin() -> None:
    rows = _dual_view_triplet_rows()
    rows[3]["metrics"]["auc"] = 0.538  # type: ignore[index]

    gate = _dual_view_triplet_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_dual_view_triplet_hold"
    assert gate["research_checks"]["seed1_candidate_beats_anchor_by_0p005"] is False


def test_uknit_dual_view_triplet_gate_rejects_protocol_drift() -> None:
    rows = _dual_view_triplet_rows()
    rows[2]["negative_mode"] = "random_ciphertext"

    gate = _dual_view_triplet_gate(rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_uknit_dual_view_triplet_protocol_invalid"


def _delta_u_query_rows() -> list[dict[str, object]]:
    rows = _passing_rows()
    for index, row in enumerate(rows):
        options = row["training"]["model_options"]
        options["sbox_context_mode"] = "edge_gate"
        options["cell_input_mode"] = (
            "state_triplet_delta_v_query"
            if index % 3 == 1
            else "state_triplet_delta_u_query"
        )
        row["parameter_count"] = 458850
        row["trainable_parameter_count"] = 458850
    return rows


def _delta_u_query_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    return adjudicate_uknit_sbox_assignment(
        run_id="u2f",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="state_triplet_delta_u_query",
        anchor_context="edge_gate",
        anchor_cell_input_mode="state_triplet_delta_v_query",
    )


def test_uknit_delta_u_query_gate_passes_two_seed_panel() -> None:
    gate = _delta_u_query_gate(_delta_u_query_rows())

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_delta_u_query_u2f"
    assert gate["decision"] == "innovation1_uknit_delta_u_query_two_seed_supported"
    assert all(gate["protocol_checks"].values())


def test_uknit_delta_u_query_gate_holds_failed_margin() -> None:
    rows = _delta_u_query_rows()
    rows[3]["metrics"]["auc"] = 0.538  # type: ignore[index]

    gate = _delta_u_query_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_delta_u_query_hold"
    assert gate["research_checks"]["seed1_candidate_beats_anchor_by_0p005"] is False


def test_uknit_delta_u_query_gate_rejects_protocol_drift() -> None:
    rows = _delta_u_query_rows()
    rows[2]["negative_mode"] = "random_ciphertext"

    gate = _delta_u_query_gate(rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_uknit_delta_u_query_protocol_invalid"


def _delta_u_cross_window_gate(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    for row in rows:
        row["rounds"] = 5
        row["runtime_structure_round_start"] = 3
        row["training"]["model_options"]["runtime_round_start"] = 3
    return adjudicate_uknit_sbox_assignment(
        run_id="u2h",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="state_triplet_delta_u_query",
        anchor_context="edge_gate",
        anchor_cell_input_mode="state_triplet_delta_v_query",
        expected_rounds=5,
        expected_round_start=3,
        experiment_stage="u2h",
    )


def test_uknit_delta_u_cross_window_gate_passes_r5_panel() -> None:
    gate = _delta_u_cross_window_gate(_delta_u_query_rows())

    assert gate["status"] == "pass"
    assert gate["task"] == "innovation1_uknit_runtime_e4_delta_u_query_u2h"
    assert gate["decision"] == "innovation1_uknit_delta_u_cross_window_supported"
    assert gate["protocol_checks"]["uknit_prefix_r5_protocol"] is True
    assert all(gate["protocol_checks"].values())


def test_uknit_delta_u_cross_window_gate_holds_nonreplication() -> None:
    rows = _delta_u_query_rows()
    rows[3]["metrics"]["auc"] = 0.538  # type: ignore[index]

    gate = _delta_u_cross_window_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_delta_u_cross_window_not_replicated"
    assert "window-specific" in gate["next_action"]


def test_uknit_delta_u_cross_window_gate_rejects_r4_window() -> None:
    rows = _delta_u_query_rows()
    for row in rows:
        row["rounds"] = 5

    gate = adjudicate_uknit_sbox_assignment(
        run_id="u2h",
        rows=rows,
        candidate_context="edge_gate",
        candidate_cell_input_mode="state_triplet_delta_u_query",
        anchor_context="edge_gate",
        anchor_cell_input_mode="state_triplet_delta_v_query",
        expected_rounds=5,
        expected_round_start=3,
        experiment_stage="u2h",
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_uknit_delta_u_cross_window_protocol_invalid"
    assert gate["protocol_checks"]["exact_descriptor_window"] is False
