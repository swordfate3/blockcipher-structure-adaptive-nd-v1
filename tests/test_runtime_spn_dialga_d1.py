from __future__ import annotations

from copy import deepcopy

from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    MODELS,
    adjudicate_runtime_spn_dialga_d1,
    adjudicate_runtime_spn_dialga_d3,
)


def _rows(
    aucs: dict[int, dict[str, float]],
    *,
    rounds: int = 4,
    runtime_round_start: int = 2,
) -> list[dict[str, object]]:
    common_options = {
        "runtime_structure_path": "configs/runtime/spn/dialga128.json",
        "runtime_round_start": runtime_round_start,
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
    }
    window_hashes = {
        "correct": "a" * 64,
        "corrupted": "b" * 64,
        "no_topology": "a" * 64,
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for role, model in MODELS.items():
            options = common_options.copy()
            if role == "corrupted":
                options["topology_corruption_seed"] = 20260725
            auc = aucs[seed][role]
            rows.append(
                {
                    "cipher": "Dialga-128",
                    "cipher_key": "dialga128",
                    "model": model,
                    "rounds": rounds,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "dataset_label_mode": "balanced_per_class",
                    "pairs_per_sample": 4,
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "difference_profile": "",
                    "difference_member": "",
                    "input_difference": 0x40,
                    "train_key": 0,
                    "validation_key": int("11" * 32, 16),
                    "parameter_count": 442466,
                    "trainable_parameter_count": 442466,
                    "runtime_structure_descriptor_name": (
                        "Dialga-128 20-round heterogeneous runtime SPN structure"
                    ),
                    "runtime_structure_descriptor_path": (
                        "/repo/configs/runtime/spn/dialga128.json"
                    ),
                    "runtime_structure_descriptor_sha256": "c" * 64,
                    "runtime_structure_round_start": runtime_round_start,
                    "runtime_structure_available_rounds": 20,
                    "runtime_structure_loaded_rounds": 2,
                    "runtime_structure_unique_transition_count": 2,
                    "runtime_structure_homogeneous": False,
                    "runtime_structure_mode": {
                        "correct": "true",
                        "corrupted": "corrupted",
                        "no_topology": "independent",
                    }[role],
                    "runtime_structure_window_control": "full",
                    "runtime_structure_transition_sha256s": ["d" * 64, "e" * 64],
                    "runtime_structure_window_sha256": window_hashes[role],
                    "metrics": {"auc": auc},
                    "history": [
                        {"epoch": epoch + 1, "val_auc": auc - 0.009 + epoch * 0.001}
                        for epoch in range(10)
                    ],
                    "training": {
                        "epochs": 10,
                        "loss": "mse",
                        "optimizer": "adam",
                        "learning_rate": 0.0001,
                        "weight_decay": 0.00001,
                        "checkpoint_metric": "val_auc",
                        "restore_best_checkpoint": True,
                        "selected_checkpoint": "best",
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "input_bits": 1024,
                        "pair_bits": 256,
                        "model_options": options,
                        "train_dataset_storage": "disk",
                        "validation_dataset_storage": "disk",
                    },
                }
            )
    return rows


def test_dialga_d1_passes_only_with_two_seed_control_margins() -> None:
    gate = adjudicate_runtime_spn_dialga_d1(
        run_id="pass",
        rows=_rows(
            {
                0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
                1: {"correct": 0.58, "corrupted": 0.56, "no_topology": 0.54},
            }
        ),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_dialga_runtime_e4_d1_two_seed_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_dialga_d1_holds_when_one_seed_misses_a_control() -> None:
    gate = adjudicate_runtime_spn_dialga_d1(
        run_id="hold",
        rows=_rows(
            {
                0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
                1: {"correct": 0.58, "corrupted": 0.579, "no_topology": 0.54},
            }
        ),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_dialga_runtime_e4_d1_not_supported"


def test_dialga_d1_fails_closed_on_protocol_drift() -> None:
    rows = _rows(
        {
            0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
            1: {"correct": 0.58, "corrupted": 0.56, "no_topology": 0.54},
        }
    )
    invalid = deepcopy(rows)
    invalid[1]["negative_mode"] = "random_ciphertext"

    gate = adjudicate_runtime_spn_dialga_d1(run_id="invalid", rows=invalid)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_dialga_runtime_e4_d1_protocol_invalid"
    assert gate["protocol_checks"]["same_data_protocol"] is False


def test_dialga_d1_fails_closed_when_history_does_not_match_best_auc() -> None:
    rows = _rows(
        {
            0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
            1: {"correct": 0.58, "corrupted": 0.56, "no_topology": 0.54},
        }
    )
    rows[0]["history"] = rows[0]["history"][:-1]  # type: ignore[index]

    gate = adjudicate_runtime_spn_dialga_d1(run_id="invalid-history", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["complete_best_checkpoint_histories"] is False


def test_dialga_d3_passes_only_on_the_frozen_adjacent_window() -> None:
    gate = adjudicate_runtime_spn_dialga_d3(
        run_id="d3-pass",
        rows=_rows(
            {
                0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
                1: {"correct": 0.58, "corrupted": 0.56, "no_topology": 0.54},
            },
            rounds=5,
            runtime_round_start=3,
        ),
    )

    assert gate["status"] == "pass"
    assert (
        gate["decision"] == "innovation1_dialga_runtime_e4_d3_adjacent_window_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "prefix-r5 adjacent-window" in gate["claim_scope"]


def test_dialga_d3_holds_when_the_adjacent_window_does_not_replicate() -> None:
    gate = adjudicate_runtime_spn_dialga_d3(
        run_id="d3-hold",
        rows=_rows(
            {
                0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
                1: {"correct": 0.51, "corrupted": 0.50, "no_topology": 0.50},
            },
            rounds=5,
            runtime_round_start=3,
        ),
    )

    assert gate["status"] == "hold"
    assert (
        gate["decision"]
        == "innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated"
    )


def test_dialga_d3_fails_closed_on_d1_round_or_window_metadata() -> None:
    rows = _rows(
        {
            0: {"correct": 0.60, "corrupted": 0.57, "no_topology": 0.55},
            1: {"correct": 0.58, "corrupted": 0.56, "no_topology": 0.54},
        },
        rounds=5,
        runtime_round_start=3,
    )
    invalid = deepcopy(rows)
    invalid[0]["runtime_structure_round_start"] = 2

    gate = adjudicate_runtime_spn_dialga_d3(run_id="d3-invalid", rows=invalid)

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_dialga_runtime_e4_d3_protocol_invalid"
    assert gate["protocol_checks"]["runtime_descriptor_contract"] is False
