from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d2 import (
    FROZEN_MODEL_OPTIONS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d4 import (
    D1_DECISION,
    D3_DECISION,
    adjudicate_factorial_dialga,
    evaluate_factorial_dialga,
)


def _row(seed: int, condition: str, auc: float) -> dict[str, object]:
    data_source = "d1_r4" if condition.startswith("r4") else "d3_r5"
    data_rounds = 4 if data_source == "d1_r4" else 5
    runtime_round_start = 2 if condition.endswith("w2") else 3
    feature_hash = ("a" if data_rounds == 4 else "b") * 64
    metadata_hash = ("c" if data_rounds == 4 else "d") * 64
    return {
        "seed": seed,
        "condition": condition,
        "cipher": "Dialga-128",
        "data_source": data_source,
        "data_rounds": data_rounds,
        "runtime_round_start": runtime_round_start,
        "runtime_loaded_rounds": 2,
        "auc": auc,
        "source_anchor_auc": 0.90,
        "auc_delta_from_r4_w2": auc - 0.90,
        "chance_excess_retention_ratio": (auc - 0.5) / 0.4,
        "max_abs_probability_delta_from_r4_w2": 0.0 if condition == "r4_w2" else 0.1,
        "mean_abs_probability_delta_from_r4_w2": 0.0 if condition == "r4_w2" else 0.05,
        "mean_probability": 0.5,
        "probability_sha256": {"r4_w2": "1", "r4_w3": "2", "r5_w2": "3", "r5_w3": "4"}[
            condition
        ]
        * 64,
        "checkpoint_sha256": ("5" if seed == 0 else "6") * 64,
        "checkpoint_selected": "best",
        "checkpoint_reported_seed": seed,
        "checkpoint_best_metric": 0.90,
        "strict_state_dict_load": True,
        "feature_sha256": feature_hash,
        "label_sha256": "7" * 64,
        "metadata_sha256": metadata_hash,
        "d1_results_sha256": "8" * 64,
        "d1_gate_sha256": "9" * 64,
        "d3_results_sha256": "a" * 64,
        "d3_gate_sha256": "b" * 64,
        "source_d1_verified": True,
        "source_d3_verified": True,
        "source_d1_decision": D1_DECISION,
        "source_d3_decision": D3_DECISION,
        "descriptor_name": "Dialga-128 20-round heterogeneous runtime SPN structure",
        "descriptor_path": "configs/runtime/spn/dialga128.json",
        "descriptor_sha256": "c" * 64,
        "runtime_structure_mode": "correct",
        "relation_mode": "true",
        "runtime_structure_window_sha256": ("d" if runtime_round_start == 2 else "e")
        * 64,
        "runtime_structure_transition_sha256s": ["f" * 64, "0" * 64],
        "runtime_structure_unique_transition_count": 2,
        "samples_total": 2048,
        "validation_seed": 10000 + seed,
        "input_bits": 1024,
        "pair_bits": 256,
        "pairs_per_sample": 4,
        "input_difference": 0x40,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "validation_key": int("11" * 32, 16),
        "parameter_count": 442466,
        "model_options": FROZEN_MODEL_OPTIONS,
        "training_performed": False,
        "data_generation_performed": False,
    }


def _rows(values: tuple[float, float, float, float]) -> list[dict[str, object]]:
    return [
        _row(seed, condition, auc)
        for seed in (0, 1)
        for condition, auc in zip(
            ("r4_w2", "r4_w3", "r5_w2", "r5_w3"), values, strict=True
        )
    ]


def test_d4_gate_isolates_fifth_round_data_loss() -> None:
    gate = adjudicate_factorial_dialga(
        run_id="d4-data", rows=_rows((0.90, 0.86, 0.51, 0.50))
    )

    assert gate["status"] == "pass"
    assert gate["diagnosis"] == "fifth_round_data_signal_loss"
    assert all(gate["protocol_checks"].values())


def test_d4_gate_isolates_runtime_window_loss() -> None:
    gate = adjudicate_factorial_dialga(
        run_id="d4-window", rows=_rows((0.90, 0.51, 0.86, 0.52))
    )

    assert gate["diagnosis"] == "runtime_window_incompatibility"


def test_d4_gate_identifies_both_factor_losses() -> None:
    gate = adjudicate_factorial_dialga(
        run_id="d4-both", rows=_rows((0.90, 0.51, 0.52, 0.50))
    )

    assert gate["diagnosis"] == "both_data_and_window_degrade"


def test_d4_gate_fails_closed_on_checkpoint_drift() -> None:
    rows = _rows((0.90, 0.86, 0.51, 0.50))
    rows[1] = deepcopy(rows[1])
    rows[1]["checkpoint_sha256"] = "0" * 64

    gate = adjudicate_factorial_dialga(run_id="d4-invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_checkpoint_within_seed"] is False


def test_d4_evaluator_crosses_only_data_and_runtime_window(tmp_path: Path) -> None:
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options=FROZEN_MODEL_OPTIONS,
    )
    checkpoint = tmp_path / "dialga.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "selected_checkpoint": "best",
                "seed": 0,
                "best_checkpoint_metric": 0.9,
            },
        },
        checkpoint,
    )
    labels = np.array([0, 1] * 8, dtype=np.uint8)
    rng = np.random.default_rng(25)
    datasets = {
        source: DifferentialDataset(
            features=rng.integers(0, 2, size=(16, 1024), dtype=np.uint8),
            labels=labels,
            metadata={
                "cipher": "Dialga-128",
                "rounds": rounds,
                "seed": 10000,
                "samples_total": 2048,
                "samples_per_class": 1024,
                "input_bits": 1024,
                "pair_bits": 256,
                "pairs_per_sample": 4,
                "input_difference": 0x40,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
            },
        )
        for source, rounds in (("d1_r4", 4), ("d3_r5", 5))
    }
    rows = evaluate_factorial_dialga(
        seed=0,
        model_options=FROZEN_MODEL_OPTIONS,
        checkpoint_path=checkpoint,
        datasets=datasets,
        dataset_hashes={
            "d1_r4": {
                "feature_sha256": "1" * 64,
                "label_sha256": "2" * 64,
                "metadata_sha256": "3" * 64,
            },
            "d3_r5": {
                "feature_sha256": "4" * 64,
                "label_sha256": "2" * 64,
                "metadata_sha256": "5" * 64,
            },
        },
        structures={
            2: dialga128_runtime_structure(2, round_start=2),
            3: dialga128_runtime_structure(2, round_start=3),
        },
        anchor_auc=0.9,
        checkpoint_sha256="6" * 64,
        source_hashes={
            "d1_results_sha256": "7" * 64,
            "d1_gate_sha256": "8" * 64,
            "d3_results_sha256": "9" * 64,
            "d3_gate_sha256": "a" * 64,
        },
        descriptor_name="Dialga-128 20-round heterogeneous runtime SPN structure",
        descriptor_path="configs/runtime/spn/dialga128.json",
        descriptor_sha256="b" * 64,
        batch_size=8,
    )

    assert [row["condition"] for row in rows] == ["r4_w2", "r4_w3", "r5_w2", "r5_w3"]
    assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert rows[0]["feature_sha256"] == rows[1]["feature_sha256"]
    assert rows[2]["feature_sha256"] == rows[3]["feature_sha256"]
    assert rows[0]["feature_sha256"] != rows[2]["feature_sha256"]
    assert (
        rows[0]["runtime_structure_window_sha256"]
        == rows[2]["runtime_structure_window_sha256"]
    )
    assert (
        rows[0]["runtime_structure_window_sha256"]
        != rows[1]["runtime_structure_window_sha256"]
    )
    assert all(row["training_performed"] is False for row in rows)
