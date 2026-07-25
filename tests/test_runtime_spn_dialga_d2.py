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
    CORRUPTION_SEED,
    FROZEN_MODEL_OPTIONS,
    adjudicate_same_checkpoint_dialga,
    evaluate_same_checkpoint_dialga,
)


def _row(
    seed: int,
    condition: str,
    auc: float,
    probability_delta: float,
) -> dict[str, object]:
    checkpoint_digest = (f"{seed + 1:x}" * 64)[:64]
    correct_window = "a" * 64
    corrupted_window = "b" * 64
    relation_mode = "independent" if condition == "no_topology" else "true"
    return {
        "seed": seed,
        "condition": condition,
        "cipher": "Dialga-128",
        "rounds": 4,
        "auc": auc,
        "source_auc": 0.60,
        "max_abs_probability_delta_from_correct": probability_delta,
        "mean_abs_probability_delta_from_correct": probability_delta / 2,
        "mean_probability": 0.5,
        "probability_sha256": {"correct": "1", "corrupted": "2", "no_topology": "3"}[condition] * 64,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_selected": "best",
        "checkpoint_reported_seed": seed,
        "checkpoint_best_metric": 0.60,
        "strict_state_dict_load": True,
        "feature_sha256": f"{seed + 3:x}" * 64,
        "label_sha256": f"{seed + 5:x}" * 64,
        "metadata_sha256": f"{seed + 7:x}" * 64,
        "source_results_sha256": "c" * 64,
        "source_gate_sha256": "d" * 64,
        "source_d1_verified": True,
        "source_d1_decision": "innovation1_dialga_runtime_e4_d1_two_seed_supported",
        "descriptor_name": "Dialga-128 20-round heterogeneous runtime SPN structure",
        "descriptor_path": "configs/runtime/spn/dialga128.json",
        "descriptor_sha256": "e" * 64,
        "source_descriptor_sha256": "e" * 64,
        "descriptor_round_start": 2,
        "descriptor_loaded_rounds": 2,
        "runtime_structure_mode": condition,
        "relation_mode": relation_mode,
        "runtime_structure_window_sha256": corrupted_window if condition == "corrupted" else correct_window,
        "runtime_structure_transition_sha256s": ["f" * 64, "9" * 64],
        "runtime_structure_unique_transition_count": 2,
        "runtime_intervention_sha256": {"correct": "6", "corrupted": "7", "no_topology": "8"}[condition] * 64,
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
    }


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        rows.extend(
            (
                _row(seed, "correct", 0.60, 0.0),
                _row(seed, "corrupted", 0.57, 0.08),
                _row(seed, "no_topology", 0.51, 0.12),
            )
        )
    return rows


def test_dialga_d2_gate_passes_two_seed_same_checkpoint_panel() -> None:
    gate = adjudicate_same_checkpoint_dialga(run_id="d2-pass", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_dialga_runtime_e4_d2_functional_topology_use_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_dialga_d2_gate_holds_when_one_seed_loses_margin() -> None:
    rows = deepcopy(_passing_rows())
    rows[4]["auc"] = 0.599

    gate = adjudicate_same_checkpoint_dialga(run_id="d2-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["research_checks"]["seed1_beats_corrupted_by_0p005"] is False


def test_dialga_d2_gate_fails_closed_on_checkpoint_drift() -> None:
    rows = deepcopy(_passing_rows())
    rows[1]["checkpoint_sha256"] = "0" * 64

    gate = adjudicate_same_checkpoint_dialga(run_id="d2-invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_checkpoint_within_seed"] is False


def test_dialga_d2_evaluator_swaps_only_runtime_intervention(tmp_path: Path) -> None:
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
                "best_checkpoint_metric": 0.5,
            },
        },
        checkpoint,
    )
    rng = np.random.default_rng(25)
    dataset = DifferentialDataset(
        features=rng.integers(0, 2, size=(16, 1024), dtype=np.uint8),
        labels=np.array([0, 1] * 8, dtype=np.uint8),
        metadata={
            "cipher": "Dialga-128",
            "rounds": 4,
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
    correct = dialga128_runtime_structure(2, round_start=2)
    corrupted = correct.corrupted(CORRUPTION_SEED)

    rows = evaluate_same_checkpoint_dialga(
        seed=0,
        model_options=FROZEN_MODEL_OPTIONS,
        checkpoint_path=checkpoint,
        dataset=dataset,
        correct_structure=correct,
        corrupted_structure=corrupted,
        source_auc=0.5,
        checkpoint_sha256="a" * 64,
        feature_sha256="b" * 64,
        label_sha256="c" * 64,
        metadata_sha256="d" * 64,
        source_results_sha256="e" * 64,
        source_gate_sha256="f" * 64,
        descriptor_name="Dialga-128 20-round heterogeneous runtime SPN structure",
        descriptor_path="configs/runtime/spn/dialga128.json",
        descriptor_sha256="1" * 64,
        source_descriptor_sha256="1" * 64,
        batch_size=8,
    )

    assert [row["condition"] for row in rows] == [
        "correct",
        "corrupted",
        "no_topology",
    ]
    assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert len({row["feature_sha256"] for row in rows}) == 1
    assert len({row["runtime_intervention_sha256"] for row in rows}) == 3
    assert rows[0]["runtime_structure_window_sha256"] == rows[2]["runtime_structure_window_sha256"]
    assert rows[0]["runtime_structure_window_sha256"] != rows[1]["runtime_structure_window_sha256"]
    assert rows[1]["max_abs_probability_delta_from_correct"] > 0.0
    assert rows[2]["max_abs_probability_delta_from_correct"] > 0.0
