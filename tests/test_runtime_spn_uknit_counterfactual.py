from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    uknit64_runtime_structure,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    adjudicate_uknit_same_checkpoint_counterfactual,
    evaluate_same_checkpoint_pair,
)


def _row(
    seed: int,
    role: str,
    mode: str,
    auc: float,
    margin: float,
    probability_delta: float,
) -> dict[str, object]:
    token = f"{seed}{role[0]}"
    digest = (token.encode().hex() + "0" * 64)[:64]
    return {
        "seed": seed,
        "source_role": role,
        "structure_mode": mode,
        "auc": auc,
        "correct_minus_shuffled_auc": margin,
        "pair_max_abs_probability_delta": probability_delta,
        "mean_probability": 0.5,
        "probability_sha256": (digest[:-1] + ("1" if mode == "correct" else "2")),
        "checkpoint_sha256": digest,
        "feature_sha256": "a" * 64,
        "label_sha256": "b" * 64,
        "descriptor_sha256": "c" * 64,
        "checkpoint_selected": "best",
        "descriptor_round_start": 2,
        "descriptor_loaded_rounds": 2,
        "samples_total": 2048,
        "input_bits": 512,
        "parameter_count": 442466,
        "training_performed": False,
        "sbox_context_mode": "late_cell" if role == "candidate" else "late_pair",
    }


def _passing_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (0, 1):
        rows.extend(
            (
                _row(seed, "candidate", "correct", 0.54, 0.01, 0.02),
                _row(seed, "candidate", "shuffled", 0.53, 0.01, 0.02),
                _row(seed, "anchor", "correct", 0.52, 0.0, 0.0),
                _row(seed, "anchor", "shuffled", 0.52, 0.0, 0.0),
            )
        )
    return rows


def test_same_checkpoint_gate_passes_two_seed_candidate_and_anchor_panel() -> None:
    gate = adjudicate_uknit_same_checkpoint_counterfactual(
        run_id="u2a",
        rows=_passing_rows(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_same_checkpoint_gate_holds_when_candidate_does_not_use_assignment() -> None:
    rows = deepcopy(_passing_rows())
    rows[1]["auc"] = 0.539

    gate = adjudicate_uknit_same_checkpoint_counterfactual(
        run_id="u2a",
        rows=rows,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_additive_late_cell_ownership_not_used"
    assert gate["research_checks"]["seed0_candidate_margin_at_least_0p005"] is False


def test_same_checkpoint_evaluation_keeps_anchor_assignment_invariant(
    tmp_path: Path,
) -> None:
    options = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "late_pair",
    }
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    checkpoint = tmp_path / "anchor.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {"selected_checkpoint": "best"},
        },
        checkpoint,
    )
    rng = np.random.default_rng(7)
    dataset = DifferentialDataset(
        features=rng.integers(0, 2, size=(16, 512), dtype=np.uint8),
        labels=np.array([0, 1] * 8, dtype=np.uint8),
        metadata={},
    )
    correct = uknit64_runtime_structure(2, round_start=2)
    shuffled = correct.shuffled_sbox_assignments(20260724)

    rows = evaluate_same_checkpoint_pair(
        seed=0,
        source_role="anchor",
        model_options=options,
        checkpoint_path=checkpoint,
        dataset=dataset,
        correct_structure=correct,
        shuffled_structure=shuffled,
        checkpoint_sha256="a" * 64,
        feature_sha256="b" * 64,
        label_sha256="c" * 64,
        descriptor_sha256="d" * 64,
    )

    assert rows[0]["auc"] == rows[1]["auc"]
    assert rows[0]["pair_max_abs_probability_delta"] <= 1e-6
    assert (
        rows[0]["pair_max_abs_probability_delta"]
        == rows[1]["pair_max_abs_probability_delta"]
    )
