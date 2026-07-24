from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.audit_runtime_spn_delta_query_counterfactual import (
    _load_validation_dataset,
)
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    uknit64_runtime_structure,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_delta_query_counterfactual import (
    adjudicate_same_checkpoint_delta_query,
    evaluate_same_checkpoint_delta_query,
)


def _row(seed: int, condition: str, auc: float) -> dict[str, object]:
    digest = f"{seed}{condition}".encode().hex().ljust(64, "0")[:64]
    probability_delta = 0.0 if condition == "correct_delta_u" else 0.02
    query_hash = "not_used" if condition == "delta_v_identity" else digest
    return {
        "seed": seed,
        "condition": condition,
        "auc": auc,
        "reference_minus_condition_auc": 0.0,
        "max_abs_probability_delta_from_reference": probability_delta,
        "mean_probability": 0.5,
        "probability_sha256": digest,
        "checkpoint_sha256": ("a" if seed == 0 else "b") * 64,
        "checkpoint_selected": "best",
        "feature_sha256": ("c" if seed == 0 else "d") * 64,
        "label_sha256": ("e" if seed == 0 else "f") * 64,
        "descriptor_sha256": "1" * 64,
        "query_sbox_truth_sha256": query_hash,
        "descriptor_round_start": 2,
        "descriptor_loaded_rounds": 2,
        "samples_total": 2048,
        "input_bits": 512,
        "parameter_count": 458850,
        "training_performed": False,
        "main_structure_mode": "correct",
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet_delta_u_query",
    }


def _passing_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (0, 1):
        rows.extend(
            (
                _row(seed, "correct_delta_u", 0.55),
                _row(seed, "shuffled_delta_u", 0.54),
                _row(seed, "delta_v_identity", 0.53),
            )
        )
    return rows


def test_delta_query_counterfactual_gate_passes_two_seed_panel() -> None:
    gate = adjudicate_same_checkpoint_delta_query(
        run_id="u2g",
        rows=_passing_rows(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_uknit_delta_u_same_checkpoint_use_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_delta_query_counterfactual_gate_holds_failed_margin() -> None:
    rows = _passing_rows()
    rows[1]["auc"] = 0.548

    gate = adjudicate_same_checkpoint_delta_query(run_id="u2g", rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_delta_u_training_distribution_only"
    assert gate["research_checks"]["seed0_beats_shuffled_by_0p005"] is False


def test_delta_query_counterfactual_gate_rejects_checkpoint_drift() -> None:
    rows = deepcopy(_passing_rows())
    rows[1]["checkpoint_sha256"] = "9" * 64

    gate = adjudicate_same_checkpoint_delta_query(run_id="u2g", rows=rows)

    assert gate["status"] == "fail"
    assert (
        gate["decision"]
        == "innovation1_uknit_delta_query_counterfactual_protocol_invalid"
    )
    assert gate["protocol_checks"]["same_checkpoint_within_seed"] is False


def test_delta_query_override_keeps_normal_forward_and_main_structure_fixed() -> None:
    options = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet_delta_u_query",
    }
    torch.manual_seed(20260730)
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    ).eval()
    correct = uknit64_runtime_structure(2, round_start=2)
    shuffled = correct.shuffled_sbox_assignments(20260724)
    model.runtime_structure = correct
    features = torch.randint(0, 2, (4, 512), dtype=torch.float32)

    with torch.no_grad():
        normal = model(features)
        explicit = model(features, query_input_mode="delta_u")
        shuffled_query = model(
            features,
            query_input_mode="delta_u",
            query_structure=shuffled,
        )
        identity_query = model(features, query_input_mode="delta_v")

    torch.testing.assert_close(normal, explicit, rtol=0.0, atol=0.0)
    assert float(torch.max(torch.abs(normal - shuffled_query))) > 1e-6
    assert float(torch.max(torch.abs(normal - identity_query))) > 1e-6
    assert sum(parameter.numel() for parameter in model.parameters()) == 458850

    relabeled, _ = correct.relabel_cells(tuple(reversed(range(correct.cells))))
    with pytest.raises(ValueError, match="only per-cell S-box ownership"):
        model(
            features,
            query_input_mode="delta_u",
            query_structure=relabeled,
        )


def test_same_checkpoint_delta_query_evaluator_uses_one_checkpoint(
    tmp_path: Path,
) -> None:
    options = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet_delta_u_query",
    }
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {"selected_checkpoint": "best"},
        },
        checkpoint,
    )
    rng = np.random.default_rng(11)
    dataset = DifferentialDataset(
        features=rng.integers(0, 2, size=(32, 512), dtype=np.uint8),
        labels=np.array([0, 1] * 16, dtype=np.uint8),
        metadata={},
    )
    correct = uknit64_runtime_structure(2, round_start=2)
    shuffled = correct.shuffled_sbox_assignments(20260724)

    rows = evaluate_same_checkpoint_delta_query(
        seed=0,
        model_options=options,
        checkpoint_path=checkpoint,
        dataset=dataset,
        correct_structure=correct,
        shuffled_structure=shuffled,
        checkpoint_sha256="a" * 64,
        feature_sha256="b" * 64,
        label_sha256="c" * 64,
        descriptor_sha256="d" * 64,
        batch_size=16,
    )

    assert [row["condition"] for row in rows] == [
        "correct_delta_u",
        "shuffled_delta_u",
        "delta_v_identity",
    ]
    assert {row["checkpoint_sha256"] for row in rows} == {"a" * 64}
    assert {row["training_performed"] for row in rows} == {False}
    assert rows[0]["query_sbox_truth_sha256"] != rows[1][
        "query_sbox_truth_sha256"
    ]
    assert rows[2]["query_sbox_truth_sha256"] == "not_used"


def test_delta_query_audit_loads_the_exact_disk_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "uknit64" / "r4" / "validation" / "seed-10000_test"
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "features.npy", np.zeros((8, 512), dtype=np.uint8))
    np.save(cache_dir / "labels.npy", np.array([0, 1] * 4, dtype=np.uint8))
    (cache_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    source = {
        "seed": 0,
        "training": {"dataset_cache_root": str(tmp_path)},
    }

    dataset, feature_path, label_path = _load_validation_dataset(source)

    assert dataset.cache_dir == cache_dir
    assert feature_path == cache_dir / "features.npy"
    assert label_path == cache_dir / "labels.npy"
    assert len(dataset.labels) == 8
