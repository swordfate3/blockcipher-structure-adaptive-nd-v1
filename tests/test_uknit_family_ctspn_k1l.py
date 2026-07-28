from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    build_k1k_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1l import (
    AUDIT_CONDITIONS,
    EXPECTED_GRADIENT_ROWS,
    EXPECTED_RESULT_ROWS,
    adjudicate_k1l,
    audit_gradient_path,
    collect_residual_path_outputs,
    label_blind_row_permutation,
    residual_metrics,
    shuffled_residual_logits,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1.csv"
)


def task(cipher: str = "uknit64", seed: int = 0) -> dict[str, object]:
    mapped = candidate_task_map(
        tasks_from_plan(
            PLAN,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=4,
            difference_profile=None,
            difference_member=0,
        )
    )
    return dict(mapped[(cipher, seed)])


def dataset(rows: int = 32) -> DifferentialDataset:
    generator = np.random.default_rng(20260728)
    return DifferentialDataset(
        features=generator.integers(0, 2, size=(rows, 512), dtype=np.uint8),
        labels=np.tile(np.array([0, 1], dtype=np.uint8), rows // 2),
        metadata={},
    )


def model() -> torch.nn.Module:
    result = build_k1k_control(
        task=task(),
        condition="exact_ordered",
        input_bits=512,
    )
    with torch.no_grad():
        result.backbone.residual_gate.fill_(0.2)
    result.eval()
    return result


def test_k1l_row_permutation_is_deterministic_bijective_and_nonidentity() -> None:
    first = label_blind_row_permutation(
        32,
        cipher="uknit64",
        seed=1,
        split="same_key_fresh",
    )
    repeated = label_blind_row_permutation(
        32,
        cipher="uknit64",
        seed=1,
        split="same_key_fresh",
    )
    other = label_blind_row_permutation(
        32,
        cipher="dialga128",
        seed=1,
        split="same_key_fresh",
    )

    assert torch.equal(first, repeated)
    assert sorted(first.tolist()) == list(range(32))
    assert not torch.equal(first, torch.arange(32))
    assert not torch.equal(first, other)


def test_k1l_exposed_residual_path_replays_forward_and_row_shuffle() -> None:
    candidate = model()
    source = dataset()
    outputs = collect_residual_path_outputs(candidate, source, batch_size=8)
    features = torch.as_tensor(source.features, dtype=torch.float32)
    with torch.inference_mode():
        direct = candidate(features).squeeze(1).numpy()

    np.testing.assert_allclose(outputs.full_logits, direct, atol=2e-7, rtol=0.0)
    assert outputs.base_embeddings.shape == (32, 384)
    assert outputs.bounded_residuals.shape == (32, 384)
    assert not np.array_equal(outputs.full_logits, outputs.zero_logits)
    metrics = residual_metrics(source.labels.astype(np.float32), outputs)
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["max_abs_residual_logit_contribution"] > 0.0

    permutation = label_blind_row_permutation(
        32,
        cipher="uknit64",
        seed=0,
        split="train_seen",
    )
    shuffled = shuffled_residual_logits(
        candidate,
        outputs,
        permutation,
        batch_size=8,
    )
    assert shuffled.shape == direct.shape
    assert not np.array_equal(shuffled, direct)


def test_k1l_gradient_proof_detects_exact_zero_starvation_and_open_gate() -> None:
    candidate = model()
    source = dataset(rows=64)
    state = {name: value.detach().clone() for name, value in candidate.state_dict().items()}

    zero = audit_gradient_path(candidate, source, effective_gate=0.0)
    opened = audit_gradient_path(candidate, source, effective_gate=0.05)

    residual_groups = (
        "cell_encoder",
        "edge_encoder",
        "cell_update",
        "residual_projection",
    )
    assert all(zero["gradient_norms"][group] == 0.0 for group in residual_groups)
    assert zero["gradient_norms"]["gate"] > 0.0
    assert any(opened["gradient_norms"][group] > 1e-8 for group in residual_groups)
    assert zero["state_restored_exact"] is True
    assert opened["state_restored_exact"] is True
    for name, value in state.items():
        torch.testing.assert_close(candidate.state_dict()[name], value)


def test_k1l_gate_classifies_closed_uknit_gradient_starvation() -> None:
    results = synthetic_results()
    gradients = synthetic_gradients()
    gate = adjudicate_k1l(
        result_rows=results,
        gradient_rows=gradients,
        source_checks={"source_binding": True},
    )

    assert len(results) == EXPECTED_RESULT_ROWS
    assert len(gradients) == EXPECTED_GRADIENT_ROWS
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("uknit_zero_gate_gradient_starvation_supported")
    assert gate["research_checks"]["uknit_both_learned_gates_effectively_closed"]
    assert gate["research_checks"][
        "exact_zero_gate_starves_residual_path_gradients"
    ]

    invalid = adjudicate_k1l(
        result_rows=results,
        gradient_rows=gradients,
        source_checks={"source_binding": False},
    )
    assert invalid["status"] == "invalid"


def synthetic_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher in ("uknit64", "dialga128"):
        gate = 0.0002 if cipher == "uknit64" else 0.02
        for seed in (0, 1):
            for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
                for condition in AUDIT_CONDITIONS:
                    contribution_auc = 0.60
                    if condition in {
                        "reversed_full",
                        "corrupted_full",
                        "no_topology_full",
                    }:
                        contribution_auc = 0.55
                    source_auc = 0.60 if condition.endswith("_full") else None
                    rows.append(
                        {
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "auc": 0.60,
                            "source_auc": source_auc,
                            "residual_contribution_auc": contribution_auc,
                            "effective_gate": gate,
                            "row_permutation_bijective": (
                                True if condition == "residual_row_shuffle" else None
                            ),
                            "row_permutation_nonidentity": (
                                True if condition == "residual_row_shuffle" else None
                            ),
                            "explained_fraction": (
                                1.0 if condition == "residual_row_shuffle" else 0.0
                            ),
                        }
                    )
    return rows


def synthetic_gradients() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher in ("uknit64", "dialga128"):
        for seed in (0, 1):
            for condition in ("exact_zero", "effective_0p05"):
                value = 0.0 if condition == "exact_zero" else 0.001
                rows.append(
                    {
                        "cipher_key": cipher,
                        "seed": seed,
                        "gate_condition": condition,
                        "gradient_norms": {
                            "gate": 0.001,
                            "cell_encoder": value,
                            "edge_encoder": value,
                            "cell_update": value,
                            "residual_projection": value,
                        },
                        "state_restored_exact": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows
