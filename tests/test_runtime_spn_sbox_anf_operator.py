from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
    inverse_sbox_anf_contributions,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    uknit64_runtime_structure,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_anf_operator import (
    OPERATOR_CONTROLS,
    SOURCE_CIPHERS,
    adjudicate_sbox_anf_operator,
    build_sbox_operator_controls,
    load_and_validate_sbox_anf_operator_config,
    run_sbox_anf_operator_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_sbox_anf_operator_s2_2048_seed0_seed1_20260726.json"
)


def _binary(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, shape, generator=generator, dtype=torch.float32)


def _candidate_spec() -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_scale=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        sbox_boolean_operator_mode="inverse_anf_contribution_gate",
        sbox_boolean_operator_scale=0.25,
    )


def test_inverse_sbox_anf_contributions_reconstruct_every_uknit_table() -> None:
    structure = uknit64_runtime_structure(round_start=3, rounds=2)
    values = torch.arange(16, dtype=torch.long)
    bits = ((values[:, None] >> torch.arange(3, -1, -1)) & 1).to(torch.float32)
    state = bits[:, None, :].repeat(1, structure.cells, 1).reshape(16, 64)

    for round_index in range(structure.rounds):
        contributions = inverse_sbox_anf_contributions(
            state,
            structure,
            round_index=round_index,
        )
        reconstructed = torch.remainder(contributions.sum(dim=-1), 2.0)
        expected_state = structure.apply_inverse_sboxes(state, round_index)
        expected = RuntimeE4EquivariantSpnDistinguisher._ordered_cell_values(
            expected_state[:, None, :],
            structure,
        )[:, 0]

        assert contributions.shape == (16, structure.cells, 4, 16)
        assert torch.equal(reconstructed, expected)


def test_sbox_anf_operator_has_fixed_geometry_and_semantic_controls() -> None:
    structure = uknit64_runtime_structure(round_start=3, rounds=2)
    controls = build_sbox_operator_controls(structure)
    pairs = _binary((3, 4, 2, 64), 26_072_611)
    torch.manual_seed(26)
    model = RuntimeE4EquivariantSpnDistinguisher(_candidate_spec()).eval()
    baseline = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=64,
            pair_embedding_dim=128,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
            cell_input_mode="state_triplet",
            round_window_mode="recurrent_window",
        )
    ).eval()

    with torch.no_grad():
        logits = {
            mode: model(
                pairs,
                structure,
                operator_structure=controls[mode],
            )
            for mode in OPERATOR_CONTROLS
        }
        pair_swapped = model(
            pairs.flip(2),
            structure,
            operator_structure=controls["exact"],
        )

    assert sum(parameter.numel() for parameter in model.parameters()) == 459234
    assert sum(parameter.numel() for parameter in baseline.parameters()) == 442466
    assert set(controls) == set(OPERATOR_CONTROLS)
    assert torch.equal(controls["exact"].linear_matrices, structure.linear_matrices)
    assert torch.equal(
        controls["input_permuted"].linear_matrices,
        structure.linear_matrices,
    )
    assert not torch.equal(
        controls["input_permuted"].sbox_truth_bits,
        structure.sbox_truth_bits,
    )
    assert not torch.equal(
        controls["identity"].sbox_truth_bits,
        structure.sbox_truth_bits,
    )
    assert all(
        float(torch.max(torch.abs(logits["exact"] - logits[mode]))) > 1e-6
        for mode in ("input_permuted", "identity")
    )
    assert torch.allclose(logits["exact"], pair_swapped, atol=1e-6, rtol=0.0)


def test_sbox_anf_operator_is_jointly_cell_relabel_invariant() -> None:
    structure = uknit64_runtime_structure(round_start=3, rounds=2)
    operator = build_sbox_operator_controls(structure)["exact"]
    permutation = tuple(reversed(range(structure.cells)))
    relabeled_structure, bit_permutation = structure.relabel_cells(permutation)
    relabeled_operator, operator_bit_permutation = operator.relabel_cells(permutation)
    pairs = _binary((2, 4, 2, 64), 26_072_612)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(27)
    model = RuntimeE4EquivariantSpnDistinguisher(_candidate_spec()).eval()

    with torch.no_grad():
        original = model(pairs, structure, operator_structure=operator)
        relabeled = model(
            relabeled_pairs,
            relabeled_structure,
            operator_structure=relabeled_operator,
        )

    assert torch.equal(bit_permutation, operator_bit_permutation)
    assert torch.allclose(original, relabeled, atol=1e-6, rtol=0.0)


def test_real_s2_config_and_readiness_pass() -> None:
    config = load_and_validate_sbox_anf_operator_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    readiness = run_sbox_anf_operator_readiness(
        config=config,
        project_root=ROOT,
    )

    assert tuple(config["source_ciphers"]) == SOURCE_CIPHERS
    assert tuple(config["operator_controls"]) == OPERATOR_CONTROLS
    assert readiness["status"] == "pass"
    assert all(readiness["checks"].values())
    assert readiness["parameter_count"] == 459234
    assert readiness["target_training_rows"] == 0
    assert readiness["target_optimizer_steps"] == 0


def test_s2_gate_distinguishes_supported_hold_and_invalid() -> None:
    payload = _gate_payload()
    passed = adjudicate_sbox_anf_operator(payload)
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("sbox_anf_operator_supported")

    held_payload = _gate_payload()
    held_payload["target_auc"]["1"]["input_permuted"] = 0.70
    held = adjudicate_sbox_anf_operator(held_payload)
    assert held["status"] == "hold"
    assert held["decision"].endswith("sbox_anf_operator_not_supported")

    invalid_payload = _gate_payload()
    invalid_payload["validation"] = {"status": "fail", "checks": {"ok": False}}
    invalid = adjudicate_sbox_anf_operator(invalid_payload)
    assert invalid["status"] == "invalid"
    assert invalid["decision"].endswith("sbox_anf_operator_protocol_invalid")


def _gate_payload() -> dict[str, Any]:
    config = load_and_validate_sbox_anf_operator_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    source = {
        "exact": 0.70,
        "input_permuted": 0.68,
        "identity": 0.67,
        "a8_anchor": 0.695,
    }
    target = {
        "exact": 0.70,
        "input_permuted": 0.68,
        "identity": 0.67,
        "a8_anchor": 0.695,
    }
    return {
        "config": config,
        "validation": {"status": "pass", "checks": {"ok": True}},
        "source_macro_auc": {"0": dict(source), "1": dict(source)},
        "target_auc": {"0": dict(target), "1": dict(target)},
        "probability_deltas": {
            "0": {"input_permuted": 0.1, "identity": 0.1},
            "1": {"input_permuted": 0.1, "identity": 0.1},
        },
    }
