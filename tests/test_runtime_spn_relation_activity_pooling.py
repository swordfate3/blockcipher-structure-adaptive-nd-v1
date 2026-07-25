from __future__ import annotations

import pytest
import torch

from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)


def _spec(mode: str) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        relation_activity_pooling_mode=mode,
    )


def _structures() -> dict[str, RuntimeSpnStructure]:
    return {
        "gift64": gift64_runtime_structure(2),
        "skinny64": skinny64_runtime_structure(2),
        "rectangle80": rectangle80_runtime_structure(2),
        "uknit64": uknit64_runtime_structure(2, round_start=3),
        "dialga128": dialga128_runtime_structure(2, round_start=2),
    }


def _weights(
    structure: RuntimeSpnStructure,
    *,
    mode: str,
    relation_mode: str = "true",
) -> torch.Tensor:
    return RuntimeE4EquivariantSpnDistinguisher.relation_activity_weights(
        structure,
        mode=mode,
        relation_mode=relation_mode,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_one_to_one_linear_layers_reduce_exactly_to_uniform_activity() -> None:
    for structure in (gift64_runtime_structure(2), rectangle80_runtime_structure(2)):
        torch.testing.assert_close(
            _weights(structure, mode="correct"),
            torch.ones((structure.cells, 4)),
        )


def test_general_gf2_weights_cycle_local_signature_types() -> None:
    structure = skinny64_runtime_structure(2)
    correct = _weights(structure, mode="correct")
    shuffled = _weights(structure, mode="shuffled")

    assert not torch.equal(correct, torch.ones_like(correct))
    assert not torch.equal(correct, shuffled)
    correct_signatures = {tuple(row.tolist()) for row in torch.unique(correct, dim=0)}
    assert all(tuple(row.tolist()) in correct_signatures for row in shuffled)


def test_independent_relation_mode_forces_uniform_pooling() -> None:
    structure = skinny64_runtime_structure(2)

    torch.testing.assert_close(
        _weights(structure, mode="correct", relation_mode="independent"),
        _weights(structure, mode="uniform", relation_mode="independent"),
    )


def test_pooling_roles_are_parameter_and_state_dict_matched() -> None:
    models = {
        mode: RuntimeE4EquivariantSpnDistinguisher(_spec(mode))
        for mode in ("uniform", "correct", "shuffled")
    }
    reference = models["correct"].state_dict()

    assert {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    } == {442_466}
    assert {tuple(model.state_dict()) for model in models.values()} == {
        tuple(reference)
    }
    for model in models.values():
        model.load_state_dict(reference, strict=True)


def test_correct_and_shuffled_pooling_change_general_gf2_logits() -> None:
    structure = skinny64_runtime_structure(2)
    models = {
        mode: RuntimeE4EquivariantSpnDistinguisher(_spec(mode)).eval()
        for mode in ("correct", "shuffled")
    }
    models["shuffled"].load_state_dict(models["correct"].state_dict(), strict=True)
    pairs = torch.randint(0, 2, (8, 4, 2, 64), dtype=torch.float32)

    with torch.no_grad():
        correct = models["correct"](pairs, structure)
        shuffled = models["shuffled"](pairs, structure)

    assert not torch.equal(correct, shuffled)


@pytest.mark.parametrize("cipher_name", tuple(_structures()))
@pytest.mark.parametrize("mode", ("correct", "shuffled"))
def test_relation_activity_pooling_preserves_cell_relabeling(
    cipher_name: str,
    mode: str,
) -> None:
    structure = _structures()[cipher_name]
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(
        0,
        2,
        (2, 4, 2, structure.block_bits),
        dtype=torch.float32,
    )
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    model = RuntimeE4EquivariantSpnDistinguisher(_spec(mode)).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def test_invalid_relation_activity_pooling_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="relation_activity_pooling_mode"):
        RuntimeParameterizedSpnSpec(relation_activity_pooling_mode="unsupported")
