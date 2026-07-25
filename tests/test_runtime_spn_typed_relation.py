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
        typed_relation_mode=mode,
        typed_relation_scale=0.1,
    )


def _structures() -> dict[str, RuntimeSpnStructure]:
    return {
        "gift64": gift64_runtime_structure(2),
        "skinny64": skinny64_runtime_structure(2),
        "rectangle80": rectangle80_runtime_structure(2),
        "uknit64": uknit64_runtime_structure(2, round_start=3),
        "dialga128": dialga128_runtime_structure(2, round_start=2),
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_typed_relation_panel_is_parameter_matched_to_true_film() -> None:
    models = {
        mode: RuntimeE4EquivariantSpnDistinguisher(_spec(mode))
        for mode in ("dense", "correct", "agnostic", "shuffled")
    }
    film = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=64,
            pair_embedding_dim=128,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
            cell_input_mode="state_triplet",
            round_window_mode="recurrent_window",
            primitive_film_mode="correct",
            primitive_film_rank=10,
            primitive_film_scale=0.1,
        )
    )

    assert {_parameter_count(model) for model in models.values()} == {446_562}
    assert _parameter_count(film) == 446_562
    reference = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(reference, strict=True)


@pytest.mark.parametrize("cipher_name", tuple(_structures()))
def test_correct_relation_channels_reconstruct_exact_gf2_edges(
    cipher_name: str,
) -> None:
    structure = _structures()[cipher_name]
    adjacency = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    indices = torch.empty((structure.cells, 4), dtype=torch.long)
    bit_indices = torch.arange(structure.block_bits)
    indices[structure.cell_membership, structure.bit_role] = bit_indices
    reconstructed = adjacency.reshape(4, 4, structure.cells, structure.cells)
    reconstructed = reconstructed.permute(2, 0, 3, 1)

    torch.testing.assert_close(
        reconstructed,
        structure.inverse_linear_matrices[0][
            indices[:, :, None, None], indices[None, None, :, :]
        ].float(),
    )


def test_relation_controls_preserve_or_remove_only_type_semantics() -> None:
    structure = uknit64_runtime_structure(2, round_start=3)
    correct = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    shuffled = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
        structure,
        round_index=0,
        mode="shuffled",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    agnostic = RuntimeE4EquivariantSpnDistinguisher.typed_relation_adjacency(
        structure,
        round_index=0,
        mode="agnostic",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(shuffled, torch.roll(correct, shifts=1, dims=0))
    torch.testing.assert_close(shuffled.sum(dim=0), correct.sum(dim=0))
    torch.testing.assert_close(agnostic.sum(dim=0), correct.sum(dim=0))
    assert not torch.equal(correct, shuffled)
    assert not torch.equal(correct, agnostic)


def test_one_typed_state_handles_all_five_runtime_structures_and_has_gradients() -> None:
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))
    state = model.state_dict()
    for structure in _structures().values():
        target = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))
        target.load_state_dict(state, strict=True)
        pairs = torch.randint(
            0,
            2,
            (2, 2, 2, structure.block_bits),
            dtype=torch.float32,
        )
        assert target(pairs, structure).shape == (2, 1)

    structure = uknit64_runtime_structure(2, round_start=3)
    pairs = torch.randint(0, 2, (4, 2, 2, 64), dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "typed_relation" in name
    ]

    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(float(parameter.grad.abs().sum()) > 0.0 for parameter in parameters)
    assert model.last_primitive_adapter_traffic["typed_relation"] > 0.0


@pytest.mark.parametrize("cipher_name", tuple(_structures()))
def test_typed_relation_logits_preserve_cell_relabeling(cipher_name: str) -> None:
    structure = _structures()[cipher_name]
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(
        0,
        2,
        (1, 2, 2, structure.block_bits),
        dtype=torch.float32,
    )
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct")).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def test_typed_relation_controls_change_logits_with_shared_weights() -> None:
    structure = uknit64_runtime_structure(2, round_start=3)
    models = {
        mode: RuntimeE4EquivariantSpnDistinguisher(_spec(mode)).eval()
        for mode in ("dense", "correct", "agnostic", "shuffled")
    }
    state = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(state, strict=True)
    pairs = torch.randint(0, 2, (3, 2, 2, 64), dtype=torch.float32)

    with torch.no_grad():
        logits = {mode: model(pairs, structure) for mode, model in models.items()}

    for control in ("dense", "agnostic", "shuffled"):
        assert not torch.equal(logits["correct"], logits[control])


def test_typed_relation_cannot_be_combined_with_old_conditioners() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RuntimeParameterizedSpnSpec(
            primitive_film_mode="correct",
            typed_relation_mode="correct",
        )
