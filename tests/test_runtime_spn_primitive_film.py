from __future__ import annotations

import pytest
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)


def _film_spec(mode: str) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        primitive_film_mode=mode,
        primitive_film_rank=10,
        primitive_film_scale=0.1,
    )


def _additive_spec() -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=64,
        pair_embedding_dim=128,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        primitive_adapter_mode="correct",
        primitive_adapter_rank=8,
        primitive_adapter_scale=0.1,
        primitive_adapter_effect="additive",
    )


def _structures() -> dict[str, RuntimeSpnStructure]:
    return {
        "gift64": gift64_runtime_structure(2),
        "skinny64": skinny64_runtime_structure(2),
        "rectangle80": rectangle80_runtime_structure(2),
        "uknit64": uknit64_runtime_structure(2, round_start=3),
        "dialga128": dialga128_runtime_structure(2, round_start=2),
    }


def _mixed_structure() -> RuntimeSpnStructure:
    linear = torch.eye(8, dtype=torch.uint8)
    linear[4:, 4:] = torch.tensor(
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
        ),
        dtype=torch.uint8,
    )
    tables = torch.tensor(PRESENT_SBOX, dtype=torch.long).repeat(2, 2, 1)
    tables[:, 1] = torch.roll(tables[:, 1], shifts=1, dims=-1)
    return runtime_spn_structure(
        cell_membership=(0, 0, 0, 0, 1, 1, 1, 1),
        bit_role=(3, 2, 1, 0, 3, 2, 1, 0),
        sbox_tables=tables,
        linear_matrices=linear.unsqueeze(0).repeat(2, 1, 1),
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_true_film_is_exactly_parameter_matched_to_additive_source() -> None:
    additive = RuntimeE4EquivariantSpnDistinguisher(_additive_spec())
    film = RuntimeE4EquivariantSpnDistinguisher(_film_spec("correct"))

    assert _parameter_count(additive) == 446_562
    assert _parameter_count(film) == 446_562


def test_true_film_roles_share_state_geometry_and_one_active_conditioner() -> None:
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_film_spec(role))
        for role in ("dense", "correct", "uniform", "shuffled")
    }
    reference = models["correct"].state_dict()

    for model in models.values():
        model.load_state_dict(reference, strict=True)
        assert model.state_dict().keys() == reference.keys()
        assert model.primitive_film_summary()["active_conditioner_evaluations"] == 1


def test_true_film_descriptor_uses_sbox_and_gf2_diffusion() -> None:
    structure = _mixed_structure()
    descriptor = RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    corrupted = structure.corrupted(seed=17)
    corrupted_descriptor = (
        RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
            corrupted,
            round_index=0,
            mode="correct",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    )

    assert descriptor.shape == (2, 128)
    assert torch.isfinite(descriptor).all()
    assert not torch.equal(descriptor[0, :64], descriptor[1, :64])
    assert not torch.equal(descriptor[:, 64:], corrupted_descriptor[:, 64:])


def test_true_film_controls_change_only_runtime_descriptor_semantics() -> None:
    structure = _mixed_structure()
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_film_spec(role)).eval()
        for role in ("dense", "correct", "uniform", "shuffled")
    }
    state = models["correct"].state_dict()
    for model in models.values():
        model.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(20260726)
    pairs = torch.randint(
        0,
        2,
        (3, 2, 2, structure.block_bits),
        generator=generator,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = {role: model(pairs, structure) for role, model in models.items()}

    for control in ("dense", "uniform", "shuffled"):
        assert not torch.equal(logits["correct"], logits[control])


def test_true_film_descriptor_and_logits_preserve_cell_relabeling() -> None:
    structure = _mixed_structure()
    permutation = tuple(reversed(range(structure.cells)))
    relabeled, bit_permutation = structure.relabel_cells(permutation)
    descriptor = RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    relabeled_descriptor = (
        RuntimeE4EquivariantSpnDistinguisher.primitive_film_descriptor(
            relabeled,
            round_index=0,
            mode="correct",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    )
    expected = torch.empty_like(descriptor)
    expected[torch.tensor(permutation)] = descriptor
    torch.testing.assert_close(relabeled_descriptor, expected)

    model = RuntimeE4EquivariantSpnDistinguisher(_film_spec("correct")).eval()
    pairs = torch.randint(0, 2, (2, 2, 2, 8), dtype=torch.float32)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    with torch.no_grad():
        original_logits = model(pairs, structure)
        relabeled_logits = model(relabeled_pairs, relabeled)
    torch.testing.assert_close(original_logits, relabeled_logits, rtol=0.0, atol=1e-6)


def test_true_film_shared_state_handles_all_five_widths_and_receives_gradients() -> (
    None
):
    model = RuntimeE4EquivariantSpnDistinguisher(_film_spec("correct"))
    state = model.state_dict()
    for structure in _structures().values():
        target = RuntimeE4EquivariantSpnDistinguisher(_film_spec("correct"))
        target.load_state_dict(state, strict=True)
        pairs = torch.randint(
            0,
            2,
            (2, 2, 2, structure.block_bits),
            dtype=torch.float32,
        )
        assert target(pairs, structure).shape == (2, 1)

    structure = _mixed_structure()
    pairs = torch.randint(0, 2, (4, 2, 2, 8), dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()
    film_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "primitive_film" in name
    ]

    assert film_parameters
    assert all(parameter.grad is not None for parameter in film_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in film_parameters)
    assert all(float(parameter.grad.abs().sum()) > 0.0 for parameter in film_parameters)
    assert model.last_primitive_adapter_traffic["film"] > 0.0


def test_true_film_and_adapter_cannot_be_enabled_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RuntimeParameterizedSpnSpec(
            primitive_adapter_mode="correct",
            primitive_film_mode="correct",
        )
