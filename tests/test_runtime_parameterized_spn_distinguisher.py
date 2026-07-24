from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers.spn.gift import GIFT64_SBOX
from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.cross_spn_typed_cell import (
    cipher_inverse_permutation_indices,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    FixedRuntimeSpnProtocolAdapter,
    RuntimeCellTokenSpnDistinguisher,
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    apply_gf2,
    load_runtime_spn_descriptor,
    permutation_matrix,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    gift64_runtime_structure,
    present_runtime_structure,
    skinny64_runtime_structure,
    standard_four_bit_cells,
)
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def _model() -> RuntimeParameterizedSpnDistinguisher:
    torch.manual_seed(23)
    return RuntimeParameterizedSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=24,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
        )
    )


def _binary(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, shape, generator=generator, dtype=torch.float32)


def _synthetic_128_structure() -> object:
    membership, roles = standard_four_bit_cells(128)
    permutation = tuple((index + 4) % 128 for index in range(128))
    return runtime_spn_structure(
        cell_membership=membership,
        bit_role=roles,
        sbox_tables=PRESENT_SBOX,
        linear_matrices=permutation_matrix(permutation),
    )


def _heterogeneous_sbox_structures() -> tuple[object, object]:
    base = present_runtime_structure()
    tables = torch.empty((1, base.cells, 16), dtype=torch.long)
    tables[:, 0::2] = torch.tensor(PRESENT_SBOX)
    tables[:, 1::2] = torch.tensor(GIFT64_SBOX)
    swapped = tables.roll(1, dims=1)
    common = {
        "cell_membership": base.cell_membership,
        "bit_role": base.bit_role,
        "linear_matrices": base.linear_matrices,
    }
    return (
        runtime_spn_structure(sbox_tables=tables, **common),
        runtime_spn_structure(sbox_tables=swapped, **common),
    )


def test_standard_runtime_cells_preserve_project_msb_role_order() -> None:
    membership, roles = standard_four_bit_cells(8)

    assert membership == (0, 0, 0, 0, 1, 1, 1, 1)
    assert roles == (3, 2, 1, 0, 3, 2, 1, 0)


def test_runtime_structure_supports_permutations_and_general_gf2() -> None:
    present = present_runtime_structure(rounds=2)
    gift = gift64_runtime_structure(rounds=2)
    skinny = skinny64_runtime_structure(rounds=2)

    assert present.linear_matrices.shape == (2, 64, 64)
    assert gift.linear_matrices.shape == (2, 64, 64)
    assert skinny.linear_matrices.shape == (2, 64, 64)
    assert torch.all(present.linear_matrices.sum(dim=2) == 1)
    assert torch.all(gift.linear_matrices.sum(dim=2) == 1)
    assert int(skinny.linear_matrices.sum(dim=2).max()) > 1
    assert not torch.equal(present.linear_matrices, gift.linear_matrices)

    identity = torch.eye(64, dtype=torch.int64)
    for structure in (present, gift, skinny):
        for linear, inverse in zip(
            structure.linear_matrices,
            structure.inverse_linear_matrices,
            strict=True,
        ):
            assert torch.equal(
                torch.remainder(linear.to(torch.int64) @ inverse.to(torch.int64), 2),
                identity,
            )


def test_runtime_model_uses_one_parameter_geometry_for_variable_structures() -> None:
    model = _model().eval()
    state_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }

    outputs = (
        model(_binary((3, 2, 2, 64), 1), present_runtime_structure(2)),
        model(_binary((3, 5, 2, 64), 2), gift64_runtime_structure(2)),
        model(_binary((3, 3, 2, 64), 3), skinny64_runtime_structure(2)),
        model(_binary((3, 4, 2, 128), 4), _synthetic_128_structure()),
    )

    assert all(output.shape == (3, 1) for output in outputs)
    assert state_shapes == {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }
    assert not any(
        token in name
        for name in state_shapes
        for token in ("cipher", "topology", "linear_matrix", "sbox_truth")
    )


def test_runtime_model_forward_backward_is_finite_for_sparse_gf2() -> None:
    model = _model().train()
    output = model(_binary((4, 3, 2, 64), 5), skinny64_runtime_structure(2))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.squeeze(1), torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_corrupted_topology_preserves_degrees_but_changes_edges_and_logits() -> None:
    structure = skinny64_runtime_structure(2)
    corrupted = structure.corrupted()
    repeated = structure.corrupted()
    true_rows = structure.linear_matrices.sum(dim=2)
    corrupted_rows = corrupted.linear_matrices.sum(dim=2)
    true_columns = torch.sort(structure.linear_matrices.sum(dim=1), dim=1).values
    corrupted_columns = torch.sort(corrupted.linear_matrices.sum(dim=1), dim=1).values

    assert torch.equal(true_rows, corrupted_rows)
    assert torch.equal(true_columns, corrupted_columns)
    assert not torch.equal(structure.linear_matrices, corrupted.linear_matrices)
    assert torch.equal(corrupted.linear_matrices, repeated.linear_matrices)

    model = _model().eval()
    pairs = _binary((2, 3, 2, 64), 6)
    with torch.no_grad():
        true_logits = model(pairs, structure)
        corrupted_logits = model(pairs, corrupted)
        independent_logits = model(pairs, structure, relation_mode="independent")
    assert float(torch.max(torch.abs(true_logits - corrupted_logits))) > 1e-6
    assert float(torch.max(torch.abs(true_logits - independent_logits))) > 1e-6


def test_corrupted_permutation_breaks_cell_and_bit_role_alignment() -> None:
    structure = gift64_runtime_structure()
    corrupted = structure.corrupted()
    true_sources = structure.inverse_linear_matrices[0].argmax(dim=1)
    corrupted_sources = corrupted.inverse_linear_matrices[0].argmax(dim=1)

    assert int((true_sources == corrupted_sources).sum()) < 4
    assert int((true_sources % 4 == corrupted_sources % 4).sum()) < 32
    assert int((true_sources // 4 == corrupted_sources // 4).sum()) < 16


def test_runtime_sbox_descriptor_changes_logits_without_changing_graph() -> None:
    present = present_runtime_structure(2)
    gift_sbox = runtime_spn_structure(
        cell_membership=present.cell_membership,
        bit_role=present.bit_role,
        sbox_tables=GIFT64_SBOX,
        linear_matrices=present.linear_matrices,
    )
    model = _model().eval()
    pairs = _binary((2, 3, 2, 64), 7)

    with torch.no_grad():
        present_logits = model(pairs, present)
        gift_sbox_logits = model(pairs, gift_sbox)
    assert float(torch.max(torch.abs(present_logits - gift_sbox_logits))) > 1e-6


def test_permutation_gather_matches_exact_gf2_inverse() -> None:
    structure = present_runtime_structure()
    inverse = structure.inverse_linear_matrices[0]
    indices = torch.argmax(inverse, dim=1)
    values = _binary((2, 4, 64), 8)

    assert torch.equal(apply_gf2(inverse, values), values.index_select(2, indices))


def test_runtime_model_is_equivariant_to_cell_relabeling() -> None:
    structure = skinny64_runtime_structure(2)
    cell_permutation = tuple(reversed(range(structure.cells)))
    relabeled_structure, bit_permutation = structure.relabel_cells(cell_permutation)
    pairs = _binary((2, 3, 2, 64), 9)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    model = _model().eval()

    with torch.no_grad():
        original = model(pairs, structure)
        relabeled = model(relabeled_pairs, relabeled_structure)
    assert torch.allclose(original, relabeled, atol=1e-6, rtol=0.0)


def test_runtime_structure_rejects_invalid_cells_sboxes_and_linear_maps() -> None:
    membership, roles = standard_four_bit_cells(64)
    identity = torch.eye(64, dtype=torch.uint8)
    bad_membership = list(membership)
    bad_membership[4] = 0

    with pytest.raises(ValueError, match="bit roles"):
        runtime_spn_structure(
            cell_membership=bad_membership,
            bit_role=roles,
            sbox_tables=PRESENT_SBOX,
            linear_matrices=identity,
        )
    with pytest.raises(ValueError, match="S-box"):
        runtime_spn_structure(
            cell_membership=membership,
            bit_role=roles,
            sbox_tables=tuple(range(15)) + (14,),
            linear_matrices=identity,
        )
    with pytest.raises(ValueError, match="invertible"):
        runtime_spn_structure(
            cell_membership=membership,
            bit_role=roles,
            sbox_tables=PRESENT_SBOX,
            linear_matrices=torch.zeros(64, 64, dtype=torch.uint8),
        )


def test_runtime_model_rejects_shape_and_nonbinary_input() -> None:
    model = _model()
    structure = present_runtime_structure()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 2, 32), structure)
    with pytest.raises(ValueError, match="binary"):
        model(torch.full((2, 3, 2, 64), 0.5), structure)


def test_legacy_protocol_adapters_share_state_geometry_and_external_structure() -> None:
    options = {"processor_steps": 2, "pair_embedding_dim": 32, "dropout": 0.0}
    names = (
        "present_runtime_spn_true",
        "present_runtime_spn_corrupted",
        "present_runtime_spn_independent",
        "gift64_runtime_spn_true",
        "gift64_runtime_spn_corrupted",
        "gift64_runtime_spn_independent",
    )
    models = [
        build_model(
            name,
            input_bits=512,
            hidden_bits=24,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for name in names
    ]
    assert all(isinstance(model, FixedRuntimeSpnProtocolAdapter) for model in models)
    assert all(
        model.input_bit_order == "project_msb_to_runtime_lsb" for model in models
    )
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert not any(
        token in name
        for name in geometries[0]
        for token in ("runtime_structure", "linear_matrix", "sbox_truth")
    )
    assert not torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[1].runtime_structure.linear_matrices,
    )
    output = models[-1](_binary((2, 512), 13))
    assert output.shape == (2, 1)

    recorded_r1_model = build_model(
        "gift64_runtime_spn_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options={
            "processor_steps": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
        },
    )
    assert sum(parameter.numel() for parameter in recorded_r1_model.parameters()) == 163971


def test_gift_adapter_converts_project_msb_features_to_runtime_lsb_coordinates() -> None:
    structure = gift64_runtime_structure()
    adapter = build_model(
        "gift64_runtime_e4_equivariant_true",
        input_bits=128,
        hidden_bits=24,
        pair_bits=128,
        structure="SPN",
        model_options={"processor_steps": 1, "pair_embedding_dim": 32},
    )
    msb_difference = torch.eye(64, dtype=torch.float32)
    runtime_difference = adapter._to_runtime_coordinates(
        torch.cat((msb_difference, torch.zeros_like(msb_difference)), dim=1)
    )[:, 0, 0]
    recovered_msb = structure.exact_inverse(runtime_difference).flip(-1)
    expected = msb_difference.index_select(
        1,
        cipher_inverse_permutation_indices("gift64", "true"),
    )

    assert torch.equal(recovered_msb, expected)


def test_runtime_cell_token_model_preserves_cells_across_pairs_and_widths() -> None:
    torch.manual_seed(29)
    model = RuntimeCellTokenSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=24,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        gift_output = model(
            _binary((2, 4, 2, 64), 14),
            gift64_runtime_structure(2),
        )
        wide_output = model(
            _binary((2, 3, 2, 128), 15),
            _synthetic_128_structure(),
        )

    assert gift_output.shape == (2, 1)
    assert wide_output.shape == (2, 1)
    assert model.last_pair_within_cell_attention is not None
    assert model.last_pair_within_cell_attention.shape == (2, 32, 3)
    assert model.last_cell_attention is not None
    assert model.last_cell_attention.shape == (2, 32)


def test_runtime_cell_token_model_is_cell_relabel_equivariant() -> None:
    structure = skinny64_runtime_structure(2)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = _binary((2, 3, 2, 64), 16)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(31)
    model = RuntimeCellTokenSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=24,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    assert torch.allclose(original, permuted, atol=1e-6, rtol=0.0)


def test_runtime_cell_token_controls_have_identical_parameter_geometry() -> None:
    options = {"processor_steps": 2, "pair_embedding_dim": 32, "dropout": 0.0}
    true_model = build_model(
        "gift64_runtime_cell_token_true",
        input_bits=512,
        hidden_bits=24,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    corrupted_model = build_model(
        "gift64_runtime_cell_token_corrupted",
        input_bits=512,
        hidden_bits=24,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )

    assert {
        name: tuple(value.shape) for name, value in true_model.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in corrupted_model.state_dict().items()
    }
    assert true_model.aggregation_mode == "cell_pair"
    assert corrupted_model.aggregation_mode == "cell_pair"
    assert not torch.equal(
        true_model.runtime_structure.linear_matrices,
        corrupted_model.runtime_structure.linear_matrices,
    )


def test_runtime_e4_equivariant_backbone_supports_widths_and_general_gf2() -> None:
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=32,
            pair_embedding_dim=64,
            processor_steps=2,
            dropout=0.0,
        )
    ).eval()
    state_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        gift = model(
            _binary((2, 4, 2, 64), 17), gift64_runtime_structure(2)
        )
        skinny = model(
            _binary((2, 3, 2, 64), 18), skinny64_runtime_structure(2)
        )
        wide = model(
            _binary((2, 5, 2, 128), 19), _synthetic_128_structure()
        )

    assert gift.shape == skinny.shape == wide.shape == (2, 1)
    assert state_shapes == {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }


def test_runtime_e4_equivariant_backbone_is_cell_relabel_invariant() -> None:
    structure = skinny64_runtime_structure(2)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = _binary((2, 3, 2, 64), 20)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(43)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=32,
            pair_embedding_dim=64,
            processor_steps=2,
            dropout=0.0,
        )
    ).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def test_runtime_e4_controls_keep_cell_and_sbox_metadata_but_change_linear_view() -> None:
    options = {
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_scale": 0.1,
        "sbox_context_mode": "late_pair",
    }
    names = (
        "gift64_runtime_e4_equivariant_true",
        "gift64_runtime_e4_equivariant_corrupted",
        "gift64_runtime_e4_equivariant_independent",
    )
    models = [
        build_model(
            name,
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for name in names
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(model.aggregation_mode == "e4_equivariant" for model in models)
    assert all(model.backbone.spec.sbox_context_scale == 0.1 for model in models)
    assert all(
        model.backbone.spec.sbox_context_mode == "late_pair" for model in models
    )
    assert torch.equal(
        models[0].runtime_structure.sbox_truth_bits,
        models[2].runtime_structure.sbox_truth_bits,
    )
    assert not torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[1].runtime_structure.linear_matrices,
    )


def test_present_runtime_e4_controls_share_geometry_and_external_metadata() -> None:
    options = {
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "late_pair",
    }
    models = [
        build_model(
            name,
            input_bits=2048,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for name in (
            "present_runtime_e4_equivariant_true",
            "present_runtime_e4_equivariant_corrupted",
            "present_runtime_e4_equivariant_independent",
        )
    ]

    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(model.aggregation_mode == "e4_equivariant" for model in models)
    assert all(model.input_bit_order == "project_msb_to_runtime_lsb" for model in models)
    assert torch.equal(
        models[0].runtime_structure.sbox_truth_bits,
        models[2].runtime_structure.sbox_truth_bits,
    )
    assert not torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[1].runtime_structure.linear_matrices,
    )


def test_runtime_e4_backbone_uses_external_sbox_descriptor() -> None:
    present = present_runtime_structure(2)
    gift_sbox = runtime_spn_structure(
        cell_membership=present.cell_membership,
        bit_role=present.bit_role,
        sbox_tables=GIFT64_SBOX,
        linear_matrices=present.linear_matrices,
    )
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=32,
            pair_embedding_dim=64,
            processor_steps=2,
            dropout=0.0,
        )
    ).eval()
    pairs = _binary((2, 3, 2, 64), 21)

    with torch.no_grad():
        present_logits = model(pairs, present)
        gift_logits = model(pairs, gift_sbox)

    assert float(torch.max(torch.abs(present_logits - gift_logits))) > 1e-6


def test_runtime_e4_nonzero_sbox_scale_preserves_descriptor_sensitivity() -> None:
    present = present_runtime_structure(2)
    gift_sbox = runtime_spn_structure(
        cell_membership=present.cell_membership,
        bit_role=present.bit_role,
        sbox_tables=GIFT64_SBOX,
        linear_matrices=present.linear_matrices,
    )
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            sbox_context_scale=0.1,
        )
    )
    model.eval()
    pairs = torch.randint(0, 2, (3, 2, 2, 64), dtype=torch.float32)

    with torch.no_grad():
        present_logits = model(pairs, present)
        gift_logits = model(pairs, gift_sbox)

    assert float(torch.max(torch.abs(present_logits - gift_logits))) > 1e-6


def test_runtime_e4_sbox_scale_does_not_change_parameter_geometry() -> None:
    common = {
        "hidden_dim": 16,
        "pair_embedding_dim": 32,
        "processor_steps": 2,
        "dropout": 0.0,
    }
    full = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_scale=1.0)
    )
    reduced = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_scale=0.1)
    )

    assert {
        name: tuple(parameter.shape) for name, parameter in full.named_parameters()
    } == {
        name: tuple(parameter.shape) for name, parameter in reduced.named_parameters()
    }


def test_runtime_spn_spec_rejects_negative_sbox_context_scale() -> None:
    with pytest.raises(ValueError, match="sbox_context_scale"):
        RuntimeParameterizedSpnSpec(sbox_context_scale=-0.1)


def test_runtime_spn_spec_rejects_unknown_sbox_context_mode() -> None:
    with pytest.raises(ValueError, match="sbox_context_mode"):
        RuntimeParameterizedSpnSpec(sbox_context_mode="unknown")  # type: ignore[arg-type]


def test_runtime_e4_late_sbox_conditioning_preserves_topology_extractor() -> None:
    present = present_runtime_structure(2)
    gift_sbox = runtime_spn_structure(
        cell_membership=present.cell_membership,
        bit_role=present.bit_role,
        sbox_tables=GIFT64_SBOX,
        linear_matrices=present.linear_matrices,
    )
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="late_pair",
        )
    ).eval()
    pairs = _binary((2, 3, 2, 64), 45)
    mixer_inputs: list[torch.Tensor] = []

    def capture_input(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        mixer_inputs.append(args[0].detach().clone())

    hook = model.mixer_blocks[0].register_forward_pre_hook(capture_input)
    with torch.no_grad():
        present_logits = model(pairs, present)
        gift_logits = model(pairs, gift_sbox)
    hook.remove()

    assert len(mixer_inputs) == 2
    torch.testing.assert_close(mixer_inputs[0], mixer_inputs[1], rtol=0.0, atol=0.0)
    assert float(torch.max(torch.abs(present_logits - gift_logits))) > 1e-6


def test_runtime_e4_sbox_location_keeps_parameter_geometry() -> None:
    common = {
        "hidden_dim": 16,
        "pair_embedding_dim": 24,
        "processor_steps": 2,
        "dropout": 0.0,
    }
    early = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="early_add")
    )
    late = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_pair")
    )
    late_cell = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_cell")
    )

    geometry = {
        name: tuple(parameter.shape) for name, parameter in early.named_parameters()
    }
    assert geometry == {
        name: tuple(parameter.shape) for name, parameter in late.named_parameters()
    }
    assert geometry == {
        name: tuple(parameter.shape) for name, parameter in late_cell.named_parameters()
    }


def test_runtime_e4_late_cell_preserves_cell_specific_sbox_assignment() -> None:
    first, swapped = _heterogeneous_sbox_structures()
    common = {
        "hidden_dim": 16,
        "pair_embedding_dim": 32,
        "processor_steps": 2,
        "dropout": 0.0,
        "sbox_context_scale": 1.0,
    }
    torch.manual_seed(53)
    late_pair = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_pair")
    ).eval()
    late_cell = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_cell")
    ).eval()
    late_cell.load_state_dict(late_pair.state_dict())
    pairs = _binary((3, 4, 2, 64), 54)

    with torch.no_grad():
        pair_first = late_pair(pairs, first)
        pair_swapped = late_pair(pairs, swapped)
        cell_first = late_cell(pairs, first)
        cell_swapped = late_cell(pairs, swapped)

    pair_delta = float(torch.max(torch.abs(pair_first - pair_swapped)))
    cell_delta = float(torch.max(torch.abs(cell_first - cell_swapped)))
    assert pair_delta <= 1e-6
    assert cell_delta > max(1e-6, pair_delta * 100.0)


def test_runtime_e4_late_cell_is_cell_relabel_invariant() -> None:
    structure, _ = _heterogeneous_sbox_structures()
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = _binary((2, 3, 2, 64), 55)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(56)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="late_cell",
        )
    ).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def _minimal_runtime_descriptor() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test-spn-8",
        "cell_membership": [0, 0, 0, 0, 1, 1, 1, 1],
        "bit_role": [3, 2, 1, 0, 3, 2, 1, 0],
        "sbox_tables": list(PRESENT_SBOX),
        "linear_layers": [
            {
                "kind": "permutation",
                "source_to_target": list(range(8)),
            }
        ],
        "repeat_single_round": True,
    }


def _write_runtime_descriptor(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    path = tmp_path / "runtime-spn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_external_runtime_permutation_descriptor_repeats_and_hashes_raw_file(
    tmp_path: Path,
) -> None:
    path = _write_runtime_descriptor(tmp_path, _minimal_runtime_descriptor())

    descriptor = load_runtime_spn_descriptor(path, rounds=2)

    assert descriptor.name == "test-spn-8"
    assert descriptor.path == path.resolve()
    assert descriptor.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert descriptor.structure.rounds == 2
    assert torch.equal(
        descriptor.structure.linear_matrices,
        torch.eye(8, dtype=torch.uint8).repeat(2, 1, 1),
    )


def test_external_runtime_sparse_gf2_descriptor_is_invertible(
    tmp_path: Path,
) -> None:
    payload = _minimal_runtime_descriptor()
    payload["linear_layers"] = [
        {
            "kind": "gf2",
            "target_sources": [
                [0, 4],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
            ],
        }
    ]
    descriptor = load_runtime_spn_descriptor(
        _write_runtime_descriptor(tmp_path, payload)
    )
    structure = descriptor.structure

    assert int(structure.linear_matrices[0, 0].sum()) == 2
    product = torch.remainder(
        structure.linear_matrices[0].to(torch.int64)
        @ structure.inverse_linear_matrices[0].to(torch.int64),
        2,
    )
    assert torch.equal(product, torch.eye(8, dtype=torch.int64))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_field", "unknown fields"),
        ("round_mismatch", "round count"),
        ("duplicate_permutation", "permutation"),
        ("duplicate_gf2_source", "must be unique"),
        ("out_of_range_gf2_source", "out of range"),
        ("singular_gf2", "must be invertible"),
        ("float_sbox", "integer arrays"),
        ("boolean_sbox", "integer arrays"),
    ],
)
def test_external_runtime_descriptor_rejects_invalid_structure(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    payload = _minimal_runtime_descriptor()
    requested_rounds = 1
    if case == "unknown_field":
        payload["unexpected"] = True
    elif case == "round_mismatch":
        payload["repeat_single_round"] = False
        requested_rounds = 2
    elif case == "duplicate_permutation":
        payload["linear_layers"] = [
            {"kind": "permutation", "source_to_target": [0, 0, 2, 3, 4, 5, 6, 7]}
        ]
    elif case in {
        "duplicate_gf2_source",
        "out_of_range_gf2_source",
        "singular_gf2",
    }:
        sources = [[index] for index in range(8)]
        if case == "duplicate_gf2_source":
            sources[0] = [0, 0]
        elif case == "out_of_range_gf2_source":
            sources[0] = [8]
        else:
            sources[1] = [0]
        payload["linear_layers"] = [
            {"kind": "gf2", "target_sources": sources}
        ]
    elif case == "float_sbox":
        payload["sbox_tables"] = [float(value) for value in PRESENT_SBOX]
    elif case == "boolean_sbox":
        payload["sbox_tables"] = [False, *list(PRESENT_SBOX)[1:]]
    else:
        raise AssertionError(f"unhandled case: {case}")

    with pytest.raises(ValueError, match=message):
        load_runtime_spn_descriptor(
            _write_runtime_descriptor(tmp_path, payload),
            rounds=requested_rounds,
        )


@pytest.mark.parametrize(
    ("filename", "factory"),
    [
        ("present64.json", present_runtime_structure),
        ("skinny64.json", skinny64_runtime_structure),
    ],
)
def test_production_runtime_descriptors_match_builtin_structures(
    filename: str,
    factory,
) -> None:
    loaded = load_runtime_spn_descriptor(
        ROOT / "configs/runtime/spn" / filename,
        rounds=2,
    ).structure
    expected = factory(2)

    for field in (
        "cell_membership",
        "bit_role",
        "sbox_truth_bits",
        "linear_matrices",
        "inverse_linear_matrices",
    ):
        assert torch.equal(getattr(loaded, field), getattr(expected, field))


def test_generic_runtime_models_share_geometry_metadata_and_forward() -> None:
    descriptor_path = ROOT / "configs/runtime/spn/present64.json"
    options = {
        "runtime_structure_path": str(descriptor_path),
        "processor_steps": 2,
        "pair_embedding_dim": 32,
        "dropout": 0.0,
        "sbox_context_mode": "late_pair",
        "topology_corruption_seed": 0,
    }
    names = (
        "runtime_spn_e4_equivariant_true",
        "runtime_spn_e4_equivariant_corrupted",
        "runtime_spn_e4_equivariant_independent",
    )
    models = [
        build_model(
            name,
            input_bits=512,
            hidden_bits=24,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        ).eval()
        for name in names
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert all(geometry == geometries[0] for geometry in geometries)
    assert torch.equal(
        models[1].runtime_structure.linear_matrices,
        load_runtime_spn_descriptor(descriptor_path, rounds=2)
        .structure.corrupted(0)
        .linear_matrices,
    )
    assert not torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[1].runtime_structure.linear_matrices,
    )
    expected_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    for model, mode in zip(models, ("true", "corrupted", "independent"), strict=True):
        metadata = model_metadata(model)
        assert metadata["runtime_structure_descriptor_name"] == (
            "PRESENT-64 runtime SPN structure"
        )
        assert metadata["runtime_structure_descriptor_path"] == str(
            descriptor_path.resolve()
        )
        assert metadata["runtime_structure_descriptor_sha256"] == expected_sha
        assert metadata["runtime_structure_mode"] == mode
        with torch.no_grad():
            assert model(_binary((2, 512), 57)).shape == (2, 1)


def test_generic_runtime_model_requires_descriptor_path() -> None:
    with pytest.raises(ValueError, match="runtime_structure_path"):
        build_model(
            "runtime_spn_e4_equivariant_true",
            input_bits=128,
            hidden_bits=16,
            pair_bits=128,
            structure="SPN",
        )
