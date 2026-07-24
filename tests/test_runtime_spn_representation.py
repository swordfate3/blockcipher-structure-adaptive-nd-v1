from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from blockcipher_nd.evaluation import (
    RuntimeE4RepresentationBatch,
    extract_runtime_e4_representation,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    FixedRuntimeSpnProtocolAdapter,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    permutation_matrix,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    gift64_runtime_structure,
    present_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    standard_four_bit_cells,
)
from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX


def _binary(rows: int, bits: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        2,
        (rows, bits),
        generator=generator,
        dtype=torch.float32,
    )


def _synthetic_128_structure(rounds: int) -> RuntimeSpnStructure:
    membership, roles = standard_four_bit_cells(128)
    permutation = tuple((index + 4) % 128 for index in range(128))
    return runtime_spn_structure(
        cell_membership=membership,
        bit_role=roles,
        sbox_tables=PRESENT_SBOX,
        linear_matrices=permutation_matrix(permutation)
        .unsqueeze(0)
        .repeat(rounds, 1, 1),
    )


def _adapter(
    structure: RuntimeSpnStructure,
    spec: RuntimeParameterizedSpnSpec,
    *,
    pairs: int,
    aggregation_mode: str = "e4_equivariant",
) -> FixedRuntimeSpnProtocolAdapter:
    pair_bits = 2 * structure.block_bits
    return FixedRuntimeSpnProtocolAdapter(
        input_bits=pairs * pair_bits,
        pair_bits=pair_bits,
        structure=structure,
        relation_mode="true",
        spec=spec,
        aggregation_mode=aggregation_mode,
    ).eval()


def test_extraction_returns_exact_classifier_input_and_logits() -> None:
    torch.manual_seed(71)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
    )
    model = _adapter(present_runtime_structure(2), spec, pairs=4)
    features = _binary(3, 512, 72)

    with torch.no_grad():
        output = extract_runtime_e4_representation(model, features)
        replayed_logits = model.backbone.classifier(output.representation)

    assert isinstance(output, RuntimeE4RepresentationBatch)
    assert output.representation.shape == (3, 3 * spec.pair_embedding_dim)
    assert output.logits.shape == (3, 1)
    assert torch.equal(output.logits, replayed_logits)


@pytest.mark.parametrize(
    ("factory", "rounds", "pairs", "block_bits"),
    [
        (present_runtime_structure, 1, 2, 64),
        (gift64_runtime_structure, 2, 3, 64),
        (skinny64_runtime_structure, 3, 4, 64),
        (rectangle80_runtime_structure, 2, 5, 64),
        (_synthetic_128_structure, 4, 3, 128),
    ],
)
def test_same_state_dict_and_representation_width_cross_runtime_structures(
    factory: Callable[[int], RuntimeSpnStructure],
    rounds: int,
    pairs: int,
    block_bits: int,
) -> None:
    torch.manual_seed(73)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
        round_window_mode="recurrent_window",
    )
    anchor = _adapter(present_runtime_structure(1), spec, pairs=2)
    model = _adapter(factory(rounds), spec, pairs=pairs)

    model.load_state_dict(anchor.state_dict(), strict=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        output = extract_runtime_e4_representation(
            model,
            _binary(2, pairs * 2 * block_bits, 74 + rounds + pairs),
        )

    assert output.representation.shape == (2, 72)
    assert tuple(model.state_dict()) == tuple(before)
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in before.items()
    )


def test_extraction_removes_hook_when_forward_raises() -> None:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
    )
    model = _adapter(present_runtime_structure(), spec, pairs=2)
    hooks_before = tuple(model.backbone.classifier._forward_pre_hooks)

    with pytest.raises(ValueError, match="must be binary"):
        extract_runtime_e4_representation(model, torch.full((2, 256), 0.5))

    assert tuple(model.backbone.classifier._forward_pre_hooks) == hooks_before


def test_extraction_rejects_non_e4_models() -> None:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
    )
    non_e4_adapter = _adapter(
        present_runtime_structure(),
        spec,
        pairs=2,
        aggregation_mode="bit_pair",
    )

    with pytest.raises(TypeError, match="E4-equivariant backbone"):
        extract_runtime_e4_representation(non_e4_adapter, _binary(2, 256, 81))
    with pytest.raises(TypeError, match="FixedRuntimeSpnProtocolAdapter"):
        extract_runtime_e4_representation(torch.nn.Identity(), _binary(2, 256, 82))
