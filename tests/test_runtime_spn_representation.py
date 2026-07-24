from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.evaluation import (
    FrozenRuntimeE4HeadAdapter,
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
from blockcipher_nd.training import TrainingConfig, train_binary_classifier
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


def _target_head(width: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(width),
        torch.nn.Linear(width, 1),
    )


def test_frozen_head_adapter_trains_only_independent_target_head() -> None:
    torch.manual_seed(83)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.2,
    )
    extractor = _adapter(skinny64_runtime_structure(2), spec, pairs=4)
    model = FrozenRuntimeE4HeadAdapter(extractor, _target_head(72))
    extractor_before = {
        name: value.clone() for name, value in extractor.state_dict().items()
    }
    head_before = {
        name: value.clone() for name, value in model.target_head.state_dict().items()
    }

    model.train()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    logits = model(_binary(4, 512, 84)).squeeze(1)
    loss = torch.nn.functional.mse_loss(
        torch.sigmoid(logits),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    loss.backward()
    optimizer.step()

    assert model.training is True
    assert model.feature_extractor.training is False
    assert model.target_head.training is True
    assert model.representation_width == 72
    assert all(
        parameter.requires_grad is False and parameter.grad is None
        for parameter in model.feature_extractor.parameters()
    )
    assert all(
        parameter.requires_grad is True and parameter.grad is not None
        for parameter in model.target_head.parameters()
    )
    assert all(
        torch.equal(extractor.state_dict()[name], value)
        for name, value in extractor_before.items()
    )
    assert any(
        not torch.equal(model.target_head.state_dict()[name], value)
        for name, value in head_before.items()
    )


def test_frozen_head_adapter_replays_public_representation_path() -> None:
    torch.manual_seed(85)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.2,
    )
    extractor = _adapter(rectangle80_runtime_structure(2), spec, pairs=3)
    model = FrozenRuntimeE4HeadAdapter(extractor, _target_head(72)).eval()
    features = _binary(3, 384, 86)

    with torch.no_grad():
        expected = model.target_head(
            extract_runtime_e4_representation(extractor, features).representation
        )
        actual = model(features)

    assert torch.equal(actual, expected)
    assert model.feature_extractor.training is False
    assert model.target_head.training is False


def test_frozen_head_checkpoint_loads_strictly_across_runtime_structures() -> None:
    torch.manual_seed(87)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
        round_window_mode="recurrent_window",
    )
    source = FrozenRuntimeE4HeadAdapter(
        _adapter(gift64_runtime_structure(1), spec, pairs=2),
        _target_head(72),
    )
    target = FrozenRuntimeE4HeadAdapter(
        _adapter(_synthetic_128_structure(3), spec, pairs=3),
        _target_head(72),
    )

    target.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        logits = target(_binary(2, 768, 88))

    assert logits.shape == (2, 1)
    assert tuple(target.state_dict()) == tuple(source.state_dict())
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in source.state_dict().items()
    )


def test_frozen_head_adapter_rejects_parameter_aliasing_and_empty_heads() -> None:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.0,
    )
    extractor = _adapter(present_runtime_structure(), spec, pairs=2)

    with pytest.raises(ValueError, match="must not share"):
        FrozenRuntimeE4HeadAdapter(extractor, extractor.backbone.classifier)
    with pytest.raises(ValueError, match="must own trainable parameters"):
        FrozenRuntimeE4HeadAdapter(extractor, torch.nn.Identity())


def test_frozen_head_adapter_standard_training_checkpoint_contract(
    tmp_path: Path,
) -> None:
    torch.manual_seed(89)
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=1,
        dropout=0.2,
    )
    extractor = _adapter(present_runtime_structure(2), spec, pairs=2)
    model = FrozenRuntimeE4HeadAdapter(extractor, _target_head(72))
    extractor_before = {
        name: value.clone() for name, value in extractor.state_dict().items()
    }
    rng = np.random.default_rng(90)
    features = rng.integers(0, 2, size=(16, 256), dtype=np.uint8)
    labels = np.bitwise_xor(features[:, 0], features[:, 1]).astype(np.uint8)
    dataset = DifferentialDataset(
        features=features,
        labels=labels,
        metadata={"feature_encoding": "ciphertext_pair_bits"},
    )
    checkpoint = tmp_path / "frozen_runtime_e4_head.pt"

    result = train_binary_classifier(
        model,
        dataset,
        dataset,
        TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=91,
            device="cpu",
            loss="mse",
            checkpoint_metric="val_auc",
            restore_best_checkpoint=True,
            checkpoint_output=checkpoint,
        ),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result.metadata["selected_checkpoint"] == "best"
    assert result.metadata["checkpoint_output"] == str(checkpoint)
    assert payload["metadata"] == result.metadata
    assert payload["final_metrics"] == result.final_metrics
    assert set(payload["state_dict"]) == set(model.state_dict())
    assert all(
        torch.equal(payload["state_dict"][name], value)
        for name, value in model.state_dict().items()
    )
    assert all(
        torch.equal(extractor.state_dict()[name], value)
        for name, value in extractor_before.items()
    )

    restored = FrozenRuntimeE4HeadAdapter(
        _adapter(gift64_runtime_structure(3), spec, pairs=2),
        _target_head(72),
    )
    restored.load_state_dict(payload["state_dict"], strict=True)
    metadata = model_metadata(restored)
    assert metadata["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in restored.target_head.parameters()
    )
    assert metadata["runtime_structure_loaded_rounds"] == 3
    assert metadata["runtime_round_window_mode"] == "last_transition"
    assert metadata["runtime_structure_transition_sha256s"] == list(
        restored.feature_extractor.runtime_structure_transition_sha256s
    )
    assert restored.adapter_mode == "frozen_runtime_e4_target_head"
    assert restored.feature_extractor_frozen is True
    assert restored.source_classifier_preserved is True
