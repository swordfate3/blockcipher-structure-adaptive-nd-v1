from __future__ import annotations

import pytest
import torch

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_round_recurrent_spn import (
    SelectedOutputRoundRecurrentSpn,
    build_round_recurrent_spn,
    round_recurrent_parameter_count,
)


OPF2_EXACT_PARAMETER_COUNT = 3_956_928


def test_opf3_model_has_frozen_shape_contexts_and_local_heads() -> None:
    model = SelectedOutputRoundRecurrentSpn(token_dim=12)

    assert model.round_position_contexts.shape == (4, 64, 12)
    assert model.final_whitening_context.shape == (1, 64, 12)
    assert len(model.heads) == 8
    assert all(head[0].in_features == 12 for head in model.heads)
    assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_opf3_uses_one_shared_round_block_exactly_four_times() -> None:
    model = SelectedOutputRoundRecurrentSpn(token_dim=8)
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.round_block.register_forward_hook(count_call)
    try:
        model(torch.zeros((1, 64)))
    finally:
        handle.remove()

    assert calls == 4
    assert len([name for name, _ in model.named_modules() if name == "round_block"]) == 1


def test_opf3_exact_and_identity_are_parameter_matched_but_route_differently() -> None:
    models = []
    for mode in ("exact", "identity"):
        torch.manual_seed(20260723)
        models.append(build_round_recurrent_spn(mode, token_dim=8))

    exact, identity = models
    assert exact.state_dict().keys() == identity.state_dict().keys()
    assert all(
        torch.equal(exact.state_dict()[name], identity.state_dict()[name])
        for name in exact.state_dict()
    )
    assert torch.equal(
        exact.round_block.source_for_destination,
        _present_topology_mapping("exact"),
    )
    assert torch.equal(
        identity.round_block.source_for_destination,
        _present_topology_mapping("identity"),
    )
    features = (torch.arange(128).reshape(2, 64) % 3 == 0).float()
    assert not torch.equal(exact(features), identity(features))


def test_opf3_exact_round_route_matches_present_for_every_one_hot_bit() -> None:
    model = build_round_recurrent_spn("exact", token_dim=1)
    with torch.no_grad():
        for module in (model.round_block.local_mlp, model.round_block.channel_mlp):
            for parameter in module.parameters():
                parameter.zero_()

    for source_msb in range(64):
        hidden = torch.zeros((1, 64, 1))
        hidden[0, source_msb, 0] = 1.0

        routed = model.round_block(hidden)[0, :, 0]
        destinations = torch.nonzero(routed, as_tuple=False).flatten().tolist()
        expected_state = Present80.permutation_layer(1 << (63 - source_msb))
        expected_destination_msb = 63 - (expected_state.bit_length() - 1)

        assert destinations == [expected_destination_msb]
        assert routed[expected_destination_msb].item() == 1.0


def test_opf3_shared_route_composes_to_four_present_p_layers() -> None:
    model = build_round_recurrent_spn("exact", token_dim=1)
    with torch.no_grad():
        for module in (model.round_block.local_mlp, model.round_block.channel_mlp):
            for parameter in module.parameters():
                parameter.zero_()

    for source_msb in range(64):
        hidden = torch.zeros((1, 64, 1))
        hidden[0, source_msb, 0] = 1.0
        expected_state = 1 << (63 - source_msb)

        for _ in range(4):
            hidden = model.round_block(hidden)
            expected_state = Present80.permutation_layer(expected_state)

        routed = hidden[0, :, 0]
        expected_destination_msb = 63 - (expected_state.bit_length() - 1)
        destinations = torch.nonzero(routed, as_tuple=False).flatten().tolist()

        assert destinations == [expected_destination_msb]
        assert routed[expected_destination_msb].item() == 1.0


def test_opf3_preregistered_width_matches_opf2_capacity_band() -> None:
    count = round_recurrent_parameter_count()

    assert count == 3_884_356
    assert abs(count - OPF2_EXACT_PARAMETER_COUNT) / OPF2_EXACT_PARAMETER_COUNT <= 0.03


def test_opf3_rejects_protocol_shape_or_topology_drift() -> None:
    model = SelectedOutputRoundRecurrentSpn(token_dim=8)

    with pytest.raises(ValueError, match=r"expected \[batch, 64\]"):
        model(torch.zeros((2, 63)))
    with pytest.raises(ValueError, match="exactly four"):
        SelectedOutputRoundRecurrentSpn(token_dim=8, rounds=5)
    with pytest.raises(ValueError, match="64-position permutation"):
        SelectedOutputRoundRecurrentSpn(
            token_dim=8,
            source_for_destination=torch.zeros(64, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="exact or identity"):
        build_round_recurrent_spn("wrong", token_dim=8)
