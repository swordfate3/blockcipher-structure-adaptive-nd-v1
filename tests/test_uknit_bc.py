from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers.spn.uknit import (
    UKNIT_SBOX_TABLES,
    UknitBc,
    uknit_linear_layer,
    uknit_round_keys,
    uknit_substitution_layer,
)
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import cipher_profile, model_metadata
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    present_runtime_structure,
    uknit64_runtime_structure,
)
from blockcipher_nd.planning.matrix import build_tasks, cipher_key_from_name
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/uknit64.json"
U2D_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1.csv"
)
U2E_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1.csv"
)
U2F_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_delta_u_query_u2f_2048_seed0_seed1.csv"
)
U2H_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1.csv"
)


def _runtime_bits(value: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> bit) & 1 for bit in range(64)]],
        dtype=torch.float32,
    )


def _runtime_int(bits: torch.Tensor) -> int:
    return sum(int(bits[0, bit]) << bit for bit in range(64))


@pytest.mark.parametrize(
    ("plaintext", "key", "ciphertext"),
    [
        (0x0000000000000000, 0x00000000000000000000000000000000, 0x034AF0B3C687E424),
        (0x0123456789ABCDEF, 0x0123456789ABCDEF0123456789ABCDEF, 0x7D4EF882C1F42DBA),
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xDB058583DF8F186F),
        (0x1111111111111111, 0xFEDCBA98765432100123456789ABCDEF, 0x7C8DDAF0FEAD3409),
    ],
)
def test_uknit_matches_official_full_round_vectors(
    plaintext: int,
    key: int,
    ciphertext: int,
) -> None:
    assert UknitBc(rounds=12, key=key).encrypt(plaintext) == ciphertext


def test_uknit_prefix_reduced_rounds_match_official_zero_vector_states() -> None:
    expected_after_linear = (
        0x305C0FBCA20690C0,
        0x34C088E84448EC46,
        0x68256D854243D5A4,
        0x2516214A84948088,
        0x36C68A03F793F502,
        0x7E73BACADA1A7B42,
        0x605A54C54E0811E4,
        0x8400D39E1401592B,
        0xF929663FA8CCB64B,
        0xF807067873ED0A40,
        0xCF719C1E491D2D2C,
    )

    assert (
        tuple(UknitBc(rounds=rounds, key=0).encrypt(0) for rounds in range(1, 12))
        == expected_after_linear
    )


def test_uknit_key_schedule_matches_official_vector() -> None:
    expected = tuple(
        int(value, 16)
        for value in (
            "0123456789abcdef",
            "0123456789abcdef",
            "59765a5e0b2f54d3",
            "50d8bc1fee76ceea",
            "e99ef9638d925310",
            "02d675b68a4617c6",
            "853132d409afbb37",
            "d0ea2d834ba7ceaf",
            "dc5516144799fbe0",
            "a742398c856f235d",
            "33ba2c15e3c72db1",
            "273a442f4caf5884",
            "dbdfa62b55d821b4",
        )
    )

    assert uknit_round_keys(0x0123456789ABCDEF0123456789ABCDEF) == expected


def test_uknit_validates_round_key_and_plaintext_ranges() -> None:
    with pytest.raises(ValueError, match="1..12"):
        UknitBc(rounds=0)
    with pytest.raises(ValueError, match="1..12"):
        UknitBc(rounds=13)
    with pytest.raises(ValueError, match="128 bits"):
        UknitBc(key=1 << 128)
    with pytest.raises(ValueError, match="64 bits"):
        UknitBc().encrypt(1 << 64)


def test_uknit_is_registered_as_a_64_bit_non_aligned_spn() -> None:
    cipher = build_cipher("uknit64", rounds=12, key=0)
    profile = cipher_profile("uknit64")

    assert isinstance(cipher, UknitBc)
    assert cipher.name == profile.name == "uKNIT-BC"
    assert profile.structure == "SPN"
    assert profile.block_bits == 64
    assert profile.key_bits == 128
    assert "cell_specific_sboxes" in profile.traits
    assert "round_specific_diffusion" in profile.traits
    assert default_difference("uknit64") == 0x40
    assert cipher_key_from_name("uKNIT-BC") == "uknit64"


def test_uknit_generates_strict_differential_pair_data() -> None:
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=build_cipher("uknit64", rounds=4, key=0),
            input_difference=0x40,
            samples_per_class=4,
            seed=7,
            pairs_per_sample=2,
            negative_mode="encrypted_random_plaintexts",
        )
    )

    assert dataset.features.shape == (8, 256)
    assert dataset.labels.shape == (8,)
    assert int(dataset.labels.sum()) == 4
    assert dataset.metadata["negative_mode"] == "encrypted_random_plaintexts"


def test_uknit_descriptor_windows_match_cipher_structure_exactly() -> None:
    for round_start in (0, 4, 9):
        loaded = load_runtime_spn_descriptor(
            DESCRIPTOR,
            rounds=2,
            round_start=round_start,
        )
        expected = uknit64_runtime_structure(2, round_start=round_start)

        assert loaded.round_start == round_start
        assert loaded.available_rounds == 11
        for field in (
            "cell_membership",
            "bit_role",
            "sbox_truth_bits",
            "linear_matrices",
            "inverse_linear_matrices",
        ):
            assert torch.equal(
                getattr(loaded.structure, field),
                getattr(expected, field),
            )


def test_uknit_descriptor_rejects_out_of_range_windows() -> None:
    with pytest.raises(ValueError, match="window"):
        load_runtime_spn_descriptor(DESCRIPTOR, rounds=2, round_start=10)
    with pytest.raises(ValueError, match="non-negative"):
        load_runtime_spn_descriptor(DESCRIPTOR, rounds=1, round_start=-1)
    with pytest.raises(ValueError, match="round_start"):
        load_runtime_spn_descriptor(
            ROOT / "configs/runtime/spn/present64.json",
            rounds=2,
            round_start=1,
        )


def test_sbox_assignment_control_rejects_shared_sbox_noop() -> None:
    with pytest.raises(ValueError, match="identical across all cells"):
        present_runtime_structure().shuffled_sbox_assignments()


def test_all_uknit_sboxes_are_permutations_and_linear_layers_are_invertible() -> None:
    expected_values = list(range(16))
    assert all(
        sorted(table) == expected_values
        for round_tables in UKNIT_SBOX_TABLES
        for table in round_tables
    )

    structure = uknit64_runtime_structure(11)
    identity = torch.eye(64, dtype=torch.int64)
    for linear, inverse in zip(
        structure.linear_matrices,
        structure.inverse_linear_matrices,
        strict=True,
    ):
        assert int(linear.sum(dim=1).min()) == 3
        assert torch.equal(
            torch.remainder(linear.to(torch.int64) @ inverse.to(torch.int64), 2),
            identity,
        )


def test_runtime_sbox_operators_match_uknit_and_invert_every_cell_table() -> None:
    structure = uknit64_runtime_structure(11)

    for round_index in range(11):
        assert torch.equal(
            structure.sbox_tables(round_index),
            torch.tensor(UKNIT_SBOX_TABLES[round_index]),
        )
        for cell_value in range(16):
            state = sum(cell_value << (4 * cell) for cell in range(16))
            values = _runtime_bits(state)
            substituted = structure.apply_sboxes(values, round_index)

            assert _runtime_int(substituted) == uknit_substitution_layer(
                state, round_index
            )
            torch.testing.assert_close(
                structure.apply_inverse_sboxes(substituted, round_index),
                values,
                rtol=0.0,
                atol=0.0,
            )


def test_runtime_inverse_linear_and_sbox_recover_known_pre_sbox_state() -> None:
    actual_round = 3
    structure = uknit64_runtime_structure(2, round_start=2)
    pre_sbox = 0x0123456789ABCDEF
    substituted = uknit_substitution_layer(pre_sbox, actual_round)
    observed = uknit_linear_layer(substituted, actual_round)

    recovered = structure.apply_inverse_sboxes(
        structure.exact_inverse(_runtime_bits(observed), -1),
        -1,
    )

    assert _runtime_int(recovered) == pre_sbox


def test_generic_runtime_models_load_uknit_window_with_equal_geometry() -> None:
    options = {
        "runtime_structure_path": str(DESCRIPTOR),
        "runtime_round_start": 4,
        "processor_steps": 2,
        "pair_embedding_dim": 32,
        "dropout": 0.0,
        "sbox_context_mode": "late_cell",
    }
    models = [
        build_model(
            name,
            input_bits=512,
            hidden_bits=24,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for name in (
            "runtime_spn_e4_equivariant_true",
            "runtime_spn_e4_equivariant_corrupted",
            "runtime_spn_e4_equivariant_sbox_shuffled",
            "runtime_spn_e4_equivariant_independent",
        )
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]
    features = torch.randint(0, 2, (2, 512), dtype=torch.float32)

    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(model(features).shape == (2, 1) for model in models)
    metadata = model_metadata(models[0])
    assert metadata["runtime_structure_round_start"] == 4
    assert metadata["runtime_structure_available_rounds"] == 11
    assert metadata["runtime_structure_loaded_rounds"] == 2
    assert [model_metadata(model)["runtime_structure_mode"] for model in models] == [
        "true",
        "corrupted",
        "sbox_shuffled",
        "independent",
    ]
    expected = uknit64_runtime_structure(2, round_start=4)
    assert torch.equal(
        models[0].runtime_structure.linear_matrices,
        expected.linear_matrices,
    )
    assert torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[2].runtime_structure.linear_matrices,
    )
    assert not torch.equal(
        models[0].runtime_structure.sbox_truth_bits,
        models[2].runtime_structure.sbox_truth_bits,
    )
    assert torch.equal(
        torch.sort(models[0].runtime_structure.sbox_truth_bits, dim=1).values,
        torch.sort(models[2].runtime_structure.sbox_truth_bits, dim=1).values,
    )


def test_real_uknit_assignment_is_visible_only_to_cell_preserving_sbox_mode() -> None:
    round_start = 4
    true_structure = uknit64_runtime_structure(2, round_start=round_start)
    shuffled_tables = torch.tensor(
        UKNIT_SBOX_TABLES[round_start : round_start + 2],
        dtype=torch.long,
    ).roll(1, dims=1)
    shuffled_structure = runtime_spn_structure(
        cell_membership=true_structure.cell_membership,
        bit_role=true_structure.bit_role,
        sbox_tables=shuffled_tables,
        linear_matrices=true_structure.linear_matrices,
    )
    common = {
        "hidden_dim": 16,
        "pair_embedding_dim": 32,
        "processor_steps": 2,
        "dropout": 0.0,
        "sbox_context_scale": 1.0,
    }
    torch.manual_seed(20260724)
    late_pair = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_pair")
    ).eval()
    late_cell = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, sbox_context_mode="late_cell")
    ).eval()
    late_cell.load_state_dict(late_pair.state_dict())
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)

    with torch.no_grad():
        pair_true = late_pair(pairs, true_structure)
        pair_shuffled = late_pair(pairs, shuffled_structure)
        cell_true = late_cell(pairs, true_structure)
        cell_shuffled = late_cell(pairs, shuffled_structure)

    pair_delta = float(torch.max(torch.abs(pair_true - pair_shuffled)))
    cell_delta = float(torch.max(torch.abs(cell_true - cell_shuffled)))
    assert pair_delta <= 1e-6
    assert cell_delta > max(1e-6, pair_delta * 100.0)


def test_uknit_edge_gate_is_assignment_sensitive_and_cell_relabel_equivariant() -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    shuffled = structure.shuffled_sbox_assignments(20260724)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(20260725)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
        )
    )

    original = model(pairs, structure)
    shuffled_logits = model(pairs, shuffled)
    relabeled_logits = model(relabeled_pairs, relabeled)
    original.square().mean().backward()

    assert torch.isfinite(original).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert float(torch.max(torch.abs(original - shuffled_logits)).detach()) > 1e-6
    torch.testing.assert_close(original, relabeled_logits, rtol=0.0, atol=1e-6)


def test_uknit_state_triplet_preserves_endpoints_and_pair_symmetry() -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)
    shifted = pairs.clone()
    shifted[..., :8] = 1.0 - shifted[..., :8]
    swapped = pairs.flip(dims=(2,))
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.testing.assert_close(
        torch.remainder(pairs[:, :, 0] + pairs[:, :, 1], 2.0),
        torch.remainder(shifted[:, :, 0] + shifted[:, :, 1], 2.0),
    )

    common = dict(
        hidden_dim=16,
        pair_embedding_dim=32,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
    )
    torch.manual_seed(20260726)
    difference_only = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common)
    ).eval()
    state_triplet = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(**common, cell_input_mode="state_triplet")
    ).eval()
    state_triplet.load_state_dict(difference_only.state_dict())

    difference_original = difference_only(pairs, structure)
    difference_shifted = difference_only(shifted, structure)
    triplet_original = state_triplet(pairs, structure)
    triplet_shifted = state_triplet(shifted, structure)
    triplet_swapped = state_triplet(swapped, structure)
    triplet_relabeled = state_triplet(relabeled_pairs, relabeled)
    triplet_original.square().mean().backward()

    torch.testing.assert_close(
        difference_original,
        difference_shifted,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        float(torch.max(torch.abs(triplet_original - triplet_shifted)).detach()) > 1e-6
    )
    torch.testing.assert_close(triplet_original, triplet_swapped, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        triplet_original,
        triplet_relabeled,
        rtol=0.0,
        atol=1e-6,
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in state_triplet.parameters()
    )


def test_uknit_inverse_sbox_triplet_is_operator_sensitive_and_equivariant() -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    shuffled = structure.shuffled_sbox_assignments(20260724)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)
    swapped = pairs.flip(dims=(2,))
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    torch.manual_seed(20260727)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
            cell_input_mode="inverse_sbox_triplet",
        )
    )
    fusion_inputs: list[torch.Tensor] = []

    def capture_fusion_input(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        fusion_inputs.append(args[0].detach().clone())

    hook = model.typed_fusion.register_forward_pre_hook(capture_fusion_input)
    correct = model(pairs, structure)
    shuffled_logits = model(pairs, shuffled)
    swapped_logits = model(swapped, structure)
    relabeled_logits = model(relabeled_pairs, relabeled)
    hook.remove()
    correct.square().mean().backward()

    assert len(fusion_inputs) == 4
    assert float(torch.max(torch.abs(fusion_inputs[0] - fusion_inputs[1]))) > 1e-6
    assert float(torch.max(torch.abs(correct - shuffled_logits)).detach()) > 1e-6
    torch.testing.assert_close(correct, swapped_logits, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(correct, relabeled_logits, rtol=0.0, atol=1e-6)
    assert torch.isfinite(correct).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_uknit_inverse_sbox_triplet_independent_mode_bypasses_inverse_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            cell_input_mode="inverse_sbox_triplet",
        )
    ).eval()

    def reject_inverse(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("independent mode must not execute inverse S-boxes")

    monkeypatch.setattr(type(structure), "apply_inverse_sboxes", reject_inverse)
    with torch.no_grad():
        output = model(
            torch.randint(0, 2, (2, 4, 2, 64), dtype=torch.float32),
            structure,
            relation_mode="independent",
        )

    assert output.shape == (2, 1)


def test_uknit_dual_view_triplet_is_exact_mean_and_preserves_symmetries() -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    shuffled = structure.shuffled_sbox_assignments(20260724)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)
    swapped = pairs.flip(dims=(2,))
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    common = dict(
        hidden_dim=16,
        pair_embedding_dim=32,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
    )
    torch.manual_seed(20260728)
    models = {
        mode: RuntimeE4EquivariantSpnDistinguisher(
            RuntimeParameterizedSpnSpec(**common, cell_input_mode=mode)
        )
        for mode in (
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
        )
    }
    state_dict = models["state_triplet"].state_dict()
    for model in models.values():
        model.load_state_dict(state_dict)
    fusion_inputs: dict[str, torch.Tensor] = {}

    for mode, model in models.items():

        def capture(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            *,
            key: str = mode,
        ) -> None:
            fusion_inputs[key] = args[0].detach().clone()

        hook = model.typed_fusion.register_forward_pre_hook(capture)
        model(pairs, structure)
        hook.remove()

    torch.testing.assert_close(
        fusion_inputs["dual_view_triplet"],
        0.5 * (fusion_inputs["state_triplet"] + fusion_inputs["inverse_sbox_triplet"]),
        rtol=0.0,
        atol=0.0,
    )

    dual = models["dual_view_triplet"]
    correct = dual(pairs, structure)
    shuffled_logits = dual(pairs, shuffled)
    swapped_logits = dual(swapped, structure)
    relabeled_logits = dual(relabeled_pairs, relabeled)
    correct.square().mean().backward()

    assert float(torch.max(torch.abs(correct - shuffled_logits)).detach()) > 1e-6
    torch.testing.assert_close(correct, swapped_logits, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(correct, relabeled_logits, rtol=0.0, atol=1e-6)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in dual.parameters()
    )


def test_uknit_dual_view_independent_mode_bypasses_runtime_inverse_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            cell_input_mode="dual_view_triplet",
        )
    ).eval()

    def reject_inverse(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("independent mode must bypass runtime inverse operators")

    monkeypatch.setattr(type(structure), "exact_inverse", reject_inverse)
    monkeypatch.setattr(type(structure), "apply_inverse_sboxes", reject_inverse)
    with torch.no_grad():
        output = model(
            torch.randint(0, 2, (2, 4, 2, 64), dtype=torch.float32),
            structure,
            relation_mode="independent",
        )

    assert output.shape == (2, 1)


def test_uknit_delta_u_query_preserves_state_views_and_runtime_symmetries() -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    shuffled = structure.shuffled_sbox_assignments(20260724)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (3, 4, 2, 64), dtype=torch.float32)
    swapped = pairs.flip(dims=(2,))
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    common = dict(
        hidden_dim=16,
        pair_embedding_dim=32,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
    )
    torch.manual_seed(20260729)
    anchor = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            **common,
            cell_input_mode="state_triplet_delta_v_query",
        )
    ).eval()
    candidate = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            **common,
            cell_input_mode="state_triplet_delta_u_query",
        )
    ).eval()
    candidate.load_state_dict(anchor.state_dict())
    fusion_inputs: dict[str, torch.Tensor] = {}

    for name, model in (("anchor", anchor), ("candidate", candidate)):

        def capture(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            fusion_inputs[key] = args[0].detach().clone()

        hook = model.typed_fusion.register_forward_pre_hook(capture)
        model(pairs, structure)
        hook.remove()

    token_dim = anchor.token_dim
    torch.testing.assert_close(
        fusion_inputs["anchor"][..., : 2 * token_dim],
        fusion_inputs["candidate"][..., : 2 * token_dim],
        rtol=0.0,
        atol=0.0,
    )
    left = pairs[:, :, 0]
    right = pairs[:, :, 1]
    previous_left = structure.exact_inverse(left, -1)
    previous_right = structure.exact_inverse(right, -1)
    delta_v = torch.remainder(previous_left + previous_right, 2.0)
    delta_u = torch.remainder(
        structure.apply_inverse_sboxes(previous_left, -1)
        + structure.apply_inverse_sboxes(previous_right, -1),
        2.0,
    )
    batch, pair_count, cell_count, _ = anchor._ordered_cell_values(
        delta_v, structure
    ).shape
    expected_delta_v = anchor.cell_encoder(
        anchor._ordered_cell_values(delta_v, structure).reshape(
            batch * pair_count, cell_count, 4
        )
    )
    expected_delta_u = candidate.cell_encoder(
        candidate._ordered_cell_values(delta_u, structure).reshape(
            batch * pair_count, cell_count, 4
        )
    )
    torch.testing.assert_close(
        fusion_inputs["anchor"][..., 2 * token_dim :],
        expected_delta_v,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fusion_inputs["candidate"][..., 2 * token_dim :],
        expected_delta_u,
        rtol=0.0,
        atol=0.0,
    )

    correct = candidate(pairs, structure)
    shuffled_logits = candidate(pairs, shuffled)
    swapped_logits = candidate(swapped, structure)
    relabeled_logits = candidate(relabeled_pairs, relabeled)
    correct.square().mean().backward()

    assert float(torch.max(torch.abs(correct - shuffled_logits)).detach()) > 1e-6
    torch.testing.assert_close(correct, swapped_logits, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(correct, relabeled_logits, rtol=0.0, atol=1e-6)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )


def test_uknit_delta_query_modes_bypass_unneeded_runtime_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = uknit64_runtime_structure(2, round_start=2)
    anchor = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            cell_input_mode="state_triplet_delta_v_query",
        )
    ).eval()

    def reject_sbox(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("deltaV identity query must not execute inverse S-boxes")

    monkeypatch.setattr(type(structure), "apply_inverse_sboxes", reject_sbox)
    with torch.no_grad():
        anchor(
            torch.randint(0, 2, (2, 4, 2, 64), dtype=torch.float32),
            structure,
        )

    candidate = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=32,
            processor_steps=2,
            dropout=0.0,
            cell_input_mode="state_triplet_delta_u_query",
        )
    ).eval()

    def reject_inverse(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("independent mode must bypass runtime inverse operators")

    monkeypatch.setattr(type(structure), "exact_inverse", reject_inverse)
    with torch.no_grad():
        output = candidate(
            torch.randint(0, 2, (2, 4, 2, 64), dtype=torch.float32),
            structure,
            relation_mode="independent",
        )

    assert output.shape == (2, 1)


def test_uknit_u2d_plan_builds_equal_geometry_six_row_matrix() -> None:
    tasks = build_tasks(parse_train_args(["--plan", str(U2D_PLAN)]))
    models = [
        build_model(
            task["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        for task in tasks
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert len(tasks) == 6
    assert [task["seed"] for task in tasks] == [0, 0, 0, 1, 1, 1]
    assert [task["model_options"]["cell_input_mode"] for task in tasks] == [
        "inverse_sbox_triplet",
        "state_triplet",
        "inverse_sbox_triplet",
        "inverse_sbox_triplet",
        "state_triplet",
        "inverse_sbox_triplet",
    ]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == 442466
        for model in models
    )


def test_uknit_u2e_plan_builds_equal_geometry_six_row_matrix() -> None:
    tasks = build_tasks(parse_train_args(["--plan", str(U2E_PLAN)]))
    models = [
        build_model(
            task["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        for task in tasks
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert len(tasks) == 6
    assert [task["seed"] for task in tasks] == [0, 0, 0, 1, 1, 1]
    assert [task["model_options"]["cell_input_mode"] for task in tasks] == [
        "dual_view_triplet",
        "state_triplet",
        "dual_view_triplet",
        "dual_view_triplet",
        "state_triplet",
        "dual_view_triplet",
    ]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == 442466
        for model in models
    )


def test_uknit_u2f_plan_builds_equal_geometry_six_row_query_matrix() -> None:
    tasks = build_tasks(parse_train_args(["--plan", str(U2F_PLAN)]))
    models = [
        build_model(
            task["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        for task in tasks
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert len(tasks) == 6
    assert [task["seed"] for task in tasks] == [0, 0, 0, 1, 1, 1]
    assert [task["model_options"]["cell_input_mode"] for task in tasks] == [
        "state_triplet_delta_u_query",
        "state_triplet_delta_v_query",
        "state_triplet_delta_u_query",
        "state_triplet_delta_u_query",
        "state_triplet_delta_v_query",
        "state_triplet_delta_u_query",
    ]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == 458850
        for model in models
    )


def test_uknit_u2h_plan_changes_only_to_r5_aligned_window() -> None:
    u2f_tasks = build_tasks(parse_train_args(["--plan", str(U2F_PLAN)]))
    tasks = build_tasks(parse_train_args(["--plan", str(U2H_PLAN)]))
    models = [
        build_model(
            task["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=task["model_options"],
        )
        for task in tasks
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]

    assert len(tasks) == 6
    assert [task["rounds"] for task in tasks] == [5] * 6
    assert [task["seed"] for task in tasks] == [0, 0, 0, 1, 1, 1]
    assert [task["model_options"]["runtime_round_start"] for task in tasks] == [3] * 6
    assert [task["model_options"]["cell_input_mode"] for task in tasks] == [
        "state_triplet_delta_u_query",
        "state_triplet_delta_v_query",
        "state_triplet_delta_u_query",
        "state_triplet_delta_u_query",
        "state_triplet_delta_v_query",
        "state_triplet_delta_u_query",
    ]
    for u2f, u2h in zip(u2f_tasks, tasks):
        assert u2f["samples_per_class"] == u2h["samples_per_class"] == 2048
        assert u2f["pairs_per_sample"] == u2h["pairs_per_sample"] == 4
        assert u2f["target_epochs"] == u2h["target_epochs"] == 10
        assert u2f["model_key"] == u2h["model_key"]
    assert all(geometry == geometries[0] for geometry in geometries)
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == 458850
        for model in models
    )
