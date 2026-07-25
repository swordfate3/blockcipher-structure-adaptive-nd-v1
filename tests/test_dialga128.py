from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers.spn.dialga import (
    Dialga128,
    dialga128_encrypt,
    dialga128_round_trace,
    dialga_inverse_linear_layer,
    dialga_inverse_round_function,
    dialga_inverse_sub_cells,
    dialga_linear_layer,
    dialga_mix_columns,
    dialga_round_function,
    dialga_sub_cells,
)
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    apply_gf2,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    present_runtime_structure,
)
from blockcipher_nd.engine.modeling import cipher_profile, model_metadata
from blockcipher_nd.planning.matrix import cipher_key_from_name, tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.model_factory import build_model


KEY0 = int(
    "00112233445566778899aabbccddeeff"
    "112233445566778899aabbccddeeff00",
    16,
)
TWEAK0 = int("2233445566778899aabbccddee00ff11", 16)
PLAINTEXT0 = int("00112233445566778899aabbccddeeff", 16)
ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/dialga128.json"
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_dialga128_runtime_e4_d1_r4_2048_seed0_seed1.csv"
)


@pytest.mark.parametrize(
    ("total_rounds", "plaintext", "key", "tweak", "expected"),
    [
        (
            20,
            PLAINTEXT0,
            KEY0,
            TWEAK0,
            int("a4a1ea948919d8996e13b1b365bb0ce6", 16),
        ),
        (
            20,
            int("0123456789abcdef0123456789abcdef", 16),
            int("fedcba9876543210fedcba9876543210" * 2, 16),
            int("00001111222233334444555566668888", 16),
            int("dc355a6376d9617723efb9a98c1b4864", 16),
        ),
        (
            16,
            PLAINTEXT0,
            KEY0,
            TWEAK0,
            int("838407143af9a876fbdc6be378e9045b", 16),
        ),
        (
            16,
            int("0123456789abcdef0123456789abcdef", 16),
            int("fedcba9876543210fedcba9876543210" * 2, 16),
            int("00001111222233334444555566668888", 16),
            int("a16812d9738333d238c23e67ac20ef16", 16),
        ),
    ],
)
def test_dialga128_matches_published_full_vectors(
    total_rounds: int,
    plaintext: int,
    key: int,
    tweak: int,
    expected: int,
) -> None:
    assert (
        dialga128_encrypt(
            plaintext,
            key,
            tweak,
            total_rounds=total_rounds,
        )
        == expected
    )


def test_dialga128_matches_published_16_round_trace() -> None:
    expected = tuple(
        int(value, 16)
        for value in (
            "9f95f3ff7a092a2c465dfdf31225ea00",
            "98871f6b568e38c69b2df2b8fecc46c4",
            "1d02f8badc70a734a02946d7584c6699",
            "0924e3460275180560387c9d89ccaef7",
            "7df1476adedc82fe587839dc0e43f31c",
            "dd68245888f9cbed8c05065ddcffc56b",
            "9da77dade1660eaf5a2877126d7bdeeb",
            "4c31411fc08dd78da9c94db8175ca087",
            "52ffe58da193d2b26faaa208042c1dfe",
            "d7beb80383bb057a8a470b61e6f1cd19",
            "d0f79a0ce3b4f519c8b06141af26a4ca",
            "ea549d58f095e1159158679953b8be8a",
            "9ef0091d1fd39ce3f836b8d14d5040eb",
            "efef7cf1a11eae91f1ed0ac3d8773621",
            "87e87173805a0eac97de2f230432bcd0",
            "92a33e3c3115979441131a892119bed7",
        )
    )
    assert dialga128_round_trace(PLAINTEXT0, KEY0, TWEAK0) == expected


@pytest.mark.parametrize(
    "state",
    [
        0,
        (1 << 128) - 1,
        0x00112233445566778899AABBCCDDEEFF,
        0xFEDCBA98765432100123456789ABCDEF,
    ],
)
def test_dialga_layers_have_exact_inverses(state: int) -> None:
    assert dialga_inverse_sub_cells(dialga_sub_cells(state)) == state
    assert dialga_mix_columns(dialga_mix_columns(state)) == state
    for round_type in range(4):
        assert (
            dialga_inverse_linear_layer(
                dialga_linear_layer(state, round_type), round_type
            )
            == state
        )
        assert (
            dialga_inverse_round_function(
                dialga_round_function(state, round_type), round_type
            )
            == state
        )


def test_dialga128_class_uses_published_prefix_and_full_semantics() -> None:
    trace = dialga128_round_trace(PLAINTEXT0, KEY0, TWEAK0)
    assert Dialga128(rounds=7, key=KEY0, tweak=TWEAK0).encrypt(PLAINTEXT0) == trace[6]
    assert Dialga128(rounds=16, key=KEY0, tweak=TWEAK0).encrypt(
        PLAINTEXT0
    ) == int("838407143af9a876fbdc6be378e9045b", 16)


def test_dialga128_is_available_to_standard_cipher_and_plan_paths() -> None:
    cipher = build_cipher("dialga128", rounds=20, key=KEY0)
    profile = cipher_profile("dialga128")

    assert isinstance(cipher, Dialga128)
    assert cipher.variant_rounds == 20
    assert cipher.encrypt(PLAINTEXT0) == dialga128_encrypt(
        PLAINTEXT0,
        KEY0,
        0,
        total_rounds=20,
    )
    assert cipher.name == profile.name == "Dialga-128"
    assert profile.structure == "SPN"
    assert profile.block_bits == 128
    assert profile.key_bits == 256
    assert "non_contiguous_sbox_cells" in profile.traits
    assert "multiple_linear_layers" in profile.traits
    assert default_difference("dialga128") == 0x40
    assert cipher_key_from_name("Dialga-128") == "dialga128"


def _integer_bits(value: int) -> torch.Tensor:
    return torch.tensor(
        [(value >> bit) & 1 for bit in range(128)],
        dtype=torch.float32,
    )


def _bits_integer(values: torch.Tensor) -> int:
    return sum(int(bit) << index for index, bit in enumerate(values.tolist()))


@pytest.mark.parametrize(
    "state",
    [
        0,
        (1 << 128) - 1,
        0x00112233445566778899AABBCCDDEEFF,
        0xFEDCBA98765432100123456789ABCDEF,
    ],
)
def test_dialga_runtime_non_contiguous_cells_match_native_subcells(state: int) -> None:
    structure = dialga128_runtime_structure(4)

    assert structure.block_bits == 128
    assert structure.cells == 32
    assert _bits_integer(structure.apply_sboxes(_integer_bits(state), 0)) == (
        dialga_sub_cells(state)
    )
    assert _bits_integer(structure.apply_inverse_sboxes(_integer_bits(state), 0)) == (
        dialga_inverse_sub_cells(state)
    )


@pytest.mark.parametrize(
    "state",
    [
        0,
        (1 << 128) - 1,
        0x00112233445566778899AABBCCDDEEFF,
        0xFEDCBA98765432100123456789ABCDEF,
    ],
)
def test_dialga_runtime_gf2_layers_match_all_native_round_types(state: int) -> None:
    structure = dialga128_runtime_structure(4)
    values = _integer_bits(state)

    for round_type in range(4):
        transformed = apply_gf2(structure.linear_matrices[round_type], values)
        assert _bits_integer(transformed) == dialga_linear_layer(state, round_type)
        assert _bits_integer(structure.exact_inverse(transformed, round_type)) == state


def test_dialga_runtime_structure_tracks_heterogeneous_round_windows() -> None:
    structure = dialga128_runtime_structure(8)
    shifted = dialga128_runtime_structure(3, round_start=3)

    assert structure.unique_transition_count == 4
    assert structure.is_homogeneous is False
    assert torch.equal(structure.linear_matrices[0], structure.linear_matrices[4])
    assert torch.equal(shifted.linear_matrices[0], structure.linear_matrices[3])
    assert torch.equal(shifted.linear_matrices[1], structure.linear_matrices[0])
    assert torch.equal(shifted.linear_matrices[2], structure.linear_matrices[1])


def test_dialga_runtime_descriptor_matches_factory_window() -> None:
    loaded = load_runtime_spn_descriptor(DESCRIPTOR, rounds=3, round_start=3)
    expected = dialga128_runtime_structure(3, round_start=3)

    assert loaded.name == "Dialga-128 20-round heterogeneous runtime SPN structure"
    assert loaded.available_rounds == 20
    assert loaded.round_start == 3
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


def test_dialga_generates_strict_128_bit_differential_pair_data() -> None:
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=build_cipher("dialga128", rounds=4, key=0),
            input_difference=default_difference("dialga128"),
            samples_per_class=4,
            seed=7,
            pairs_per_sample=2,
            negative_mode="encrypted_random_plaintexts",
        )
    )

    assert dataset.features.shape == (8, 512)
    assert dataset.labels.shape == (8,)
    assert int(dataset.labels.sum()) == 4
    assert dataset.metadata["negative_mode"] == "encrypted_random_plaintexts"


def test_runtime_e4_reuses_weights_between_present_and_dialga_widths() -> None:
    spec = RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=2,
        dropout=0.0,
    )
    present_model = RuntimeE4EquivariantSpnDistinguisher(spec).eval()
    dialga_model = RuntimeE4EquivariantSpnDistinguisher(spec).eval()
    dialga_model.load_state_dict(present_model.state_dict(), strict=True)

    present_pairs = torch.randint(0, 2, (2, 3, 2, 64), dtype=torch.float32)
    dialga_pairs = torch.randint(0, 2, (2, 3, 2, 128), dtype=torch.float32)
    with torch.no_grad():
        assert present_model(present_pairs, present_runtime_structure(2)).shape == (2, 1)
        assert dialga_model(dialga_pairs, dialga128_runtime_structure(2)).shape == (2, 1)


def test_runtime_e4_is_equivariant_to_dialga_cell_relabeling() -> None:
    structure = dialga128_runtime_structure(2)
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(0, 2, (2, 3, 2, 128), dtype=torch.float32)
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    model = RuntimeE4EquivariantSpnDistinguisher(
        RuntimeParameterizedSpnSpec(
            hidden_dim=16,
            pair_embedding_dim=24,
            processor_steps=2,
            dropout=0.0,
            sbox_context_mode="edge_gate",
            round_window_mode="recurrent_window",
        )
    ).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def test_generic_runtime_entry_builds_all_dialga_topology_controls() -> None:
    options = {
        "runtime_structure_path": str(DESCRIPTOR),
        "runtime_rounds": 2,
        "runtime_round_start": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 32,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "round_window_mode": "recurrent_window",
        "topology_corruption_seed": 20260725,
    }
    names = (
        "runtime_spn_e4_equivariant_true",
        "runtime_spn_e4_equivariant_corrupted",
        "runtime_spn_e4_equivariant_independent",
    )
    models = [
        build_model(
            name,
            input_bits=1024,
            hidden_bits=24,
            pair_bits=256,
            structure="SPN",
            model_options=options,
        ).eval()
        for name in names
    ]
    shared_state = models[0].state_dict()
    geometry = {name: tuple(value.shape) for name, value in shared_state.items()}
    features = torch.randint(0, 2, (2, 1024), dtype=torch.float32)

    for model, mode in zip(models, ("true", "corrupted", "independent"), strict=True):
        model.load_state_dict(shared_state, strict=True)
        assert {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        } == geometry
        assert model.runtime_structure.block_bits == 128
        assert model.runtime_structure.cells == 32
        assert model_metadata(model)["runtime_structure_mode"] == mode
        with torch.no_grad():
            assert model(features).shape == (2, 1)

    assert not torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[1].runtime_structure.linear_matrices,
    )
    assert torch.equal(
        models[0].runtime_structure.linear_matrices,
        models[2].runtime_structure.linear_matrices,
    )


def test_dialga_d1_plan_parses_the_frozen_six_role_protocol() -> None:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )

    assert len(tasks) == 6
    assert {(task["seed"], task["model_key"]) for task in tasks} == {
        (seed, model)
        for seed in (0, 1)
        for model in (
            "runtime_spn_e4_equivariant_true",
            "runtime_spn_e4_equivariant_corrupted",
            "runtime_spn_e4_equivariant_independent",
        )
    }
    for task in tasks:
        assert task["cipher_key"] == "dialga128"
        assert task["rounds"] == 4
        assert task["samples_per_class"] == 2048
        assert task["pairs_per_sample"] == 4
        assert task["input_difference"] == 0x40
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["train_key"] == 0
        assert task["validation_key"] == int("11" * 32, 16)
        assert task["model_options"]["runtime_structure_path"] == str(
            DESCRIPTOR.relative_to(ROOT)
        )
        assert task["model_options"]["runtime_round_start"] == 2
        assert task["model_options"]["runtime_rounds"] == 2
        assert task["model_options"]["round_window_mode"] == "recurrent_window"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"rounds": 17},
        {"variant_rounds": 15},
        {"key": -1},
        {"key": 1 << 256},
        {"tweak": -1},
        {"tweak": 1 << 128},
    ],
)
def test_dialga128_rejects_invalid_protocol_values(kwargs: dict[str, int]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Dialga128(**kwargs)
