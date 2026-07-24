from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers.spn.uknit import (
    UKNIT_SBOX_TABLES,
    UknitBc,
    uknit_round_keys,
)
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import cipher_profile, model_metadata
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
from blockcipher_nd.planning.matrix import cipher_key_from_name
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/uknit64.json"


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
