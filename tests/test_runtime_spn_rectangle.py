from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.rectangle import (
    Rectangle80,
    rectangle_shift_rows,
    rectangle_sub_columns,
)
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import cipher_profile
from blockcipher_nd.features.profile import (
    STRUCTURE_FEATURE_NAMES,
    structure_feature_vector,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    apply_gf2,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    rectangle80_runtime_structure,
)
from blockcipher_nd.planning.matrix import cipher_key_from_name
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/runtime/spn/rectangle64.json"


def _bits(value: int) -> torch.Tensor:
    return torch.tensor(
        [[(value >> index) & 1 for index in range(64)]],
        dtype=torch.float32,
    )


def _integer(bits: torch.Tensor) -> int:
    return sum(int(bit) << index for index, bit in enumerate(bits.tolist()))


def _runtime_model(descriptor: Path):
    return build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=512,
        hidden_bits=24,
        pair_bits=128,
        structure="SPN",
        model_options={
            "runtime_structure_path": str(descriptor),
            "runtime_rounds": 2,
            "processor_steps": 2,
            "pair_embedding_dim": 32,
            "dropout": 0.0,
            "sbox_context_mode": "late_pair",
        },
    )


def test_rectangle_runtime_cells_are_real_non_contiguous_columns() -> None:
    structure = rectangle80_runtime_structure()

    for column in range(16):
        indices = torch.nonzero(
            structure.cell_membership == column,
            as_tuple=False,
        ).flatten()
        assert indices.tolist() == [column, 16 + column, 32 + column, 48 + column]
        assert structure.bit_role[indices].tolist() == [3, 2, 1, 0]


def test_rectangle_runtime_sbox_and_linear_layer_match_cipher_operations() -> None:
    structure = rectangle80_runtime_structure()
    values = (
        0,
        1,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0xFFFFFFFFFFFFFFFF,
    )

    for value in values:
        bits = _bits(value)
        substituted = structure.apply_sboxes(bits)
        shifted = apply_gf2(structure.linear_matrices[0], bits)
        assert _integer(substituted[0]) == rectangle_sub_columns(value)
        assert _integer(shifted[0]) == rectangle_shift_rows(value)


def test_rectangle_production_descriptor_matches_builtin_structure() -> None:
    loaded = load_runtime_spn_descriptor(DESCRIPTOR, rounds=2).structure
    expected = rectangle80_runtime_structure(2)

    for field in (
        "cell_membership",
        "bit_role",
        "sbox_truth_bits",
        "linear_matrices",
        "inverse_linear_matrices",
    ):
        assert torch.equal(getattr(loaded, field), getattr(expected, field))


def test_rectangle_is_available_to_standard_cipher_data_and_profile_paths() -> None:
    cipher = build_cipher("rectangle80", rounds=25, key=0)
    profile = cipher_profile("rectangle80")

    assert isinstance(cipher, Rectangle80)
    assert cipher.encrypt(0) == 0x0874E8B1E3542D96
    assert cipher_key_from_name("RECTANGLE-80") == "rectangle80"
    assert default_difference("rectangle80") == 0x40
    assert profile.name == "RECTANGLE-80"
    assert profile.structure == "SPN"
    assert profile.block_bits == 64
    assert profile.key_bits == 80
    assert "non_contiguous_sbox_cells" in profile.traits
    structure_features = dict(
        zip(
            STRUCTURE_FEATURE_NAMES,
            structure_feature_vector(profile, rounds=4),
            strict=True,
        )
    )
    assert structure_features["has_sbox_layer"] == 1.0
    assert structure_features["has_permutation_layer"] == 1.0
    assert structure_features["has_bit_permutation"] == 1.0
    assert structure_features["has_rotation"] == 1.0

    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=build_cipher("rectangle80", rounds=4, key=0),
            input_difference=default_difference("rectangle80"),
            samples_per_class=8,
            seed=87,
            shuffle=False,
            pairs_per_sample=4,
            feature_encoding="ciphertext_pair_bits",
            negative_mode="encrypted_random_plaintexts",
        )
    )
    assert dataset.features.shape == (16, 512)
    np.testing.assert_array_equal(
        dataset.labels,
        np.asarray([1] * 8 + [0] * 8, dtype=np.uint8),
    )
    assert dataset.metadata["negative_mode"] == "encrypted_random_plaintexts"
    assert dataset.metadata["pairs_per_sample"] == 4


def test_rectangle_six_round_best_trail_profile_uses_physical_row_bits() -> None:
    cells = [0] * 16
    cells[0] = 0x6
    cells[5] = 0x5
    expected = sum(
        ((cells[column] >> row) & 1) << (16 * row + column)
        for column in range(16)
        for row in range(4)
    )

    assert expected == 0x0000002100010020
    assert (
        difference_for_profile("rectangle80_weng_repo_best_trail_r6")
        == expected
    )


def test_generic_runtime_model_loads_rectangle_without_new_parameter_geometry() -> None:
    rectangle = _runtime_model(DESCRIPTOR).eval()
    present = _runtime_model(ROOT / "configs/runtime/spn/present64.json").eval()
    rectangle_geometry = {
        name: tuple(value.shape) for name, value in rectangle.state_dict().items()
    }
    present_geometry = {
        name: tuple(value.shape) for name, value in present.state_dict().items()
    }

    assert rectangle_geometry == present_geometry
    assert rectangle.runtime_structure_descriptor_name == (
        "RECTANGLE-80 runtime SPN structure"
    )
    assert rectangle.runtime_structure.cell_membership.tolist()[:20] == [
        *range(16),
        0,
        1,
        2,
        3,
    ]
    with torch.no_grad():
        output = rectangle(
            torch.randint(0, 2, (2, 512), dtype=torch.float32)
        )
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
