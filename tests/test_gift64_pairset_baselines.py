from __future__ import annotations

import torch

from blockcipher_nd.engine.task_config import build_dataset_config
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.models.structure.spn.cross_spn_typed_cell import (
    GiftCrossSpnTypedCellTrueDistinguisher,
)
from blockcipher_nd.models.structure.spn.gift_pairset_baselines import (
    Gift64GohrStyleResNetPairSetDistinguisher,
    Gift64SunStyleLstmPairSetDistinguisher,
)
from blockcipher_nd.registry.model_factory import build_model


def _features(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260715)
    return torch.randint(0, 2, (batch, 4 * 128), generator=generator).float()


def test_gift64_pairset_baselines_return_one_logit_per_sample() -> None:
    features = _features()
    for model in (
        Gift64SunStyleLstmPairSetDistinguisher(input_bits=512),
        Gift64GohrStyleResNetPairSetDistinguisher(input_bits=512),
    ):
        assert model(features).shape == (3, 1)


def test_gift64_pairset_baselines_are_pair_order_invariant() -> None:
    features = _features(batch=2)
    reordered = features.reshape(2, 4, 128)[:, [2, 0, 3, 1], :].reshape(2, 512)
    for model in (
        Gift64SunStyleLstmPairSetDistinguisher(input_bits=512),
        Gift64GohrStyleResNetPairSetDistinguisher(input_bits=512),
    ):
        model.eval()
        torch.testing.assert_close(model(features), model(reordered))


def test_gift64_mainstream_adaptations_are_capacity_matched() -> None:
    typed = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=512)
    typed_parameters = sum(parameter.numel() for parameter in typed.parameters())
    for model in (
        Gift64SunStyleLstmPairSetDistinguisher(input_bits=512),
        Gift64GohrStyleResNetPairSetDistinguisher(input_bits=512),
    ):
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert 0.75 * typed_parameters <= parameters <= 1.35 * typed_parameters


def test_gift64_pairset_baselines_and_source_aliases_build_from_registry() -> None:
    options_by_model = {
        "gift64_sun_style_lstm_pairset": {"lstm_hidden_bits": 128},
        "gift64_gohr_style_resnet_pairset": {
            "resnet_channels": 64,
            "resnet_blocks": 7,
        },
        "gift_cross_spn_typed_cell_true_from_present_true_s0": {},
        "gift_cross_spn_typed_cell_true_from_present_true_s1": {},
    }
    for model_key, options in options_by_model.items():
        model = build_model(
            model_key,
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        assert model(_features(batch=2)).shape == (2, 1)


def test_balanced_validation_total_is_converted_to_per_class_only() -> None:
    task = {
        "dataset_label_mode": "balanced_per_class",
        "input_difference": 0x40,
        "feature_encoding": "ciphertext_pair_bits",
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "integral_active_nibble": 0,
        "integral_active_nibbles": (),
        "validation_integral_active_nibbles": (),
        "selected_bit_indices": (),
    }

    config = build_dataset_config(
        task,
        cipher=build_cipher("gift64", rounds=6, key=0),
        samples_per_class=64,
        samples_total=128,
        seed=10_006,
        split="validation",
    )

    assert config.samples_per_class == 64
    assert config.samples_total is None
