from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.ciphers.feistel.des import Des
from blockcipher_nd.tasks.innovation2.feistel.des_output_prediction_data import (
    DesOutputPredictionDataConfig,
    encrypt_des_numpy,
    prepare_des_output_prediction_data,
    serializable_config,
    validate_des_output_prediction_data,
)


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def test_feistel1_formal_data_protocols_are_frozen() -> None:
    f1_a = DesOutputPredictionDataConfig.f1_a()
    f1_b = DesOutputPredictionDataConfig.f1_b()
    f1_c = DesOutputPredictionDataConfig.f1_c()
    f1_r_a = DesOutputPredictionDataConfig.f1_r(key_seed=31)
    f1_r_b = DesOutputPredictionDataConfig.f1_r(key_seed=32)

    assert (f1_a.rounds, f1_a.train_rows, f1_a.test_rows, f1_a.key_seed) == (
        2,
        1 << 20,
        1 << 15,
        31,
    )
    assert (f1_b.rounds, f1_b.train_rows, f1_b.test_rows, f1_b.key_seed) == (
        2,
        1 << 20,
        1 << 15,
        32,
    )
    assert (f1_c.rounds, f1_c.train_rows, f1_c.test_rows, f1_c.key_seed) == (
        2,
        1 << 22,
        1 << 15,
        31,
    )
    assert (f1_r_a.mode, f1_r_a.rounds, f1_r_a.key_seed) == ("f1_r_a", 3, 31)
    assert (f1_r_b.mode, f1_r_b.rounds, f1_r_b.key_seed) == ("f1_r_b", 3, 32)
    assert serializable_config(f1_c)["mode"] == "f1_c"

    with pytest.raises(ValueError, match="formal f1_a"):
        replace(f1_a, train_rows=(1 << 20) - 1)
    with pytest.raises(ValueError, match="key_seed must be 31 or 32"):
        DesOutputPredictionDataConfig.f1_r(key_seed=33)


def test_numpy_des_matches_official_vector_and_scalar_round_reductions() -> None:
    plaintexts = np.asarray(
        [
            0,
            1,
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
            0xFFFFFFFFFFFFFFFF,
        ],
        dtype=np.uint64,
    )
    keys = (
        0,
        1,
        0x133457799BBCDFF1,
        0x0F1571C947D9E859,
        0xFFFFFFFFFFFFFFFF,
    )

    for rounds in (1, 2, 3, 16):
        for key in keys:
            expected = np.asarray(
                [Des(rounds=rounds, key=key).encrypt(int(word)) for word in plaintexts],
                dtype=np.uint64,
            )
            np.testing.assert_array_equal(
                encrypt_des_numpy(plaintexts, rounds=rounds, key=key),
                expected,
            )

    official = encrypt_des_numpy(
        np.asarray([0x0123456789ABCDEF], dtype=np.uint64),
        rounds=16,
        key=0x133457799BBCDFF1,
    )
    assert int(official[0]) == 0x85E813540F0AB405


def test_des_output_data_are_true_outputs_half_preserving_and_resumable(
    tmp_path: Path,
) -> None:
    config = DesOutputPredictionDataConfig(
        train_rows=19,
        test_rows=13,
        chunk_rows=7,
    )
    events: list[tuple[str, dict[str, object]]] = []
    data = prepare_des_output_prediction_data(
        config,
        tmp_path,
        progress=lambda event, payload: events.append((event, payload)),
    )
    checks = validate_des_output_prediction_data(config, data)
    reused = prepare_des_output_prediction_data(config, tmp_path)

    assert all(checks.values()), checks
    assert data["features"].shape == (32, 64)
    assert data["full_targets"].shape == (32, 64)
    assert data["train_features"].shape == (19, 64)
    assert data["test_targets"].shape == (13, 64)
    assert data["metadata"]["cipher"] == "DES"
    assert data["metadata"]["serialized_half_order"] == [
        "plaintext_msw32",
        "plaintext_lsw32",
    ]
    assert data["metadata"]["sample_classification"] is False
    assert data["metadata"]["target"] == (
        "64_msb_first_true_round_reduced_des_ciphertext_bits"
    )
    assert len(data["metadata"]["secret_key_hex"]) == 16
    assert events[0][1]["split"] == "test"
    assert events[-1][1]["split"] == "train"
    assert events[-1][1]["completed_train_rows"] == 19
    assert reused["cache_reused"] is True

    for index, plaintext in enumerate(data["plaintexts"]):
        feature_word = _bits_to_word(data["features"][index])
        target_word = _bits_to_word(data["full_targets"][index])
        assert feature_word == int(plaintext)
        assert _bits_to_word(data["features"][index, :32]) == int(plaintext) >> 32
        assert _bits_to_word(data["features"][index, 32:]) == (
            int(plaintext) & 0xFFFFFFFF
        )
        assert target_word == Des(
            rounds=config.rounds,
            key=int(data["secret_key"]),
        ).encrypt(int(plaintext))


def test_des_key_seed_changes_only_key_and_true_targets(tmp_path: Path) -> None:
    config = DesOutputPredictionDataConfig(
        train_rows=12,
        test_rows=8,
        chunk_rows=5,
    )
    original = prepare_des_output_prediction_data(config, tmp_path / "key31")
    new_key = prepare_des_output_prediction_data(
        replace(config, key_seed=32),
        tmp_path / "key32",
    )

    np.testing.assert_array_equal(original["plaintexts"], new_key["plaintexts"])
    np.testing.assert_array_equal(original["features"], new_key["features"])
    assert original["secret_key"] != new_key["secret_key"]
    assert not np.array_equal(original["full_targets"], new_key["full_targets"])


def test_des_training_prefix_and_test_set_survive_scale_extension(
    tmp_path: Path,
) -> None:
    small_config = DesOutputPredictionDataConfig(
        train_rows=17,
        test_rows=11,
        chunk_rows=6,
    )
    large_config = replace(
        small_config,
        run_id="extended",
        train_rows=29,
    )
    small = prepare_des_output_prediction_data(small_config, tmp_path / "small")
    large = prepare_des_output_prediction_data(large_config, tmp_path / "large")

    np.testing.assert_array_equal(
        small["plaintexts"][: small_config.train_rows],
        large["plaintexts"][: small_config.train_rows],
    )
    np.testing.assert_array_equal(
        small["plaintexts"][small_config.train_rows :],
        large["plaintexts"][large_config.train_rows :],
    )
    small_test = set(small["plaintexts"][small_config.train_rows :])
    large_train = set(large["plaintexts"][: large_config.train_rows])
    assert small_test.isdisjoint(large_train)


def test_des_cache_rejects_parameter_mismatch_and_partial_bundle(
    tmp_path: Path,
) -> None:
    config = DesOutputPredictionDataConfig(
        train_rows=8,
        test_rows=8,
        chunk_rows=4,
    )
    prepare_des_output_prediction_data(config, tmp_path / "mismatch")

    with pytest.raises(ValueError, match="parameters do not match"):
        prepare_des_output_prediction_data(
            replace(config, rounds=3),
            tmp_path / "mismatch",
        )

    partial_root = tmp_path / "partial" / "data"
    partial_root.mkdir(parents=True)
    np.save(partial_root / "plaintexts.npy", np.zeros(16, dtype=np.uint64))
    with pytest.raises(ValueError, match="arrays are incomplete"):
        prepare_des_output_prediction_data(config, tmp_path / "partial")


def test_partial_des_test_reserve_resumes_to_identical_complete_cache(
    tmp_path: Path,
) -> None:
    config = DesOutputPredictionDataConfig(
        train_rows=17,
        test_rows=13,
        chunk_rows=5,
    )

    class PlannedInterruption(RuntimeError):
        pass

    def stop_after_first_chunk(event: str, payload: dict[str, object]) -> None:
        assert event == "cache_chunk"
        assert payload["split"] == "test"
        raise PlannedInterruption

    with pytest.raises(PlannedInterruption):
        prepare_des_output_prediction_data(
            config,
            tmp_path / "resumed",
            progress=stop_after_first_chunk,
        )

    resumed = prepare_des_output_prediction_data(config, tmp_path / "resumed")
    uninterrupted = prepare_des_output_prediction_data(
        config,
        tmp_path / "uninterrupted",
    )

    assert resumed["metadata"]["status"] == "complete"
    np.testing.assert_array_equal(resumed["plaintexts"], uninterrupted["plaintexts"])
    np.testing.assert_array_equal(resumed["features"], uninterrupted["features"])
    np.testing.assert_array_equal(
        resumed["full_targets"],
        uninterrupted["full_targets"],
    )
