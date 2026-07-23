from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_data import (
    Speck32OutputPredictionDataConfig,
    prepare_speck32_output_prediction_data,
    serializable_config,
    validate_speck32_output_prediction_data,
)


def test_arx1_formal_data_protocols_are_frozen() -> None:
    arx1_a = Speck32OutputPredictionDataConfig.arx1_a()
    arx1_b = Speck32OutputPredictionDataConfig.arx1_b()
    arx1_c = Speck32OutputPredictionDataConfig.arx1_c()

    assert (arx1_a.train_rows, arx1_a.test_rows, arx1_a.key_seed) == (
        1 << 20,
        1 << 15,
        21,
    )
    assert (arx1_b.train_rows, arx1_b.test_rows, arx1_b.key_seed) == (
        1 << 20,
        1 << 15,
        22,
    )
    assert (arx1_c.train_rows, arx1_c.test_rows, arx1_c.key_seed) == (
        1 << 22,
        1 << 15,
        21,
    )
    assert serializable_config(arx1_c)["mode"] == "arx1_c"
    with pytest.raises(ValueError, match="formal arx1_a"):
        replace(arx1_a, train_rows=(1 << 20) - 1)


def test_speck32_output_data_are_true_outputs_word_preserving_and_resumable(
    tmp_path: Path,
) -> None:
    config = Speck32OutputPredictionDataConfig(
        train_rows=19,
        test_rows=13,
        chunk_rows=7,
    )
    events: list[tuple[str, dict[str, object]]] = []
    data = prepare_speck32_output_prediction_data(
        config,
        tmp_path,
        progress=lambda event, payload: events.append((event, payload)),
    )
    checks = validate_speck32_output_prediction_data(config, data)
    reused = prepare_speck32_output_prediction_data(config, tmp_path)

    assert all(checks.values()), checks
    assert data["features"].shape == (32, 32)
    assert data["full_targets"].shape == (32, 32)
    assert data["train_features"].shape == (19, 32)
    assert data["test_targets"].shape == (13, 32)
    assert data["metadata"]["cipher"] == "SPECK32/64"
    assert data["metadata"]["word_order"] == ["x_msw", "y_lsw"]
    assert data["metadata"]["sample_classification"] is False
    assert data["metadata"]["target"] == ("32_msb_first_true_speck32_ciphertext_bits")
    assert len(data["metadata"]["secret_key_hex"]) == 16
    assert events[0][1]["split"] == "test"
    assert events[-1][1]["split"] == "train"
    assert events[-1][1]["completed_train_rows"] == 19
    assert reused["cache_reused"] is True


def test_key_seed_changes_only_key_and_true_targets(tmp_path: Path) -> None:
    config = Speck32OutputPredictionDataConfig(
        train_rows=12,
        test_rows=8,
        chunk_rows=5,
    )
    original = prepare_speck32_output_prediction_data(config, tmp_path / "key21")
    new_key = prepare_speck32_output_prediction_data(
        replace(config, key_seed=22),
        tmp_path / "key22",
    )

    np.testing.assert_array_equal(original["plaintexts"], new_key["plaintexts"])
    np.testing.assert_array_equal(original["features"], new_key["features"])
    assert original["secret_key"] != new_key["secret_key"]
    assert not np.array_equal(original["full_targets"], new_key["full_targets"])


def test_training_prefix_and_test_set_survive_scale_extension(tmp_path: Path) -> None:
    small_config = Speck32OutputPredictionDataConfig(
        train_rows=17,
        test_rows=11,
        chunk_rows=6,
    )
    large_config = replace(
        small_config,
        run_id="extended",
        train_rows=29,
    )
    small = prepare_speck32_output_prediction_data(
        small_config,
        tmp_path / "small",
    )
    large = prepare_speck32_output_prediction_data(
        large_config,
        tmp_path / "large",
    )

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


def test_cache_rejects_parameter_mismatch_and_partial_bundle(tmp_path: Path) -> None:
    config = Speck32OutputPredictionDataConfig(
        train_rows=8,
        test_rows=8,
        chunk_rows=4,
    )
    prepare_speck32_output_prediction_data(config, tmp_path / "mismatch")

    with pytest.raises(ValueError, match="parameters do not match"):
        prepare_speck32_output_prediction_data(
            replace(config, rounds=4),
            tmp_path / "mismatch",
        )

    partial_root = tmp_path / "partial" / "data"
    partial_root.mkdir(parents=True)
    np.save(partial_root / "plaintexts.npy", np.zeros(16, dtype=np.uint32))
    with pytest.raises(ValueError, match="arrays are incomplete"):
        prepare_speck32_output_prediction_data(config, tmp_path / "partial")


def test_partial_test_reserve_resumes_to_identical_complete_cache(
    tmp_path: Path,
) -> None:
    config = Speck32OutputPredictionDataConfig(
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
        prepare_speck32_output_prediction_data(
            config,
            tmp_path / "resumed",
            progress=stop_after_first_chunk,
        )

    resumed = prepare_speck32_output_prediction_data(
        config,
        tmp_path / "resumed",
    )
    uninterrupted = prepare_speck32_output_prediction_data(
        config,
        tmp_path / "uninterrupted",
    )

    assert resumed["metadata"]["status"] == "complete"
    np.testing.assert_array_equal(resumed["plaintexts"], uninterrupted["plaintexts"])
    np.testing.assert_array_equal(resumed["features"], uninterrupted["features"])
    np.testing.assert_array_equal(
        resumed["full_targets"], uninterrupted["full_targets"]
    )
