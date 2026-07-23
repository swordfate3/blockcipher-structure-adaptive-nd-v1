from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2.gift64_output_prediction_data import (
    Gift64OutputPredictionDataConfig,
    prepare_gift64_fresh_data,
    prepare_gift64_source_data,
    serializable_config,
    validate_gift64_fresh_data,
    validate_gift64_source_data,
)


def test_gift64_output_data_config_freezes_formal_protocol() -> None:
    config = Gift64OutputPredictionDataConfig.formal()

    assert config.rounds == 3
    assert config.seed == 11
    assert config.key_seed == 11
    assert config.train_rows == 1 << 17
    assert config.discovery_rows == 1 << 16
    assert config.fresh_rows == 1 << 16
    assert config.chunk_rows == 4096
    assert serializable_config(config)["mode"] == "formal"

    with pytest.raises(ValueError, match="formal GX1 data protocol is frozen"):
        replace(config, key_seed=12)


def test_gift64_source_data_are_true_outputs_and_disk_resumable(
    tmp_path: Path,
) -> None:
    config = Gift64OutputPredictionDataConfig(
        train_rows=17,
        discovery_rows=11,
        fresh_rows=13,
        chunk_rows=7,
    )
    events: list[tuple[str, dict[str, object]]] = []
    data = prepare_gift64_source_data(
        config,
        tmp_path,
        progress=lambda event, payload: events.append((event, payload)),
    )
    checks = validate_gift64_source_data(config, data)
    reused = prepare_gift64_source_data(config, tmp_path)

    assert all(checks.values()), checks
    assert data["features"].shape == (28, 64)
    assert data["full_targets"].shape == (28, 64)
    assert data["train_features"].shape == (17, 64)
    assert data["discovery_targets"].shape == (11, 64)
    assert data["metadata"]["cipher"] == "GIFT-64"
    assert len(data["metadata"]["secret_key_hex"]) == 32
    assert events[-1][1]["completed_rows"] == 28
    assert reused["cache_reused"] is True


def test_gift64_key_seed_changes_only_key_and_targets(tmp_path: Path) -> None:
    config = Gift64OutputPredictionDataConfig(
        train_rows=8,
        discovery_rows=8,
        fresh_rows=8,
        chunk_rows=4,
    )
    original = prepare_gift64_source_data(config, tmp_path / "original")
    new_key = prepare_gift64_source_data(
        replace(config, key_seed=12), tmp_path / "new_key"
    )

    assert np.array_equal(original["plaintexts"], new_key["plaintexts"])
    assert np.array_equal(original["features"], new_key["features"])
    assert original["secret_key"] != new_key["secret_key"]
    assert not np.array_equal(original["full_targets"], new_key["full_targets"])


def test_gift64_source_cache_rejects_parameter_mismatch(tmp_path: Path) -> None:
    config = Gift64OutputPredictionDataConfig(
        train_rows=8,
        discovery_rows=8,
        fresh_rows=8,
        chunk_rows=4,
    )
    prepare_gift64_source_data(config, tmp_path)

    with pytest.raises(ValueError, match="parameters do not match"):
        prepare_gift64_source_data(replace(config, rounds=4), tmp_path)


def test_gift64_fresh_data_are_post_freeze_and_disjoint(tmp_path: Path) -> None:
    config = Gift64OutputPredictionDataConfig(
        train_rows=12,
        discovery_rows=10,
        fresh_rows=9,
        chunk_rows=5,
    )
    source = prepare_gift64_source_data(config, tmp_path)
    candidate_sha256 = "a" * 64
    fresh = prepare_gift64_fresh_data(
        config,
        source,
        tmp_path,
        candidate_sha256=candidate_sha256,
    )
    checks = validate_gift64_fresh_data(
        config,
        source,
        fresh,
        candidate_sha256=candidate_sha256,
    )

    assert all(checks.values()), checks
    assert set(source["plaintexts"]).isdisjoint(set(fresh["plaintexts"]))
    assert fresh["metadata"]["candidate_sha256"] == candidate_sha256
    assert fresh["metadata"]["split"] == "fresh_after_candidate_freeze"

    with pytest.raises(ValueError, match="lowercase SHA256"):
        prepare_gift64_fresh_data(
            config,
            source,
            tmp_path / "bad",
            candidate_sha256="not-a-hash",
        )
