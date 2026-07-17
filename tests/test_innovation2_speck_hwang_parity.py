import json

import numpy as np
import pytest

from blockcipher_nd.ciphers.arx.speck import Speck32_64
from blockcipher_nd.tasks.innovation2 import speck_hwang_parity as parity
from blockcipher_nd.tasks.innovation2.speck_hwang_parity import (
    HWANG_SPECK_R6_BASIS_BITS,
    HWANG_SPECK_R7_BASIS_BITS,
    SPECK32_ACTIVE_BITS,
    SPECK32_FIXED_MASK,
    SpeckParityCacheConfig,
    assignments_to_plaintexts,
    assignments_to_plaintexts_torch,
    chunked_speck_parity_word,
    encrypt_speck32_numpy,
    encrypt_speck32_torch,
    exhaustive_scalar_speck_parity_word,
    hwang_speck_basis_masks,
    run_cached_speck_parity_rows,
)


def test_e25_freezes_wang_speck_input_and_hwang_output_masks() -> None:
    assert len(SPECK32_ACTIVE_BITS) == 30
    assert set(SPECK32_ACTIVE_BITS) == set(range(32)) - {5, 6}
    assert SPECK32_FIXED_MASK == 0x60
    assert HWANG_SPECK_R6_BASIS_BITS == (
        (2, 18),
        (3, 19),
        (4, 20),
        (5, 21),
        (6, 22),
        (7, 23),
        (8, 24),
        (9, 25),
        (16,),
    )
    assert HWANG_SPECK_R7_BASIS_BITS == ((2, 9, 16, 18, 25),)
    assert hwang_speck_basis_masks(7) == (
        (1 << 2) | (1 << 9) | (1 << 16) | (1 << 18) | (1 << 25),
    )


def test_e25_project_speck_matches_official_vector() -> None:
    cipher = Speck32_64(rounds=22, key=0x1918111009080100)
    assert cipher.encrypt(0x6574694C) == 0xA86842F2


def test_e25_numpy_batch_matches_scalar_random_plaintexts() -> None:
    rng = np.random.default_rng(2501)
    plaintexts = rng.integers(0, 1 << 32, size=257, dtype=np.uint32)
    for rounds in (1, 6, 7, 22):
        key = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        expected = np.asarray(
            [Speck32_64(rounds=rounds, key=key).encrypt(int(p)) for p in plaintexts],
            dtype=np.uint32,
        )
        actual = encrypt_speck32_numpy(plaintexts, rounds=rounds, key=key)
        np.testing.assert_array_equal(actual, expected)


def test_e25_torch_batch_matches_numpy_random_plaintexts() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2502)
    plaintexts = rng.integers(0, 1 << 32, size=257, dtype=np.uint32)
    torch_plaintexts = torch.from_numpy(plaintexts.astype(np.int64))
    for rounds in (1, 6, 7, 22):
        key = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        expected = encrypt_speck32_numpy(plaintexts, rounds=rounds, key=key)
        actual = encrypt_speck32_torch(
            torch_plaintexts,
            rounds=rounds,
            key=key,
        ).numpy().astype(np.uint32)
        np.testing.assert_array_equal(actual, expected)


def test_e25_exact_assignment_mapping_keeps_only_bits_five_and_six_fixed() -> None:
    assignments = np.asarray([0, 1, (1 << 30) - 1], dtype=np.uint32)
    plaintexts = assignments_to_plaintexts(
        assignments,
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=0x40,
    )
    assert plaintexts.tolist() == [0x40, 0x41, 0xFFFFFFDF]
    assert all((int(value) & 0x60) == 0x40 for value in plaintexts)


def test_e25_torch_exact_assignment_mapping_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    assignments = np.asarray([0, 1, 31, 32, (1 << 30) - 1], dtype=np.uint32)
    expected = assignments_to_plaintexts(
        assignments,
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=0x20,
    )
    actual = assignments_to_plaintexts_torch(
        torch.from_numpy(assignments.astype(np.int64)),
        active_bits=SPECK32_ACTIVE_BITS,
        fixed_plaintext=0x20,
    )
    np.testing.assert_array_equal(actual.numpy().astype(np.uint32), expected)


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1000])
def test_e25_chunked_parity_equals_scalar_exhaustive(chunk_size: int) -> None:
    active_bits = (0, 2, 4, 7, 9, 12)
    params = {
        "rounds": 7,
        "key": 0x1918111009080100,
        "active_bits": active_bits,
        "fixed_plaintext": 0xA5A50020,
    }
    expected = exhaustive_scalar_speck_parity_word(**params)
    actual = chunked_speck_parity_word(**params, chunk_size=chunk_size)
    assert actual == expected


@pytest.mark.parametrize("chunk_size", [7, 64, 1000])
def test_e25_torch_chunked_parity_matches_numpy(chunk_size: int) -> None:
    pytest.importorskip("torch")
    params = {
        "rounds": 7,
        "key": 0x1918111009080100,
        "active_bits": (0, 2, 4, 7, 9, 12),
        "fixed_plaintext": 0xA5A50020,
        "chunk_size": chunk_size,
    }
    expected = chunked_speck_parity_word(**params)
    actual = chunked_speck_parity_word(
        **params,
        backend="torch_int32",
        device="cpu",
    )
    assert actual == expected


def test_e25_cache_resumes_only_missing_rows(monkeypatch, tmp_path) -> None:
    config = SpeckParityCacheConfig(
        run_id="cache",
        rounds=(6, 7),
        keys=(1, 2, 3),
        active_bits=(0, 1, 2, 3),
        fixed_plaintext=0x20,
        chunk_size=3,
    )
    calls: list[tuple[int, int]] = []
    real = parity.chunked_speck_parity_word

    def recording_parity(**kwargs):
        calls.append((kwargs["rounds"], kwargs["key"]))
        return real(**kwargs)

    monkeypatch.setattr(parity, "chunked_speck_parity_word", recording_parity)
    first = run_cached_speck_parity_rows(config, cache_root=tmp_path / "cache")
    second = run_cached_speck_parity_rows(config, cache_root=tmp_path / "cache")

    assert first["cache_status"] == "created"
    assert first["rows_generated"] == 6
    assert second["cache_status"] == "resumed"
    assert second["rows_generated"] == 0
    assert calls == [(6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3)]
    np.testing.assert_array_equal(first["parity_rows"], second["parity_rows"])
    assert second["completed"].all()
    metadata = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    assert metadata["assignments_per_key"] == 16
    assert metadata["backend"] == "numpy_uint32"
    assert metadata["device"] == "cpu"


def test_e25_cache_rejects_parameter_mismatch(tmp_path) -> None:
    first = SpeckParityCacheConfig(
        run_id="cache",
        rounds=(6,),
        keys=(1,),
        active_bits=(0, 1),
        fixed_plaintext=0x20,
        chunk_size=2,
    )
    changed = SpeckParityCacheConfig(
        run_id="cache",
        rounds=(7,),
        keys=(1,),
        active_bits=(0, 1),
        fixed_plaintext=0x20,
        chunk_size=2,
    )
    run_cached_speck_parity_rows(first, cache_root=tmp_path / "cache")

    with pytest.raises(ValueError, match="metadata does not match"):
        run_cached_speck_parity_rows(changed, cache_root=tmp_path / "cache")
