from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.ciphers.arx.speck import Speck32_64


SPECK32_ACTIVE_BITS = tuple(bit for bit in range(32) if bit not in {5, 6})
SPECK32_FIXED_MASK = (1 << 5) | (1 << 6)
HWANG_SPECK_R6_BASIS_BITS = (
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
HWANG_SPECK_R7_BASIS_BITS = ((2, 9, 16, 18, 25),)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpeckParityCacheConfig:
    run_id: str
    rounds: tuple[int, ...]
    keys: tuple[int, ...]
    active_bits: tuple[int, ...]
    fixed_plaintext: int
    chunk_size: int
    backend: str = "numpy_uint32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.rounds or any(rounds <= 0 for rounds in self.rounds):
            raise ValueError("rounds must contain positive values")
        if len(set(self.rounds)) != len(self.rounds):
            raise ValueError("rounds must be unique")
        if not self.keys or any(key < 0 or key >= (1 << 64) for key in self.keys):
            raise ValueError("keys must contain 64-bit values")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys must be unique")
        _validate_structure(self.active_bits, self.fixed_plaintext)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.backend not in {"numpy_uint32", "torch_int32"}:
            raise ValueError("backend must be numpy_uint32 or torch_int32")
        if self.backend == "numpy_uint32" and self.device != "cpu":
            raise ValueError("numpy_uint32 backend requires device=cpu")
        if not self.device:
            raise ValueError("device must be non-empty")

    @property
    def assignments(self) -> int:
        return 1 << len(self.active_bits)


def hwang_speck_basis_masks(rounds: int) -> tuple[int, ...]:
    if rounds == 6:
        basis_bits = HWANG_SPECK_R6_BASIS_BITS
    elif rounds == 7:
        basis_bits = HWANG_SPECK_R7_BASIS_BITS
    else:
        raise ValueError("Hwang Table 7 only defines 6- and 7-round bases")
    return tuple(sum(1 << bit for bit in bits) for bits in basis_bits)


def encrypt_speck32_numpy(
    plaintexts: np.ndarray,
    *,
    rounds: int,
    key: int,
) -> np.ndarray:
    values = np.asarray(plaintexts)
    if values.dtype != np.uint32:
        raise ValueError("plaintexts must have dtype uint32")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if key < 0 or key >= (1 << 64):
        raise ValueError("key must fit in 64 bits")

    x = (values >> np.uint32(16)) & np.uint32(0xFFFF)
    y = values & np.uint32(0xFFFF)
    round_keys = Speck32_64(rounds=rounds, key=key)._round_keys()
    for round_key in round_keys:
        x = ((x >> np.uint32(7)) | (x << np.uint32(9))) & np.uint32(0xFFFF)
        x = (x + y) & np.uint32(0xFFFF)
        x ^= np.uint32(round_key)
        y = ((y << np.uint32(2)) | (y >> np.uint32(14))) & np.uint32(0xFFFF)
        y ^= x
    return (x << np.uint32(16)) | y


def encrypt_speck32_torch(plaintexts, *, rounds: int, key: int):
    import torch

    if plaintexts.dtype != torch.int64:
        raise ValueError("torch plaintexts must have dtype int64")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if key < 0 or key >= (1 << 64):
        raise ValueError("key must fit in 64 bits")

    x = ((plaintexts >> 16) & 0xFFFF).to(torch.int32)
    y = (plaintexts & 0xFFFF).to(torch.int32)
    round_keys = Speck32_64(rounds=rounds, key=key)._round_keys()
    for round_key in round_keys:
        x = ((x >> 7) | (x << 9)) & 0xFFFF
        x = (x + y) & 0xFFFF
        x ^= round_key
        y = ((y << 2) | (y >> 14)) & 0xFFFF
        y ^= x
    return (x.to(torch.int64) << 16) | y.to(torch.int64)


def assignments_to_plaintexts(
    assignments: np.ndarray,
    *,
    active_bits: tuple[int, ...],
    fixed_plaintext: int,
) -> np.ndarray:
    _validate_structure(active_bits, fixed_plaintext)
    values = np.asarray(assignments)
    if values.dtype != np.uint32:
        raise ValueError("assignments must have dtype uint32")
    if values.size and int(values.max()) >= (1 << len(active_bits)):
        raise ValueError("assignment exceeds active-bit width")

    fixed_mask = _fixed_mask(active_bits)
    fixed = np.uint32(fixed_plaintext & fixed_mask)
    fixed_bits = _two_fixed_bits(active_bits)
    if fixed_bits is not None:
        first, second = fixed_bits
        low_mask = (1 << first) - 1
        middle_width = second - first - 1
        middle_mask = (1 << middle_width) - 1
        low = values & np.uint32(low_mask)
        middle = (
            ((values >> np.uint32(first)) & np.uint32(middle_mask))
            << np.uint32(first + 1)
        )
        upper = (values >> np.uint32(second - 1)) << np.uint32(second + 1)
        return low | middle | upper | fixed

    plaintexts = np.full(values.shape, fixed, dtype=np.uint32)
    for assignment_bit, plaintext_bit in enumerate(active_bits):
        plaintexts |= (
            ((values >> np.uint32(assignment_bit)) & np.uint32(1))
            << np.uint32(plaintext_bit)
        )
    return plaintexts


def assignments_to_plaintexts_torch(
    assignments,
    *,
    active_bits: tuple[int, ...],
    fixed_plaintext: int,
):
    import torch

    _validate_structure(active_bits, fixed_plaintext)
    if assignments.dtype != torch.int64:
        raise ValueError("torch assignments must have dtype int64")
    if assignments.numel() and int(assignments.max().item()) >= (
        1 << len(active_bits)
    ):
        raise ValueError("assignment exceeds active-bit width")

    fixed = fixed_plaintext & _fixed_mask(active_bits)
    fixed_bits = _two_fixed_bits(active_bits)
    if fixed_bits is not None:
        first, second = fixed_bits
        low_mask = (1 << first) - 1
        middle_width = second - first - 1
        middle_mask = (1 << middle_width) - 1
        low = assignments & low_mask
        middle = ((assignments >> first) & middle_mask) << (first + 1)
        upper = (assignments >> (second - 1)) << (second + 1)
        return low | middle | upper | fixed

    plaintexts = torch.full_like(assignments, fixed)
    for assignment_bit, plaintext_bit in enumerate(active_bits):
        plaintexts |= ((assignments >> assignment_bit) & 1) << plaintext_bit
    return plaintexts


def chunked_speck_parity_word(
    *,
    rounds: int,
    key: int,
    active_bits: tuple[int, ...],
    fixed_plaintext: int,
    chunk_size: int,
    backend: str = "numpy_uint32",
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> int:
    _validate_structure(active_bits, fixed_plaintext)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if backend not in {"numpy_uint32", "torch_int32"}:
        raise ValueError("backend must be numpy_uint32 or torch_int32")
    if backend == "numpy_uint32" and device != "cpu":
        raise ValueError("numpy_uint32 backend requires device=cpu")
    assignments = 1 << len(active_bits)
    parity = np.uint32(0)
    for start in range(0, assignments, chunk_size):
        stop = min(start + chunk_size, assignments)
        if backend == "numpy_uint32":
            assignment_values = np.arange(start, stop, dtype=np.uint32)
            plaintexts = assignments_to_plaintexts(
                assignment_values,
                active_bits=active_bits,
                fixed_plaintext=fixed_plaintext,
            )
            ciphertexts = encrypt_speck32_numpy(
                plaintexts,
                rounds=rounds,
                key=key,
            )
        else:
            import torch

            assignment_values = torch.arange(
                start,
                stop,
                dtype=torch.int64,
                device=device,
            )
            plaintexts = assignments_to_plaintexts_torch(
                assignment_values,
                active_bits=active_bits,
                fixed_plaintext=fixed_plaintext,
            )
            torch_ciphertexts = encrypt_speck32_torch(
                plaintexts,
                rounds=rounds,
                key=key,
            )
            ciphertexts = torch_ciphertexts.detach().cpu().numpy().astype(
                np.uint32,
                copy=False,
            )
        parity ^= np.bitwise_xor.reduce(ciphertexts, initial=np.uint32(0))
        _emit(
            progress_callback,
            "speck_parity_chunk_done",
            {
                "rounds": rounds,
                "key": f"0x{key:016X}",
                "start": start,
                "stop": stop,
                "assignments": assignments,
                "backend": backend,
                "device": device,
            },
        )
    return int(parity)


def exhaustive_scalar_speck_parity_word(
    *,
    rounds: int,
    key: int,
    active_bits: tuple[int, ...],
    fixed_plaintext: int,
) -> int:
    _validate_structure(active_bits, fixed_plaintext)
    if len(active_bits) > 20:
        raise ValueError("scalar exhaustive parity is limited to at most 20 active bits")
    parity = 0
    cipher = Speck32_64(rounds=rounds, key=key)
    for assignment in range(1 << len(active_bits)):
        plaintext = fixed_plaintext & _fixed_mask(active_bits)
        for assignment_bit, plaintext_bit in enumerate(active_bits):
            plaintext |= ((assignment >> assignment_bit) & 1) << plaintext_bit
        parity ^= cipher.encrypt(plaintext)
    return parity


def run_cached_speck_parity_rows(
    config: SpeckParityCacheConfig,
    *,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / "metadata.json"
    rows_path = cache_root / "parity_rows.npy"
    completed_path = cache_root / "completed.npy"
    expected_metadata = _cache_metadata(config)
    shape = (len(config.rounds), len(config.keys))

    if metadata_path.exists() or rows_path.exists() or completed_path.exists():
        if not (metadata_path.exists() and rows_path.exists() and completed_path.exists()):
            raise ValueError("partial SPECK parity cache is missing required files")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != expected_metadata:
            raise ValueError("SPECK parity cache metadata does not match config")
        parity_rows = np.load(rows_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if parity_rows.shape != shape or parity_rows.dtype != np.uint32:
            raise ValueError("cached parity_rows.npy has wrong shape or dtype")
        if completed.shape != shape or completed.dtype != np.bool_:
            raise ValueError("cached completed.npy has wrong shape or dtype")
        cache_status = "resumed"
    else:
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        parity_rows = np.lib.format.open_memmap(
            rows_path,
            mode="w+",
            dtype=np.uint32,
            shape=shape,
        )
        completed = np.lib.format.open_memmap(
            completed_path,
            mode="w+",
            dtype=np.bool_,
            shape=shape,
        )
        parity_rows[:] = 0
        completed[:] = False
        parity_rows.flush()
        completed.flush()
        cache_status = "created"

    rows_generated = 0
    for round_index, rounds in enumerate(config.rounds):
        for key_index, key in enumerate(config.keys):
            if bool(completed[round_index, key_index]):
                continue
            _emit(
                progress_callback,
                "speck_parity_row_start",
                {
                    "rounds": rounds,
                    "round_index": round_index,
                    "key_index": key_index,
                    "keys": len(config.keys),
                },
            )
            parity_rows[round_index, key_index] = chunked_speck_parity_word(
                rounds=rounds,
                key=key,
                active_bits=config.active_bits,
                fixed_plaintext=config.fixed_plaintext,
                chunk_size=config.chunk_size,
                backend=config.backend,
                device=config.device,
                progress_callback=progress_callback,
            )
            parity_rows.flush()
            completed[round_index, key_index] = True
            completed.flush()
            rows_generated += 1
            _emit(
                progress_callback,
                "speck_parity_row_done",
                {
                    "rounds": rounds,
                    "round_index": round_index,
                    "key_index": key_index,
                    "keys": len(config.keys),
                },
            )
    return {
        "parity_rows": np.asarray(parity_rows).copy(),
        "completed": np.asarray(completed).copy(),
        "metadata": expected_metadata,
        "cache_status": cache_status,
        "rows_generated": rows_generated,
    }


def _cache_metadata(config: SpeckParityCacheConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "cipher": "SPECK32/64",
        "rounds": list(config.rounds),
        "keys": [f"0x{key:016X}" for key in config.keys],
        "active_bits": list(config.active_bits),
        "active_bit_mask": f"0x{sum(1 << bit for bit in config.active_bits):08X}",
        "fixed_plaintext": f"0x{config.fixed_plaintext:08X}",
        "fixed_mask": f"0x{_fixed_mask(config.active_bits):08X}",
        "assignments_per_key": config.assignments,
        "chunk_size": config.chunk_size,
        "output_bit_order": "LSB-first",
        "backend": config.backend,
        "device": config.device,
    }


def _validate_structure(active_bits: tuple[int, ...], fixed_plaintext: int) -> None:
    if tuple(sorted(active_bits)) != active_bits:
        raise ValueError("active_bits must be sorted")
    if len(set(active_bits)) != len(active_bits):
        raise ValueError("active_bits must be unique")
    if any(bit < 0 or bit >= 32 for bit in active_bits):
        raise ValueError("active_bits must fit in a 32-bit plaintext")
    if fixed_plaintext < 0 or fixed_plaintext >= (1 << 32):
        raise ValueError("fixed_plaintext must fit in 32 bits")


def _fixed_mask(active_bits: tuple[int, ...]) -> int:
    active_mask = sum(1 << bit for bit in active_bits)
    return ((1 << 32) - 1) ^ active_mask


def _two_fixed_bits(active_bits: tuple[int, ...]) -> tuple[int, int] | None:
    fixed_bits = tuple(bit for bit in range(32) if bit not in set(active_bits))
    if len(fixed_bits) != 2:
        return None
    return fixed_bits


def _emit(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)
