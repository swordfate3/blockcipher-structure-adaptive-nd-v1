from __future__ import annotations

import numpy as np

from blockcipher_nd.ciphers.feistel.des import (
    DES_E,
    DES_FP,
    DES_IP,
    DES_P,
    DES_SBOXES,
    Des,
)


_DES_SBOX_ARRAYS = tuple(np.asarray(sbox, dtype=np.uint8) for sbox in DES_SBOXES)


def encrypt_des_numpy(
    plaintexts: np.ndarray,
    *,
    rounds: int,
    key: int,
) -> np.ndarray:
    values = np.asarray(plaintexts, dtype=np.uint64)
    if values.ndim != 1:
        raise ValueError("DES plaintexts must be a one-dimensional uint64 array")
    cipher = Des(rounds=rounds, key=key)
    round_keys = cipher._round_keys()
    permuted = _permute_numpy(values, DES_IP, 64)
    left = (permuted >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
    right = permuted & np.uint64(0xFFFFFFFF)
    for round_key in round_keys:
        left, right = right, left ^ _des_f_numpy(right, round_key)
    preoutput = (right << np.uint64(32)) | left
    return _permute_numpy(preoutput, DES_FP, 64)


def _des_f_numpy(right: np.ndarray, round_key: int) -> np.ndarray:
    expanded = _permute_numpy(right, DES_E, 32) ^ np.uint64(round_key)
    substituted = np.zeros(len(right), dtype=np.uint64)
    for index, sbox in enumerate(_DES_SBOX_ARRAYS):
        chunk = (expanded >> np.uint64(42 - 6 * index)) & np.uint64(0x3F)
        row = ((chunk & np.uint64(0x20)) >> np.uint64(4)) | (chunk & np.uint64(0x01))
        column = (chunk >> np.uint64(1)) & np.uint64(0x0F)
        lookup = (row * np.uint64(16) + column).astype(np.intp)
        substituted = (substituted << np.uint64(4)) | sbox[lookup].astype(np.uint64)
    return _permute_numpy(substituted, DES_P, 32)


def _permute_numpy(
    values: np.ndarray,
    table: tuple[int, ...],
    input_bits: int,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint64)
    output = np.zeros(source.shape, dtype=np.uint64)
    for position in table:
        output = (output << np.uint64(1)) | (
            (source >> np.uint64(input_bits - position)) & np.uint64(1)
        )
    return output


__all__ = ["encrypt_des_numpy"]
