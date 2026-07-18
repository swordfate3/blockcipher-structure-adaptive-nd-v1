from __future__ import annotations

import numpy as np


DESTINATION_CELL_ROTATION = np.asarray(
    [4 * ((node // 4 + 1) % 4) + node % 4 for node in range(16)],
    dtype=np.int64,
)


def destination_cell_rotation(state_bits: int) -> np.ndarray:
    if state_bits <= 0 or state_bits % 4:
        raise ValueError("state_bits must be a positive multiple of four")
    cells = state_bits // 4
    return np.asarray(
        [4 * ((node // 4 + 1) % cells) + node % 4 for node in range(state_bits)],
        dtype=np.int64,
    )


def topology_players(players: np.ndarray, mode: str) -> np.ndarray:
    player_array = np.asarray(players, dtype=np.int64)
    if mode == "true":
        return player_array.copy()
    if mode == "shuffled":
        return np.roll(player_array, shift=1, axis=0)
    if mode == "corrupted":
        return destination_cell_rotation(player_array.shape[1])[player_array]
    raise ValueError("topology mode must be true, shuffled, or corrupted")
