from __future__ import annotations

import numpy as np


DESTINATION_CELL_ROTATION = np.asarray(
    [4 * ((node // 4 + 1) % 4) + node % 4 for node in range(16)],
    dtype=np.int64,
)


def topology_players(players: np.ndarray, mode: str) -> np.ndarray:
    player_array = np.asarray(players, dtype=np.int64)
    if mode == "true":
        return player_array.copy()
    if mode == "shuffled":
        return np.roll(player_array, shift=1, axis=0)
    if mode == "corrupted":
        return DESTINATION_CELL_ROTATION[player_array]
    raise ValueError("topology mode must be true, shuffled, or corrupted")
