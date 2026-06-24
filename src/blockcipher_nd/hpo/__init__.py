from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _as_options(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def expand_search_space(space: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    option_lists = [_as_options(space[key]) for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*option_lists)]


def sample_search_space(space: Mapping[str, Any], trials: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(space.keys())
    option_lists = [_as_options(space[key]) for key in keys]
    sampled: list[dict[str, Any]] = []
    for _ in range(trials):
        sampled.append({key: rng.choice(options) for key, options in zip(keys, option_lists)})
    return sampled


def select_trials(
    space: Mapping[str, Any],
    mode: str,
    max_trials: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if mode == "grid":
        trials = expand_search_space(space)
        return trials[:max_trials] if max_trials is not None else trials
    if mode == "random":
        if max_trials is None:
            raise ValueError("random search requires max_trials")
        return sample_search_space(space, max_trials, seed=seed)
    raise ValueError(f"unsupported HPO mode: {mode}")
