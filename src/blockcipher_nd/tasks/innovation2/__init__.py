from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "INPUT_BITS",
    "IntegralExperimentConfig",
    "IntegralStructure",
    "IntegralRankingThresholds",
    "adjudicate_joint_integral_ranking",
    "build_integral_split",
    "evaluate_integral_ranking",
    "integral_mask_parity",
    "run_integral_property_experiment",
    "spearman_correlation",
]

_EXPORT_MODULES = {
    "INPUT_BITS": "integral_property_prediction",
    "IntegralExperimentConfig": "integral_property_prediction",
    "IntegralStructure": "integral_property_prediction",
    "build_integral_split": "integral_property_prediction",
    "integral_mask_parity": "integral_property_prediction",
    "run_integral_property_experiment": "integral_property_prediction",
    "IntegralRankingThresholds": "integral_property_ranking",
    "adjudicate_joint_integral_ranking": "integral_property_ranking",
    "evaluate_integral_ranking": "integral_property_ranking",
    "spearman_correlation": "integral_property_ranking",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
