from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "HPARAM_SUMMARY_FIELDS": "blockcipher_nd.evaluation.summary",
    "INNOVATION_ONE_GROUP_FIELDS": "blockcipher_nd.evaluation.summary",
    "INNOVATION_ONE_METRIC_FIELDS": "blockcipher_nd.evaluation.summary",
    "EnsembleScoreArtifact": "blockcipher_nd.evaluation.neural_ensemble",
    "PairSetAggregationConfig": "blockcipher_nd.evaluation.pairset_aggregation",
    "aggregate_pair_logits": "blockcipher_nd.evaluation.pairset_aggregation",
    "evaluate_frozen_score_ensemble": "blockcipher_nd.evaluation.neural_ensemble",
    "hparam_summary_row": "blockcipher_nd.evaluation.summary",
    "hparam_summary_rows": "blockcipher_nd.evaluation.summary",
    "innovation_one_summary_fields": "blockcipher_nd.evaluation.summary",
    "innovation_one_summary_rows": "blockcipher_nd.evaluation.summary",
    "load_score_artifact": "blockcipher_nd.evaluation.neural_ensemble",
    "load_jsonl_rows": "blockcipher_nd.evaluation.summary",
    "pairset_aggregation_metrics": "blockcipher_nd.evaluation.pairset_aggregation",
    "pairset_aggregation_scores": "blockcipher_nd.evaluation.pairset_aggregation",
    "plot_jsonl_training_curves": "blockcipher_nd.evaluation.plots",
    "split_pairset_features": "blockcipher_nd.evaluation.pairset_aggregation",
    "training_curve_series": "blockcipher_nd.evaluation.plots",
    "write_score_artifact": "blockcipher_nd.evaluation.neural_ensemble",
    "write_history_csv": "blockcipher_nd.evaluation.plots",
    "write_csv_rows": "blockcipher_nd.evaluation.summary",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
