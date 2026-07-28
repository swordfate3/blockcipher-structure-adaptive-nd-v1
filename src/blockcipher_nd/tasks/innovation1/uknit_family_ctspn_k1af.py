from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    build_k1ac_control,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_dialga_single_pair_replay_k1af_20260729"
EXPECTED_SEEDS = (0, 1)
CONDITIONS = ("exact", "wrong_sbox")
PAIR_COUNT = 16
PAIR_BITS = 256
POOLED_ROWS = 32768
AUC_FLOOR = 0.550
AUC_CEILING = 0.950
SEMANTIC_MARGIN = 0.010
REPLAY_TOLERANCE = 1e-6


def evaluate_single_pair_replay(
    *,
    seed: int,
    task: Mapping[str, Any],
    checkpoint_path: Path,
    dataset: DiskDifferentialDataset,
    cache_digests: Mapping[str, str],
    source_k1ae_gate_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected K1-AF seed: {seed}")
    features = np.asarray(dataset.features, dtype=np.uint8)
    labels = np.asarray(dataset.labels, dtype=np.uint8)
    if features.shape != (2048, PAIR_COUNT * PAIR_BITS) or labels.shape != (2048,):
        raise ValueError("K1-AF source validation geometry is invalid")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("K1-AF checkpoint must contain a state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("selected_checkpoint") != "best" or metadata.get("seed") != seed:
        raise ValueError("K1-AF requires a seed-specific exact best checkpoint")
    state = payload["state_dict"]
    state_sha256 = tensor_mapping_sha256(state)
    checkpoint_sha256 = file_sha256(checkpoint_path)

    pairs = features.reshape(2048, PAIR_COUNT, PAIR_BITS)
    pooled_features = pairs.transpose(1, 0, 2).reshape(POOLED_ROWS, PAIR_BITS)
    pooled_labels = np.tile(labels, PAIR_COUNT)
    pooled_dataset = DifferentialDataset(
        features=pooled_features,
        labels=pooled_labels,
        metadata={**dataset.metadata, "pairs_per_sample": 1, "input_bits": PAIR_BITS},
    )
    probabilities: dict[str, np.ndarray] = {}
    replay_errors: dict[str, float] = {}
    for condition in CONDITIONS:
        model = _build_one_pair_model(task, condition)
        model.load_state_dict(state, strict=True)
        if tensor_mapping_sha256(model.state_dict()) != state_sha256:
            raise ValueError("strict K1-AF load changed the source state")
        replay_errors[condition] = _direct_repeat_error(
            task=task,
            condition=condition,
            state=state,
            one_pair_model=model,
            fixture=pooled_features[:8],
        )
        probabilities[condition] = predict_binary_probabilities(
            model,
            pooled_dataset,
            batch_size=batch_size,
            device=device,
        ).reshape(PAIR_COUNT, 2048)

    rows: list[dict[str, Any]] = []
    for scope, position in [("pooled", -1), *[("pair_position", index) for index in range(PAIR_COUNT)], ("mean_query_aggregate", -1)]:
        scope_probabilities: dict[str, np.ndarray] = {}
        scope_labels = pooled_labels if scope == "pooled" else labels
        for condition in CONDITIONS:
            values = probabilities[condition]
            if scope == "pooled":
                scope_probabilities[condition] = values.reshape(-1)
            elif scope == "pair_position":
                scope_probabilities[condition] = values[position]
            else:
                scope_probabilities[condition] = values.mean(axis=0)
        aucs = {condition: binary_auc(scope_labels, values) for condition, values in scope_probabilities.items()}
        reference = scope_probabilities["exact"]
        for condition in CONDITIONS:
            values = scope_probabilities[condition]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "seed": seed,
                    "condition": condition,
                    "scope": scope,
                    "pair_position": position,
                    "auc": aucs[condition],
                    "exact_minus_condition_auc": 0.0 if condition == "exact" else aucs["exact"] - aucs[condition],
                    "max_abs_probability_delta_from_exact": float(np.max(np.abs(reference - values))),
                    "mean_abs_probability_delta_from_exact": float(np.mean(np.abs(reference - values))),
                    "observation_rows": int(len(scope_labels)),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_selected": metadata.get("selected_checkpoint"),
                    "checkpoint_reported_seed": metadata.get("seed"),
                    "state_dict_sha256": state_sha256,
                    **dict(cache_digests),
                    "cache_dir": str(dataset.cache_dir),
                    "source_k1ae_gate_sha256": source_k1ae_gate_sha256,
                    "one_pair_input_bits": PAIR_BITS,
                    "source_pairs_per_sample": PAIR_COUNT,
                    "audit_pairs_per_observation": 1,
                    "parameter_count": EXPECTED_PARAMETER_COUNT,
                    "direct_repeat_logit_max_error": replay_errors[condition],
                    "negative_mode": dataset.metadata["negative_mode"],
                    "sample_structure": dataset.metadata["sample_structure"],
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "data_generation_performed": False,
                }
            )
    return rows


def adjudicate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row.get("seed", -1)), str(row.get("condition")), str(row.get("scope")), int(row.get("pair_position", -2)))].append(row)
    expected = {
        (seed, condition, scope, position)
        for seed in EXPECTED_SEEDS
        for condition in CONDITIONS
        for scope, position in [("pooled", -1), *[("pair_position", index) for index in range(PAIR_COUNT)], ("mean_query_aggregate", -1)]
    }
    complete = set(grouped) == expected and all(len(value) == 1 for value in grouped.values())
    seed_results = {str(seed): _seed_result(grouped, seed) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "seventy_two_rows_complete": len(rows) == 72 and complete,
        "same_source_within_seed": complete and all(
            len({row.get(field) for row in rows if row.get("seed") == seed}) == 1
            for seed in EXPECTED_SEEDS
            for field in ("checkpoint_sha256", "state_dict_sha256", "feature_sha256", "label_sha256", "metadata_sha256", "cache_dir")
        ),
        "distinct_seed_checkpoints": len({row.get("checkpoint_sha256") for row in rows if row.get("condition") == "exact"}) == 2,
        "source_k1ae_bound": len(rows) == 72 and len({row.get("source_k1ae_gate_sha256") for row in rows}) == 1 and all(_sha256(row.get("source_k1ae_gate_sha256")) for row in rows),
        "best_checkpoints_strictly_loaded": all(row.get("checkpoint_selected") == "best" and row.get("checkpoint_reported_seed") == row.get("seed") and row.get("strict_state_dict_load") is True for row in rows),
        "one_pair_geometry_parameter_invariant": all(row.get("one_pair_input_bits") == PAIR_BITS and row.get("source_pairs_per_sample") == PAIR_COUNT and row.get("audit_pairs_per_observation") == 1 and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT for row in rows),
        "direct_repeat_equivalence": all(_finite(row.get("direct_repeat_logit_max_error")) and float(row["direct_repeat_logit_max_error"]) <= REPLAY_TOLERANCE for row in rows),
        "scope_row_counts_exact": all(
            row.get("observation_rows") == (POOLED_ROWS if row.get("scope") == "pooled" else 2048)
            for row in rows
        ),
        "finite_metrics": all(_finite(row.get(field)) for row in rows for field in ("auc", "exact_minus_condition_auc", "max_abs_probability_delta_from_exact", "mean_abs_probability_delta_from_exact")),
        "inference_only_no_new_data": all(row.get("training_performed") is False and row.get("optimizer_steps") == 0 and row.get("data_generation_performed") is False for row in rows),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_pooled_signal_at_least_0p550"] = result["pooled_exact_auc"] >= AUC_FLOOR
        research_checks[f"seed{seed}_pooled_not_saturated_at_most_0p950"] = result["pooled_exact_auc"] <= AUC_CEILING
        research_checks[f"seed{seed}_pooled_exact_beats_wrong_by_0p010"] = result["pooled_exact_minus_wrong_auc"] >= SEMANTIC_MARGIN

    protocol_valid = all(protocol_checks.values())
    all_research = all(research_checks.values())
    weak = any(not value for name, value in research_checks.items() if "signal_at_least" in name)
    saturated = any(not value for name, value in research_checks.items() if "not_saturated" in name)
    semantic_failed = any(not value for name, value in research_checks.items() if "beats_wrong" in name)
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1af_protocol_invalid"
        next_action = "repair only the replay or source binding and rerun unchanged"
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1af_one_pair_semantic_surface_supported"
        next_action = "preregister the same local K1-AA one-pair exact/wrong-S-box training matrix"
    elif saturated:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1af_one_pair_still_saturated"
        next_action = "close mechanical Dialga query reduction and retain Dialga only as GF(2) signal calibration"
    elif weak:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1af_one_pair_signal_too_weak"
        next_action = "do not infer fresh one-pair training success; close the Dialga reduction route"
    elif semantic_failed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1af_one_pair_semantic_attribution_failed"
        next_action = "do not train one pair; seek family semantics on another shared-primitive cipher surface"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1af_unclassified_hold"
        next_action = "inspect the frozen per-seed pooled checks before any new experiment"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(name for name, passed in protocol_checks.items() if not passed),
        "research_checks": research_checks,
        "failed_research_checks": sorted(name for name, passed in research_checks.items() if not passed),
        "seed_results": seed_results,
        "thresholds": {"pooled_auc_floor": AUC_FLOOR, "pooled_auc_ceiling": AUC_CEILING, "exact_minus_wrong_sbox": SEMANTIC_MARGIN, "direct_repeat_logit_tolerance": REPLAY_TOLERANCE},
        "next_action": next_action,
        "claim_scope": (
            "two-seed zero-training Dialga-128 r4 frozen-checkpoint single-pair replay; "
            "mean-query aggregation is application-level support, not raw single-pair SOTA evidence"
        ),
        "blocked_actions": ["training inside K1-AF", "remote scale or new data", "attack, SOTA, or family-success claims"],
    }


def _build_one_pair_model(task: Mapping[str, Any], condition: str) -> torch.nn.Module:
    return build_model(
        CONTROL_MODELS["virtual_slot_exact" if condition == "exact" else "virtual_slot_wrong_sbox"],
        input_bits=PAIR_BITS,
        hidden_bits=32,
        pair_bits=PAIR_BITS,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def _direct_repeat_error(
    *,
    task: Mapping[str, Any],
    condition: str,
    state: Mapping[str, torch.Tensor],
    one_pair_model: torch.nn.Module,
    fixture: np.ndarray,
) -> float:
    one_pair_fixture_model = deepcopy(one_pair_model).double()
    sixteen_pair_model = build_k1ac_control(
        task=task,
        condition="virtual_slot_exact" if condition == "exact" else "virtual_slot_wrong_sbox",
    ).double()
    sixteen_pair_model.load_state_dict(state, strict=True)
    one_pair_fixture_model.eval()
    sixteen_pair_model.eval()
    direct = torch.as_tensor(fixture, dtype=torch.float64)
    repeated = direct.repeat(1, PAIR_COUNT)
    with torch.no_grad():
        return float(
            (
                one_pair_fixture_model(direct)
                - sixteen_pair_model(repeated)
            )
            .abs()
            .max()
        )


def _seed_result(
    grouped: Mapping[tuple[int, str, str, int], Sequence[Mapping[str, Any]]], seed: int
) -> dict[str, Any]:
    exact = grouped[(seed, "exact", "pooled", -1)][0]
    wrong = grouped[(seed, "wrong_sbox", "pooled", -1)][0]
    aggregate_exact = grouped[(seed, "exact", "mean_query_aggregate", -1)][0]
    aggregate_wrong = grouped[(seed, "wrong_sbox", "mean_query_aggregate", -1)][0]
    positions = []
    for position in range(PAIR_COUNT):
        exact_row = grouped[(seed, "exact", "pair_position", position)][0]
        wrong_row = grouped[(seed, "wrong_sbox", "pair_position", position)][0]
        positions.append(
            {
                "pair_position": position,
                "exact_auc": float(exact_row["auc"]),
                "wrong_sbox_auc": float(wrong_row["auc"]),
                "exact_minus_wrong_auc": float(exact_row["auc"]) - float(wrong_row["auc"]),
            }
        )
    return {
        "pooled_exact_auc": float(exact["auc"]),
        "pooled_wrong_sbox_auc": float(wrong["auc"]),
        "pooled_exact_minus_wrong_auc": float(exact["auc"]) - float(wrong["auc"]),
        "mean_query_exact_auc": float(aggregate_exact["auc"]),
        "mean_query_wrong_sbox_auc": float(aggregate_wrong["auc"]),
        "mean_query_exact_minus_wrong_auc": float(aggregate_exact["auc"]) - float(aggregate_wrong["auc"]),
        "per_position": positions,
    }


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = ["CONDITIONS", "EXPECTED_SEEDS", "RUN_ID", "adjudicate", "evaluate_single_pair_replay"]
