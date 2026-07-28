from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import (
    EXPECTED_INPUT_BITS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PAIRS,
    build_k1ac_control,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_dialga_branch_ablation_k1ae_20260729"
EXPECTED_SEEDS = (0, 1)
CONDITIONS = ("full", "histogram_off", "edge_off", "base_only")
NECESSITY_MARGIN = 0.010
SOURCE_REPLAY_TOLERANCE = 1e-7


def evaluate_branch_ablation(
    *,
    seed: int,
    task: Mapping[str, Any],
    source_row: Mapping[str, Any],
    checkpoint_path: Path,
    dataset: DiskDifferentialDataset,
    cache_digests: Mapping[str, str],
    source_k1ac_gate_sha256: str,
    source_k1ad_results_sha256: str,
    source_k1ad_gate_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected K1-AE seed: {seed}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("K1-AE source checkpoint must contain a state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("selected_checkpoint") != "best" or metadata.get("seed") != seed:
        raise ValueError("K1-AE requires the seed-specific exact best checkpoint")
    state = payload["state_dict"]
    state_sha256 = tensor_mapping_sha256(state)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    labels = np.asarray(dataset.labels, dtype=np.float32)
    probabilities: dict[str, np.ndarray] = {}
    gate_values: dict[str, dict[str, float]] = {}
    runtime_sha256: str | None = None

    for condition in CONDITIONS:
        model = build_k1ac_control(task=task, condition="virtual_slot_exact")
        model.load_state_dict(state, strict=True)
        if tensor_mapping_sha256(model.state_dict()) != state_sha256:
            raise ValueError("strict K1-AE load changed the source state")
        learned_edge = float(torch.tanh(model.backbone.residual_gate.detach()))
        learned_histogram = float(torch.tanh(model.backbone.histogram_gate.detach()))
        with torch.no_grad():
            if condition in {"edge_off", "base_only"}:
                model.backbone.residual_gate.zero_()
            if condition in {"histogram_off", "base_only"}:
                model.backbone.histogram_gate.zero_()
        applied_edge = float(torch.tanh(model.backbone.residual_gate.detach()))
        applied_histogram = float(torch.tanh(model.backbone.histogram_gate.detach()))
        gate_values[condition] = {
            "learned_edge_gate": learned_edge,
            "learned_histogram_gate": learned_histogram,
            "applied_edge_gate": applied_edge,
            "applied_histogram_gate": applied_histogram,
        }
        current_runtime_sha = model.runtime_structure.window_sha256()
        if runtime_sha256 is None:
            runtime_sha256 = current_runtime_sha
        elif runtime_sha256 != current_runtime_sha:
            raise ValueError("K1-AE runtime descriptor drifted across interventions")
        probabilities[condition] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )

    aucs = {condition: binary_auc(labels, values) for condition, values in probabilities.items()}
    reference = probabilities["full"]
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        values = probabilities[condition]
        intervention = gate_values[condition]
        rows.append(
            {
                "run_id": RUN_ID,
                "seed": seed,
                "condition": condition,
                "cipher": "Dialga-128",
                "cipher_key": "dialga128",
                "rounds": 4,
                "auc": aucs[condition],
                "source_full_auc": float(source_row["auc"]),
                "full_minus_condition_auc": 0.0 if condition == "full" else aucs["full"] - aucs[condition],
                "max_abs_probability_delta_from_full": float(np.max(np.abs(reference - values))),
                "mean_abs_probability_delta_from_full": float(np.mean(np.abs(reference - values))),
                "probability_sha256": hashlib.sha256(values.astype(np.float32, copy=False).tobytes()).hexdigest(),
                **intervention,
                "intervention_sha256": _intervention_sha256(condition, intervention),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_selected": metadata.get("selected_checkpoint"),
                "checkpoint_reported_seed": metadata.get("seed"),
                "pre_intervention_state_dict_sha256": state_sha256,
                **dict(cache_digests),
                "cache_dir": str(dataset.cache_dir),
                "source_k1ac_gate_sha256": source_k1ac_gate_sha256,
                "source_k1ad_results_sha256": source_k1ad_results_sha256,
                "source_k1ad_gate_sha256": source_k1ad_gate_sha256,
                "runtime_structure_window_sha256": runtime_sha256,
                "samples_total": int(dataset.features.shape[0]),
                "input_bits": int(dataset.features.shape[1]),
                "pair_bits": int(dataset.metadata["pair_bits"]),
                "pairs_per_sample": int(dataset.metadata["pairs_per_sample"]),
                "input_difference": int(dataset.metadata["input_difference"]),
                "negative_mode": dataset.metadata["negative_mode"],
                "sample_structure": dataset.metadata["sample_structure"],
                "validation_seed": int(dataset.metadata["seed"]),
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "strict_state_dict_load": True,
                "training_performed": False,
                "optimizer_steps": 0,
            }
        )
    return rows


def adjudicate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row.get("seed", -1))][str(row.get("condition"))].append(row)
    complete = all(len(grouped[seed][condition]) == 1 for seed in EXPECTED_SEEDS for condition in CONDITIONS)
    seed_results = {str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "eight_rows_complete": len(rows) == 8 and complete and set(grouped) == set(EXPECTED_SEEDS),
        "same_source_checkpoint_state_and_cache": complete and _same_seed_fields(
            grouped,
            (
                "checkpoint_sha256",
                "pre_intervention_state_dict_sha256",
                "feature_sha256",
                "label_sha256",
                "metadata_sha256",
                "cache_dir",
                "runtime_structure_window_sha256",
            ),
        ),
        "distinct_seed_checkpoints": complete and len({grouped[seed]["full"][0].get("checkpoint_sha256") for seed in EXPECTED_SEEDS}) == 2,
        "source_provenance_bound": len(rows) == 8
        and all(_sha256(row.get(field)) for row in rows for field in ("source_k1ac_gate_sha256", "source_k1ad_results_sha256", "source_k1ad_gate_sha256"))
        and all(len({row.get(field) for row in rows}) == 1 for field in ("source_k1ac_gate_sha256", "source_k1ad_results_sha256", "source_k1ad_gate_sha256")),
        "best_checkpoints_strictly_loaded": all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_reported_seed") == row.get("seed")
            and row.get("strict_state_dict_load") is True
            for row in rows
        ),
        "declared_gate_interventions_exact": complete and all(_intervention_exact(grouped[seed], condition) for seed in EXPECTED_SEEDS for condition in CONDITIONS),
        "full_auc_reproduces_k1ad": complete and all(abs(seed_results[str(seed)]["full_auc"] - seed_results[str(seed)]["source_full_auc"]) <= SOURCE_REPLAY_TOLERANCE for seed in EXPECTED_SEEDS),
        "frozen_validation_geometry": all(
            row.get("cipher_key") == "dialga128"
            and row.get("rounds") == 4
            and row.get("samples_total") == 2048
            and row.get("input_bits") == EXPECTED_INPUT_BITS
            and row.get("pair_bits") == 256
            and row.get("pairs_per_sample") == EXPECTED_PAIRS
            and row.get("input_difference") == 0x40
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("validation_seed") == 10000 + int(row.get("seed", -1))
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            for row in rows
        ),
        "finite_metrics_and_learned_gates": all(
            _finite(row.get(field))
            for row in rows
            for field in (
                "auc",
                "source_full_auc",
                "max_abs_probability_delta_from_full",
                "mean_abs_probability_delta_from_full",
                "learned_edge_gate",
                "learned_histogram_gate",
            )
        ),
        "inference_only": all(row.get("training_performed") is False and row.get("optimizer_steps") == 0 for row in rows),
    }

    protocol_valid = all(protocol_checks.values())
    base_retains = all(seed_results[str(seed)]["full_minus_base_only_auc"] <= NECESSITY_MARGIN for seed in EXPECTED_SEEDS)
    histogram_necessary = all(seed_results[str(seed)]["full_minus_histogram_off_auc"] >= NECESSITY_MARGIN for seed in EXPECTED_SEEDS)
    edge_necessary = all(seed_results[str(seed)]["full_minus_edge_off_auc"] >= NECESSITY_MARGIN for seed in EXPECTED_SEEDS)
    joint_necessary = all(seed_results[str(seed)]["full_minus_base_only_auc"] >= NECESSITY_MARGIN for seed in EXPECTED_SEEDS)
    research_checks = {
        "both_seeds_full_auc_at_least_0p950": all(seed_results[str(seed)]["full_auc"] >= 0.950 for seed in EXPECTED_SEEDS),
        "both_seeds_base_only_retains_within_0p010": base_retains,
        "both_seeds_histogram_residual_necessary": histogram_necessary,
        "both_seeds_edge_residual_necessary": edge_necessary,
        "both_seeds_joint_residuals_necessary": joint_necessary,
    }
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1ae_protocol_invalid"
        next_action = "repair only the failed source or intervention binding and rerun unchanged"
    elif base_retains:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ae_gf2_base_path_dominates"
        next_action = (
            "stop using Dialga r4 sixteen-pair AUC as an S-box semantic gate; "
            "select one already evidenced less-saturated Dialga condition before retraining"
        )
    elif histogram_necessary or edge_necessary:
        status = "hold"
        branch = "histogram" if histogram_necessary else "edge"
        decision = f"innovation1_uknit_family_ctspn_k1ae_{branch}_residual_necessary"
        next_action = f"redesign only the {branch} residual attribution at the same local budget"
    elif joint_necessary:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ae_residuals_jointly_necessary"
        next_action = "run one frozen residual-interaction audit without new training or data"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ae_branch_effect_unstable"
        next_action = "retain the frozen protocol and inspect per-seed gate signs before redesign"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(name for name, passed in protocol_checks.items() if not passed),
        "research_checks": research_checks,
        "seed_results": seed_results,
        "thresholds": {"branch_necessity_auc": NECESSITY_MARGIN, "source_replay_tolerance": SOURCE_REPLAY_TOLERANCE},
        "next_action": next_action,
        "claim_scope": (
            "two-seed zero-training Dialga-128 r4 K1-AA sixteen-pair branch ablation; "
            "not formal scale, attack, SOTA, or family-transfer evidence"
        ),
        "blocked_actions": ["remote scale", "new training or data", "family or S-box understanding claims"],
    }


def _seed_result(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    if any(len(grouped.get(condition, ())) != 1 for condition in CONDITIONS):
        return {name: math.nan for name in (
            "full_auc", "histogram_off_auc", "edge_off_auc", "base_only_auc", "source_full_auc",
            "full_minus_histogram_off_auc", "full_minus_edge_off_auc", "full_minus_base_only_auc",
        )}
    aucs = {condition: float(grouped[condition][0]["auc"]) for condition in CONDITIONS}
    result = {f"{condition}_auc": value for condition, value in aucs.items()}
    result["source_full_auc"] = float(grouped["full"][0]["source_full_auc"])
    for condition in CONDITIONS[1:]:
        result[f"full_minus_{condition}_auc"] = aucs["full"] - aucs[condition]
        result[f"max_probability_delta_{condition}"] = float(grouped[condition][0]["max_abs_probability_delta_from_full"])
        result[f"mean_probability_delta_{condition}"] = float(grouped[condition][0]["mean_abs_probability_delta_from_full"])
    result["learned_edge_gate"] = float(grouped["full"][0]["learned_edge_gate"])
    result["learned_histogram_gate"] = float(grouped["full"][0]["learned_histogram_gate"])
    return result


def _intervention_exact(grouped: Mapping[str, Sequence[Mapping[str, Any]]], condition: str) -> bool:
    row = grouped[condition][0]
    learned_edge = float(row["learned_edge_gate"])
    learned_histogram = float(row["learned_histogram_gate"])
    expected_edge = 0.0 if condition in {"edge_off", "base_only"} else learned_edge
    expected_histogram = 0.0 if condition in {"histogram_off", "base_only"} else learned_histogram
    return (
        abs(float(row["applied_edge_gate"]) - expected_edge) <= 1e-12
        and abs(float(row["applied_histogram_gate"]) - expected_histogram) <= 1e-12
        and _sha256(row.get("intervention_sha256"))
    )


def _same_seed_fields(
    grouped: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]], fields: Sequence[str]
) -> bool:
    return all(
        len({grouped[seed][condition][0].get(field) for condition in CONDITIONS}) == 1
        for seed in EXPECTED_SEEDS
        for field in fields
    )


def _intervention_sha256(condition: str, values: Mapping[str, float]) -> str:
    payload = f"{condition}|{values['applied_edge_gate']:.17g}|{values['applied_histogram_gate']:.17g}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = ["CONDITIONS", "EXPECTED_SEEDS", "RUN_ID", "adjudicate", "evaluate_branch_ablation"]
