from __future__ import annotations

from collections import defaultdict
import hashlib
import json
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


RUN_ID = "i1_uknit_family_ctspn_dialga_same_checkpoint_k1ad_20260729"
SOURCE_DECISION = "innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed"
EXPECTED_SEEDS = (0, 1)
CONDITIONS = ("exact", "wrong_sbox")
MARGIN = 0.010
AUC_FLOOR = 0.950
PROBABILITY_DELTA_FLOOR = 1e-6
SOURCE_REPLAY_TOLERANCE = 1e-7


def load_validation_cache(cache_dir: Path) -> tuple[DiskDifferentialDataset, dict[str, str]]:
    paths = {
        "feature_sha256": cache_dir / "features.npy",
        "label_sha256": cache_dir / "labels.npy",
        "metadata_sha256": cache_dir / "metadata.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"incomplete K1-AD validation cache: {cache_dir}")
    metadata = json.loads(paths["metadata_sha256"].read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(paths["feature_sha256"], mmap_mode="r"),
        labels=np.load(paths["label_sha256"], mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, {name: file_sha256(path) for name, path in paths.items()}


def evaluate_same_checkpoint(
    *,
    seed: int,
    task: Mapping[str, Any],
    source_row: Mapping[str, Any],
    checkpoint_path: Path,
    dataset: DiskDifferentialDataset,
    cache_digests: Mapping[str, str],
    source_results_sha256: str,
    source_gate_sha256: str,
    source_progress_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected K1-AD seed: {seed}")
    _validate_dataset(dataset, seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("K1-AD source checkpoint must contain a state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("K1-AD source checkpoint must contain metadata")
    if metadata.get("selected_checkpoint") != "best" or metadata.get("seed") != seed:
        raise ValueError("K1-AD requires the seed-specific exact best checkpoint")

    source_state = payload["state_dict"]
    state_sha256 = tensor_mapping_sha256(source_state)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    models: dict[str, torch.nn.Module] = {}
    probabilities: dict[str, np.ndarray] = {}
    runtime_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        model = build_k1ac_control(
            task=task,
            condition="virtual_slot_exact" if condition == "exact" else "virtual_slot_wrong_sbox",
        )
        model.load_state_dict(source_state, strict=True)
        if tensor_mapping_sha256(model.state_dict()) != state_sha256:
            raise ValueError("strict K1-AD load changed the source state")
        models[condition] = model
        probabilities[condition] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        runtime_hashes[condition] = model.runtime_structure.window_sha256()

    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {condition: binary_auc(labels, values) for condition, values in probabilities.items()}
    reference = probabilities["exact"]
    source_auc = float(source_row["metrics"]["auc"])
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        values = probabilities[condition]
        rows.append(
            {
                "run_id": RUN_ID,
                "seed": seed,
                "condition": condition,
                "cipher": "Dialga-128",
                "cipher_key": "dialga128",
                "rounds": 4,
                "auc": aucs[condition],
                "source_exact_auc": source_auc,
                "exact_minus_condition_auc": 0.0 if condition == "exact" else aucs["exact"] - aucs[condition],
                "max_abs_probability_delta_from_exact": float(np.max(np.abs(reference - values))),
                "mean_abs_probability_delta_from_exact": float(np.mean(np.abs(reference - values))),
                "probability_sha256": hashlib.sha256(values.astype(np.float32, copy=False).tobytes()).hexdigest(),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_selected": metadata.get("selected_checkpoint"),
                "checkpoint_reported_seed": metadata.get("seed"),
                "checkpoint_best_metric": metadata.get("best_checkpoint_metric"),
                "state_dict_sha256": state_sha256,
                **dict(cache_digests),
                "cache_dir": str(dataset.cache_dir),
                "source_results_sha256": source_results_sha256,
                "source_gate_sha256": source_gate_sha256,
                "source_progress_sha256": source_progress_sha256,
                "source_decision": SOURCE_DECISION,
                "runtime_structure_mode": condition,
                "runtime_structure_window_sha256": runtime_hashes[condition],
                "samples_total": int(dataset.features.shape[0]),
                "input_bits": int(dataset.features.shape[1]),
                "pair_bits": int(dataset.metadata["pair_bits"]),
                "pairs_per_sample": int(dataset.metadata["pairs_per_sample"]),
                "input_difference": int(dataset.metadata["input_difference"]),
                "negative_mode": dataset.metadata["negative_mode"],
                "sample_structure": dataset.metadata["sample_structure"],
                "validation_seed": int(dataset.metadata["seed"]),
                "parameter_count": sum(value.numel() for value in models[condition].parameters()),
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
        "four_rows_complete": len(rows) == 4 and complete and set(grouped) == set(EXPECTED_SEEDS),
        "same_checkpoint_and_state_within_seed": complete and _same_seed_fields(grouped, ("checkpoint_sha256", "state_dict_sha256")),
        "same_cache_within_seed": complete and _same_seed_fields(grouped, ("feature_sha256", "label_sha256", "metadata_sha256", "cache_dir")),
        "distinct_seed_checkpoints": complete and len({grouped[seed]["exact"][0].get("checkpoint_sha256") for seed in EXPECTED_SEEDS}) == 2,
        "exact_best_checkpoints_strictly_loaded": all(
            row.get("checkpoint_selected") == "best"
            and row.get("checkpoint_reported_seed") == row.get("seed")
            and row.get("strict_state_dict_load") is True
            for row in rows
        ),
        "source_provenance_bound": len(rows) == 4
        and all(row.get("source_decision") == SOURCE_DECISION for row in rows)
        and all(_sha256(row.get(field)) for row in rows for field in ("source_results_sha256", "source_gate_sha256", "source_progress_sha256"))
        and len({row.get("source_results_sha256") for row in rows}) == 1
        and len({row.get("source_gate_sha256") for row in rows}) == 1
        and len({row.get("source_progress_sha256") for row in rows}) == 1,
        "runtime_sbox_intervention_distinct": complete
        and all(grouped[seed]["exact"][0].get("runtime_structure_window_sha256") != grouped[seed]["wrong_sbox"][0].get("runtime_structure_window_sha256") for seed in EXPECTED_SEEDS),
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
        "exact_auc_reproduces_source": complete and all(abs(seed_results[str(seed)]["exact_auc"] - seed_results[str(seed)]["source_exact_auc"]) <= SOURCE_REPLAY_TOLERANCE for seed in EXPECTED_SEEDS),
        "finite_metrics": all(
            _finite(row.get(field))
            for row in rows
            for field in ("auc", "source_exact_auc", "max_abs_probability_delta_from_exact", "mean_abs_probability_delta_from_exact")
        ),
        "inference_only": all(row.get("training_performed") is False and row.get("optimizer_steps") == 0 for row in rows),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_exact_auc_at_least_0p950"] = result["exact_auc"] >= AUC_FLOOR
        research_checks[f"seed{seed}_exact_beats_wrong_sbox_by_0p010"] = result["exact_minus_wrong_sbox_auc"] >= MARGIN
        research_checks[f"seed{seed}_prediction_changes"] = result["max_abs_probability_delta"] > PROBABILITY_DELTA_FLOOR

    protocol_valid = all(protocol_checks.values())
    margins_pass = all(value for name, value in research_checks.items() if "beats_wrong_sbox" in name)
    sensitivity_pass = all(value for name, value in research_checks.items() if name.endswith("prediction_changes"))
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1ad_protocol_invalid"
        next_action = "repair only the failed source or artifact binding and rerun unchanged"
    elif margins_pass and sensitivity_pass:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1ad_functional_sbox_use_supported"
        next_action = (
            "test one same-budget training-time counterfactual attribution constraint "
            "while keeping the K1-AA architecture and sixteen-pair data fixed"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ad_discriminative_sbox_use_failed"
        next_action = (
            "run zero-training K1-AE base-path versus histogram-residual ablation on "
            "the identical checkpoints and validation caches before redesign or scale"
        )
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
        "thresholds": {
            "exact_auc": AUC_FLOOR,
            "exact_minus_wrong_sbox_auc": MARGIN,
            "max_abs_probability_delta": PROBABILITY_DELTA_FLOOR,
            "source_replay_tolerance": SOURCE_REPLAY_TOLERANCE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed zero-training same-checkpoint Dialga-128 r4 K1-AA sixteen-pair "
            "S-box intervention audit; not formal scale, attack, SOTA, or family-transfer evidence"
        ),
        "blocked_actions": [
            "remote scale",
            "new data, pairs, epochs, seeds, differences, or model families",
            "claiming arbitrary-SPN semantic understanding",
        ],
    }


def _validate_dataset(dataset: DiskDifferentialDataset, seed: int) -> None:
    metadata = dataset.metadata
    checks = (
        dataset.features.shape == (2048, EXPECTED_INPUT_BITS),
        dataset.labels.shape == (2048,),
        metadata.get("cipher") == "Dialga-128",
        metadata.get("rounds") == 4,
        metadata.get("seed") == 10000 + seed,
        metadata.get("pair_bits") == 256,
        metadata.get("pairs_per_sample") == EXPECTED_PAIRS,
        metadata.get("input_difference") == 0x40,
        metadata.get("negative_mode") == "encrypted_random_plaintexts",
        metadata.get("sample_structure") == "independent_pairs",
    )
    if not all(checks):
        raise ValueError(f"K1-AD validation cache protocol mismatch for seed {seed}")


def _seed_result(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    if any(len(grouped.get(condition, ())) != 1 for condition in CONDITIONS):
        return {
            "exact_auc": math.nan,
            "wrong_sbox_auc": math.nan,
            "source_exact_auc": math.nan,
            "exact_minus_wrong_sbox_auc": math.nan,
            "max_abs_probability_delta": math.nan,
            "mean_abs_probability_delta": math.nan,
        }
    exact = grouped["exact"][0]
    wrong = grouped["wrong_sbox"][0]
    return {
        "exact_auc": float(exact["auc"]),
        "wrong_sbox_auc": float(wrong["auc"]),
        "source_exact_auc": float(exact["source_exact_auc"]),
        "exact_minus_wrong_sbox_auc": float(exact["auc"]) - float(wrong["auc"]),
        "max_abs_probability_delta": float(wrong["max_abs_probability_delta_from_exact"]),
        "mean_abs_probability_delta": float(wrong["mean_abs_probability_delta_from_exact"]),
    }


def _same_seed_fields(
    grouped: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    fields: Sequence[str],
) -> bool:
    return all(
        len({grouped[seed][condition][0].get(field) for condition in CONDITIONS}) == 1
        for seed in EXPECTED_SEEDS
        for field in fields
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "CONDITIONS",
    "EXPECTED_SEEDS",
    "RUN_ID",
    "SOURCE_DECISION",
    "adjudicate",
    "evaluate_same_checkpoint",
    "load_validation_cache",
]
