from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1aa import (
    BASE_LR,
    CONTROL_MODELS,
    EXPECTED_KEYS,
    EXPECTED_PARAMETER_COUNT,
    MODEL_TO_CONDITION,
    VIRTUAL_PARAMETER,
    VIRTUAL_SHAPE,
    build_k1aa_control,
)
from blockcipher_nd.training.optim import make_optimizer
from blockcipher_nd.training.types import TrainingConfig


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_ctspn_virtual_slot_pair_count_"
    "k1ab_16pair_2048_seed3_seed4_20260729"
)
EXPECTED_PAIRS = 16
EXPECTED_INPUT_BITS = 2048
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
SEMANTIC_MARGIN = 0.010
ADDED_VALUE_MARGIN = 0.010
K1V_RETENTION_TOLERANCE = 0.020
K1AA_4PAIR_AUCS = {3: 0.5708699226379395, 4: 0.5909538269042969}
K1V_INVARIANT_16PAIR_AUCS = {3: 0.5914902687072754, 4: 0.6975908279418945}
K1AA_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_virtual_slot_projection_"
    "k1aa_2048_seed3_seed4_20260728"
)
K1V_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_pair_count_"
    "k1v_16pair_2048_seed3_seed4_20260728_clean"
)
SOURCE_PATHS = {
    "k1aa_gate": K1AA_ROOT / "gate.json",
    "k1aa_results": K1AA_ROOT / "results.jsonl",
    "k1v_gate": K1V_ROOT / "gate.json",
}
SOURCE_DIGESTS = {
    "k1aa_gate": "973c2a4fe7abdba36f14edf4ae5978089df2c5d86b82f3655b88740572472835",
    "k1aa_results": "9007ba446905683449d3802222bbd75043af07c080f4247e7186ccbb64e0439b",
    "k1v_gate": "64cb951014b232fea996e9be19c2697c8143e8528eabf945f63c72c8f2722e4a",
}
SOURCE_CACHE_ROWS = (
    (
        3,
        "train",
        K1V_ROOT / "cache/uknit64/r5/train/seed-3_f127a1f8908cac11",
        "0301d3bbb04083c99c3f488bc01923c972ab5c625edc74ebd18ade807fb97384",
    ),
    (
        3,
        "validation",
        K1V_ROOT / "cache/uknit64/r5/validation/seed-10003_47889cf64d9415e2",
        "e355bcf18dcb266a143f029eef01b7aea589f52ac027797e06b7fc5bfaf4a4f4",
    ),
    (
        4,
        "train",
        K1V_ROOT / "cache/uknit64/r5/train/seed-4_799708c8cdcb079f",
        "82354b606866a19e63d01690543fc30c5282996b00505fd8dad9f3ea9407b50f",
    ),
    (
        4,
        "validation",
        K1V_ROOT / "cache/uknit64/r5/validation/seed-10004_4410ce0a8fb4cef5",
        "df2d112c255efc2f60140f35fb2814ce989e492d4a57da28f3c3803aa522a3f9",
    ),
)


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=EXPECTED_PAIRS,
        difference_profile=None,
        difference_member=0,
    )


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        condition = MODEL_TO_CONDITION.get(str(task.get("model_key")))
        if condition is None:
            continue
        key = (int(task["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AB task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AB task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    expected_keys = {
        3: (
            0x44444444444444444444444444444444,
            0x55555555555555555555555555555555,
        ),
        4: (
            0x66666666666666666666666666666666,
            0x77777777777777777777777777777777,
        ),
    }
    return (
        len(tasks) == 4
        and set(mapped) == EXPECTED_KEYS
        and all(
            task.get("cipher_key") == "uknit64"
            and task.get("rounds") == 5
            and task.get("seed") == seed
            and task.get("model_key") == CONTROL_MODELS[condition]
            and task.get("samples_per_class") == 2048
            and task.get("validation_samples_total") == 2048
            and task.get("pairs_per_sample") == EXPECTED_PAIRS
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and task.get("learning_rate") == BASE_LR
            and task.get("weight_decay") == 1e-5
            and task.get("lr_scheduler") == "none"
            and task.get("target_epochs") == EXPECTED_EPOCHS
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and task.get("optimizer_state_transition") == "reset_each_stage"
            and (task.get("train_key"), task.get("validation_key"))
            == expected_keys[seed]
            and task.get("model_options", {}).get("virtual_projection_slots") == 16
            and task.get("model_options", {}).get("input_difference_hex")
            == "0x0000400000000000"
            for (seed, condition), task in mapped.items()
        )
    )


def source_cache_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, split, path, expected_digest in SOURCE_CACHE_ROWS:
        dataset = load_cache(path)
        observed = differential_dataset_sha256(dataset)
        rows.append(
            {
                "seed": seed,
                "split": split,
                "cache_dir": str(path),
                "rows": int(dataset.labels.shape[0]),
                "dataset_sha256": observed,
                "expected_dataset_sha256": expected_digest,
                "digest_matches": observed == expected_digest,
            }
        )
    return rows


def source_binding_checks(cache_rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    k1aa_gate = read_json(SOURCE_PATHS["k1aa_gate"])
    k1v_gate = read_json(SOURCE_PATHS["k1v_gate"])
    return {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == SOURCE_DIGESTS[name]
            for name, path in SOURCE_PATHS.items()
        ),
        "k1aa_completed_pass_exact": (
            k1aa_gate.get("status") == "pass"
            and k1aa_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1aa_virtual_slots_supported"
            and not k1aa_gate.get("failed_protocol_checks")
            and not k1aa_gate.get("failed_research_checks")
        ),
        "k1v_completed_pass_exact": (
            k1v_gate.get("status") == "pass"
            and k1v_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1v_16pair_added_value_supported"
            and not k1v_gate.get("failed_protocol_checks")
        ),
        "four_bound_16pair_caches_exact": len(cache_rows) == 4
        and all(bool(row.get("digest_matches")) for row in cache_rows)
        and {
            (int(row["seed"]), str(row["split"])) for row in cache_rows
        }
        == {(seed, split) for seed in (3, 4) for split in ("train", "validation")},
    }


def build_readiness(
    tasks: Sequence[Mapping[str, Any]],
    *,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    mapped = task_map(tasks)
    rows = source_cache_manifest() if cache_rows is None else list(cache_rows)
    source_checks = source_binding_checks(rows)
    models = {
        key: build_k1aa_control(
            task=task,
            condition=key[1],
            input_bits=EXPECTED_INPUT_BITS,
        )
        for key, task in mapped.items()
    }
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }
    parameter_counts = {
        int(model_metadata(model)["trainable_parameter_count"])
        for model in models.values()
    }
    config = TrainingConfig(
        epochs=EXPECTED_EPOCHS,
        batch_size=64,
        learning_rate=BASE_LR,
        optimizer="adam",
        weight_decay=1e-5,
        lr_scheduler="none",
        loss="mse",
    )
    model_checks: dict[str, bool] = {}
    for key, model in models.items():
        named = dict(model.named_parameters())
        optimizer = make_optimizer(model, config)
        label = f"seed{key[0]}_{key[1]}"
        model_checks[f"{label}_geometry_optimizer_exact"] = (
            VIRTUAL_PARAMETER in named
            and tuple(named[VIRTUAL_PARAMETER].shape) == VIRTUAL_SHAPE
            and len(optimizer.param_groups) == 1
            and float(optimizer.param_groups[0]["lr"]) == BASE_LR
            and not hasattr(model, "optimizer_parameter_lr_multipliers")
            and EXPECTED_INPUT_BITS
            // (2 * int(model.runtime_structure.block_bits))
            == EXPECTED_PAIRS
        )
    fixture = torch.as_tensor(
        np.random.default_rng(20260729).integers(
            0, 2, size=(8, EXPECTED_INPUT_BITS), dtype=np.uint8
        ),
        dtype=torch.float32,
    )
    exact = models[(3, "virtual_slot_exact")]
    wrong = models[(3, "virtual_slot_wrong_sbox")]
    wrong.load_state_dict(deepcopy(exact.state_dict()), strict=True)
    exact.eval()
    wrong.eval()
    with torch.no_grad():
        exact_logits = exact(fixture)
        wrong_logits = wrong(fixture)
    protocol_checks = {
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        **source_checks,
        "identical_state_geometry": len(geometries) == 1,
        "parameter_count_unchanged": parameter_counts == {EXPECTED_PARAMETER_COUNT},
        **model_checks,
        "sixteen_pair_shared_state_controls_observable": (
            exact_logits.shape == wrong_logits.shape == (8, 1)
            and torch.isfinite(exact_logits).all()
            and torch.isfinite(wrong_logits).all()
            and not torch.equal(exact_logits, wrong_logits)
        ),
        "no_identity_or_runtime_position_parameters": all(
            model.uses_cipher_identity is False
            and model.uses_absolute_cell_or_bit_identity is False
            and model.uses_runtime_native_cell_slots is False
            for model in models.values()
        ),
    }
    status = "pass" if protocol_checks and all(protocol_checks.values()) else "fail"
    return {
        "run_id": RUN_ID,
        "status": status,
        "optimizer_step_authorized": status == "pass",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "source_cache_manifest": rows,
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": readiness.get("status") == "pass"
        and readiness.get("optimizer_step_authorized") is True
        and all(readiness.get("protocol_checks", {}).values()),
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "four_training_rows_complete": len(result_rows) == 4
        and set(rows) == EXPECTED_KEYS,
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "eight_source_cache_reuses_exact": cache_reuse_protocol(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(auc(row)) for row in rows.values()),
    }
    research_checks: dict[str, bool] = {}
    seed_results: dict[str, Any] = {}
    for seed in (3, 4):
        exact = auc(rows[(seed, "virtual_slot_exact")]) if rows else math.nan
        wrong = auc(rows[(seed, "virtual_slot_wrong_sbox")]) if rows else math.nan
        four_pair = K1AA_4PAIR_AUCS[seed]
        k1v_invariant = K1V_INVARIANT_16PAIR_AUCS[seed]
        research_checks[f"seed{seed}_sixteen_pair_added_value"] = (
            exact - four_pair >= ADDED_VALUE_MARGIN
        )
        research_checks[f"seed{seed}_beats_wrong_sbox"] = (
            exact - wrong >= SEMANTIC_MARGIN
        )
        research_checks[f"seed{seed}_retains_k1v_invariant"] = (
            exact >= k1v_invariant - K1V_RETENTION_TOLERANCE
        )
        seed_results[str(seed)] = {
            "exact_16pair_auc": exact,
            "wrong_sbox_16pair_auc": wrong,
            "k1aa_exact_4pair_auc": four_pair,
            "k1v_invariant_16pair_auc": k1v_invariant,
            "exact16_minus_exact4": exact - four_pair,
            "exact16_minus_wrong_sbox16": exact - wrong,
            "exact16_minus_k1v_invariant16": exact - k1v_invariant,
        }
    protocol_valid = all(protocol_checks.values())
    all_research = bool(research_checks) and all(research_checks.values())
    semantic_failed = any(
        not passed
        for name, passed in research_checks.items()
        if name.endswith("beats_wrong_sbox")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1ab_protocol_invalid"
        next_action = "repair only the failed protocol binding and rerun unchanged"
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1ab_16pair_supported"
        next_action = (
            "retain sixteen pairs for K1-AA and run a separate Dialga r4 "
            "retention check before choosing any remote uKNIT scale"
        )
    elif semantic_failed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ab_semantic_attribution_failed"
        next_action = "retain four pairs and audit sixteen-pair semantic aggregation"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ab_pair_added_value_failed"
        next_action = (
            "retain four pairs; do not spend additional query budget or remote scale"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "added_value_margin": ADDED_VALUE_MARGIN,
            "semantic_margin": SEMANTIC_MARGIN,
            "k1v_retention_tolerance": K1V_RETENTION_TOLERANCE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class K1-AA four-to-sixteen-pair diagnostic; "
            "not formal scale, attack, SOTA, transfer, or optimal-query evidence"
        ),
        "blocked_actions": [
            "remote scale or pairs beyond sixteen",
            "architecture, data, difference, epoch, seed, or learning-rate changes",
            "averaging seeds to hide a failed pair or semantic gate",
        ],
    }


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AB result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AB result panel is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == 4 and all(
            row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == EXPECTED_PAIRS
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("virtual_projection_slots") == 16
            and row.get("virtual_projection_parameter") == VIRTUAL_PARAMETER
            and row.get("training", {}).get("input_bits") == EXPECTED_INPUT_BITS
            and row.get("training", {}).get("train_rows") == EXPECTED_TRAIN_ROWS
            and row.get("training", {}).get("validation_rows")
            == EXPECTED_VALIDATION_ROWS
            and row.get("training", {}).get("epochs") == EXPECTED_EPOCHS
            and row.get("training", {}).get("epochs_ran") == EXPECTED_EPOCHS
            and row.get("training", {}).get("learning_rate") == BASE_LR
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and Path(
                str(row.get("training", {}).get("checkpoint_output", ""))
            ).is_file()
            for row in rows
        )
    except (TypeError, ValueError):
        return False


def cache_reuse_protocol(rows: Sequence[Mapping[str, Any]]) -> bool:
    reuses = [row for row in rows if row.get("event") == "cache_reuse"]
    creates = [row for row in rows if row.get("event") in {"cache_start", "cache_done"}]
    return len(reuses) == 8 and not creates


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **dict(result)}
        for seed, result in sorted(gate.get("seed_results", {}).items())
    ]


def load_cache(path: Path) -> DiskDifferentialDataset:
    metadata = path / "metadata.json"
    features = path / "features.npy"
    labels = path / "labels.npy"
    if not all(item.is_file() for item in (metadata, features, labels)):
        raise ValueError(f"missing K1-AB source cache payload: {path}")
    return DiskDifferentialDataset(
        features=np.load(features, mmap_mode="r"),
        labels=np.load(labels, mmap_mode="r"),
        metadata=read_json(metadata),
        cache_dir=path,
    )


def auc(row: Mapping[str, Any]) -> float:
    return float(row["metrics"]["auc"])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "source_cache_manifest",
    "task_map",
]
