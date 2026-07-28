from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1aa import (
    BASE_LR,
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    MODEL_TO_CONDITION,
    VIRTUAL_PARAMETER,
    VIRTUAL_SHAPE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import input_geometry
from blockcipher_nd.training.optim import make_optimizer
from blockcipher_nd.training.types import TrainingConfig


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1_20260729"
)
EXPECTED_SEEDS = (0, 1)
EXPECTED_KEYS = {
    (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_MODELS
}
EXPECTED_PAIRS = 16
EXPECTED_INPUT_BITS = 4096
EXPECTED_INPUT_DIFFERENCE = 0x40
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
RETENTION_TOLERANCE = 0.020
SEMANTIC_MARGIN = 0.010
K1W_EXACT_AUCS = {0: 0.9604473114013672, 1: 0.9604053497314453}
K1W_WRONG_SBOX_AUCS = {0: 0.9609260559082031, 1: 0.9583034515380859}
K1N_EXACT_AUCS = {0: 0.9597501754760742, 1: 0.9547367095947266}
K1AB_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_virtual_slot_pair_count_"
    "k1ab_16pair_2048_seed3_seed4_20260729"
)
K1W_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_compact_invariant_"
    "k1w_2048_seed_panel_20260728"
)
SOURCE_PATHS = {
    "k1ab_gate": K1AB_ROOT / "gate.json",
    "k1ab_results": K1AB_ROOT / "results.jsonl",
    "k1w_gate": K1W_ROOT / "gate.json",
    "k1w_results": K1W_ROOT / "results.jsonl",
}
SOURCE_DIGESTS = {
    "k1ab_gate": "bf2969d29ecf531071c8524f4b2facc83c40927b1d14aab498aed49ea16a30f4",
    "k1ab_results": "aa3e17f9972a08f77aae7c47b8a4f8a060bc367876e280960c0fd5177b844f1a",
    "k1w_gate": "8f94cd31798638313d21c632445004ceb9d3fee545b5d3813b1ed6e4b998e338",
    "k1w_results": "75a7bdad3fb64b562c92545f4734e14dfad6c2d002b0099c5c02c0a1495a37e7",
}


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
            raise ValueError(f"duplicate K1-AC task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AC task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    train_key = 0
    validation_key = int("1" * 64, 16)
    return (
        len(tasks) == 4
        and set(mapped) == EXPECTED_KEYS
        and all(
            task.get("cipher_key") == "dialga128"
            and task.get("rounds") == 4
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
            == (train_key, validation_key)
            and task.get("difference_profile") in (None, "")
            and task.get("model_options", {}).get("runtime_structure_path")
            == "configs/runtime/spn/dialga128.json"
            and task.get("model_options", {}).get("runtime_round_start") == 2
            and task.get("model_options", {}).get("runtime_rounds") == 2
            and task.get("model_options", {}).get("virtual_projection_slots") == 16
            for (seed, condition), task in mapped.items()
        )
    )


def build_k1ac_control(
    *,
    task: Mapping[str, Any],
    condition: str,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-AC condition")
    _, pair_bits = input_geometry("dialga128")
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=EXPECTED_INPUT_BITS,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def source_binding_checks() -> dict[str, bool]:
    k1ab_gate = read_json(SOURCE_PATHS["k1ab_gate"])
    k1w_gate = read_json(SOURCE_PATHS["k1w_gate"])
    dialga_checks = {
        name: passed
        for name, passed in k1w_gate.get("research_checks", {}).items()
        if name.startswith("dialga128_")
    }
    return {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == SOURCE_DIGESTS[name]
            for name, path in SOURCE_PATHS.items()
        ),
        "k1ab_completed_pass_exact": (
            k1ab_gate.get("status") == "pass"
            and k1ab_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1ab_16pair_supported"
            and not k1ab_gate.get("failed_protocol_checks")
        ),
        "k1w_dialga_anchor_valid_exact": (
            k1w_gate.get("status") == "hold"
            and k1w_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1w_semantic_attribution_failed"
            and not k1w_gate.get("failed_protocol_checks")
            and len(dialga_checks) == 2
            and all(dialga_checks.values())
        ),
    }


def build_readiness(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks)
    models = {
        key: build_k1ac_control(task=task, condition=key[1])
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
    exact = models[(0, "virtual_slot_exact")]
    wrong = models[(0, "virtual_slot_wrong_sbox")]
    wrong.load_state_dict(deepcopy(exact.state_dict()), strict=True)
    exact.eval()
    wrong.eval()
    with torch.no_grad():
        exact_logits = exact(fixture)
        wrong_logits = wrong(fixture)
    protocol_checks = {
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        **source_binding_checks(),
        "identical_state_geometry": len(geometries) == 1,
        "parameter_count_unchanged": parameter_counts == {EXPECTED_PARAMETER_COUNT},
        **model_checks,
        "dialga_sixteen_pair_shared_state_controls_observable": (
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
    status = "pass" if all(protocol_checks.values()) else "fail"
    return {
        "run_id": RUN_ID,
        "status": status,
        "optimizer_step_authorized": status == "pass",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
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
        "four_cache_creations_then_four_reuses": cache_protocol_frozen(
            progress_rows
        ),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(auc(row)) for row in rows.values()),
    }
    research_checks: dict[str, bool] = {}
    seed_results: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        exact = auc(rows[(seed, "virtual_slot_exact")]) if rows else math.nan
        wrong = (
            auc(rows[(seed, "virtual_slot_wrong_sbox")]) if rows else math.nan
        )
        anchor = K1W_EXACT_AUCS[seed]
        research_checks[f"seed{seed}_retains_k1w_signal"] = (
            exact >= anchor - RETENTION_TOLERANCE
        )
        research_checks[f"seed{seed}_beats_wrong_sbox"] = (
            exact - wrong >= SEMANTIC_MARGIN
        )
        seed_results[str(seed)] = {
            "exact_16pair_auc": exact,
            "wrong_sbox_16pair_auc": wrong,
            "k1w_exact_4pair_auc": anchor,
            "k1w_wrong_sbox_4pair_auc": K1W_WRONG_SBOX_AUCS[seed],
            "k1n_exact_4pair_auc": K1N_EXACT_AUCS[seed],
            "exact16_minus_k1w_exact4": exact - anchor,
            "exact16_minus_wrong_sbox16": exact - wrong,
        }
    protocol_valid = all(protocol_checks.values())
    retention_failed = any(
        not passed
        for name, passed in research_checks.items()
        if name.endswith("retains_k1w_signal")
    )
    semantic_failed = any(
        not passed
        for name, passed in research_checks.items()
        if name.endswith("beats_wrong_sbox")
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1ac_protocol_invalid"
        next_action = "repair only the failed protocol binding and rerun unchanged"
    elif retention_failed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ac_signal_retention_failed"
        next_action = (
            "keep sixteen pairs uKNIT-specific, retain Dialga four-pair settings, "
            "and audit Dialga pair aggregation before family or remote-scale claims"
        )
    elif semantic_failed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed"
        next_action = (
            "run a zero-training same-checkpoint K1-AD audit on the identical "
            "Dialga validation caches before changing data, capacity, or network"
        )
    else:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1ac_"
            "retention_and_semantics_supported"
        )
        next_action = (
            "preregister a remote uKNIT r5 K1-AA sixteen-pair 65536/class "
            "seed3/4 exact-versus-wrong-Sbox medium diagnostic with disk cache"
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
            "k1w_signal_retention_tolerance": RETENTION_TOLERANCE,
            "exact_minus_wrong_sbox": SEMANTIC_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class Dialga r4 K1-AA sixteen-pair retention "
            "diagnostic; not formal scale, attack, SOTA, transfer, or optimal-query evidence"
        ),
        "blocked_actions": [
            "remote scale from K1-AC alone",
            "more samples, pairs, epochs, seeds, differences, MoE, or architecture changes",
            "averaging seeds or reporting Dialga AUC as correct-S-box attribution",
        ],
    }


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("cipher_key") != "dialga128":
            continue
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-AC result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AC result panel is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == 4 and all(
            row.get("cipher_key") == "dialga128"
            and row.get("rounds") in (None, 4)
            and row.get("input_difference") == EXPECTED_INPUT_DIFFERENCE
            and row.get("samples_per_class") == 2048
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


def cache_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    starts = [row for row in rows if row.get("event") == "cache_start"]
    completions = [row for row in rows if row.get("event") == "cache_done"]
    reuses = [row for row in rows if row.get("event") == "cache_reuse"]
    expected = {
        (seed, split) for seed in EXPECTED_SEEDS for split in ("train", "validation")
    }

    def keys(items: Sequence[Mapping[str, Any]]) -> set[tuple[int, str]]:
        return {(int(row["seed"]), str(row["split"])) for row in items}

    def geometry_exact(items: Sequence[Mapping[str, Any]]) -> bool:
        return all(
            row.get("pairs_per_sample") == EXPECTED_PAIRS
            and row.get("input_bits") == EXPECTED_INPUT_BITS
            for row in items
        )

    return (
        len(starts) == len(completions) == len(reuses) == 4
        and keys(starts) == keys(completions) == keys(reuses) == expected
        and geometry_exact(starts)
        and geometry_exact(completions)
        and geometry_exact(reuses)
    )


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **dict(result)}
        for seed, result in sorted(gate.get("seed_results", {}).items())
    ]


def auc(row: Mapping[str, Any]) -> float:
    return float(row["metrics"]["auc"])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CONTROL_MODELS",
    "EXPECTED_KEYS",
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "task_map",
]
