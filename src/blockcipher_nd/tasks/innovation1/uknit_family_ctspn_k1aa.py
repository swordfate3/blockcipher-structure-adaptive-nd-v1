from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import input_geometry
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1y import (
    K1T_INVARIANT_AUCS,
    source_cache_manifest,
)
from blockcipher_nd.training.optim import make_optimizer
from blockcipher_nd.training.types import TrainingConfig


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_ctspn_virtual_slot_projection_"
    "k1aa_2048_seed3_seed4_20260728"
)
CONTROL_MODELS = {
    "virtual_slot_exact": "runtime_spn_ct_k1aa_virtual_slot_histogram_true",
    "virtual_slot_wrong_sbox": (
        "runtime_spn_ct_k1aa_virtual_slot_histogram_wrong_sbox"
    ),
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
EXPECTED_SEEDS = (3, 4)
EXPECTED_KEYS = {
    (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_MODELS
}
EXPECTED_PARAMETER_COUNT = 214_316
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
BASE_LR = 1e-4
VIRTUAL_SLOTS = 16
VIRTUAL_PARAMETER = "backbone.histogram_projection.0.virtual_slot_weights"
VIRTUAL_SHAPE = (16, 128, 40)
SHARED_BIAS_PARAMETER = "backbone.histogram_projection.0.bias"
SHARED_BIAS_SHAPE = (128,)
SEMANTIC_MARGIN = 0.010
K1T_TOLERANCE = 0.010
K1Y_TOLERANCE = 0.005
AUC_FLOOR = 0.550
K1Y_EXACT_AUCS = {3: 0.5488901138305664, 4: 0.5938735008239746}
K1Y_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_compact_projection_update_"
    "k1y_2048_seed3_seed4_20260728"
)
SOURCE_PATHS = {
    "k1y_gate": K1Y_ROOT / "gate.json",
    "k1y_results": K1Y_ROOT / "results.jsonl",
    "k1y_validation": K1Y_ROOT / "validation.json",
}
SOURCE_DIGESTS = {
    "k1y_gate": "2d6106a1af8cfc6c3a1f2a2ed98a96db890e83908e8b49dec27b0f251a194de0",
    "k1y_results": "0961855ad6406ab71933c55a217940252861bd7e8e6ab48189ce3f45f022bb54",
    "k1y_validation": "a12b3c4f312f15bde5cb7421bb886020e7ea6b4cdde7d29fc2e9c1b89bb6abd9",
}


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
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
            raise ValueError(f"duplicate K1-AA task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AA task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == len(EXPECTED_KEYS)
        and set(mapped) == EXPECTED_KEYS
        and all(task_protocol_frozen(key, task) for key, task in mapped.items())
    )


def task_protocol_frozen(
    key: tuple[int, str],
    task: Mapping[str, Any],
) -> bool:
    seed, condition = key
    options = task.get("model_options", {})
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
    try:
        return (
            task.get("cipher_key") == "uknit64"
            and task.get("rounds") == 5
            and task.get("seed") == seed
            and task.get("model_key") == CONTROL_MODELS[condition]
            and task.get("samples_per_class") == 2048
            and task.get("validation_samples_total") == 2048
            and task.get("pairs_per_sample") == 4
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
            and options.get("runtime_structure_path")
            == "configs/runtime/spn/uknit64.json"
            and options.get("runtime_round_start") == 3
            and options.get("runtime_rounds") == 2
            and options.get("virtual_projection_slots") == VIRTUAL_SLOTS
            and "histogram_projection_lr_multiplier" not in options
            and options.get("input_difference_hex")
            == "0x0000400000000000"
        )
    except (TypeError, ValueError):
        return False


def build_k1aa_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-AA condition")
    _, pair_bits = input_geometry("uknit64")
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def source_binding_checks(
    cache_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    gate = read_json(SOURCE_PATHS["k1y_gate"])
    return {
        "k1y_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == SOURCE_DIGESTS[name]
            for name, path in SOURCE_PATHS.items()
        ),
        "k1y_near_retention_hold_exact": (
            gate.get("status") == "hold"
            and gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1y_anchor_retention_failed"
            and gate.get("failed_research_checks")
            == ["seed3_retains_k1t_anchor"]
            and not gate.get("failed_protocol_checks")
        ),
        "four_bound_source_caches_exact": len(cache_rows) == 4
        and all(bool(row.get("digest_matches")) for row in cache_rows)
        and {
            (int(row["seed"]), str(row["split"])) for row in cache_rows
        }
        == {(seed, split) for seed in EXPECTED_SEEDS for split in ("train", "validation")},
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
        key: build_k1aa_control(task=task, condition=key[1])
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
    tensor_checks: dict[str, bool] = {}
    optimizer_checks: dict[str, bool] = {}
    config = TrainingConfig(
        epochs=EXPECTED_EPOCHS,
        batch_size=EXPECTED_BATCH_SIZE,
        learning_rate=BASE_LR,
        optimizer="adam",
        weight_decay=1e-5,
        lr_scheduler="none",
        loss="mse",
    )
    for key, model in models.items():
        named = dict(model.named_parameters())
        virtual = named.get(VIRTUAL_PARAMETER)
        bias = named.get(SHARED_BIAS_PARAMETER)
        label = f"seed{key[0]}_{key[1]}"
        tensor_checks[f"{label}_virtual_geometry_exact"] = (
            virtual is not None
            and tuple(virtual.shape) == VIRTUAL_SHAPE
            and bias is not None
            and tuple(bias.shape) == SHARED_BIAS_SHAPE
            and getattr(model, "virtual_projection_slots", None) == VIRTUAL_SLOTS
            and getattr(model, "virtual_projection_parameter", None)
            == VIRTUAL_PARAMETER
        )
        optimizer = make_optimizer(model, config)
        optimizer_checks[f"{label}_ordinary_optimizer_exact"] = (
            len(optimizer.param_groups) == 1
            and float(optimizer.param_groups[0]["lr"]) == BASE_LR
            and not hasattr(model, "optimizer_parameter_lr_multipliers")
        )

    rng = np.random.default_rng(20260728)
    fixture = torch.as_tensor(
        rng.integers(0, 2, size=(9, 512), dtype=np.uint8),
        dtype=torch.float32,
    )
    forward_errors: dict[str, float] = {}
    gradient_audits: dict[str, dict[str, float]] = {}
    for seed in EXPECTED_SEEDS:
        task = mapped[(seed, "virtual_slot_exact")]
        virtual_model = models[(seed, "virtual_slot_exact")]
        compact_options = deepcopy(dict(task["model_options"]))
        compact_options.pop("virtual_projection_slots", None)
        compact_model = build_model(
            "runtime_spn_ct_k1w_compact_histogram_true",
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=compact_options,
        )
        load_effective_compact_state(virtual_model, compact_model)
        virtual_model.eval()
        compact_model.eval()
        with torch.no_grad():
            forward_errors[str(seed)] = float(
                (virtual_model(fixture) - compact_model(fixture)).abs().max()
            )
        gradient_audits[str(seed)] = audit_virtual_projection_gradient(
            virtual_model.backbone.histogram_projection[0]
        )

    protocol_checks = {
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        **source_checks,
        "identical_state_geometry": len(geometries) == 1,
        "parameter_count_matches_k1t": parameter_counts == {EXPECTED_PARAMETER_COUNT},
        **tensor_checks,
        "effective_compact_forward_equivalent": all(
            error <= 1e-7 for error in forward_errors.values()
        ),
        "virtual_slot_gradient_geometry_exact": all(
            audit["slot_gradient_relative_error"] <= 1e-9
            and audit["effective_gradient_ratio_error"] <= 1e-9
            for audit in gradient_audits.values()
        ),
        **optimizer_checks,
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
        "effective_compact_forward_max_errors": forward_errors,
        "gradient_audits": gradient_audits,
    }


def load_effective_compact_state(
    virtual_model: nn.Module,
    compact_model: nn.Module,
) -> None:
    virtual_state = virtual_model.state_dict()
    compact_state = compact_model.state_dict()
    virtual_weight = virtual_state[VIRTUAL_PARAMETER]
    for name in compact_state:
        if name == "backbone.histogram_projection.0.weight":
            compact_state[name] = virtual_weight.sum(dim=0)
        else:
            compact_state[name] = virtual_state[name]
    compact_model.load_state_dict(compact_state, strict=True)


def audit_virtual_projection_gradient(layer: nn.Module) -> dict[str, float]:
    source = layer.double()
    compact = nn.Linear(40, 128, bias=True).double()
    with torch.no_grad():
        compact.weight.copy_(source.effective_weight())
        compact.bias.copy_(source.bias)
    inputs = torch.as_tensor(
        np.random.default_rng(20260728).normal(size=(7, 40)),
        dtype=torch.float64,
    )
    source.zero_grad(set_to_none=True)
    compact.zero_grad(set_to_none=True)
    source(inputs).square().sum().backward()
    compact(inputs).square().sum().backward()
    slot_gradients = source.virtual_slot_weights.grad
    compact_gradient = compact.weight.grad
    if slot_gradients is None or compact_gradient is None:
        raise ValueError("K1-AA gradient audit did not populate projection gradients")
    denominator = max(float(compact_gradient.norm()), torch.finfo(torch.float64).eps)
    slot_error = float(
        (slot_gradients - compact_gradient.unsqueeze(0)).norm()
        / (math.sqrt(VIRTUAL_SLOTS) * denominator)
    )
    effective_error = float(
        (slot_gradients.sum(dim=0) - VIRTUAL_SLOTS * compact_gradient).norm()
        / (VIRTUAL_SLOTS * denominator)
    )
    return {
        "slot_gradient_relative_error": slot_error,
        "effective_gradient_ratio": float(
            slot_gradients.sum(dim=0).norm() / denominator
        ),
        "effective_gradient_ratio_error": effective_error,
        "optimizer_steps": 0,
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
    for seed in EXPECTED_SEEDS:
        exact = auc(rows[(seed, "virtual_slot_exact")]) if rows else math.nan
        wrong = auc(rows[(seed, "virtual_slot_wrong_sbox")]) if rows else math.nan
        k1t = K1T_INVARIANT_AUCS[seed]
        k1y = K1Y_EXACT_AUCS[seed]
        retention = max(AUC_FLOOR, k1t - K1T_TOLERANCE, k1y - K1Y_TOLERANCE)
        research_checks[f"seed{seed}_retains_anchor"] = exact >= retention
        research_checks[f"seed{seed}_beats_wrong_sbox"] = (
            exact - wrong >= SEMANTIC_MARGIN
        )
        seed_results[str(seed)] = {
            "virtual_slot_exact_auc": exact,
            "virtual_slot_wrong_sbox_auc": wrong,
            "k1t_invariant_auc": k1t,
            "k1y_exact_auc": k1y,
            "retention_threshold": retention,
            "exact_minus_wrong_sbox": exact - wrong,
            "exact_minus_k1t": exact - k1t,
            "exact_minus_k1y": exact - k1y,
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
        decision = "innovation1_uknit_family_ctspn_k1aa_protocol_invalid"
        next_action = "repair only the failed protocol binding and rerun unchanged"
    elif all_research:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1aa_virtual_slots_supported"
        next_action = (
            "retain K1-AA and compare 4 versus 16 pairs inside the same architecture "
            "as one separate local diagnostic"
        )
    elif semantic_failed:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1aa_semantic_attribution_failed"
        next_action = (
            "reject virtual-slot scaling and audit why correct S-box semantics vanish"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1aa_anchor_retention_failed"
        next_action = (
            "audit virtual-slot initialization and checkpoint dynamics without "
            "changing pairs, data, epochs, seeds, or learning rate"
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
            "auc_floor": AUC_FLOOR,
            "k1t_tolerance": K1T_TOLERANCE,
            "k1y_tolerance": K1Y_TOLERANCE,
            "semantic_margin": SEMANTIC_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class virtual-slot optimizer-geometry diagnostic; "
            "not formal scale, attack, SOTA, transfer, or family ceiling"
        ),
        "blocked_actions": [
            "remote scale or sixteen pairs inside K1-AA",
            "other slot counts, LR tuning, epochs, data, seeds, or differences",
            "averaging seeds to hide failed retention or semantic attribution",
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
            raise ValueError(f"duplicate K1-AA result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-AA result panel is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == 4 and all(
            row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("virtual_projection_slots") == VIRTUAL_SLOTS
            and row.get("virtual_projection_parameter") == VIRTUAL_PARAMETER
            and "histogram_projection_lr_multiplier" not in row
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


def auc(row: Mapping[str, Any]) -> float:
    return float(row["metrics"]["auc"])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BASE_LR",
    "CONTROL_MODELS",
    "EXPECTED_KEYS",
    "EXPECTED_PARAMETER_COUNT",
    "RUN_ID",
    "VIRTUAL_PARAMETER",
    "VIRTUAL_SHAPE",
    "adjudicate",
    "audit_virtual_projection_gradient",
    "build_k1aa_control",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "load_effective_compact_state",
    "read_tasks",
    "source_binding_checks",
    "task_map",
]
