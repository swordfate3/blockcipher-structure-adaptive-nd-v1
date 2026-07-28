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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import input_geometry
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1w import (
    source_cache_manifest as k1w_source_cache_manifest,
)
from blockcipher_nd.training.optim import make_optimizer
from blockcipher_nd.training.types import TrainingConfig


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_ctspn_compact_projection_update_k1y_2048_seed3_seed4_20260728"
CONTROL_MODELS = {
    "projection16x_exact": "runtime_spn_ct_k1y_compact_histogram_true",
    "projection16x_wrong_sbox": "runtime_spn_ct_k1y_compact_histogram_wrong_sbox",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
EXPECTED_SEEDS = (3, 4)
EXPECTED_KEYS = {
    (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_MODELS
}
EXPECTED_PARAMETER_COUNT = 137_516
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
BASE_LR = 1e-4
PROJECTION_LR_MULTIPLIER = 16.0
PROJECTION_LR = BASE_LR * PROJECTION_LR_MULTIPLIER
PROJECTION_PARAMETER = "backbone.histogram_projection.0.weight"
PROJECTION_SHAPE = (128, 40)
SEMANTIC_MARGIN = 0.010
IMPROVEMENT_MARGIN = 0.020
AUC_FLOOR = 0.550
ANCHOR_TOLERANCE = 0.020
K1W_EXACT_AUCS = {3: 0.5083932876586914, 4: 0.528264045715332}
K1T_INVARIANT_AUCS = {3: 0.5654244422912598, 4: 0.5940475463867188}
K1X_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_uknit_family_ctspn_compact_optimization_geometry_k1x_20260728"
)
K1W_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel_20260728"
)
K1Z_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_uknit_family_ctspn_compact_branch_interference_k1z_20260728_clean"
)
SOURCE_PATHS = {
    "k1w_gate": K1W_ROOT / "gate.json",
    "k1w_results": K1W_ROOT / "results.jsonl",
    "k1x_gate": K1X_ROOT / "gate.json",
    "k1z_gate": K1Z_ROOT / "gate.json",
    "k1z_results": K1Z_ROOT / "results.jsonl",
    "k1z_alpha_grid": K1Z_ROOT / "alpha_grid.jsonl",
}
SOURCE_DIGESTS = {
    "k1w_gate": "8f94cd31798638313d21c632445004ceb9d3fee545b5d3813b1ed6e4b998e338",
    "k1w_results": "75a7bdad3fb64b562c92545f4734e14dfad6c2d002b0099c5c02c0a1495a37e7",
    "k1x_gate": "ceae8bca25b0b3a9af034d02898d1233c491b4865f8a28e7cebfb1489f17b0d9",
    "k1z_gate": "1855270a3c8a589c1e1d62c0822907bf0eeaebab7490c1e30d9e365e9c67bf94",
    "k1z_results": "8ab1dca2c5a631769d3a0e0d21c3b1577e8e22e05f8792e23a294e49eb176e0d",
    "k1z_alpha_grid": "5434881b688d556c2bb255dc37a60f24e1ca3c9114984668fa24d14fdd2ddc04",
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
            raise ValueError(f"duplicate K1-Y task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-Y task matrix is incomplete")
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
            and options.get("histogram_projection_lr_multiplier")
            == PROJECTION_LR_MULTIPLIER
            and options.get("input_difference_hex")
            == "0x0000400000000000"
        )
    except (TypeError, ValueError):
        return False


def build_k1y_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = 512,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-Y condition")
    _, pair_bits = input_geometry("uknit64")
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=deepcopy(dict(task["model_options"])),
    )


def source_cache_manifest() -> list[dict[str, Any]]:
    return [
        row
        for row in k1w_source_cache_manifest()
        if row.get("cipher_key") == "uknit64"
    ]


def source_binding_checks(cache_rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    k1x_gate = read_json(SOURCE_PATHS["k1x_gate"])
    k1z_gate = read_json(SOURCE_PATHS["k1z_gate"])
    return {
        "source_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == SOURCE_DIGESTS[name]
            for name, path in SOURCE_PATHS.items()
        ),
        "four_bound_source_caches_exact": len(cache_rows) == 4
        and all(bool(row.get("digest_matches")) for row in cache_rows)
        and {
            (int(row["seed"]), str(row["split"])) for row in cache_rows
        }
        == {(seed, split) for seed in EXPECTED_SEEDS for split in ("train", "validation")},
        "k1x_gradient_relation_verified_but_held": (
            k1x_gate.get("status") == "hold"
            and k1x_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1x_optimization_geometry_not_sufficient"
            and all(
                k1x_gate.get("research_checks", {}).get(
                    f"seed{seed}_effective_update_ratio_16x"
                )
                is True
                for seed in EXPECTED_SEEDS
            )
        ),
        "k1z_inference_rescaling_insufficient": (
            k1z_gate.get("status") == "hold"
            and k1z_gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1z_inference_rescaling_insufficient_optimization_unresolved"
            and not k1z_gate.get("failed_protocol_checks")
        ),
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
        key: build_k1y_control(task=task, condition=key[1])
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
    optimizer_groups: dict[str, Any] = {}
    group_checks: dict[str, bool] = {}
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
        optimizer = make_optimizer(model, config)
        groups = [
            {
                "name": str(group.get("parameter_group_name")),
                "lr": float(group["lr"]),
                "lr_multiplier": float(group.get("lr_multiplier", math.nan)),
                "parameter_count": sum(
                    int(parameter.numel()) for parameter in group["params"]
                ),
                "tensor_count": len(group["params"]),
            }
            for group in optimizer.param_groups
        ]
        optimizer_groups[f"seed{key[0]}:{key[1]}"] = groups
        accelerated = [group for group in groups if group["name"] == PROJECTION_PARAMETER]
        default = [group for group in groups if group["name"] == "default"]
        parameter = dict(model.named_parameters())[PROJECTION_PARAMETER]
        group_checks[f"seed{key[0]}_{key[1]}_optimizer_groups_exact"] = (
            len(groups) == 2
            and len(accelerated) == 1
            and len(default) == 1
            and accelerated[0]["lr"] == PROJECTION_LR
            and accelerated[0]["lr_multiplier"] == PROJECTION_LR_MULTIPLIER
            and accelerated[0]["tensor_count"] == 1
            and accelerated[0]["parameter_count"] == parameter.numel()
            and tuple(parameter.shape) == PROJECTION_SHAPE
            and default[0]["lr"] == BASE_LR
            and default[0]["lr_multiplier"] == 1.0
        )
    rng = np.random.default_rng(20260728)
    fixture = torch.as_tensor(
        rng.integers(0, 2, size=(9, 512), dtype=np.uint8),
        dtype=torch.float32,
    )
    forward_errors: dict[str, float] = {}
    for seed in EXPECTED_SEEDS:
        task = mapped[(seed, "projection16x_exact")]
        k1y = models[(seed, "projection16x_exact")]
        k1w = build_model(
            "runtime_spn_ct_k1w_compact_histogram_true",
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options={
                key: value
                for key, value in deepcopy(dict(task["model_options"])).items()
                if key != "histogram_projection_lr_multiplier"
            },
        )
        k1w.load_state_dict(k1y.state_dict(), strict=True)
        k1w.eval()
        k1y.eval()
        with torch.no_grad():
            forward_errors[str(seed)] = float((k1w(fixture) - k1y(fixture)).abs().max())
    protocol_checks = {
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        **source_checks,
        "identical_state_geometry": len(geometries) == 1,
        "parameter_count_unchanged": parameter_counts == {EXPECTED_PARAMETER_COUNT},
        "k1w_k1y_forward_equivalent": all(
            error <= 1e-7 for error in forward_errors.values()
        ),
        **group_checks,
        "no_identity_or_position_parameters": all(
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
        "optimizer_parameter_groups": optimizer_groups,
        "k1w_k1y_forward_max_errors": forward_errors,
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
        exact = auc(rows[(seed, "projection16x_exact")]) if rows else math.nan
        wrong = auc(rows[(seed, "projection16x_wrong_sbox")]) if rows else math.nan
        k1w = K1W_EXACT_AUCS[seed]
        k1t = K1T_INVARIANT_AUCS[seed]
        retention = max(AUC_FLOOR, k1t - ANCHOR_TOLERANCE)
        research_checks[f"seed{seed}_retains_k1t_anchor"] = exact >= retention
        research_checks[f"seed{seed}_beats_wrong_sbox"] = (
            exact - wrong >= SEMANTIC_MARGIN
        )
        research_checks[f"seed{seed}_improves_k1w"] = (
            exact - k1w >= IMPROVEMENT_MARGIN
        )
        seed_results[str(seed)] = {
            "projection16x_exact_auc": exact,
            "projection16x_wrong_sbox_auc": wrong,
            "k1w_exact_auc": k1w,
            "k1t_invariant_auc": k1t,
            "retention_threshold": retention,
            "exact_minus_wrong_sbox": exact - wrong,
            "exact_minus_k1w": exact - k1w,
            "exact_minus_k1t": exact - k1t,
        }
    protocol_valid = all(protocol_checks.values())
    research_valid = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1y_protocol_invalid"
        next_action = "repair only the failed plan, cache, optimizer-group, checkpoint, or artifact binding"
    elif research_valid:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1y_projection_update_supported"
        next_action = "freeze K1-Y and compare four versus sixteen pairs inside the selected compact architecture"
    elif any(
        not passed
        for name, passed in research_checks.items()
        if name.endswith("beats_wrong_sbox")
    ):
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1y_semantic_attribution_failed"
        next_action = "stop compact projection optimization and redesign the invariant parameterization"
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1y_anchor_retention_failed"
        next_action = "stop LR-multiplier tuning and redesign runtime-stable forward/backward parameterization"
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
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "semantic_margin": SEMANTIC_MARGIN,
            "k1w_improvement_margin": IMPROVEMENT_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class compact projection optimizer diagnostic; "
            "not formal scale, attack, SOTA, transfer, or family ceiling"
        ),
        "blocked_actions": [
            "remote scale or sixteen pairs",
            "other LR multipliers, sweeps, epochs, data, seeds, or differences",
            "averaging seeds to hide a failed retention or semantic gate",
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
            raise ValueError(f"duplicate K1-Y result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != EXPECTED_KEYS:
        raise ValueError("K1-Y result panel is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == 4 and all(
            row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("histogram_projection_lr_multiplier")
            == PROJECTION_LR_MULTIPLIER
            and row.get("histogram_projection_lr_parameter")
            == PROJECTION_PARAMETER
            and row.get("training", {}).get("train_rows") == EXPECTED_TRAIN_ROWS
            and row.get("training", {}).get("validation_rows")
            == EXPECTED_VALIDATION_ROWS
            and row.get("training", {}).get("epochs") == EXPECTED_EPOCHS
            and row.get("training", {}).get("epochs_ran") == EXPECTED_EPOCHS
            and row.get("training", {}).get("learning_rate") == BASE_LR
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and Path(str(row.get("training", {}).get("checkpoint_output", ""))).is_file()
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
    "PROJECTION_LR",
    "PROJECTION_LR_MULTIPLIER",
    "PROJECTION_PARAMETER",
    "RUN_ID",
    "adjudicate",
    "build_k1y_control",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "source_cache_manifest",
    "task_map",
]
