from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    EXPECTED_PARAMETER_COUNT,
    build_k1t_control,
)


RUN_ID = "i1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_20260731"
EXPECTED_SEEDS = (3, 4)
CONDITIONS = {
    "exact4": "runtime_spn_ct_k1t_position_histogram_true",
    "exact16": "runtime_spn_ct_k1t_position_histogram_true",
    "wrong16": "runtime_spn_ct_k1t_position_histogram_wrong_sbox",
}
EXPECTED_PAIRS = {"exact4": 4, "exact16": 16, "wrong16": 16}
EXPECTED_INPUT_BITS = {name: pairs * 128 for name, pairs in EXPECTED_PAIRS.items()}
EXPECTED_RESULT_ROWS = 6
EXPECTED_SAMPLES_PER_CLASS = 2_048
EXPECTED_TRAIN_ROWS = 4_096
EXPECTED_VALIDATION_ROWS = 2_048
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_CACHE_CREATIONS = 8
EXPECTED_CACHE_REUSES = 4
INPUT_DIFFERENCE = 0x0000400000000000
DIFFERENCE_PROFILE = "uknit64_k1q_cell11_r5"
KEYS = {
    3: (int("44" * 16, 16), int("55" * 16, 16)),
    4: (int("66" * 16, 16), int("77" * 16, 16)),
}
WEAK_AUC_FLOOR = 0.510
STRONG_AUC_FLOOR = 0.550
PAIR_GAIN_FLOOR = 0.010
SEMANTIC_GAP_FLOOR = 0.010


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def expected_keys() -> set[tuple[int, str]]:
    return {(seed, condition) for seed in EXPECTED_SEEDS for condition in CONDITIONS}


def _condition(value: Mapping[str, Any], *, result: bool) -> str | None:
    model = str(value.get("model" if result else "model_key", ""))
    pairs = int(value.get("pairs_per_sample", -1))
    if model == CONDITIONS["exact4"] and pairs == 4:
        return "exact4"
    if model == CONDITIONS["exact16"] and pairs == 16:
        return "exact16"
    if model == CONDITIONS["wrong16"] and pairs == 16:
        return "wrong16"
    return None


def _map(
    rows: Sequence[Mapping[str, Any]], *, result: bool, fail_closed: bool = True
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        condition = _condition(row, result=result)
        if condition is None:
            continue
        key = (int(row.get("seed", -1)), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-BV row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BV matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = _map(tasks, result=False, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and all(_task_frozen(task, seed=seed, condition=condition)
                for (seed, condition), task in mapped.items())
    )


def _task_frozen(task: Mapping[str, Any], *, seed: int, condition: str) -> bool:
    options = task.get("model_options", {})
    train_key, validation_key = KEYS[seed]
    return (
        task.get("cipher_key") == "uknit64"
        and int(task.get("rounds", -1)) == 6
        and int(task.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(task.get("validation_samples_total", -1)) == EXPECTED_VALIDATION_ROWS
        and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS[condition]
        and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
        and task.get("difference_profile") == DIFFERENCE_PROFILE
        and task.get("feature_encoding") == "ciphertext_pair_bits"
        and task.get("negative_mode") == "encrypted_random_plaintexts"
        and task.get("sample_structure") == "independent_pairs"
        and int(task.get("key_rotation_interval", -1)) == 0
        and int(task.get("train_key", -1)) == train_key
        and int(task.get("validation_key", -1)) == validation_key
        and task.get("loss") == "mse"
        and task.get("optimizer") == "adam"
        and task.get("optimizer_state_transition") == "reset_each_stage"
        and float(task.get("learning_rate", math.nan)) == 1e-4
        and float(task.get("weight_decay", math.nan)) == 1e-5
        and task.get("lr_scheduler") == "none"
        and task.get("checkpoint_metric") == "val_auc"
        and task.get("restore_best_checkpoint") is True
        and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
        and int(options.get("runtime_round_start", -1)) == 4
        and int(options.get("runtime_rounds", -1)) == 2
        and int(options.get("pair_embedding_dim", -1)) == 128
        and int(options.get("histogram_value_dim", -1)) == 8
    )


def build_readiness(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = _map(tasks, result=False, fail_closed=False)
    checks = {
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "mixed_pair_geometry_frozen": set(EXPECTED_PAIRS.values()) == {4, 16},
    }
    metrics: dict[str, Any] = {}
    error = ""
    if all(checks.values()):
        try:
            torch.manual_seed(20260731)
            fixtures = {
                name: torch.randint(0, 2, (8, bits), dtype=torch.float32)
                for name, bits in EXPECTED_INPUT_BITS.items()
            }
            models = {}
            for condition in CONDITIONS:
                control = (
                    "wrong_sbox_position_histogram_residual"
                    if condition == "wrong16"
                    else "exact_position_histogram_residual"
                )
                models[condition] = build_k1t_control(
                    task=mapped[(3, condition)], condition=control,
                    input_bits=EXPECTED_INPUT_BITS[condition],
                )
            geometries = {
                condition: tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
                for condition, model in models.items()
            }
            parameter_counts = {
                condition: sum(p.numel() for p in model.parameters() if p.requires_grad)
                for condition, model in models.items()
            }
            models["wrong16"].load_state_dict(models["exact16"].state_dict(), strict=True)
            outputs = {name: models[name](fixtures[name]) for name in CONDITIONS}
            target = torch.arange(8, dtype=torch.float32).remainder(2)
            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(outputs["exact16"]).flatten(), target
            )
            loss.backward()
            checks.update({
                "input_bits_512_and_2048_exact": {
                    name: int(value.shape[1]) for name, value in fixtures.items()
                } == EXPECTED_INPUT_BITS,
                "identical_state_geometry": len(set(geometries.values())) == 1,
                "parameter_count_214316_exact": set(parameter_counts.values()) == {EXPECTED_PARAMETER_COUNT},
                "shared_state_wrong_sbox_is_observable": not torch.equal(outputs["exact16"], outputs["wrong16"]),
                "finite_forward_and_backward": all(torch.isfinite(value).all() for value in outputs.values())
                and math.isfinite(float(loss.detach()))
                and any(p.grad is not None and torch.isfinite(p.grad).all() for p in models["exact16"].parameters()),
            })
            metrics = {
                "input_bits": dict(EXPECTED_INPUT_BITS),
                "parameter_counts": parameter_counts,
                "exact16_wrong16_max_abs_delta": float(
                    (outputs["exact16"] - outputs["wrong16"]).detach().abs().max()
                ),
                "backward_loss": float(loss.detach()),
            }
        except Exception as exc:  # readiness must fail closed with the concrete error
            error = f"{type(exc).__name__}: {exc}"
    status = "pass" if checks and all(checks.values()) and not error else "fail"
    return {
        "run_id": RUN_ID,
        "status": status,
        "checks": checks,
        "failed_checks": sorted(name for name, ok in checks.items() if not ok),
        "metrics": metrics,
        "error": error,
        "training_performed": False,
        "next_action": "launch the frozen six-row remote A6000 diagnostic" if status == "pass" else "repair only the failed readiness invariant",
    }


def result_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        mapped = _map(rows, result=True)
    except ValueError:
        return False
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        _result_frozen(row, seed=seed, condition=condition)
        for (seed, condition), row in mapped.items()
    )


def _result_frozen(row: Mapping[str, Any], *, seed: int, condition: str) -> bool:
    training = row.get("training", {})
    metrics = row.get("metrics", {})
    checkpoint = str(training.get("checkpoint_output", ""))
    return (
        row.get("cipher_key") == "uknit64"
        and int(row.get("rounds", -1)) == 6
        and int(row.get("seed", -1)) == seed
        and int(row.get("samples_per_class", -1)) == EXPECTED_SAMPLES_PER_CLASS
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS[condition]
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and int(row.get("runtime_structure_round_start", -1)) == 4
        and int(row.get("runtime_structure_loaded_rounds", -1)) == 2
        and int(training.get("input_bits", -1)) == EXPECTED_INPUT_BITS[condition]
        and int(training.get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(training.get("validation_rows", -1)) == EXPECTED_VALIDATION_ROWS
        and int(training.get("epochs", -1)) == EXPECTED_EPOCHS
        and int(training.get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and int(training.get("batch_size", -1)) == EXPECTED_BATCH_SIZE
        and str(training.get("device", "")).startswith("cuda")
        and training.get("selected_checkpoint") == "best"
        and training.get("restore_best_checkpoint") is True
        and _is_remote_path(checkpoint)
        and all(math.isfinite(float(metrics.get(name, math.nan))) for name in ("auc", "accuracy", "loss"))
    )


def cache_protocol_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    creations = [row for row in events if row.get("event") == "cache_start"]
    completions = [row for row in events if row.get("event") == "cache_done"]
    reuses = [row for row in events if row.get("event") == "cache_reuse"]
    return {
        "eight_disk_caches_created": len(creations) == EXPECTED_CACHE_CREATIONS,
        "eight_disk_caches_completed": len(completions) == EXPECTED_CACHE_CREATIONS,
        "four_exact16_wrong16_cache_reuses": len(reuses) == EXPECTED_CACHE_REUSES,
        "all_cache_paths_remote": bool(creations) and all(_is_remote_path(row.get("cache_path")) for row in creations),
        "cache_chunk_workers_frozen": all(
            int(row.get("chunk_size", -1)) == 1024 and int(row.get("workers", -1)) == 1
            for row in (*creations, *reuses)
        ),
        "run_done_present": any(row.get("event") == "run_done" for row in events),
    }


def adjudicate(
    *, tasks: Sequence[Mapping[str, Any]], result_rows: Sequence[Mapping[str, Any]],
    progress_events: Sequence[Mapping[str, Any]], source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    mapped = _map(result_rows, result=True, fail_closed=False)
    protocol_checks = {
        **dict(source_checks),
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS and set(mapped) == expected_keys(),
        "result_protocol_frozen_without_local_remote_path_probe": result_protocol_frozen(result_rows),
        **cache_protocol_checks(progress_events),
    }
    seed_results: dict[str, Any] = {}
    tiers: dict[int, str] = {}
    if set(mapped) == expected_keys():
        for seed in EXPECTED_SEEDS:
            aucs = {condition: float(mapped[(seed, condition)]["metrics"]["auc"]) for condition in CONDITIONS}
            pair_gain = aucs["exact16"] - aucs["exact4"]
            semantic_gap = aucs["exact16"] - aucs["wrong16"]
            strong = aucs["exact16"] >= STRONG_AUC_FLOOR and pair_gain >= PAIR_GAIN_FLOOR and semantic_gap >= SEMANTIC_GAP_FLOOR
            weak = aucs["exact16"] >= WEAK_AUC_FLOOR and pair_gain >= PAIR_GAIN_FLOOR and semantic_gap >= SEMANTIC_GAP_FLOOR
            tiers[seed] = "strong" if strong else "weak" if weak else "unsupported"
            seed_results[str(seed)] = {"aucs": aucs, "pair_gain": pair_gain, "semantic_gap": semantic_gap, "tier": tiers[seed]}
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    if not protocol_valid:
        status, tier = "invalid", "invalid"
        decision = "innovation1_uknit_k1bv_protocol_invalid"
        remote_scale = "no"
        next_action = "repair only the failed plan, cache, checkpoint-manifest, source, or result binding and rerun unchanged"
    elif tiers and all(value == "strong" for value in tiers.values()):
        status, tier = "pass", "strong"
        decision = "innovation1_uknit_k1bv_pair_amplification_strong"
        remote_scale = "authorize_65536_per_class_confirmation"
        next_action = "preregister and run the identical three-condition matrix at 65536/class on seeds3/4 remotely"
    elif tiers and all(value in {"strong", "weak"} for value in tiers.values()):
        status, tier = "pass", "weak"
        decision = "innovation1_uknit_k1bv_pair_amplification_weak"
        remote_scale = "fresh_seed_submedium_only"
        next_action = "repeat the identical 2048/class matrix on fresh seeds5/6; do not increase scale"
    else:
        status, tier = "hold", "unsupported"
        decision = "innovation1_uknit_k1bv_pair_amplification_not_supported"
        remote_scale = "no"
        next_action = "close pair amplification for this frozen r6 difference/network and require a new data or representation hypothesis"
    return {
        "run_id": RUN_ID, "status": status, "tier": tier, "decision": decision,
        "remote_scale": remote_scale, "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(name for name, ok in protocol_checks.items() if not ok),
        "seed_results": seed_results,
        "thresholds": {"weak_auc": WEAK_AUC_FLOOR, "strong_auc": STRONG_AUC_FLOOR, "pair_gain": PAIR_GAIN_FLOOR, "semantic_gap": SEMANTIC_GAP_FLOOR},
        "next_action": next_action,
        "claim_scope": "two-seed remote 2048/class uKNIT r6 pair-amplification diagnostic; not formal, paper-scale, attack, SOTA, breakthrough, route-ceiling, or universal-r6 evidence",
        "blocked_actions": ["calling 2048/class formal evidence", "scaling unless both seeds pass the same gate", "changing data, difference, keys, window, epochs, capacity, or negatives"],
    }


def _is_remote_path(value: Any) -> bool:
    return str(value or "").replace("/", "\\").lower().startswith("g:\\lxy\\")


__all__ = [
    "CONDITIONS", "EXPECTED_CACHE_CREATIONS", "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_RESULT_ROWS", "RUN_ID", "adjudicate", "build_readiness",
    "candidate_protocol_frozen", "read_tasks", "result_protocol_frozen",
]
