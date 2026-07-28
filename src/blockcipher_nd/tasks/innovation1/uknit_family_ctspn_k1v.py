from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    MODEL_TO_CONDITION,
    RUN_ID as K1T_RUN_ID,
    build_k1t_control,
)


RUN_ID = (
    "i1_uknit_family_ctspn_pair_count_k1v_16pair_"
    "2048_seed3_seed4_20260728"
)
EXPECTED_ANCHOR_GATE_SHA256 = (
    "f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf"
)
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 16
EXPECTED_INPUT_BITS = 2048
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_MODELS)
SEMANTIC_MARGIN = 0.010
ADDED_VALUE_MARGIN = 0.010


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
            raise ValueError(f"duplicate K1-V task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-V task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1)) == 2048
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == "independent_pairs"
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == CONFIRMATION_KEYS[seed][0]
            and int(task.get("validation_key", -1)) == CONFIRMATION_KEYS[seed][1]
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == 10
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1)) == 128
            and int(task.get("model_options", {}).get("histogram_value_dim", -1)) == 8
            for (seed, _), task in mapped.items()
        )
    )


def build_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    anchor_gate: Mapping[str, Any],
    anchor_gate_sha256: str,
) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    checks = {
        "six_frozen_tasks_exact": (
            len(tasks) == EXPECTED_TRAINING_ROWS and set(mapped) == expected_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "k1t_anchor_digest_exact": (
            anchor_gate_sha256 == EXPECTED_ANCHOR_GATE_SHA256
        ),
        "k1t_anchor_completed_pass": anchor_gate_valid(anchor_gate),
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(checks.values()):
        try:
            rng = np.random.default_rng(20260728)
            fixture = torch.as_tensor(
                rng.integers(0, 2, size=(8, EXPECTED_INPUT_BITS), dtype=np.uint8),
                dtype=torch.float32,
            )
            models = {
                condition: build_k1t_control(
                    task=mapped[(3, condition)],
                    condition=condition,
                    input_bits=EXPECTED_INPUT_BITS,
                )
                for condition in CONTROL_MODELS
            }
            geometries = {
                condition: tuple(
                    (name, tuple(value.shape))
                    for name, value in model.state_dict().items()
                )
                for condition, model in models.items()
            }
            parameter_counts = {
                condition: int(model_metadata(model)["trainable_parameter_count"])
                for condition, model in models.items()
            }
            decoded_pair_counts = {
                condition: int(
                    fixture.shape[1]
                    // (2 * int(model.runtime_structure.block_bits))
                )
                for condition, model in models.items()
            }
            exact = models["exact_position_histogram_residual"]
            wrong = models["wrong_sbox_position_histogram_residual"]
            invariant = models["invariant_histogram_residual"]
            shared_state = deepcopy(exact.state_dict())
            wrong.load_state_dict(shared_state, strict=True)
            invariant.load_state_dict(shared_state, strict=True)
            for model in models.values():
                model.eval()
            with torch.no_grad():
                logits = {name: model(fixture) for name, model in models.items()}
            exact.train()
            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(exact(fixture)).flatten(),
                torch.arange(len(fixture), dtype=torch.float32).remainder(2),
            )
            loss.backward()
            gradient_l1 = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in exact.parameters()
                if parameter.grad is not None
            )
            evidence_checks = {
                "input_width_is_2048_bits": fixture.shape == (8, EXPECTED_INPUT_BITS),
                "all_models_decode_sixteen_pairs": all(
                    count == EXPECTED_PAIRS for count in decoded_pair_counts.values()
                ),
                "three_controls_identical_geometry": len(set(geometries.values())) == 1,
                "parameter_count_exact": set(parameter_counts.values())
                == {EXPECTED_PARAMETER_COUNT},
                "finite_equal_shape_logits": (
                    len({tuple(values.shape) for values in logits.values()}) == 1
                    and all(torch.isfinite(values).all() for values in logits.values())
                ),
                "shared_state_controls_observable": (
                    not torch.equal(
                        logits["exact_position_histogram_residual"],
                        logits["wrong_sbox_position_histogram_residual"],
                    )
                    and not torch.equal(
                        logits["exact_position_histogram_residual"],
                        logits["invariant_histogram_residual"],
                    )
                ),
                "finite_nonzero_backward_gradient": math.isfinite(gradient_l1)
                and gradient_l1 > 0.0,
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "pairs_per_sample": decoded_pair_counts,
                "parameter_counts": parameter_counts,
                "gradient_l1": gradient_l1,
            }
        except Exception as exc:  # pragma: no cover - fail-closed artifact path
            errors.append(f"{type(exc).__name__}: {exc}")
            evidence_checks["readiness_execution_succeeded"] = False
    status = (
        "pass"
        if checks
        and evidence_checks
        and all(checks.values())
        and all(evidence_checks.values())
        and not errors
        else "fail"
    )
    return {
        "run_id": RUN_ID,
        "status": status,
        "optimizer_step_authorized": status == "pass",
        "protocol_checks": checks,
        "evidence_checks": evidence_checks,
        "evidence_metrics": evidence_metrics,
        "errors": errors,
    }


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    anchor_gate: Mapping[str, Any],
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("optimizer_step_authorized") is True
            and all(readiness.get("protocol_checks", {}).values())
            and all(readiness.get("evidence_checks", {}).values())
        ),
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_training_rows_complete": set(rows) == expected_keys()
        and len(result_rows) == EXPECTED_TRAINING_ROWS,
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "disk_cache_created_and_reused": cache_protocol_frozen(progress_rows),
        "k1t_anchor_completed_pass": anchor_gate_valid(anchor_gate),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
    }
    seed_results: dict[str, dict[str, float]] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        if all((seed, condition) in rows for condition in CONTROL_MODELS):
            exact_auc = _auc(rows[(seed, "exact_position_histogram_residual")])
            wrong_auc = _auc(
                rows[(seed, "wrong_sbox_position_histogram_residual")]
            )
            invariant_auc = _auc(rows[(seed, "invariant_histogram_residual")])
            anchor_auc = float(
                anchor_gate["seed_results"][str(seed)]["cross_key_validation"][
                    "exact_position_histogram_residual_auc"
                ]
            )
            semantic_margin = exact_auc - wrong_auc
            pair_gain = exact_auc - anchor_auc
            position_margin = exact_auc - invariant_auc
            seed_results[str(seed)] = {
                "exact_16pair_auc": exact_auc,
                "wrong_sbox_16pair_auc": wrong_auc,
                "invariant_16pair_auc": invariant_auc,
                "exact_4pair_anchor_auc": anchor_auc,
                "exact_minus_wrong_sbox": semantic_margin,
                "exact_16pair_minus_exact_4pair": pair_gain,
                "exact_minus_invariant": position_margin,
            }
            research_checks[f"seed{seed}_semantic_margin"] = (
                semantic_margin >= SEMANTIC_MARGIN
            )
            research_checks[f"seed{seed}_added_value"] = (
                pair_gain >= ADDED_VALUE_MARGIN
                or position_margin >= ADDED_VALUE_MARGIN
            )
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    semantic_pass = all(
        research_checks.get(f"seed{seed}_semantic_margin") is True
        for seed in EXPECTED_SEEDS
    )
    added_value_pass = all(
        research_checks.get(f"seed{seed}_added_value") is True
        for seed in EXPECTED_SEEDS
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1v_protocol_invalid"
        next_action = (
            "repair only the failed K1-V plan, shape, cache, checkpoint, or artifact "
            "binding and rerun unchanged"
        )
    elif semantic_pass and added_value_pass:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1v_16pair_added_value_supported"
        next_action = (
            "retain 16 pairs as a promising query-budget setting; first run the separate "
            "K1-W compact invariant experiment, then confirm pair count in that selected architecture"
        )
    elif semantic_pass:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1v_16pair_no_added_value"
        next_action = (
            "keep four pairs and proceed to K1-W compact invariant; do not mechanically add pairs"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1v_16pair_semantic_attribution_lost"
        next_action = (
            "reject the 16-pair route and audit pair aggregation dilution before changing capacity"
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
            "exact_minus_wrong_sbox": SEMANTIC_MARGIN,
            "exact_16pair_minus_exact_4pair_or_exact_minus_invariant": (
                ADDED_VALUE_MARGIN
            ),
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class uKNIT r5 cell11 pair-count diagnostic; "
            "not formal scale, attack, SOTA, transfer, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale or more pairs from this local diagnostic alone",
            "changing model, difference, keys, epochs, labels, or negative definition",
            "averaging across seeds to hide a failed per-seed gate",
        ],
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **dict(values)}
        for seed, values in sorted(gate.get("seed_results", {}).items())
    ]


def anchor_gate_valid(gate: Mapping[str, Any]) -> bool:
    try:
        return (
            gate.get("run_id") == K1T_RUN_ID
            and gate.get("status") == "pass"
            and all(gate.get("protocol_checks", {}).values())
            and all(
                math.isfinite(
                    float(
                        gate["seed_results"][str(seed)]["cross_key_validation"][
                            "exact_position_histogram_residual_auc"
                        ]
                    )
                )
                for seed in EXPECTED_SEEDS
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


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
            raise ValueError(f"duplicate K1-V result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-V result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and int(row.get("rounds", -1)) == 5
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and int(row.get("training", {}).get("input_bits", -1))
        == EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1))
        == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == 10
        and int(row.get("training", {}).get("epochs_ran", -1)) == 10
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and Path(str(row.get("training", {}).get("checkpoint_output", ""))).is_file()
        for row in rows
    )


def cache_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    events = [
        row
        for row in rows
        if row.get("event") in {"cache_start", "cache_reuse"}
        and row.get("split") in {"train", "validation"}
    ]
    created = [row for row in events if row.get("event") == "cache_start"]
    reused = [row for row in events if row.get("event") == "cache_reuse"]
    return len(created) == 4 and len(reused) == 8


def expected_keys() -> set[tuple[int, str]]:
    return {
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_MODELS
    }


def _auc(row: Mapping[str, Any]) -> float:
    return float(row.get("metrics", {}).get("auc", math.nan))


__all__ = [
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "task_map",
]
