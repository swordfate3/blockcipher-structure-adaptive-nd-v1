from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    INPUT_DIFFERENCE,
)


RUN_ID = (
    "i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_"
    "2048_seed3_seed4_20260731"
)
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 16
PAIR_BITS = 128
EXPECTED_INPUT_BITS = EXPECTED_PAIRS * PAIR_BITS
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
EXPERT_SIGNAL_FLOOR = 0.550
EXPERT_MARGIN = 0.010

ARCHITECTURES = {
    "uknit_structure_expert": "runtime_spn_ct_k1t_position_histogram_true",
    "autond_dbitnet": "autond_dbitnet2023",
    "generic_spn_cell_pairset": "spn_pairset_dbitnet_v2",
    "generic_spn_token_mixer": "spn_token_mixer_pairset",
}
MODEL_TO_ARCHITECTURE = {
    model: architecture for architecture, model in ARCHITECTURES.items()
}
EXPECTED_PARAMETER_COUNTS = {
    "uknit_structure_expert": 214316,
    "autond_dbitnet": 985985,
    "generic_spn_cell_pairset": 1045763,
    "generic_spn_token_mixer": 313634,
}
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(ARCHITECTURES)


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
        architecture = MODEL_TO_ARCHITECTURE.get(str(task.get("model_key")))
        if architecture is None:
            continue
        key = (int(task["seed"]), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-BS task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BS task matrix is incomplete")
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
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and int(task.get("model_options", {}).get("runtime_round_start", -1))
            == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1))
            == 128
            and int(task.get("model_options", {}).get("histogram_value_dim", -1))
            == 8
            for (seed, _), task in mapped.items()
        )
    )


def build_architecture(
    *, task: Mapping[str, Any], architecture: str, input_bits: int
) -> torch.nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown K1-BS architecture")
    return build_model(
        ARCHITECTURES[architecture],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=PAIR_BITS,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )


def build_readiness(*, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        "eight_frozen_tasks_exact": (
            len(tasks) == EXPECTED_TRAINING_ROWS and set(mapped) == expected_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            rng = np.random.default_rng(20260731)
            fixture = torch.as_tensor(
                rng.integers(0, 2, size=(4, EXPECTED_INPUT_BITS), dtype=np.uint8),
                dtype=torch.float32,
            )
            models = {
                architecture: build_architecture(
                    task=mapped[(3, architecture)],
                    architecture=architecture,
                    input_bits=EXPECTED_INPUT_BITS,
                )
                for architecture in ARCHITECTURES
            }
            parameter_counts = {
                architecture: int(model_metadata(model)["trainable_parameter_count"])
                for architecture, model in models.items()
            }
            output_shapes: dict[str, list[int]] = {}
            gradient_l1: dict[str, float] = {}
            finite_outputs: dict[str, bool] = {}
            for architecture, model in models.items():
                model.train()
                logits = model(fixture)
                output_shapes[architecture] = list(logits.shape)
                finite_outputs[architecture] = bool(torch.isfinite(logits).all())
                loss = torch.nn.functional.mse_loss(
                    torch.sigmoid(logits).flatten(),
                    torch.arange(len(fixture), dtype=torch.float32).remainder(2),
                )
                loss.backward()
                gradient_l1[architecture] = sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            evidence_checks = {
                "input_width_is_sixteen_128_bit_pairs": (
                    tuple(fixture.shape) == (4, EXPECTED_INPUT_BITS)
                    and EXPECTED_INPUT_BITS // PAIR_BITS == EXPECTED_PAIRS
                ),
                "all_architectures_accept_same_input": (
                    set(tuple(shape) for shape in output_shapes.values()) == {(4, 1)}
                    and all(finite_outputs.values())
                ),
                "parameter_counts_exact_and_recorded": (
                    parameter_counts == EXPECTED_PARAMETER_COUNTS
                ),
                "all_backward_gradients_finite_nonzero": all(
                    math.isfinite(value) and value > 0.0
                    for value in gradient_l1.values()
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "pair_bits": PAIR_BITS,
                "pairs_per_sample": EXPECTED_PAIRS,
                "parameter_counts": parameter_counts,
                "output_shapes": output_shapes,
                "gradient_l1": gradient_l1,
            }
        except Exception as exc:  # pragma: no cover - fail-closed artifact path
            errors.append(f"{type(exc).__name__}: {exc}")
            evidence_checks["readiness_execution_succeeded"] = False
    status = (
        "pass"
        if protocol_checks
        and evidence_checks
        and all(protocol_checks.values())
        and all(evidence_checks.values())
        and not errors
        else "fail"
    )
    return {
        "run_id": RUN_ID,
        "status": status,
        "optimizer_step_authorized": status == "pass",
        "protocol_checks": protocol_checks,
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
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("optimizer_step_authorized") is True
            and all(readiness.get("protocol_checks", {}).values())
            and all(readiness.get("evidence_checks", {}).values())
        ),
        "eight_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "eight_training_rows_complete": (
            set(rows) == expected_keys()
            and len(result_rows) == EXPECTED_TRAINING_ROWS
        ),
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "disk_cache_created_and_reused": cache_protocol_frozen(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
    }
    seed_results: dict[str, dict[str, Any]] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        if all((seed, architecture) in rows for architecture in ARCHITECTURES):
            aucs = {
                architecture: _auc(rows[(seed, architecture)])
                for architecture in ARCHITECTURES
            }
            generic_aucs = {
                architecture: auc
                for architecture, auc in aucs.items()
                if architecture != "uknit_structure_expert"
            }
            best_generic = max(generic_aucs, key=generic_aucs.get)
            expert_auc = aucs["uknit_structure_expert"]
            margin = expert_auc - generic_aucs[best_generic]
            seed_results[str(seed)] = {
                "auc_by_architecture": aucs,
                "best_generic_architecture": best_generic,
                "best_generic_auc": generic_aucs[best_generic],
                "expert_minus_best_generic": margin,
            }
            research_checks[f"seed{seed}_expert_signal"] = (
                expert_auc >= EXPERT_SIGNAL_FLOOR
            )
            research_checks[f"seed{seed}_expert_margin"] = margin >= EXPERT_MARGIN

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    signal_pass = all(
        research_checks.get(f"seed{seed}_expert_signal") is True
        for seed in EXPECTED_SEEDS
    )
    margin_pass = all(
        research_checks.get(f"seed{seed}_expert_margin") is True
        for seed in EXPECTED_SEEDS
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_k1bs_architecture_protocol_invalid"
        remote_scale = "no"
        next_action = (
            "repair only the failed K1-BS plan, input, cache, checkpoint, or result binding "
            "and rerun the unchanged architecture matrix"
        )
    elif signal_pass and margin_pass:
        status = "pass"
        decision = "innovation1_uknit_k1bs_structure_expert_retained"
        remote_scale = "candidate"
        next_action = (
            "compare only the uKNIT structure expert and the strongest generic baseline at "
            "65536/class on the remote GPU, with the same r5 data protocol and seeds"
        )
    elif signal_pass:
        status = "hold"
        decision = "innovation1_uknit_k1bs_structure_expert_not_necessary"
        remote_scale = "no"
        next_action = (
            "retain the strongest generic architecture as the local candidate and run one "
            "fresh-seed local confirmation before any remote allocation"
        )
    else:
        status = "hold"
        decision = "innovation1_uknit_k1bs_expert_signal_not_reproduced"
        remote_scale = "no"
        next_action = (
            "audit optimization parity against the completed K1-V anchor before changing "
            "data, model capacity, or experiment scale"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": remote_scale,
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "thresholds": {
            "expert_auc": EXPERT_SIGNAL_FLOOR,
            "expert_minus_best_generic": EXPERT_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class uKNIT r5 architecture diagnostic with sixteen "
            "pairs per sample; same training budget but architectures have different parameter counts"
        ),
        "blocked_actions": [
            "calling this local diagnostic formal or paper-scale evidence",
            "claiming structure attribution from a capacity-unmatched architecture comparison",
            "changing difference, keys, labels, negatives, epochs, or pair count",
            "remote-running all four architectures",
        ],
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        aucs = values["auc_by_architecture"]
        rows.append(
            {
                "seed": int(seed),
                **{f"{name}_auc": aucs[name] for name in ARCHITECTURES},
                "best_generic_architecture": values["best_generic_architecture"],
                "best_generic_auc": values["best_generic_auc"],
                "expert_minus_best_generic": values[
                    "expert_minus_best_generic"
                ],
            }
        )
    return rows


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        architecture = MODEL_TO_ARCHITECTURE.get(str(row.get("model")))
        if architecture is None:
            continue
        key = (int(row["seed"]), architecture)
        if key in mapped:
            raise ValueError(f"duplicate K1-BS result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BS result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_ARCHITECTURE
        and int(row.get("rounds", -1)) == 5
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNTS[
            MODEL_TO_ARCHITECTURE[str(row.get("model"))]
        ]
        and int(row.get("training", {}).get("input_bits", -1))
        == EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1))
        == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1))
        == EXPECTED_EPOCHS
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
    return len(created) == 4 and len(reused) == 12


def expected_keys() -> set[tuple[int, str]]:
    return {
        (seed, architecture)
        for seed in EXPECTED_SEEDS
        for architecture in ARCHITECTURES
    }


def _auc(row: Mapping[str, Any]) -> float:
    return float(row.get("metrics", {}).get("auc", math.nan))


__all__ = [
    "ARCHITECTURES",
    "EXPECTED_PARAMETER_COUNTS",
    "RUN_ID",
    "adjudicate",
    "build_architecture",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "task_map",
]
