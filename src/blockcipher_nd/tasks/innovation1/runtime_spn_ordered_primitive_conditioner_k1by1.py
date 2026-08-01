from __future__ import annotations

import hashlib
import json
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


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_ordered_primitive_conditioner_k1by1_"
    "16pair_2048_seed3_seed4_20260801"
)
PLAN_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_ordered_primitive_conditioner_k1by1_"
    "16pair_2048_seed3_seed4.csv"
)
SOURCE_ROOT = ROOT / (
    "outputs/local_audit/i1_runtime_spn_ordered_primitive_compiler_k1by0_20260801"
)
SOURCE_CONFIG = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_ordered_primitive_compiler_k1by0_20260801.json"
)
SOURCE_DIGESTS = {
    "config": "290cd74a7d589137f2de8b98f1f190fa2f187820eafd8195819b835f75a114e6",
    "gate.json": "279976a489a9cfec7c168da9df72777052baa6b064e11305a08c82280118211d",
    "validation.json": "899f31f7e7b9db136dbc28292756040cc09bd995656d075ff1fce4fa713105a6",
}
EXPECTED_SEEDS = (3, 4)
EXPECTED_PAIRS = 16
PAIR_BITS = 128
EXPECTED_INPUT_BITS = EXPECTED_PAIRS * PAIR_BITS
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
EXPECTED_PARAMETER_COUNT = 235780
SIGNAL_FLOOR = 0.550
NO_CONDITIONER_MARGIN = 0.010
ROUTING_MARGIN = 0.005

CONDITIONS = {
    "correct_compiler_routing": "runtime_spn_k1by1_compiler_correct",
    "wrong_order_routing": "runtime_spn_k1by1_compiler_wrong_order",
    "wrong_target_binding_routing": "runtime_spn_k1by1_compiler_wrong_binding",
    "no_compiler_conditioner": "runtime_spn_k1by1_no_compiler_conditioner",
}
MODEL_TO_CONDITION = {model: name for name, model in CONDITIONS.items()}
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(CONDITIONS)


def read_tasks(path: Path = PLAN_PATH) -> list[dict[str, Any]]:
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
            raise ValueError(f"duplicate K1-BY1 task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY1 task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
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
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 3
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("primitive_hidden_dim", -1)) == 32
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1)) == 128
            and int(task.get("model_options", {}).get("wrong_binding_seed", -1)) == 11
            for (seed, _condition), task in mapped.items()
        )
    )


def source_binding_checks() -> dict[str, bool]:
    paths = {
        "config": SOURCE_CONFIG,
        "gate.json": SOURCE_ROOT / "gate.json",
        "validation.json": SOURCE_ROOT / "validation.json",
    }
    checks = {
        f"k1by0_{name}_digest_exact": path.is_file()
        and _file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in paths.items()
    }
    try:
        gate = _read_json(paths["gate.json"])
        validation = _read_json(paths["validation.json"])
    except (OSError, json.JSONDecodeError):
        gate = {}
        validation = {}
    checks["k1by0_ordered_compiler_exact_pass"] = (
        gate.get("status") == "pass"
        and gate.get("decision")
        == "innovation1_runtime_spn_k1by0_ordered_compiler_ready"
        and validation.get("status") == "pass"
    )
    return checks


def build_condition(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int = EXPECTED_INPUT_BITS,
) -> torch.nn.Module:
    if condition not in CONDITIONS:
        raise ValueError("unknown K1-BY1 condition")
    return build_model(
        CONDITIONS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=PAIR_BITS,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )


def build_readiness(*, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        **source_binding_checks(),
        "eight_frozen_tasks_exact": (
            len(tasks) == EXPECTED_RESULT_ROWS and set(mapped) == expected_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            fixture = torch.as_tensor(
                np.random.default_rng(20260801).integers(
                    0,
                    2,
                    size=(4, EXPECTED_INPUT_BITS),
                    dtype=np.uint8,
                ),
                dtype=torch.float32,
            )
            models: dict[str, torch.nn.Module] = {}
            for condition in CONDITIONS:
                torch.manual_seed(20260801)
                models[condition] = build_condition(
                    task=mapped[(3, condition)],
                    condition=condition,
                )
            parameter_counts = {
                name: int(model_metadata(model)["trainable_parameter_count"])
                for name, model in models.items()
            }
            outputs: dict[str, torch.Tensor] = {}
            gradient_l1: dict[str, float] = {}
            for name, model in models.items():
                model.train()
                output = model(fixture)
                outputs[name] = output.detach()
                loss = torch.nn.functional.mse_loss(
                    torch.sigmoid(output).flatten(),
                    torch.arange(len(fixture), dtype=torch.float32).remainder(2),
                )
                loss.backward()
                gradient_l1[name] = sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            correct = models["correct_compiler_routing"]
            wrong_order = models["wrong_order_routing"]
            wrong_binding = models["wrong_target_binding_routing"]
            no_conditioner = models["no_compiler_conditioner"]
            output_deltas = {
                name: float((outputs["correct_compiler_routing"] - value).abs().max())
                for name, value in outputs.items()
                if name != "correct_compiler_routing"
            }
            evidence_checks = {
                "all_conditions_accept_same_input": all(
                    output.shape == (4, 1) and torch.isfinite(output).all()
                    for output in outputs.values()
                ),
                "parameter_geometry_exactly_equal": (
                    set(parameter_counts.values()) == {EXPECTED_PARAMETER_COUNT}
                ),
                "all_backward_gradients_finite_nonzero": all(
                    math.isfinite(value) and value > 0.0
                    for value in gradient_l1.values()
                ),
                "correct_and_wrong_program_digests_differ": (
                    correct.compiled_program_semantic_sha256
                    != wrong_order.compiled_program_semantic_sha256
                    and correct.compiled_program_semantic_sha256
                    != wrong_binding.compiled_program_semantic_sha256
                ),
                "only_no_conditioner_disables_residual": (
                    correct.primitive_conditioner_enabled is True
                    and wrong_order.primitive_conditioner_enabled is True
                    and wrong_binding.primitive_conditioner_enabled is True
                    and no_conditioner.primitive_conditioner_enabled is False
                ),
                "all_models_exclude_cipher_identity": all(
                    model.uses_cipher_identity is False for model in models.values()
                ),
                "routing_controls_change_fixture_output": all(
                    output_deltas[name] > 0.0
                    for name in (
                        "wrong_order_routing",
                        "wrong_target_binding_routing",
                        "no_compiler_conditioner",
                    )
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "parameter_counts": parameter_counts,
                "gradient_l1": gradient_l1,
                "correct_output_max_deltas": output_deltas,
                "program_control_modes": {
                    name: model.runtime_structure_window_control
                    for name, model in models.items()
                },
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
            len(result_rows) == EXPECTED_RESULT_ROWS and set(rows) == expected_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "disk_cache_created_and_reused": cache_protocol_frozen(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
    }
    seed_results: dict[str, dict[str, Any]] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        if all((seed, condition) in rows for condition in CONDITIONS):
            aucs = {
                condition: _auc(rows[(seed, condition)]) for condition in CONDITIONS
            }
            correct = aucs["correct_compiler_routing"]
            margins = {
                "no_compiler_conditioner": correct - aucs["no_compiler_conditioner"],
                "wrong_order_routing": correct - aucs["wrong_order_routing"],
                "wrong_target_binding_routing": correct
                - aucs["wrong_target_binding_routing"],
            }
            seed_results[str(seed)] = {
                "auc_by_condition": aucs,
                "correct_minus_control": margins,
            }
            research_checks[f"seed{seed}_signal"] = correct >= SIGNAL_FLOOR
            research_checks[f"seed{seed}_no_conditioner_margin"] = (
                margins["no_compiler_conditioner"] >= NO_CONDITIONER_MARGIN
            )
            research_checks[f"seed{seed}_wrong_order_margin"] = (
                margins["wrong_order_routing"] >= ROUTING_MARGIN
            )
            research_checks[f"seed{seed}_wrong_binding_margin"] = (
                margins["wrong_target_binding_routing"] >= ROUTING_MARGIN
            )
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    signal_checks = [
        research_checks.get(f"seed{seed}_signal", False) for seed in EXPECTED_SEEDS
    ]
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by1_protocol_invalid"
        next_action = "Repair only the failed protocol invariant and rerun unchanged."
    elif not failed_research:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by1_compiler_conditioner_supported"
        next_action = (
            "Run one fresh-seed local confirmation with the correct conditioner, "
            "the strongest failed-routing control and no-conditioner anchor."
        )
    elif all(signal_checks):
        status = "hold"
        decision = "innovation1_runtime_spn_k1by1_structure_attribution_not_supported"
        next_action = (
            "Inspect only the failed order or target-binding route; do not add "
            "samples, epochs, width, ciphers or remote scale."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by1_conditioner_signal_not_reproduced"
        next_action = (
            "Compare the deterministic conditioner features with the retained "
            "K1-BS position-histogram expert before redesigning the interface."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "seed_results": seed_results,
        "parameter_count_per_condition": EXPECTED_PARAMETER_COUNT,
        "next_action": next_action,
        "claim_scope": (
            "Local uKNIT r5 2048/class ordered-primitive conditioner diagnostic; "
            "not formal scale, unseen-cipher transfer, attack or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        aucs = values["auc_by_condition"]
        margins = values["correct_minus_control"]
        rows.append(
            {
                "seed": int(seed),
                **{f"{name}_auc": aucs[name] for name in CONDITIONS},
                **{f"correct_minus_{name}": margins[name] for name in margins},
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
        condition = MODEL_TO_CONDITION.get(str(row.get("model")))
        if condition is None:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-BY1 result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY1 result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and int(row.get("rounds", -1)) == 5
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == "independent_pairs"
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and int(row.get("training", {}).get("input_bits", -1)) == EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1)) == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
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
    return (
        sum(row.get("event") == "cache_start" for row in events) == 4
        and sum(row.get("event") == "cache_reuse" for row in events) == 12
    )


def expected_keys() -> set[tuple[int, str]]:
    return {(seed, condition) for seed in EXPECTED_SEEDS for condition in CONDITIONS}


def _auc(row: Mapping[str, Any]) -> float:
    return float(row.get("metrics", {}).get("auc", math.nan))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


__all__ = [
    "CONDITIONS",
    "EXPECTED_PARAMETER_COUNT",
    "PLAN_PATH",
    "RUN_ID",
    "adjudicate",
    "build_condition",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "source_binding_checks",
    "task_map",
]
