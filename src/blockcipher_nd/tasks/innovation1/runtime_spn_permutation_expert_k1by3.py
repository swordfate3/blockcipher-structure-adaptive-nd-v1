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
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_conditioner_k1by1 import (
    EXPECTED_INPUT_BITS,
    EXPECTED_PAIRS,
    EXPECTED_PARAMETER_COUNT,
    build_condition,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_permutation_expert_k1by3_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
PLAN_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_permutation_expert_k1by3_"
    "present_r7_16pair_2048_seed2_seed3.csv"
)
K1BY2_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_runtime_spn_ordered_primitive_conditioner_k1by2_"
    "fresh_seed5_seed6_20260801"
)
K1BY2_PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_ordered_primitive_conditioner_k1by2_"
    "fresh_seed5_seed6.csv"
)
PRESENT_SOURCE_ROOTS = {
    0: ROOT / "outputs/local_diagnostic/i1_rtg1_present_runtime_e4_transfer_t1_2048_seed0",
    1: ROOT / "outputs/local_diagnostic/i1_rtg1_present_runtime_e4_transfer_t1_2048_seed1",
}
PRESENT_SOURCE_PLANS = {
    0: ROOT
    / "configs/experiment/innovation1/innovation1_spn_present_runtime_e4_transfer_t1_2048_seed0.csv",
    1: ROOT
    / "configs/experiment/innovation1/innovation1_spn_present_runtime_e4_transfer_t1_2048_seed1.csv",
}
SOURCE_DIGESTS = {
    "k1by2_plan": "c914c2624748dbd920641b07eb60c92ae772505570c6f3e18e9c4dfef6c75572",
    "k1by2_gate": "a8b22c43cf8d670c5063a2d5ee87e945a41fc5a1c605b08122c48c7fced29673",
    "k1by2_results": "4838ef8ca5fc906b9bff68cc9e6af476e4eee396dcfda664dee7131b103a3793",
    "k1by2_validation": "b82b4f9b9656791334b57f04d19b6b2d4f499b71904fed6305af75a5d1f8f98d",
    "present_seed0_plan": "975caccde1a6e24cd060cb3d2ae753cb68118edc948533f8403ff1181f99444f",
    "present_seed0_gate": "397ada6e7733e8bc2c7e83e1f271f099f4c73173ea854102da308290baf244c0",
    "present_seed0_results": "a78d86a0491f0d665290602751ba7de3bb62b935bcdfa1d362246222b4f07eee",
    "present_seed1_plan": "f55b5cd3d27faf70785fd8e73cded107261c73e8167427d37e39a4b98c64f44e",
    "present_seed1_gate": "5d156152fc65ecfb63866fe3d116cb3c9a39d54979b7d531465453c3a7175d92",
    "present_seed1_results": "766600d6bf628b14524287b15940c88356804a3be535635987410b920f439622",
}
EXPECTED_SEEDS = (2, 3)
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
INPUT_DIFFERENCE = 0x0000000000000009
DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
SAMPLE_STRUCTURE = "zhang_wang_case2_official_mcnd"
TRAIN_KEY = 0
VALIDATION_KEY = 0x11111111111111111111
SIGNAL_FLOOR = 0.550
NO_CONDITIONER_MARGIN = 0.010
ROUTING_MARGIN = 0.005

CONDITIONS = {
    "correct_permutation_routing": "runtime_spn_k1by1_compiler_correct",
    "wrong_permutation_binding": "runtime_spn_k1by1_compiler_wrong_binding",
    "no_compiler_conditioner": "runtime_spn_k1by1_no_compiler_conditioner",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONDITIONS.items()}
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
            raise ValueError(f"duplicate K1-BY3 task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY3 task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == expected_keys()
        and all(
            task.get("cipher_key") == "present80"
            and int(task.get("rounds", -1)) == 7
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1)) == 2048
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == INPUT_DIFFERENCE
            and task.get("difference_profile") == DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == SAMPLE_STRUCTURE
            and int(task.get("key_rotation_interval", -1)) == 0
            and int(task.get("train_key", -1)) == TRAIN_KEY
            and int(task.get("validation_key", -1)) == VALIDATION_KEY
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and task.get("model_options", {}).get("runtime_structure_path")
            == "configs/runtime/spn/present64.json"
            and int(task.get("model_options", {}).get("runtime_round_start", -1)) == 0
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("primitive_hidden_dim", -1))
            == 32
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1))
            == 128
            and int(task.get("model_options", {}).get("wrong_binding_seed", -1))
            == 11
            for (seed, _condition), task in mapped.items()
        )
    )


def source_binding_checks() -> dict[str, bool]:
    paths = {
        "k1by2_plan": K1BY2_PLAN,
        "k1by2_gate": K1BY2_ROOT / "gate.json",
        "k1by2_results": K1BY2_ROOT / "results.jsonl",
        "k1by2_validation": K1BY2_ROOT / "validation.json",
        "present_seed0_plan": PRESENT_SOURCE_PLANS[0],
        "present_seed0_gate": PRESENT_SOURCE_ROOTS[0] / "gate.json",
        "present_seed0_results": PRESENT_SOURCE_ROOTS[0] / "results.jsonl",
        "present_seed1_plan": PRESENT_SOURCE_PLANS[1],
        "present_seed1_gate": PRESENT_SOURCE_ROOTS[1] / "gate.json",
        "present_seed1_results": PRESENT_SOURCE_ROOTS[1] / "results.jsonl",
    }
    checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in paths.items()
    }
    try:
        k1by2_gate = _read_json(paths["k1by2_gate"])
        k1by2_validation = _read_json(paths["k1by2_validation"])
        present_gates = {
            seed: _read_json(paths[f"present_seed{seed}_gate"])
            for seed in (0, 1)
        }
    except (OSError, json.JSONDecodeError, ValueError):
        k1by2_gate = {}
        k1by2_validation = {}
        present_gates = {0: {}, 1: {}}
    checks["k1by2_fresh_seed_exact_pass"] = (
        k1by2_gate.get("status") == "pass"
        and k1by2_gate.get("decision")
        == "innovation1_runtime_spn_k1by2_fresh_seed_confirmed"
        and k1by2_validation.get("status") == "pass"
    )
    checks["present_t1_two_seed_exact_pass"] = all(
        gate.get("status") == "pass"
        and gate.get("decision")
        == f"innovation1_runtime_spn_present_transfer_seed{seed}_supported"
        for seed, gate in present_gates.items()
    )
    return checks


def build_readiness(*, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        **source_binding_checks(),
        "six_frozen_tasks_exact": (
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
                np.random.default_rng(20260802).integers(
                    0,
                    2,
                    size=(4, EXPECTED_INPUT_BITS),
                    dtype=np.uint8,
                ),
                dtype=torch.float32,
            )
            models: dict[str, torch.nn.Module] = {}
            for condition in CONDITIONS:
                torch.manual_seed(20260802)
                models[condition] = build_condition(
                    task=mapped[(EXPECTED_SEEDS[0], condition)],
                    condition=(
                        "correct_compiler_routing"
                        if condition == "correct_permutation_routing"
                        else "wrong_target_binding_routing"
                        if condition == "wrong_permutation_binding"
                        else condition
                    ),
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
            correct = models["correct_permutation_routing"]
            wrong = models["wrong_permutation_binding"]
            no_conditioner = models["no_compiler_conditioner"]
            output_deltas = {
                name: float(
                    (outputs["correct_permutation_routing"] - value).abs().max()
                )
                for name, value in outputs.items()
                if name != "correct_permutation_routing"
            }
            expected_usage = {
                "sbox4_table": 32,
                "linear_permutation": 32,
                "linear_gf2": 0,
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
                "permutation_expert_only_linear_routing": all(
                    model.compiled_program_expert_usage == expected_usage
                    for model in models.values()
                ),
                "correct_and_wrong_program_digests_differ": (
                    correct.compiled_program_semantic_sha256
                    != wrong.compiled_program_semantic_sha256
                ),
                "wrong_binding_changes_fixture_output": (
                    output_deltas["wrong_permutation_binding"] > 0.0
                ),
                "only_no_conditioner_disables_residual": (
                    correct.primitive_conditioner_enabled is True
                    and wrong.primitive_conditioner_enabled is True
                    and no_conditioner.primitive_conditioner_enabled is False
                ),
                "all_models_exclude_cipher_identity": all(
                    model.uses_cipher_identity is False for model in models.values()
                ),
                "all_models_exclude_absolute_identity": all(
                    model.uses_absolute_cell_or_bit_identity is False
                    for model in models.values()
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
                "compiled_program_expert_usage": {
                    name: model.compiled_program_expert_usage
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
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_training_rows_complete": (
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
            correct = aucs["correct_permutation_routing"]
            margins = {
                "wrong_permutation_binding": correct
                - aucs["wrong_permutation_binding"],
                "no_compiler_conditioner": correct - aucs["no_compiler_conditioner"],
            }
            seed_results[str(seed)] = {
                "auc_by_condition": aucs,
                "correct_minus_control": margins,
            }
            research_checks[f"seed{seed}_signal"] = correct >= SIGNAL_FLOOR
            research_checks[f"seed{seed}_wrong_binding_margin"] = (
                margins["wrong_permutation_binding"] >= ROUTING_MARGIN
            )
            research_checks[f"seed{seed}_no_conditioner_margin"] = (
                margins["no_compiler_conditioner"] >= NO_CONDITIONER_MARGIN
            )
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    signal_pass = all(
        research_checks.get(f"seed{seed}_signal", False) for seed in EXPECTED_SEEDS
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by3_protocol_invalid"
        next_action = "Repair only the failed protocol invariant and rerun unchanged."
    elif not failed_research:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by3_permutation_expert_supported"
        next_action = (
            "At the same PRESENT budget, isolate deterministic compiled inverse "
            "execution from learned primitive descriptors before transfer or scale."
        )
    elif signal_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by3_permutation_attribution_not_supported"
        next_action = (
            "Inspect only the failed wrong-binding or no-conditioner control; "
            "do not add ciphers, scale or capacity."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by3_permutation_signal_not_reproduced"
        next_action = (
            "Compare deterministic compiled PRESENT features with the retained T1 "
            "representation before redesigning the learned interface."
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
            "Local PRESENT-80 r7 2048/class permutation-expert diagnostic on fresh "
            "seeds; not formal scale, shared-weight transfer, attack or SOTA evidence."
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
            raise ValueError(f"duplicate K1-BY3 result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY3 result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and row.get("cipher_key") == "present80"
        and int(row.get("rounds", -1)) == 7
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == SAMPLE_STRUCTURE
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and row.get("compiled_program_expert_usage")
        == {"sbox4_table": 32, "linear_permutation": 32, "linear_gf2": 0}
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
        and sum(row.get("event") == "cache_reuse" for row in events) == 8
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
    "PLAN_PATH",
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "candidate_protocol_frozen",
    "comparison_rows",
    "read_tasks",
    "source_binding_checks",
    "task_map",
]
