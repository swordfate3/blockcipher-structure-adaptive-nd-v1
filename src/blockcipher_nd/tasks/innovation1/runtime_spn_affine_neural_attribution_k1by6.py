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
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_conditioner_k1by1 import (
    EXPECTED_INPUT_BITS,
    EXPECTED_PAIRS,
    EXPECTED_PARAMETER_COUNT,
    PAIR_BITS,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_affine_neural_attribution_k1by6_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
PLAN_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_affine_neural_attribution_k1by6_"
    "present_r7_16pair_2048_seed2_seed3.csv"
)
K1BY3_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_runtime_spn_permutation_expert_k1by3_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
K1BY3_PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_permutation_expert_k1by3_"
    "present_r7_16pair_2048_seed2_seed3.csv"
)
K1BY3_CACHE_ROOT = K1BY3_ROOT / "cache"
K1BY5_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_runtime_spn_affine_endpoint_control_k1by5_"
    "present_r7_seed2_seed3_20260801"
)
K1BY5_CONFIG = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_affine_endpoint_control_k1by5_20260801.json"
)
SOURCE_DIGESTS = {
    "k1by3_plan": "68c35c24b3416a13856d7eecfb8c360081a2e7924750ce140689e2c51ab978a2",
    "k1by3_results": "54d06976024662b274981b195eb36756c4bbdf43f5a2b8f16de407f75052dc94",
    "k1by3_gate": "62b6c44e9c36153b903cc7131eb9c4717badd6843d406b064568db0f1cb99c68",
    "k1by3_preflight": "16bb5ee4377bbc278b5f4af2270f3bb0244bb7176cba6905eb423560c7f7990f",
    "k1by3_validation": "de27d159224762aa4dc08ec04fb948d14a42ac9d79524b6396e5d0269e178875",
    "k1by5_config": "447b81724f6c4d2824b26bff52b2cbe1d804cb5a30b906a0857ccf3c65b0f071",
    "k1by5_results": "b3c31a7e0c9d85c036cd16a96f1f7a3efbd59e46046e52ab76a168f115973b6d",
    "k1by5_gate": "e450f71418ce1e0ec9f4be328c57a42c40386042b028c3ff49cb6e3cddd2a25f",
    "k1by5_preflight": "03be0b5f43d46df78a43f29f06234220801c16604f9fb737240fa25d43e08f25",
    "k1by5_validation": "6ce50ab43f124d0f2db251eaee063ca41511ece3844b7d252f7b348f1cf894dc",
}
EXPECTED_SEEDS = (2, 3)
EXPECTED_RESULT_ROWS = 2
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_EPOCHS = 10
INPUT_DIFFERENCE = 0x0000000000000009
DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
SAMPLE_STRUCTURE = "zhang_wang_case2_official_mcnd"
TRAIN_KEY = 0
VALIDATION_KEY = 0x11111111111111111111
AFFINE_MODEL = "runtime_spn_k1by1_compiler_affine_wrong_endpoint"
CORRECT_MODEL = "runtime_spn_k1by1_compiler_correct"
NO_CONDITIONER_MODEL = "runtime_spn_k1by1_no_compiler_conditioner"
AFFINE_CONTROL = "source_endpoint_affine_m5_b1_mod64"
ROUTING_MARGIN = 0.005
EXPECTED_USAGE = {
    "sbox4_table": 32,
    "linear_permutation": 32,
    "linear_gf2": 0,
}
EXPECTED_ANCHORS = {
    2: {
        "correct_auc": 0.6837368011474609,
        "correct_accuracy": 0.63525390625,
        "no_conditioner_auc": 0.543799877166748,
        "no_conditioner_accuracy": 0.52783203125,
    },
    3: {
        "correct_auc": 0.6655435562133789,
        "correct_accuracy": 0.50048828125,
        "no_conditioner_auc": 0.5272355079650879,
        "no_conditioner_accuracy": 0.50341796875,
    },
}


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
) -> dict[int, Mapping[str, Any]]:
    mapped: dict[int, Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != AFFINE_MODEL:
            continue
        seed = int(task["seed"])
        if seed in mapped:
            raise ValueError(f"duplicate K1-BY6 task seed: {seed}")
        mapped[seed] = task
    if fail_closed and set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-BY6 task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_RESULT_ROWS
        and set(mapped) == set(EXPECTED_SEEDS)
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
            and int(task.get("model_options", {}).get("runtime_round_start", -1))
            == 0
            and int(task.get("model_options", {}).get("runtime_rounds", -1)) == 2
            and int(task.get("model_options", {}).get("primitive_hidden_dim", -1))
            == 32
            and int(task.get("model_options", {}).get("pair_embedding_dim", -1))
            == 128
            and int(
                task.get("model_options", {}).get("affine_endpoint_multiplier", -1)
            )
            == 5
            and int(task.get("model_options", {}).get("affine_endpoint_offset", -1))
            == 1
            for seed, task in mapped.items()
        )
    )


def source_binding_checks() -> dict[str, bool]:
    paths = _source_paths()
    checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in paths.items()
    }
    try:
        k1by3_gate = _read_json(paths["k1by3_gate"])
        k1by3_validation = _read_json(paths["k1by3_validation"])
        k1by5_gate = _read_json(paths["k1by5_gate"])
        k1by5_validation = _read_json(paths["k1by5_validation"])
        anchors = historical_anchors()
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        k1by3_gate = {}
        k1by3_validation = {}
        k1by5_gate = {}
        k1by5_validation = {}
        anchors = {}
    checks["k1by3_exact_completed_hold"] = (
        k1by3_gate.get("status") == "hold"
        and k1by3_gate.get("decision")
        == "innovation1_runtime_spn_k1by3_permutation_attribution_not_supported"
        and k1by3_validation.get("status") == "pass"
    )
    checks["k1by5_exact_identifiability_pass"] = (
        k1by5_gate.get("status") == "pass"
        and k1by5_gate.get("decision")
        == "innovation1_runtime_spn_k1by5_affine_endpoint_control_ready"
        and k1by5_gate.get("all_taps_identifiable") is True
        and k1by5_validation.get("status") == "pass"
    )
    checks["historical_anchor_values_exact"] = anchors == EXPECTED_ANCHORS
    checks.update(cache_authority_checks())
    return checks


def historical_anchors() -> dict[int, dict[str, float]]:
    rows = [
        json.loads(line)
        for line in (K1BY3_ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchors: dict[int, dict[str, float]] = {}
    for seed in EXPECTED_SEEDS:
        correct = _one_source_row(rows, seed=seed, model=CORRECT_MODEL)
        no_conditioner = _one_source_row(
            rows,
            seed=seed,
            model=NO_CONDITIONER_MODEL,
        )
        anchors[seed] = {
            "correct_auc": _metric(correct, "auc"),
            "correct_accuracy": _metric(correct, "accuracy"),
            "no_conditioner_auc": _metric(no_conditioner, "auc"),
            "no_conditioner_accuracy": _metric(no_conditioner, "accuracy"),
        }
    return anchors


def build_model_for_task(
    task: Mapping[str, Any],
    *,
    model_key: str,
) -> torch.nn.Module:
    return build_model(
        model_key,
        input_bits=EXPECTED_INPUT_BITS,
        hidden_bits=32,
        pair_bits=PAIR_BITS,
        structure="SPN",
        model_options=dict(task["model_options"]),
    )


def build_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    selected_device: str,
) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        **source_binding_checks(),
        "two_frozen_tasks_exact": (
            len(tasks) == EXPECTED_RESULT_ROWS
            and set(mapped) == set(EXPECTED_SEEDS)
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "cpu_fallback_frozen_before_optimizer": selected_device == "cpu",
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {
        "selected_device": selected_device,
        "local_cuda_available_at_readiness": torch.cuda.is_available(),
        "local_cuda_device_count_at_readiness": torch.cuda.device_count(),
    }
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            fixture = torch.as_tensor(
                np.random.default_rng(20260806).integers(
                    0,
                    2,
                    size=(4, EXPECTED_INPUT_BITS),
                    dtype=np.uint8,
                ),
                dtype=torch.float32,
            )
            torch.manual_seed(20260806)
            correct = build_model_for_task(
                mapped[EXPECTED_SEEDS[0]],
                model_key=CORRECT_MODEL,
            )
            torch.manual_seed(20260806)
            affine = build_model_for_task(
                mapped[EXPECTED_SEEDS[0]],
                model_key=AFFINE_MODEL,
            )
            models = {"correct": correct, "affine_wrong_endpoint": affine}
            parameter_counts = {
                name: int(model_metadata(model)["trainable_parameter_count"])
                for name, model in models.items()
            }
            outputs: dict[str, torch.Tensor] = {}
            gradients: dict[str, float] = {}
            for name, model in models.items():
                output = model(fixture)
                outputs[name] = output.detach()
                loss = torch.nn.functional.mse_loss(
                    torch.sigmoid(output).flatten(),
                    torch.arange(len(fixture), dtype=torch.float32).remainder(2),
                )
                loss.backward()
                gradients[name] = sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            evidence_checks = {
                "both_models_accept_exact_input": all(
                    output.shape == (4, 1) and torch.isfinite(output).all()
                    for output in outputs.values()
                ),
                "parameter_geometry_exactly_equal": (
                    set(parameter_counts.values()) == {EXPECTED_PARAMETER_COUNT}
                ),
                "all_backward_gradients_finite_nonzero": all(
                    math.isfinite(value) and value > 0.0
                    for value in gradients.values()
                ),
                "correct_and_affine_program_digests_differ": (
                    correct.compiled_program_semantic_sha256
                    != affine.compiled_program_semantic_sha256
                ),
                "affine_control_mode_exact": (
                    affine.runtime_structure_window_control == AFFINE_CONTROL
                ),
                "permutation_expert_usage_exact": all(
                    model.compiled_program_expert_usage == EXPECTED_USAGE
                    for model in models.values()
                ),
                "conditioner_enabled_for_both": all(
                    model.primitive_conditioner_enabled is True
                    for model in models.values()
                ),
                "models_exclude_cipher_identity": all(
                    model.uses_cipher_identity is False for model in models.values()
                ),
                "models_exclude_absolute_identity": all(
                    model.uses_absolute_cell_or_bit_identity is False
                    for model in models.values()
                ),
                "affine_changes_equal_initialization_output": (
                    not torch.equal(outputs["correct"], outputs["affine_wrong_endpoint"])
                ),
            }
            evidence_metrics.update(
                {
                    "fixture_shape": list(fixture.shape),
                    "parameter_counts": parameter_counts,
                    "gradient_l1": gradients,
                    "program_semantic_sha256": {
                        name: model.compiled_program_semantic_sha256
                        for name, model in models.items()
                    },
                    "compiled_program_expert_usage": {
                        name: model.compiled_program_expert_usage
                        for name, model in models.items()
                    },
                    "affine_control_mode": affine.runtime_structure_window_control,
                    "correct_affine_output_max_delta": float(
                        (outputs["correct"] - outputs["affine_wrong_endpoint"])
                        .abs()
                        .max()
                    ),
                }
            )
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
    cache_unchanged: bool,
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("optimizer_step_authorized") is True
            and all(readiness.get("protocol_checks", {}).values())
            and all(readiness.get("evidence_checks", {}).values())
        ),
        "two_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "two_training_rows_complete": (
            len(result_rows) == EXPECTED_RESULT_ROWS
            and set(rows) == set(EXPECTED_SEEDS)
        ),
        "training_protocol_frozen": training_protocol_frozen(result_rows),
        "source_cache_reused_without_generation": cache_protocol_frozen(progress_rows),
        "source_cache_unchanged": cache_unchanged,
        "source_bindings_still_exact": all(source_binding_checks().values()),
        "finite_metrics": bool(rows)
        and all(
            math.isfinite(_metric(row, metric))
            for row in rows.values()
            for metric in ("auc", "accuracy")
        ),
    }
    anchors = historical_anchors()
    seed_results: dict[str, dict[str, Any]] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        if seed not in rows or seed not in anchors:
            continue
        affine_auc = _metric(rows[seed], "auc")
        affine_accuracy = _metric(rows[seed], "accuracy")
        margin = anchors[seed]["correct_auc"] - affine_auc
        seed_results[str(seed)] = {
            **anchors[seed],
            "affine_wrong_endpoint_auc": affine_auc,
            "affine_wrong_endpoint_accuracy": affine_accuracy,
            "correct_minus_affine_auc": margin,
            "correct_minus_no_conditioner_auc": (
                anchors[seed]["correct_auc"]
                - anchors[seed]["no_conditioner_auc"]
            ),
        }
        research_checks[f"seed{seed}_correct_minus_affine_margin"] = (
            margin >= ROUTING_MARGIN
        )
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by6_protocol_invalid"
        next_action = (
            "Repair only the failed source binding, plan, cache, model, device, "
            "training or artifact invariant and rerun unchanged."
        )
    elif research_checks and not failed_research:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by6_permutation_attribution_supported"
        next_action = (
            "At the same local diagnostic budget, apply the validated permutation "
            "expert and affine endpoint control to GIFT with frozen controls before "
            "any remote expansion."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by6_permutation_attribution_not_supported"
        next_action = (
            "Audit learned access at the linear histogram, primitive expert, cell "
            "fusion, invariant pooling and final taps using K1-BY3/K1-BY6 checkpoints. "
            "Do not increase samples, pairs, epochs, width, seeds or ciphers."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": {"correct_minus_affine_auc_min_each_seed": ROUTING_MARGIN},
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "seed_results": seed_results,
        "parameter_count_per_condition": EXPECTED_PARAMETER_COUNT,
        "new_training_rows": EXPECTED_RESULT_ROWS,
        "historical_anchor_rows_retrained": 0,
        "next_action": next_action,
        "claim_scope": (
            "Local PRESENT-80 r7 2048/class, 16-pair, two-seed neural attribution "
            "diagnostic using frozen historical anchors; not formal scale, shared-weight "
            "transfer, attack or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **values}
        for seed, values in sorted(gate.get("seed_results", {}).items())
    ]


def result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[int, Mapping[str, Any]]:
    mapped: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("model") != AFFINE_MODEL:
            continue
        seed = int(row["seed"])
        if seed in mapped:
            raise ValueError(f"duplicate K1-BY6 result seed: {seed}")
        mapped[seed] = row
    if fail_closed and set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-BY6 result matrix is incomplete")
    return mapped


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        row.get("model") == AFFINE_MODEL
        and row.get("cipher_key") == "present80"
        and int(row.get("rounds", -1)) == 7
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == SAMPLE_STRUCTURE
        and int(row.get("trainable_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNT
        and row.get("compiled_program_expert_usage") == EXPECTED_USAGE
        and row.get("runtime_structure_window_control") == AFFINE_CONTROL
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
        and row.get("training", {}).get("device") == "cpu"
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
        len(events) == 4
        and all(row.get("event") == "cache_reuse" for row in events)
        and {int(row.get("seed", -1)) for row in events} == set(EXPECTED_SEEDS)
        and all(
            Path(str(row.get("cache_path", ""))).is_relative_to(K1BY3_CACHE_ROOT)
            for row in events
        )
    )


def cache_authority_checks() -> dict[str, bool]:
    expected = {
        ("train", 2): (4096, 2048),
        ("train", 3): (4096, 2048),
        ("validation", 10002): (2048, 1024),
        ("validation", 10003): (2048, 1024),
    }
    found: dict[tuple[str, int], tuple[int, int]] = {}
    required_files = True
    for metadata_path in K1BY3_CACHE_ROOT.glob("present80/r7/*/*/metadata.json"):
        metadata = _read_json(metadata_path)
        key = (metadata_path.parents[1].name, int(metadata.get("seed", -1)))
        found[key] = (
            int(metadata.get("total_rows", -1)),
            int(metadata.get("samples_per_class", -1)),
        )
        required_files = required_files and all(
            (metadata_path.parent / name).is_file()
            for name in ("metadata.json", "features.npy", "labels.npy")
        )
    return {
        "k1by3_cache_geometry_exact": found == expected,
        "k1by3_cache_arrays_complete": required_files and len(found) == 4,
    }


def cache_file_digests() -> dict[str, str]:
    return {
        str(path.relative_to(K1BY3_CACHE_ROOT)): _file_sha256(path)
        for path in sorted(K1BY3_CACHE_ROOT.glob("present80/r7/*/*/*"))
        if path.is_file()
    }


def _source_paths() -> dict[str, Path]:
    return {
        "k1by3_plan": K1BY3_PLAN,
        "k1by3_results": K1BY3_ROOT / "results.jsonl",
        "k1by3_gate": K1BY3_ROOT / "gate.json",
        "k1by3_preflight": K1BY3_ROOT / "preflight.json",
        "k1by3_validation": K1BY3_ROOT / "validation.json",
        "k1by5_config": K1BY5_CONFIG,
        "k1by5_results": K1BY5_ROOT / "results.jsonl",
        "k1by5_gate": K1BY5_ROOT / "gate.json",
        "k1by5_preflight": K1BY5_ROOT / "preflight.json",
        "k1by5_validation": K1BY5_ROOT / "validation.json",
    }


def _one_source_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    model: str,
) -> Mapping[str, Any]:
    matched = [
        row
        for row in rows
        if int(row.get("seed", -1)) == seed and row.get("model") == model
    ]
    if len(matched) != 1:
        raise ValueError(f"expected one K1-BY3 source row for seed={seed}, model={model}")
    return matched[0]


def _metric(row: Mapping[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, math.nan))


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
    "AFFINE_MODEL",
    "K1BY3_CACHE_ROOT",
    "PLAN_PATH",
    "RUN_ID",
    "adjudicate",
    "build_readiness",
    "cache_file_digests",
    "candidate_protocol_frozen",
    "comparison_rows",
    "historical_anchors",
    "read_tasks",
    "source_binding_checks",
    "task_map",
]
