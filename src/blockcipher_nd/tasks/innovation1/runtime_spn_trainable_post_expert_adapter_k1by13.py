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
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_permutation_expert_k1by3 as k1by3,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
PLAN_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_trainable_post_expert_adapter_k1by13_"
    "present_r7_16pair_2048_seed2_seed3.csv"
)
K1BY3_ROOT = ROOT / (
    "outputs/local_diagnostic/"
    "i1_runtime_spn_permutation_expert_k1by3_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
K1BY12_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_runtime_spn_post_expert_edge_residual_k1by12_"
    "present_r7_seed2_seed3_20260801"
)
K1BY12_CONFIG = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_post_expert_edge_residual_k1by12_20260801.json"
)
SOURCE_DIGESTS = {
    "k1by3_plan": "68c35c24b3416a13856d7eecfb8c360081a2e7924750ce140689e2c51ab978a2",
    "k1by3_results": "54d06976024662b274981b195eb36756c4bbdf43f5a2b8f16de407f75052dc94",
    "k1by3_gate": "62b6c44e9c36153b903cc7131eb9c4717badd6843d406b064568db0f1cb99c68",
    "k1by3_validation": "de27d159224762aa4dc08ec04fb948d14a42ac9d79524b6396e5d0269e178875",
    "k1by12_config": "e5bf57ace0bdd13891c0a3f0f80030a6a05e86acb6c64b75f4b9273897a42c77",
    "k1by12_results": "1d656469a2da2a1e39ad3d0a2c9596c847da47213c8a82487f9c4452c3596d06",
    "k1by12_gate": "da31a802ceb623f8e4b2b587c53a37f472713ccd8e7bac113986e0e90b458664",
    "k1by12_validation": "2d66a8cbf2979741dfd03dc0865b0c9a0164befa097c4a6e6d90acf05acd9c6a",
}
EXPECTED_PLAN_SHA256 = (
    "29bbcce189c4229e71b12e8568b2624f632f4178319edc3cb969d4d40bdf72a5"
)
EXPECTED_SEEDS = (2, 3)
EXPECTED_EPOCHS = 10
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
ANCHOR_PARAMETER_COUNT = 235780
ADAPTER_PARAMETER_COUNT = 237876
ADAPTER_PARAMETER_DELTA = ADAPTER_PARAMETER_COUNT - ANCHOR_PARAMETER_COUNT
ADAPTER_BOTTLENECK_DIM = 16
SIGNAL_FLOOR = 0.550
RETENTION_FLOOR = -0.005
STRUCTURE_MARGIN = 0.005
SHUFFLED_SOURCE_CELLS = (3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9, 0, 7, 14, 5, 12)

CONDITIONS = {
    "anchor_correct": "runtime_spn_k1by13_anchor_correct",
    "adapter_correct": "runtime_spn_k1by13_adapter_correct",
    "adapter_affine": "runtime_spn_k1by13_adapter_affine",
    "adapter_shuffled": "runtime_spn_k1by13_adapter_shuffled",
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONDITIONS.items()}
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(CONDITIONS)


def read_tasks(path: Path = PLAN_PATH) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=k1by3.EXPECTED_PAIRS,
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
            raise ValueError(f"duplicate K1-BY13 task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY13 task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    if _file_sha256(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        return False
    mapped = task_map(tasks, fail_closed=False)
    if len(tasks) != EXPECTED_RESULT_ROWS or set(mapped) != expected_keys():
        return False
    for (seed, condition), task in mapped.items():
        options = task.get("model_options", {})
        expected_adapter = "none" if condition == "anchor_correct" else (
            "edge_conditioned_bottleneck"
        )
        if not (
            task.get("cipher_key") == "present80"
            and int(task.get("rounds", -1)) == 7
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1)) == 2048
            and int(task.get("validation_samples_total", -1))
            == EXPECTED_VALIDATION_ROWS
            and int(task.get("pairs_per_sample", -1)) == k1by3.EXPECTED_PAIRS
            and int(task.get("input_difference", -1)) == k1by3.INPUT_DIFFERENCE
            and task.get("difference_profile") == k1by3.DIFFERENCE_PROFILE
            and task.get("feature_encoding") == "ciphertext_pair_bits"
            and task.get("negative_mode") == "encrypted_random_plaintexts"
            and task.get("sample_structure") == k1by3.SAMPLE_STRUCTURE
            and int(task.get("train_key", -1)) == k1by3.TRAIN_KEY
            and int(task.get("validation_key", -1)) == k1by3.VALIDATION_KEY
            and task.get("loss") == "mse"
            and task.get("optimizer") == "adam"
            and float(task.get("learning_rate", math.nan)) == 1e-4
            and float(task.get("weight_decay", math.nan)) == 1e-5
            and task.get("checkpoint_metric") == "val_auc"
            and task.get("restore_best_checkpoint") is True
            and int(task.get("target_epochs", -1)) == EXPECTED_EPOCHS
            and options.get("runtime_structure_path")
            == "configs/runtime/spn/present64.json"
            and int(options.get("runtime_round_start", -1)) == 0
            and int(options.get("runtime_rounds", -1)) == 2
            and int(options.get("primitive_hidden_dim", -1)) == 32
            and int(options.get("pair_embedding_dim", -1)) == 128
            and options.get("post_expert_adapter_mode") == expected_adapter
        ):
            return False
        if condition != "anchor_correct" and int(
            options.get("post_expert_adapter_bottleneck_dim", -1)
        ) != ADAPTER_BOTTLENECK_DIM:
            return False
        if condition == "adapter_affine":
            if (
                int(options.get("affine_endpoint_multiplier", -1)) != 5
                or int(options.get("affine_endpoint_offset", -1)) != 1
            ):
                return False
        elif (
            "affine_endpoint_multiplier" in options
            or "affine_endpoint_offset" in options
        ):
            return False
        if condition == "adapter_shuffled":
            if tuple(options.get("post_expert_source_cell_permutation", ())) != (
                SHUFFLED_SOURCE_CELLS
            ):
                return False
        elif "post_expert_source_cell_permutation" in options:
            return False
    return True


def source_binding_checks() -> dict[str, bool]:
    paths = {
        "k1by3_plan": k1by3.PLAN_PATH,
        "k1by3_results": K1BY3_ROOT / "results.jsonl",
        "k1by3_gate": K1BY3_ROOT / "gate.json",
        "k1by3_validation": K1BY3_ROOT / "validation.json",
        "k1by12_config": K1BY12_CONFIG,
        "k1by12_results": K1BY12_ROOT / "results.jsonl",
        "k1by12_gate": K1BY12_ROOT / "gate.json",
        "k1by12_validation": K1BY12_ROOT / "validation.json",
    }
    checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in paths.items()
    }
    try:
        gate3 = _read_json(paths["k1by3_gate"])
        validation3 = _read_json(paths["k1by3_validation"])
        gate12 = _read_json(paths["k1by12_gate"])
        validation12 = _read_json(paths["k1by12_validation"])
    except (OSError, ValueError, json.JSONDecodeError):
        gate3 = {}
        validation3 = {}
        gate12 = {}
        validation12 = {}
    checks["k1by3_anchor_protocol_exact"] = (
        gate3.get("status") == "hold"
        and gate3.get("decision")
        == "innovation1_runtime_spn_k1by3_permutation_attribution_not_supported"
        and validation3.get("status") == "pass"
    )
    checks["k1by12_trainable_route_authorized"] = (
        gate12.get("status") == "pass"
        and gate12.get("research_gate_passed") is False
        and gate12.get("decision")
        == "innovation1_runtime_spn_k1by12_deterministic_interventions_exhausted"
        and validation12.get("status") == "pass"
        and validation12.get("optimizer_steps") == 0
    )
    return checks


def build_readiness(*, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        **source_binding_checks(),
        "eight_frozen_tasks_exact": (
            len(tasks) == EXPECTED_RESULT_ROWS and set(mapped) == expected_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "local_readiness_has_zero_optimizer_steps": True,
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            fixture = torch.as_tensor(
                np.random.default_rng(20260813).integers(
                    0,
                    2,
                    size=(8, k1by3.EXPECTED_INPUT_BITS),
                    dtype=np.uint8,
                ),
                dtype=torch.float32,
            )
            models: dict[str, torch.nn.Module] = {}
            for condition in CONDITIONS:
                torch.manual_seed(20260813)
                task = mapped[(EXPECTED_SEEDS[0], condition)]
                models[condition] = k1by6.build_model_for_task(
                    task,
                    model_key=CONDITIONS[condition],
                )
            parameter_counts = {
                condition: int(model_metadata(model)["trainable_parameter_count"])
                for condition, model in models.items()
            }
            outputs = {}
            for condition, model in models.items():
                model.eval()
                with torch.inference_mode():
                    outputs[condition] = model(fixture)
            common_parameters_equal = _common_parameters_equal(models)
            candidate_geometry = {
                condition: _adapter_parameter_geometry(models[condition])
                for condition in CONDITIONS
                if condition != "anchor_correct"
            }
            output_projection_zero = {
                condition: _adapter_output_is_zero(models[condition])
                for condition in CONDITIONS
                if condition != "anchor_correct"
            }
            correct = models["adapter_correct"]
            correct.train()
            correct.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(correct(fixture)).flatten(),
                torch.arange(len(fixture), dtype=torch.float32).remainder(2),
            )
            loss.backward()
            output_gradient_l1 = _adapter_output_gradient_l1(correct)
            edge_fingerprints = {
                condition: _tensor_sha256(
                    model.conditioner.post_expert_edge_source_cells
                )
                for condition, model in models.items()
                if condition != "anchor_correct"
            }
            program_digests = {
                condition: model.compiled_program_semantic_sha256
                for condition, model in models.items()
            }
            evidence_checks = {
                "all_outputs_finite_and_equal_shape": all(
                    output.shape == (len(fixture), 1)
                    and torch.isfinite(output).all()
                    for output in outputs.values()
                ),
                "parameter_counts_exact": (
                    parameter_counts["anchor_correct"] == ANCHOR_PARAMETER_COUNT
                    and set(
                        value
                        for condition, value in parameter_counts.items()
                        if condition != "anchor_correct"
                    )
                    == {ADAPTER_PARAMETER_COUNT}
                    and ADAPTER_PARAMETER_DELTA == 2096
                ),
                "all_common_named_parameters_equal": common_parameters_equal,
                "candidate_adapter_geometry_equal": (
                    len(set(candidate_geometry.values())) == 1
                ),
                "candidate_output_projections_exact_zero": all(
                    output_projection_zero.values()
                ),
                "correct_adapter_exactly_replays_anchor_at_initialization": (
                    torch.equal(outputs["anchor_correct"], outputs["adapter_correct"])
                ),
                "shuffled_adapter_exactly_replays_correct_at_initialization": (
                    torch.equal(
                        outputs["adapter_correct"],
                        outputs["adapter_shuffled"],
                    )
                ),
                "adapter_zero_output_layer_receives_gradient": (
                    math.isfinite(output_gradient_l1) and output_gradient_l1 > 0.0
                ),
                "candidate_edge_bindings_pairwise_distinct": (
                    len(set(edge_fingerprints.values())) == 3
                ),
                "shuffled_preserves_correct_program_semantics": (
                    program_digests["adapter_shuffled"]
                    == program_digests["adapter_correct"]
                ),
                "affine_changes_program_semantics": (
                    program_digests["adapter_affine"]
                    != program_digests["adapter_correct"]
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
                "adapter_parameter_delta": ADAPTER_PARAMETER_DELTA,
                "adapter_parameter_delta_fraction": (
                    ADAPTER_PARAMETER_DELTA / ANCHOR_PARAMETER_COUNT
                ),
                "candidate_adapter_geometry": {
                    condition: list(geometry)
                    for condition, geometry in candidate_geometry.items()
                },
                "adapter_output_gradient_l1": output_gradient_l1,
                "edge_binding_fingerprints": edge_fingerprints,
                "program_semantic_sha256": program_digests,
                "initial_output_max_delta": {
                    condition: float(
                        (outputs["anchor_correct"] - output).abs().max()
                    )
                    for condition, output in outputs.items()
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
        "local_training_authorized": False,
        "remote_cuda_training_authorized": status == "pass",
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
    checkpoint_root: Path,
) -> dict[str, Any]:
    rows = result_map(result_rows, fail_closed=False)
    checkpoint_norms, checkpoint_errors = checkpoint_adapter_norms(
        rows,
        checkpoint_root=checkpoint_root,
    )
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
        "training_protocol_frozen": training_protocol_frozen(
            result_rows,
            checkpoint_root=checkpoint_root,
        ),
        "disk_cache_created_and_reused": cache_protocol_frozen(progress_rows),
        "finite_auc_metrics": bool(rows)
        and all(math.isfinite(_auc(row)) for row in rows.values()),
        "candidate_checkpoint_norms_recomputed": (
            len(checkpoint_norms) == 6 and not checkpoint_errors
        ),
    }
    seed_results: dict[str, dict[str, Any]] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        if all((seed, condition) in rows for condition in CONDITIONS):
            aucs = {
                condition: _auc(rows[(seed, condition)]) for condition in CONDITIONS
            }
            correct = aucs["adapter_correct"]
            margins = {
                "anchor_correct": correct - aucs["anchor_correct"],
                "adapter_affine": correct - aucs["adapter_affine"],
                "adapter_shuffled": correct - aucs["adapter_shuffled"],
            }
            norms = {
                condition: checkpoint_norms.get((seed, condition), math.nan)
                for condition in CONDITIONS
                if condition != "anchor_correct"
            }
            seed_results[str(seed)] = {
                "auc_by_condition": aucs,
                "correct_minus_control": margins,
                "adapter_output_projection_l2": norms,
            }
            research_checks[f"seed{seed}_signal"] = correct >= SIGNAL_FLOOR
            research_checks[f"seed{seed}_retention"] = (
                margins["anchor_correct"] >= RETENTION_FLOOR
            )
            research_checks[f"seed{seed}_affine_margin"] = (
                margins["adapter_affine"] >= STRUCTURE_MARGIN
            )
            research_checks[f"seed{seed}_shuffled_margin"] = (
                margins["adapter_shuffled"] >= STRUCTURE_MARGIN
            )
            research_checks[f"seed{seed}_adapter_weights_moved"] = all(
                math.isfinite(value) and value > 1e-8 for value in norms.values()
            )
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    signal_retention_pass = all(
        research_checks.get(f"seed{seed}_signal", False)
        and research_checks.get(f"seed{seed}_retention", False)
        for seed in EXPECTED_SEEDS
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by13_protocol_invalid"
        next_action = "Repair only the failed protocol invariant and rerun unchanged."
    elif not failed_research:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by13_trainable_adapter_supported"
        next_action = (
            "Run one fresh-seed local-scale confirmation with the same four "
            "conditions before any 65536/class gate."
        )
    elif signal_retention_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by13_capacity_without_edge_use"
        next_action = (
            "Discard this adapter because it retained signal without stable correct-"
            "edge attribution; do not tune its bottleneck or depth."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by13_signal_or_retention_failed"
        next_action = (
            "Return to the K1-BY8 anchor and revisit the training objective rather "
            "than adding another deterministic or trainable adapter."
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
        "checkpoint_adapter_norm_errors": checkpoint_errors,
        "parameter_counts": {
            "anchor": ANCHOR_PARAMETER_COUNT,
            "adapter": ADAPTER_PARAMETER_COUNT,
            "delta": ADAPTER_PARAMETER_DELTA,
        },
        "thresholds": {
            "signal_auc_min": SIGNAL_FLOOR,
            "correct_minus_anchor_auc_min": RETENTION_FLOOR,
            "correct_minus_affine_auc_min": STRUCTURE_MARGIN,
            "correct_minus_shuffled_auc_min": STRUCTURE_MARGIN,
            "adapter_output_projection_l2_min_exclusive": 1e-8,
        },
        "next_action": next_action,
        "claim_scope": (
            "PRESENT-80 r7 2048/class trainable post-expert adapter diagnostic "
            "on seeds 2/3; remote A6000 is a local-CUDA availability exception, "
            "not a scale or formal-evidence claim."
        ),
    }


def checkpoint_adapter_norms(
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    checkpoint_root: Path,
) -> tuple[dict[tuple[int, str], float], list[str]]:
    norms: dict[tuple[int, str], float] = {}
    errors: list[str] = []
    for key, row in rows.items():
        seed, condition = key
        if condition == "anchor_correct":
            continue
        checkpoint = _local_checkpoint_path(row, checkpoint_root)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload["state_dict"]
            tensors = [
                tensor.detach().to(torch.float64)
                for name, tensor in state.items()
                if "post_expert_trainable_adapter.output_projection" in name
            ]
            if len(tensors) != 2:
                raise ValueError("expected adapter output weight and bias")
            norms[(seed, condition)] = math.sqrt(
                sum(float(tensor.square().sum()) for tensor in tensors)
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"seed{seed}/{condition}: {type(exc).__name__}: {exc}")
    return norms, errors


def training_protocol_frozen(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_root: Path,
) -> bool:
    return len(rows) == EXPECTED_RESULT_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and row.get("cipher_key") == "present80"
        and int(row.get("rounds", -1)) == 7
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == k1by3.EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == k1by3.INPUT_DIFFERENCE
        and row.get("difference_profile") == k1by3.DIFFERENCE_PROFILE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == k1by3.SAMPLE_STRUCTURE
        and int(row.get("trainable_parameter_count", -1))
        == (
            ANCHOR_PARAMETER_COUNT
            if MODEL_TO_CONDITION.get(str(row.get("model"))) == "anchor_correct"
            else ADAPTER_PARAMETER_COUNT
        )
        and row.get("compiled_program_expert_usage")
        == {"sbox4_table": 32, "linear_permutation": 32, "linear_gf2": 0}
        and int(row.get("training", {}).get("input_bits", -1))
        == k1by3.EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1))
        == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and _local_checkpoint_path(row, checkpoint_root).is_file()
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
            raise ValueError(f"duplicate K1-BY13 result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_keys():
        raise ValueError("K1-BY13 result matrix is incomplete")
    return mapped


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        aucs = values["auc_by_condition"]
        margins = values["correct_minus_control"]
        norms = values["adapter_output_projection_l2"]
        rows.append(
            {
                "seed": int(seed),
                **{f"{name}_auc": aucs[name] for name in CONDITIONS},
                **{f"correct_minus_{name}": margins[name] for name in margins},
                **{f"{name}_output_projection_l2": norms[name] for name in norms},
            }
        )
    return rows


def expected_keys() -> set[tuple[int, str]]:
    return {(seed, condition) for seed in EXPECTED_SEEDS for condition in CONDITIONS}


def _common_parameters_equal(models: Mapping[str, torch.nn.Module]) -> bool:
    named = {
        condition: dict(model.named_parameters()) for condition, model in models.items()
    }
    common_names = set.intersection(*(set(values) for values in named.values()))
    return bool(common_names) and all(
        torch.equal(named["anchor_correct"][name], values[name])
        for condition, values in named.items()
        if condition != "anchor_correct"
        for name in common_names
    )


def _adapter_parameter_geometry(
    model: torch.nn.Module,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(parameter.shape))
        for name, parameter in model.conditioner.post_expert_trainable_adapter.named_parameters()
    )


def _adapter_output_is_zero(model: torch.nn.Module) -> bool:
    output = model.conditioner.post_expert_trainable_adapter.output_projection
    return (
        output is not None
        and int(torch.count_nonzero(output.weight)) == 0
        and int(torch.count_nonzero(output.bias)) == 0
    )


def _adapter_output_gradient_l1(model: torch.nn.Module) -> float:
    output = model.conditioner.post_expert_trainable_adapter.output_projection
    if output is None:
        return 0.0
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in output.parameters()
        if parameter.grad is not None
    )


def _local_checkpoint_path(row: Mapping[str, Any], checkpoint_root: Path) -> Path:
    raw = str(row.get("training", {}).get("checkpoint_output", ""))
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    return checkpoint_root / name


def _auc(row: Mapping[str, Any]) -> float:
    return float(row.get("metrics", {}).get("auc", math.nan))


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


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
    "checkpoint_adapter_norms",
    "comparison_rows",
    "read_tasks",
    "source_binding_checks",
    "task_map",
]
