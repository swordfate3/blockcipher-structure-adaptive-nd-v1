from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_permutation_expert_k1by3 as k1by3,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_same_checkpoint_runtime_swap_k1by8 as k1by8,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_trainable_post_expert_adapter_k1by13 as k1by13,
)
from blockcipher_nd.training.metrics import evaluate_binary_classifier


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_paired_runtime_objective_k1by14_present_r7_"
    "16pair_2048_seed2_seed3_20260801"
)
PLAN_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_paired_runtime_objective_k1by14_"
    "present_r7_16pair_2048_seed2_seed3.csv"
)
K1BY8_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_runtime_spn_same_checkpoint_runtime_swap_k1by8_"
    "present_r7_seed2_seed3_20260801"
)
K1BY13_ROOT = ROOT / (
    "outputs/remote_results_incomplete/"
    "i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_"
    "16pair_2048_seed2_seed3_20260801/output"
)
EXPECTED_PLAN_SHA256 = (
    "bba750d57ce0b9c77ff9a68257d5b238f5d449720e259f14962f0e13f8c59590"
)
SOURCE_DIGESTS = {
    "k1by8_config": "59f16610a530fd2e903cf74b25f4055f4282aaba39b50ce1da488f26a97d302f",
    "k1by8_gate": "995fc8302cd28d33b2a493ebc0dfc45aaffeb0f6a23d50ad541dce654734aa27",
    "k1by8_validation": "7c4b9d533db37958b47a1be39eb9e52c94b7a9d8fa7b484f158645cb74f7998c",
    "k1by8_results": "54af296f551d86d2af239d800b3c4ef52e86f074ac9b9b86806e2397be7aad79",
    "k1by13_gate": "002ddcf20e23921825beeef3171e4c0fcb6344f2aa81b9fea7e8145047b04b9c",
    "k1by13_validation": "f4b4c373572434d9603dea23fb4211d96b0f44816f3dbad8316996a58f24b84a",
    "k1by13_results": "221650784a3262df28b580f00102935c97763c2cc69867a8e8490af51f91e5b4",
}
EXPECTED_SEEDS = (2, 3)
ORIENTATIONS = ("correct_oriented", "swapped_orientation")
ORIENTATION_MODELS = {
    "correct_oriented": "runtime_spn_k1by14_paired_correct",
    "swapped_orientation": "runtime_spn_k1by14_paired_affine",
}
MODEL_TO_ORIENTATION = {
    model: orientation for orientation, model in ORIENTATION_MODELS.items()
}
RUNTIME_CONDITIONS = ("correct_runtime", "affine_runtime", "heldout_shuffled")
RUNTIME_MODELS = {
    "correct_runtime": "runtime_spn_k1by1_compiler_correct",
    "affine_runtime": "runtime_spn_k1by1_compiler_affine_wrong_endpoint",
    "heldout_shuffled": "runtime_spn_k1by1_compiler_wrong_binding",
}
EXPECTED_TRAINING_ROWS = 4
EXPECTED_EVALUATION_ROWS = 12
EXPECTED_PARAMETER_COUNT = 235780
EXPECTED_EPOCHS = 10
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
CONTRAST_SCALE = 0.25
CONTRAST_MARGIN = 0.02
HELDOUT_WRONG_BINDING_SEED = 20260814
SIGNAL_FLOOR = 0.550
RETENTION_FLOOR = -0.005
STRUCTURE_MARGIN = 0.005


def read_tasks(path: Path = PLAN_PATH) -> list[dict[str, Any]]:
    return k1by13.read_tasks(path)


def task_map(
    tasks: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for task in tasks:
        orientation = MODEL_TO_ORIENTATION.get(str(task.get("model_key")))
        if orientation is None:
            continue
        key = (int(task["seed"]), orientation)
        if key in mapped:
            raise ValueError(f"duplicate K1-BY14 task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-BY14 task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    if _file_sha256(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        return False
    mapped = task_map(tasks, fail_closed=False)
    if len(tasks) != EXPECTED_TRAINING_ROWS or set(mapped) != expected_training_keys():
        return False
    for (seed, orientation), task in mapped.items():
        options = task.get("model_options", {})
        expected_contrast = (
            "correct_vs_affine"
            if orientation == "correct_oriented"
            else "affine_vs_correct"
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
            and options.get("runtime_contrast_orientation") == expected_contrast
            and float(options.get("runtime_contrast_scale", math.nan))
            == CONTRAST_SCALE
            and float(options.get("runtime_contrast_margin", math.nan))
            == CONTRAST_MARGIN
            and int(options.get("affine_endpoint_multiplier", -1)) == 5
            and int(options.get("affine_endpoint_offset", -1)) == 1
            and int(options.get("heldout_wrong_binding_seed", -1))
            == HELDOUT_WRONG_BINDING_SEED
        ):
            return False
    return True


def source_binding_checks() -> dict[str, bool]:
    paths = source_artifact_paths()
    digest_checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in paths.items()
    }
    try:
        gate8 = _read_json(paths["k1by8_gate"])
        validation8 = _read_json(paths["k1by8_validation"])
        gate13 = _read_json(paths["k1by13_gate"])
        validation13 = _read_json(paths["k1by13_validation"])
    except (OSError, ValueError, json.JSONDecodeError):
        gate8 = {}
        validation8 = {}
        gate13 = {}
        validation13 = {}
    return {
        **k1by13.source_binding_checks(),
        **digest_checks,
        "k1by8_decision_exact": (
            gate8.get("status") == "pass"
            and gate8.get("method_status") == "hold"
            and gate8.get("decision")
            == "innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss"
            and validation8.get("status") == "pass"
        ),
        "k1by13_decision_exact": (
            gate13.get("status") == "hold"
            and gate13.get("decision")
            == "innovation1_runtime_spn_k1by13_signal_or_retention_failed"
            and validation13.get("status") == "pass"
        ),
    }


def build_model_for_task(
    task: Mapping[str, Any],
    *,
    model_key: str | None = None,
) -> torch.nn.Module:
    return k1by6.build_model_for_task(
        task,
        model_key=str(model_key or task["model_key"]),
    )


def build_readiness(*, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = task_map(tasks, fail_closed=False)
    protocol_checks = {
        **source_binding_checks(),
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "task_keys_exact": set(mapped) == expected_training_keys(),
        "zero_optimizer_steps_before_readiness": True,
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            models: dict[tuple[int, str], torch.nn.Module] = {}
            for key, task in mapped.items():
                torch.manual_seed(key[0])
                models[key] = build_model_for_task(task)
            parameter_geometry = {
                key: tuple(
                    (name, tuple(parameter.shape))
                    for name, parameter in model.named_parameters()
                )
                for key, model in models.items()
            }
            parameter_fingerprints = {
                key: k1by8.learned_parameter_fingerprint(model)
                for key, model in models.items()
            }
            parameter_counts = {
                key: int(model_metadata(model)["trainable_parameter_count"])
                for key, model in models.items()
            }
            auxiliary_metrics: dict[str, Any] = {}
            runtime_program_digests: dict[str, str] = {}
            for key, model in models.items():
                fixture = torch.randint(
                    0,
                    2,
                    (4, k1by3.EXPECTED_INPUT_BITS),
                    generator=torch.Generator().manual_seed(20260814 + key[0]),
                    dtype=torch.int64,
                ).to(torch.float32)
                labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
                model.train()
                model.zero_grad(set_to_none=True)
                logits = model(fixture).squeeze(1)
                auxiliary = model.compute_auxiliary_loss(logits, labels, "mse")
                if auxiliary is None:
                    raise RuntimeError("paired runtime auxiliary loss is missing")
                auxiliary.backward()
                gradient_l1 = sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                counterfactual = model._last_runtime_counterfactual_logits
                auxiliary_metrics[f"seed{key[0]}_{key[1]}"] = {
                    "auxiliary_loss": float(auxiliary.detach()),
                    "gradient_l1": gradient_l1,
                    "primary_counterfactual_max_abs_delta": float(
                        (logits.detach() - counterfactual.detach().squeeze(1))
                        .abs()
                        .max()
                    ),
                }
            reference_task = mapped[(EXPECTED_SEEDS[0], "correct_oriented")]
            for runtime_condition in RUNTIME_CONDITIONS:
                options = dict(reference_task["model_options"])
                for field in (
                    "runtime_contrast_orientation",
                    "runtime_contrast_scale",
                    "runtime_contrast_margin",
                    "heldout_wrong_binding_seed",
                ):
                    options.pop(field, None)
                if runtime_condition == "heldout_shuffled":
                    options["wrong_binding_seed"] = HELDOUT_WRONG_BINDING_SEED
                runtime_task = {**reference_task, "model_options": options}
                runtime_model = build_model_for_task(
                    runtime_task,
                    model_key=RUNTIME_MODELS[runtime_condition],
                )
                runtime_program_digests[runtime_condition] = (
                    runtime_model.compiled_program_semantic_sha256
                )
            evidence_checks = {
                "parameter_count_exact": set(parameter_counts.values())
                == {EXPECTED_PARAMETER_COUNT},
                "parameter_geometry_identical": len(
                    set(parameter_geometry.values())
                )
                == 1,
                "paired_initialization_exact_within_seed": all(
                    parameter_fingerprints[(seed, "correct_oriented")]
                    == parameter_fingerprints[(seed, "swapped_orientation")]
                    for seed in EXPECTED_SEEDS
                ),
                "distinct_seed_initializations": len(
                    {
                        parameter_fingerprints[(seed, "correct_oriented")]
                        for seed in EXPECTED_SEEDS
                    }
                )
                == len(EXPECTED_SEEDS),
                "contrast_configuration_exact": all(
                    model.runtime_contrast_scale == CONTRAST_SCALE
                    and model.runtime_contrast_margin == CONTRAST_MARGIN
                    and model.runtime_contrast_primary_sha256
                    != model.runtime_contrast_counterfactual_sha256
                    for model in models.values()
                ),
                "no_registered_counterfactual_parameters": all(
                    not any("counterfactual" in name for name, _ in model.named_parameters())
                    for model in models.values()
                ),
                "auxiliary_loss_finite_positive": all(
                    math.isfinite(values["auxiliary_loss"])
                    and values["auxiliary_loss"] > 0.0
                    for values in auxiliary_metrics.values()
                ),
                "auxiliary_gradient_finite_positive": all(
                    math.isfinite(values["gradient_l1"])
                    and values["gradient_l1"] > 0.0
                    for values in auxiliary_metrics.values()
                ),
                "runtime_intervention_changes_logits": all(
                    values["primary_counterfactual_max_abs_delta"] > 0.0
                    for values in auxiliary_metrics.values()
                ),
                "evaluation_runtime_programs_pairwise_distinct": (
                    len(set(runtime_program_digests.values()))
                    == len(RUNTIME_CONDITIONS)
                ),
                "no_cipher_or_absolute_identity": all(
                    model.uses_cipher_identity is False
                    and model.uses_absolute_cell_or_bit_identity is False
                    for model in models.values()
                ),
            }
            evidence_metrics = {
                "parameter_counts": {
                    f"seed{seed}_{orientation}": value
                    for (seed, orientation), value in parameter_counts.items()
                },
                "parameter_fingerprints": {
                    f"seed{seed}_{orientation}": value
                    for (seed, orientation), value in parameter_fingerprints.items()
                },
                "auxiliary": auxiliary_metrics,
                "evaluation_runtime_program_sha256": runtime_program_digests,
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


def evaluate_checkpoints(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
    cache_root: Path,
    device: str,
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    rows_by_key = training_map(result_rows)
    evaluation_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        dataset = load_validation_dataset(cache_root, seed=seed)
        for orientation in ORIENTATIONS:
            task = tasks_by_key[(seed, orientation)]
            source_model = build_model_for_task(task)
            checkpoint = local_checkpoint_path(
                rows_by_key[(seed, orientation)],
                checkpoint_root,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            source_model.load_state_dict(payload["state_dict"], strict=True)
            source_fingerprint = k1by8.learned_parameter_fingerprint(source_model)
            for runtime_condition in RUNTIME_CONDITIONS:
                options = dict(task["model_options"])
                for field in (
                    "runtime_contrast_orientation",
                    "runtime_contrast_scale",
                    "runtime_contrast_margin",
                    "heldout_wrong_binding_seed",
                ):
                    options.pop(field, None)
                if runtime_condition == "heldout_shuffled":
                    options["wrong_binding_seed"] = HELDOUT_WRONG_BINDING_SEED
                runtime_task = {**task, "model_options": options}
                model = build_model_for_task(
                    runtime_task,
                    model_key=RUNTIME_MODELS[runtime_condition],
                )
                k1by8.copy_named_parameters(model, source_model)
                target_fingerprint = k1by8.learned_parameter_fingerprint(model)
                metrics = evaluate_binary_classifier(
                    model,
                    dataset,
                    batch_size=64,
                    device=device,
                )
                evaluation_rows.append(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "orientation": orientation,
                        "runtime_condition": runtime_condition,
                        "metrics": metrics,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _file_sha256(checkpoint),
                        "learned_parameter_fingerprint": target_fingerprint,
                        "source_parameter_fingerprint": source_fingerprint,
                        "runtime_program_sha256": (
                            model.compiled_program_semantic_sha256
                        ),
                        "dataset_cache_dir": str(dataset.cache_dir),
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return evaluation_rows


def adjudicate(
    *,
    tasks: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    checkpoint_root: Path,
) -> dict[str, Any]:
    training = training_map(result_rows, fail_closed=False)
    evaluation = evaluation_map(evaluation_rows, fail_closed=False)
    anchors = k1by3_anchor_auc()
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("optimizer_step_authorized") is True
            and all(readiness.get("protocol_checks", {}).values())
            and all(readiness.get("evidence_checks", {}).values())
        ),
        "four_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "four_training_rows_complete": (
            len(result_rows) == EXPECTED_TRAINING_ROWS
            and set(training) == expected_training_keys()
        ),
        "twelve_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(evaluation) == expected_evaluation_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(
            result_rows,
            checkpoint_root=checkpoint_root,
        ),
        "disk_cache_created_and_reused": cache_protocol_frozen(progress_rows),
        "evaluation_uses_same_checkpoint_parameters": bool(evaluation)
        and all(
            row.get("learned_parameter_fingerprint")
            == row.get("source_parameter_fingerprint")
            and row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in evaluation.values()
        ),
        "source_anchor_complete": set(anchors) == set(EXPECTED_SEEDS),
    }
    research_checks: dict[str, bool] = {}
    seed_results: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        if all(key in evaluation for key in expected_evaluation_keys(seed=seed)):
            aucs = {
                orientation: {
                    runtime: _auc(evaluation[(seed, orientation, runtime)])
                    for runtime in RUNTIME_CONDITIONS
                }
                for orientation in ORIENTATIONS
            }
            correct = aucs["correct_oriented"]["correct_runtime"]
            swapped_primary = aucs["swapped_orientation"]["affine_runtime"]
            margins = {
                "anchor": correct - anchors[seed],
                "swapped_primary": correct - swapped_primary,
                "same_checkpoint_affine": (
                    correct - aucs["correct_oriented"]["affine_runtime"]
                ),
                "same_checkpoint_heldout_shuffled": (
                    correct - aucs["correct_oriented"]["heldout_shuffled"]
                ),
            }
            seed_results[str(seed)] = {
                "auc_by_orientation_and_runtime": aucs,
                "ordinary_k1by3_anchor_auc": anchors[seed],
                "correct_oriented_margins": margins,
            }
            research_checks[f"seed{seed}_signal"] = correct >= SIGNAL_FLOOR
            research_checks[f"seed{seed}_anchor_retention"] = (
                margins["anchor"] >= RETENTION_FLOOR
            )
            research_checks[f"seed{seed}_orientation_placebo"] = (
                margins["swapped_primary"] >= STRUCTURE_MARGIN
            )
            research_checks[f"seed{seed}_affine_margin"] = (
                margins["same_checkpoint_affine"] >= STRUCTURE_MARGIN
            )
            research_checks[f"seed{seed}_heldout_shuffle_margin"] = (
                margins["same_checkpoint_heldout_shuffled"] >= STRUCTURE_MARGIN
            )
            for orientation in ORIENTATIONS:
                history = training[(seed, orientation)].get("history", [])
                research_checks[f"seed{seed}_{orientation}_auxiliary_active"] = (
                    isinstance(history, list)
                    and any(
                        math.isfinite(float(epoch.get("train_auxiliary_loss", math.nan)))
                        and float(epoch.get("train_auxiliary_loss", 0.0)) > 0.0
                        for epoch in history
                        if isinstance(epoch, Mapping)
                    )
                    and all(
                        math.isfinite(float(epoch.get("train_runtime_loss_gap", math.nan)))
                        for epoch in history
                        if isinstance(epoch, Mapping)
                    )
                )
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    failed_research = sorted(
        name for name, passed in research_checks.items() if not passed
    )
    orientation_pass = all(
        research_checks.get(f"seed{seed}_orientation_placebo", False)
        for seed in EXPECTED_SEEDS
    )
    heldout_pass = all(
        research_checks.get(f"seed{seed}_heldout_shuffle_margin", False)
        for seed in EXPECTED_SEEDS
    )
    retention_pass = all(
        research_checks.get(f"seed{seed}_signal", False)
        and research_checks.get(f"seed{seed}_anchor_retention", False)
        for seed in EXPECTED_SEEDS
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by14_protocol_invalid"
        next_action = "Repair only the failed protocol invariant and rerun unchanged."
    elif not failed_research:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by14_paired_preference_supported"
        next_action = (
            "Repeat the unchanged four-row protocol on one compatible non-PRESENT "
            "SPN before any sample-scale increase."
        )
    elif not orientation_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by14_orientation_placebo_failed"
        next_action = (
            "Close this supervised runtime-preference objective because it can "
            "impose an arbitrary orientation; retain deterministic compilation."
        )
    elif not heldout_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by14_counterexample_overfit"
        next_action = (
            "Close this objective because it does not generalize beyond the affine "
            "counterexample; do not add more hand-selected wrong runtimes."
        )
    elif not retention_pass:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by14_anchor_retention_failed"
        next_action = (
            "Close this objective because it destroys the existing PRESENT signal; "
            "do not tune scale, margin, capacity or data."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by14_research_gate_failed"
        next_action = (
            "Close supervised topology preference on this PRESENT surface and "
            "retain deterministic primitive compilation as the supported method."
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
        "thresholds": {
            "signal_auc_min": SIGNAL_FLOOR,
            "correct_minus_anchor_auc_min": RETENTION_FLOOR,
            "all_structure_margins_auc_min": STRUCTURE_MARGIN,
            "contrast_scale": CONTRAST_SCALE,
            "contrast_margin": CONTRAST_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "PRESENT-80 r7 2048/class paired-runtime objective diagnostic on "
            "seeds 2/3; not formal-scale, universal-SPN or publication evidence."
        ),
    }


def training_protocol_frozen(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_root: Path,
) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_ORIENTATION
        and row.get("cipher_key") == "present80"
        and int(row.get("rounds", -1)) == 7
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == k1by3.EXPECTED_PAIRS
        and int(row.get("input_difference", -1)) == k1by3.INPUT_DIFFERENCE
        and row.get("difference_profile") == k1by3.DIFFERENCE_PROFILE
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and row.get("sample_structure") == k1by3.SAMPLE_STRUCTURE
        and int(row.get("trainable_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNT
        and int(row.get("training", {}).get("input_bits", -1))
        == k1by3.EXPECTED_INPUT_BITS
        and int(row.get("training", {}).get("train_rows", -1))
        == EXPECTED_TRAIN_ROWS
        and int(row.get("training", {}).get("validation_rows", -1))
        == EXPECTED_VALIDATION_ROWS
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("selected_checkpoint") == "best"
        and row.get("runtime_contrast_orientation")
        in {"correct_vs_affine", "affine_vs_correct"}
        and float(row.get("runtime_contrast_scale", math.nan)) == CONTRAST_SCALE
        and float(row.get("runtime_contrast_margin", math.nan)) == CONTRAST_MARGIN
        and local_checkpoint_path(row, checkpoint_root).is_file()
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
        and sum(row.get("event") == "cache_reuse" for row in events) == 4
    )


def training_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        orientation = MODEL_TO_ORIENTATION.get(str(row.get("model")))
        if orientation is None:
            continue
        key = (int(row["seed"]), orientation)
        if key in mapped:
            raise ValueError(f"duplicate K1-BY14 training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-BY14 training rows are incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["orientation"]),
            str(row["runtime_condition"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-BY14 evaluation row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_evaluation_keys():
        raise ValueError("K1-BY14 evaluation rows are incomplete")
    return mapped


def load_validation_dataset(cache_root: Path, *, seed: int) -> DiskDifferentialDataset:
    matches = list(
        (cache_root / "present80" / "r7" / "validation").glob(
            f"seed-{10000 + seed}_*"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"expected one K1-BY14 validation cache for seed {seed}")
    cache_dir = matches[0]
    metadata = _read_json(cache_dir / "metadata.json")
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
    if features.shape != (EXPECTED_VALIDATION_ROWS, k1by3.EXPECTED_INPUT_BITS):
        raise ValueError("K1-BY14 validation feature geometry drifted")
    if labels.shape != (EXPECTED_VALIDATION_ROWS,):
        raise ValueError("K1-BY14 validation label geometry drifted")
    return DiskDifferentialDataset(
        features=features,
        labels=labels,
        metadata=metadata,
        cache_dir=cache_dir,
    )


def k1by3_anchor_auc() -> dict[int, float]:
    rows = [
        json.loads(line)
        for line in (k1by13.K1BY3_ROOT / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {
        seed: float(
            next(
                row["metrics"]["auc"]
                for row in rows
                if int(row["seed"]) == seed
                and row["model"] == k1by3.CONDITIONS["correct_permutation_routing"]
            )
        )
        for seed in EXPECTED_SEEDS
    }


def local_checkpoint_path(row: Mapping[str, Any], checkpoint_root: Path) -> Path:
    raw = str(row.get("training", {}).get("checkpoint_output", ""))
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    path = checkpoint_root / name
    if not path.is_file():
        raise ValueError(f"missing K1-BY14 checkpoint: {path}")
    return path


def expected_training_keys() -> set[tuple[int, str]]:
    return {(seed, orientation) for seed in EXPECTED_SEEDS for orientation in ORIENTATIONS}


def expected_evaluation_keys(
    *,
    seed: int | None = None,
) -> set[tuple[int, str, str]]:
    seeds = EXPECTED_SEEDS if seed is None else (seed,)
    return {
        (value, orientation, runtime)
        for value in seeds
        for orientation in ORIENTATIONS
        for runtime in RUNTIME_CONDITIONS
    }


def source_artifact_paths() -> dict[str, Path]:
    return {
        "k1by8_config": k1by8.CONFIG_PATH,
        "k1by8_gate": K1BY8_ROOT / "gate.json",
        "k1by8_validation": K1BY8_ROOT / "validation.json",
        "k1by8_results": K1BY8_ROOT / "results.jsonl",
        "k1by13_gate": K1BY13_ROOT / "gate.json",
        "k1by13_validation": K1BY13_ROOT / "validation.json",
        "k1by13_results": K1BY13_ROOT / "results.jsonl",
    }


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
    "ORIENTATIONS",
    "PLAN_PATH",
    "RUN_ID",
    "RUNTIME_CONDITIONS",
    "adjudicate",
    "build_model_for_task",
    "build_readiness",
    "candidate_protocol_frozen",
    "evaluate_checkpoints",
    "evaluation_map",
    "expected_evaluation_keys",
    "expected_training_keys",
    "read_tasks",
    "source_binding_checks",
    "task_map",
    "training_map",
]
