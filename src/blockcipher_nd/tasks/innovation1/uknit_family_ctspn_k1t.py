from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    deterministic_position_histogram,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    input_geometry,
    load_bound_state,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import project_features
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONFIRMATION_KEYS,
    DIFFERENCE_PROFILE,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_PAIRS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
    RUN_ID as K1R_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import (
    EXPECTED_SOURCE_DIGESTS as K1S_BOUND_SOURCE_DIGESTS,
    RUN_ID as K1S_RUN_ID,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = (
    "i1_uknit_family_ctspn_deterministic_position_residual_"
    "k1t_2048_seed3_seed4_20260728"
)
K1S_DECISION = (
    "innovation1_uknit_family_ctspn_k1s_"
    "learned_representation_access_not_supported"
)
EXPECTED_K1S_DIGESTS = {
    "k1s_gate": "b7b16cef0c14f27c3325b65deaaca4acb206e811397a1839d8f28aca45ecc2e6",
    "k1s_validation": (
        "f264c54343986647594935d0f9b0aee78ec1a986f01eb1d135ef8048b23cdafa"
    ),
    "k1s_results": "c9d2b0d899cbe132359755cc19ddc9588e1d97e46dc3e5cdd08ca3e5356be077",
    "k1s_feature_manifest": (
        "b9d62cab840069e34588d6a609fff589ba66d04e26c20a8820036b3b13b36d91"
    ),
    "k1s_scorer_manifest": (
        "93e2d9f3d843be91334fe8640d9ce2555dddf3e5635f42bd21e32e1e9894afc9"
    ),
    "k1s_checkpoint_manifest": (
        "73758c90a9564fa35b61e0d2bb707ca88bf3b912dd2926de1c457b0c461ee046"
    ),
}
EXPECTED_SOURCE_DIGESTS = {
    **{f"bound_{name}": digest for name, digest in K1S_BOUND_SOURCE_DIGESTS.items()},
    **EXPECTED_K1S_DIGESTS,
}
CONTROL_MODELS = {
    "exact_position_histogram_residual": (
        "runtime_spn_ct_k1t_position_histogram_true"
    ),
    "wrong_sbox_position_histogram_residual": (
        "runtime_spn_ct_k1t_position_histogram_wrong_sbox"
    ),
    "invariant_histogram_residual": (
        "runtime_spn_ct_k1t_position_histogram_invariant"
    ),
}
MODEL_TO_CONDITION = {model: condition for condition, model in CONTROL_MODELS.items()}
ANCHOR_CONDITION = "current_k1r_exact_anchor"
EVALUATION_CONDITIONS = (*CONTROL_MODELS, ANCHOR_CONDITION)
EXPECTED_PARAMETER_COUNT = 214316
EXPECTED_PARAMETER_CAP = 225000
EXPECTED_TRAINING_ROWS = len(EXPECTED_SEEDS) * len(CONTROL_MODELS)
EXPECTED_EVALUATION_ROWS = (
    len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(EVALUATION_CONDITIONS)
)
AUC_FLOOR = 0.600
ANCHOR_MARGIN = 0.050
WRONG_SBOX_MARGIN = 0.010
INVARIANT_MARGIN = 0.030
REPLAY_TOLERANCE = 0.0


def build_k1t_control(
    *,
    task: Mapping[str, Any],
    condition: str,
    input_bits: int,
) -> torch.nn.Module:
    if condition not in CONTROL_MODELS:
        raise ValueError("unknown K1-T condition")
    options = deepcopy(dict(task["model_options"]))
    _, pair_bits = input_geometry(str(task["cipher_key"]))
    return build_model(
        CONTROL_MODELS[condition],
        input_bits=input_bits,
        hidden_bits=32,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=options,
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
            raise ValueError(f"duplicate K1-T task: {key}")
        mapped[key] = task
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-T task matrix is incomplete")
    return mapped


def candidate_protocol_frozen(tasks: Sequence[Mapping[str, Any]]) -> bool:
    mapped = task_map(tasks, fail_closed=False)
    return (
        len(tasks) == EXPECTED_TRAINING_ROWS
        and set(mapped) == expected_training_keys()
        and all(
            task.get("cipher_key") == "uknit64"
            and int(task.get("rounds", -1)) == 5
            and int(task.get("seed", -1)) == seed
            and int(task.get("samples_per_class", -1)) == 2048
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
            and int(task.get("model_options", {}).get("histogram_value_dim", -1)) == 8
            and float(
                task.get("model_options", {}).get(
                    "histogram_gate_initial_effective", math.nan
                )
            )
            == 0.05
            for (seed, _), task in mapped.items()
        )
    )


def source_binding_checks(
    *,
    source_digests: Mapping[str, str],
    k1s_gate: Mapping[str, Any],
    k1s_validation: Mapping[str, Any],
    bound_source_checks: Mapping[str, bool],
) -> dict[str, bool]:
    tap_access = k1s_gate.get("tap_accessible_on_all_fresh_splits", {})
    return {
        "source_artifact_digests_exact": (
            dict(source_digests) == EXPECTED_SOURCE_DIGESTS
        ),
        "k1s_clean_hold_exact": (
            k1s_gate.get("run_id") == K1S_RUN_ID
            and k1s_gate.get("status") == "hold"
            and k1s_gate.get("decision") == K1S_DECISION
            and bool(k1s_gate.get("protocol_checks"))
            and all(k1s_gate.get("protocol_checks", {}).values())
            and k1s_validation.get("run_id") == K1S_RUN_ID
            and k1s_validation.get("status") == "pass"
            and not k1s_validation.get("errors")
        ),
        "k1s_tap_decision_exact": (
            tap_access.get("T0_exact_position_histogram") is True
            and tap_access.get("T1_bit_encoder_position") is False
            and tap_access.get("T2_topology_delta_position") is False
            and tap_access.get("T3_invariant_cell_pool") is False
        ),
        "bound_k1q_k1r_sources_exact": (
            bool(bound_source_checks) and all(bound_source_checks.values())
        ),
    }


def build_k1t_readiness(
    *,
    tasks: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    tasks_by_key = task_map(tasks, fail_closed=False)
    expected_datasets = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    protocol_checks = {
        **dict(source_checks),
        "six_frozen_tasks_exact": (
            len(tasks) == EXPECTED_TRAINING_ROWS
            and set(tasks_by_key) == expected_training_keys()
        ),
        "candidate_protocol_frozen": candidate_protocol_frozen(tasks),
        "six_bound_source_caches": set(datasets) == expected_datasets,
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            fixture = torch.as_tensor(
                np.asarray(datasets[(3, "train_seen")].features[:8]).copy(),
                dtype=torch.float32,
            )
            models = {
                condition: build_k1t_control(
                    task=tasks_by_key[(3, condition)],
                    condition=condition,
                    input_bits=int(fixture.shape[1]),
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
            exact = models["exact_position_histogram_residual"]
            wrong = models["wrong_sbox_position_histogram_residual"]
            invariant = models["invariant_histogram_residual"]
            runtime = project_features(fixture, exact.runtime_structure)
            exact_histogram = deterministic_position_histogram(
                runtime, exact.runtime_structure
            )
            wrong_histogram = deterministic_position_histogram(
                project_features(fixture, wrong.runtime_structure),
                wrong.runtime_structure,
            )
            invariant_histogram = deterministic_position_histogram(
                runtime,
                exact.runtime_structure,
                invariant_cells=True,
            )
            expected_invariant = exact_histogram.mean(
                dim=2, keepdim=True
            ).expand_as(exact_histogram)
            state = exact.state_dict()
            wrong.load_state_dict(state, strict=True)
            invariant.load_state_dict(state, strict=True)
            exact.eval()
            wrong.eval()
            invariant.eval()
            with torch.no_grad():
                exact_logits = exact(fixture)
                exact_replay = exact(fixture)
                wrong_logits = wrong(fixture)
                invariant_logits = invariant(fixture)
            exact.train()
            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(exact(fixture)).flatten(),
                torch.arange(len(fixture), dtype=torch.float32).remainder(2),
            )
            loss.backward()
            histogram_gradients = [
                parameter.grad
                for name, parameter in exact.named_parameters()
                if "histogram_" in name and parameter.grad is not None
            ]
            gradient_l1 = sum(
                float(gradient.detach().abs().sum())
                for gradient in histogram_gradients
            )
            evidence_checks = {
                "three_controls_identical_geometry": len(set(geometries.values())) == 1,
                "parameter_count_exact_and_bounded": (
                    set(parameter_counts.values()) == {EXPECTED_PARAMETER_COUNT}
                    and EXPECTED_PARAMETER_COUNT <= EXPECTED_PARAMETER_CAP
                ),
                "wrong_sbox_changes_histogram_only_with_same_linear_operators": (
                    torch.equal(
                        exact.runtime_structure.linear_matrices,
                        wrong.runtime_structure.linear_matrices,
                    )
                    and not torch.equal(exact_histogram, wrong_histogram)
                ),
                "invariant_control_exact_and_nonidentity": (
                    torch.equal(invariant_histogram, expected_invariant)
                    and not torch.equal(exact_histogram, invariant_histogram)
                ),
                "finite_equal_shape_logits": (
                    exact_logits.shape == wrong_logits.shape == invariant_logits.shape
                    and all(
                        torch.isfinite(values).all()
                        for values in (exact_logits, wrong_logits, invariant_logits)
                    )
                ),
                "shared_state_semantic_controls_observable": (
                    not torch.equal(exact_logits, wrong_logits)
                    and not torch.equal(exact_logits, invariant_logits)
                ),
                "both_bounded_gates_open": (
                    0.0
                    < abs(
                        float(torch.tanh(exact.backbone.residual_gate.detach()))
                    )
                    < 1.0
                    and 0.0
                    < abs(
                        float(torch.tanh(exact.backbone.histogram_gate.detach()))
                    )
                    < 1.0
                ),
                "histogram_branch_gradient_finite_nonzero": (
                    bool(histogram_gradients)
                    and all(torch.isfinite(gradient).all() for gradient in histogram_gradients)
                    and gradient_l1 > 0.0
                ),
                "evaluation_forward_deterministic": torch.equal(
                    exact_logits, exact_replay
                ),
            }
            evidence_metrics = {
                "parameter_counts": parameter_counts,
                "histogram_gradient_l1": gradient_l1,
                "exact_wrong_max_logit_delta": float(
                    (exact_logits - wrong_logits).abs().max()
                ),
                "exact_invariant_max_logit_delta": float(
                    (exact_logits - invariant_logits).abs().max()
                ),
                "edge_effective_gate": float(
                    torch.tanh(exact.backbone.residual_gate.detach())
                ),
                "histogram_effective_gate": float(
                    torch.tanh(exact.backbone.histogram_gate.detach())
                ),
            }
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    ready = (
        all(protocol_checks.values())
        and bool(evidence_checks)
        and all(evidence_checks.values())
        and not errors
    )
    return {
        "run_id": RUN_ID,
        "status": "pass" if ready else "fail",
        "decision": (
            "innovation1_uknit_family_ctspn_k1t_execution_authorized"
            if ready
            else "innovation1_uknit_family_ctspn_k1t_not_ready"
        ),
        "execution_authorized": ready,
        "optimizer_step_authorized": ready,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "failed_evidence_checks": sorted(
            name for name, passed in evidence_checks.items() if not passed
        ),
        "evidence_metrics": evidence_metrics,
        "errors": errors,
        "training_rows": 0,
        "optimizer_steps": 0,
    }


def evaluate_k1t_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    k1r_anchor_rows: Sequence[Mapping[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    trained = training_map(training_rows)
    checkpoints = checkpoint_map(checkpoint_manifest)
    anchors = {
        (int(row["seed"]), str(row["split"])): row
        for row in k1r_anchor_rows
        if row.get("condition") == "exact_composition"
    }
    expected_anchors = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    if set(anchors) != expected_anchors:
        raise ValueError("K1-T requires six exact K1-R anchor rows")

    rows: list[dict[str, Any]] = []
    for seed, condition in sorted(expected_training_keys()):
        task = tasks_by_key[(seed, condition)]
        source = trained[(seed, condition)]
        checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
        state, checkpoint_sha = load_bound_state(
            checkpoint_path,
            checkpoints[(seed, condition)],
        )
        state_sha = tensor_mapping_sha256(state)
        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            model = build_k1t_control(
                task=task,
                condition=condition,
                input_bits=int(dataset.features.shape[1]),
            )
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha:
                raise ValueError("K1-T strict checkpoint load changed learned state")
            probabilities = predict_binary_probabilities(
                model,
                dataset,
                batch_size=EXPECTED_BATCH_SIZE,
                device=device,
            )
            labels = np.asarray(dataset.labels, dtype=np.float32)
            rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "uknit64",
                    "rounds": 5,
                    "seed": seed,
                    "condition": condition,
                    "model": CONTROL_MODELS[condition],
                    "split": split,
                    "rows": int(len(labels)),
                    "auc": binary_auc(labels, probabilities),
                    "dataset_sha256": differential_dataset_sha256(dataset),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "state_dict_sha256": state_sha,
                    "histogram_semantics_sha256": model.histogram_semantics_sha256,
                    "edge_effective_gate": float(
                        torch.tanh(model.backbone.residual_gate.detach())
                    ),
                    "histogram_effective_gate": float(
                        torch.tanh(model.backbone.histogram_gate.detach())
                    ),
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    for seed in EXPECTED_SEEDS:
        for split in EXPECTED_SPLITS:
            source = anchors[(seed, split)]
            rows.append(
                {
                    "run_id": RUN_ID,
                    "source_run_id": K1R_RUN_ID,
                    "cipher_key": "uknit64",
                    "rounds": 5,
                    "seed": seed,
                    "condition": ANCHOR_CONDITION,
                    "model": source["model"],
                    "split": split,
                    "rows": int(source["rows"]),
                    "auc": float(source["auc"]),
                    "source_auc": float(source["auc"]),
                    "dataset_sha256": source["dataset_sha256"],
                    "checkpoint_path": source["checkpoint_path"],
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "state_dict_sha256": source["state_dict_sha256"],
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def adjudicate_k1t(
    *,
    tasks: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    cache_checks: Mapping[str, bool],
) -> dict[str, Any]:
    trained = training_map(training_rows, fail_closed=False)
    checkpoints = checkpoint_map(checkpoint_manifest, fail_closed=False)
    evaluated = evaluation_map(evaluation_rows)
    expected_evaluations = {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in EVALUATION_CONDITIONS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(cache_checks),
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("optimizer_step_authorized") is True
            and all(readiness.get("protocol_checks", {}).values())
            and all(readiness.get("evidence_checks", {}).values())
        ),
        "six_frozen_tasks_exact": candidate_protocol_frozen(tasks),
        "six_training_rows_complete": (
            len(training_rows) == EXPECTED_TRAINING_ROWS
            and set(trained) == expected_training_keys()
        ),
        "training_protocol_frozen": training_protocol_frozen(training_rows),
        "six_checkpoint_entries_exact": (
            len(checkpoint_manifest.get("entries", [])) == EXPECTED_TRAINING_ROWS
            and set(checkpoints) == expected_training_keys()
        ),
        "twenty_four_evaluation_rows_complete": (
            len(evaluation_rows) == EXPECTED_EVALUATION_ROWS
            and set(evaluated) == expected_evaluations
        ),
        "evaluation_rows_zero_training": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("strict_state_dict_load") is True
            for row in evaluation_rows
        ),
        "split_rows_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in evaluation_rows
        ),
        "same_dataset_per_seed_split": all(
            len(
                {
                    evaluated[(seed, split, condition)].get("dataset_sha256")
                    for condition in EVALUATION_CONDITIONS
                }
            )
            == 1
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "anchor_metrics_replayed_exactly": all(
            abs(
                float(evaluated[(seed, split, ANCHOR_CONDITION)]["auc"])
                - float(
                    evaluated[(seed, split, ANCHOR_CONDITION)]["source_auc"]
                )
            )
            <= REPLAY_TOLERANCE
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "finite_metrics": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and 0.0 <= float(row.get("auc", math.nan)) <= 1.0
            for row in evaluation_rows
        ),
    }
    seed_results: dict[str, Any] = {}
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        seed_results[str(seed)] = {}
        for split in EXPECTED_SPLITS:
            aucs = {
                condition: float(evaluated[(seed, split, condition)]["auc"])
                for condition in EVALUATION_CONDITIONS
            }
            exact = aucs["exact_position_histogram_residual"]
            result = {
                **{f"{condition}_auc": auc for condition, auc in aucs.items()},
                "exact_minus_anchor": exact - aucs[ANCHOR_CONDITION],
                "exact_minus_wrong_sbox": (
                    exact - aucs["wrong_sbox_position_histogram_residual"]
                ),
                "exact_minus_invariant": (
                    exact - aucs["invariant_histogram_residual"]
                ),
            }
            seed_results[str(seed)][split] = result
            if split in FRESH_SPLITS:
                prefix = f"seed{seed}_{split}"
                research_checks[f"{prefix}_exact_auc_floor"] = exact >= AUC_FLOOR
                research_checks[f"{prefix}_beats_anchor"] = (
                    result["exact_minus_anchor"] >= ANCHOR_MARGIN
                )
                research_checks[f"{prefix}_beats_wrong_sbox"] = (
                    result["exact_minus_wrong_sbox"] >= WRONG_SBOX_MARGIN
                )
                research_checks[f"{prefix}_beats_invariant"] = (
                    result["exact_minus_invariant"] >= INVARIANT_MARGIN
                )
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_pass = bool(research_checks) and all(research_checks.values())
    exact_signal = all(
        research_checks[f"seed{seed}_{split}_exact_auc_floor"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    semantic_pass = all(
        research_checks[f"seed{seed}_{split}_beats_wrong_sbox"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    invariant_pass = all(
        research_checks[f"seed{seed}_{split}_beats_invariant"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    same_key_pass = all(
        research_checks[f"seed{seed}_same_key_fresh_{suffix}"]
        for seed in EXPECTED_SEEDS
        for suffix in (
            "exact_auc_floor",
            "beats_anchor",
            "beats_wrong_sbox",
            "beats_invariant",
        )
    )
    cross_key_pass = all(
        research_checks[f"seed{seed}_cross_key_validation_{suffix}"]
        for seed in EXPECTED_SEEDS
        for suffix in (
            "exact_auc_floor",
            "beats_anchor",
            "beats_wrong_sbox",
            "beats_invariant",
        )
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1t_protocol_invalid"
        next_action = (
            "repair only the failed K1-T source, readiness, cache, checkpoint, or "
            "artifact binding and rerun unchanged"
        )
    elif research_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1t_deterministic_position_"
            "residual_supported"
        )
        next_action = (
            "preregister a remote 65536/class medium diagnostic with exact, strongest "
            "semantic control and invariant control using disk-backed cache/progress/reuse"
        )
    elif exact_signal and not semantic_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1t_signal_without_wrong_sbox_"
            "attribution"
        )
        next_action = (
            "hold scale and isolate exact stage contributions without adding data or capacity"
        )
    elif exact_signal and not invariant_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1t_signal_without_position_necessity"
        )
        next_action = (
            "replace the candidate by the simpler invariant histogram branch before scale"
        )
    elif same_key_pass and not cross_key_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1t_key_specific_position_residual"
        )
        next_action = (
            "hold scale and test one key-invariance change at the identical local budget"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1t_trainable_position_residual_"
            "not_supported"
        )
        next_action = (
            "audit fixed Fisher initialization versus learned random initialization on "
            "the frozen exact histogram; do not add samples first"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "authorized_65536_per_class" if research_pass else "no",
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
            "exact_auc_floor": AUC_FLOOR,
            "exact_minus_anchor": ANCHOR_MARGIN,
            "exact_minus_wrong_sbox": WRONG_SBOX_MARGIN,
            "exact_minus_invariant": INVARIANT_MARGIN,
        },
        "descriptive_diagnostics": {
            "exact_signal_all_fresh": exact_signal,
            "wrong_sbox_attribution_all_fresh": semantic_pass,
            "position_necessity_all_fresh": invariant_pass,
            "same_key_full_gate": same_key_pass,
            "cross_key_full_gate": cross_key_pass,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local 2048/class uKNIT r5 cell11 deterministic-position-residual "
            "diagnostic; not formal scale, attack, SOTA, transfer, or ceiling evidence"
        ),
        "blocked_actions": [
            "more local samples, pairs, positions, epochs, seeds, or keys",
            "MoE, DDT/trails, cipher identity, another cipher, or another network family",
            "remote scale unless every frozen fresh gate passes",
        ],
    }


def training_protocol_frozen(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == EXPECTED_TRAINING_ROWS and all(
        row.get("model") in MODEL_TO_CONDITION
        and int(row.get("trainable_parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
        and int(row.get("rounds", -1)) == 5
        and int(row.get("seed", -1)) in EXPECTED_SEEDS
        and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
        and row.get("difference_profile") == DIFFERENCE_PROFILE
        and int(row.get("samples_per_class", -1)) == 2048
        and int(row.get("pairs_per_sample", -1)) == EXPECTED_PAIRS
        and row.get("negative_mode") == "encrypted_random_plaintexts"
        and int(row.get("training", {}).get("batch_size", -1)) == EXPECTED_BATCH_SIZE
        and int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
        and int(row.get("training", {}).get("epochs_ran", -1)) == EXPECTED_EPOCHS
        and row.get("training", {}).get("checkpoint_metric") == "val_auc"
        and row.get("training", {}).get("selected_checkpoint") == "best"
        for row in rows
    )


def training_map(
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
            raise ValueError(f"duplicate K1-T training row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-T training panel is incomplete")
    return mapped


def checkpoint_map(
    manifest: Mapping[str, Any],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in manifest.get("entries", []):
        condition = str(row.get("condition"))
        if condition not in CONTROL_MODELS:
            continue
        key = (int(row["seed"]), condition)
        if key in mapped:
            raise ValueError(f"duplicate K1-T checkpoint: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != expected_training_keys():
        raise ValueError("K1-T checkpoint panel is incomplete")
    return mapped


def evaluation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-T evaluation row: {key}")
        mapped[key] = row
    return mapped


def expected_training_keys() -> set[tuple[int, str]]:
    return {
        (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_MODELS
    }


__all__ = [
    "ANCHOR_CONDITION",
    "AUC_FLOOR",
    "CONTROL_MODELS",
    "EVALUATION_CONDITIONS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_K1S_DIGESTS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_SOURCE_DIGESTS",
    "EXPECTED_TRAINING_ROWS",
    "INVARIANT_MARGIN",
    "MODEL_TO_CONDITION",
    "RUN_ID",
    "WRONG_SBOX_MARGIN",
    "adjudicate_k1t",
    "build_k1t_control",
    "build_k1t_readiness",
    "candidate_protocol_frozen",
    "checkpoint_map",
    "evaluate_k1t_panel",
    "expected_training_keys",
    "source_binding_checks",
    "task_map",
]
