from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN,
    POST_EXPERT_RESIDUAL_NONE,
    PostExpertStructuralResidual,
    post_expert_edge_gated_laplacian,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_learned_access_audit_k1by7 as k1by7,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_same_checkpoint_runtime_swap_k1by8 as k1by8,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_post_expert_edge_residual_k1by12_present_r7_"
    "seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_post_expert_edge_residual_k1by12_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "e5bf57ace0bdd13891c0a3f0f80030a6a05e86acb6c64b75f4b9273897a42c77"
)
EXPECTED_SEEDS = (2, 3)
CONDITIONS = (
    "anchor_local__correct_runtime",
    "anchor_local__affine_runtime",
    "candidate_post_expert__correct_runtime",
    "candidate_post_expert__affine_runtime",
    "candidate_post_expert__correct_state_shuffled_edges",
)
TAPS = (
    "post_expert_structural_residual",
    "cell_fusion",
    "pooled_stage_summary",
    "pre_classifier_representation",
)
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(CONDITIONS) * len(TAPS)
MARGIN_FLOOR = 0.005
RETENTION_FLOOR = -0.005
SHUFFLE_MULTIPLIER = 7
SHUFFLE_OFFSET = 3


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY12 config digest drifted")
    config = _read_json(path)
    audit = config.get("audit", {})
    gates = config.get("gates", {})
    if (
        config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_post_expert_edge_residual_k1by12"
        or audit.get("cipher") != "PRESENT-80"
        or audit.get("rounds") != 7
        or tuple(audit.get("seeds", ())) != EXPECTED_SEEDS
        or audit.get("validation_rows_per_seed") != 2048
        or audit.get("pairs_per_sample") != 16
        or audit.get("input_bits") != 2048
        or audit.get("batch_size") != 128
        or audit.get("device") != "cpu"
        or audit.get("neural_training_performed") is not False
        or audit.get("optimizer_steps") != 0
        or audit.get("parameter_source") != "k1by3_correct_checkpoint_only"
        or tuple(audit.get("conditions", ())) != CONDITIONS
        or audit.get("parameter_transfer") != "named_parameters_only"
        or audit.get("runtime_buffers_preserved_from_target_model") is not True
        or audit.get("candidate_scope")
        != "after_linear_primitive_expert_before_cell_fusion"
        or audit.get("linear_histogram_mode") != "local_unchanged"
        or audit.get("sbox_histogram_mode") != "local_unchanged"
        or audit.get("edge_message")
        != "masked_mean_of_source_cell_linear_expert_outputs_per_target_cell"
        or audit.get("edge_gate")
        != "tanh_of_frozen_learned_edge_role_embedding"
        or audit.get("candidate_formula")
        != "x_plus_tanh(source_edge_mean_minus_x)_times_tanh(edge_role_embedding)"
        or float(audit.get("candidate_residual_bound_per_coordinate", math.nan))
        != 1.0
        or audit.get("candidate_has_tunable_coefficient") is not False
        or audit.get("shuffled_source_cell_formula")
        != "(7*source_cell+3)%16"
        or audit.get("shuffled_control_changes_inverse_state") is not False
        or audit.get("uses_cipher_identity") is not False
        or audit.get("uses_absolute_cell_or_bit_identity_as_feature") is not False
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("tap_order") != "forward_information_flow_from_intervention"
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or float(audit.get("anchor_replay_tolerance", math.nan)) != 1e-6
        or float(audit.get("joint_cell_relabel_tolerance", math.nan)) != 1e-6
        or float(
            gates.get("candidate_correct_minus_affine_probe_auc_min_each_tap")
        )
        != MARGIN_FLOOR
        or float(
            gates.get("candidate_correct_minus_shuffled_probe_auc_min_each_tap")
        )
        != MARGIN_FLOOR
        or float(gates.get("candidate_correct_minus_affine_final_auc_min"))
        != MARGIN_FLOOR
        or float(gates.get("candidate_correct_minus_shuffled_final_auc_min"))
        != MARGIN_FLOOR
        or float(gates.get("candidate_correct_final_auc_min_relative_to_anchor"))
        != RETENTION_FLOOR
        or gates.get("require_both_seeds") is not True
        or gates.get("remote_scale") != "no"
    ):
        raise ValueError("K1-BY12 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file() and _file_sha256(path) == expected[name]
        for name, path in paths.items()
    }
    try:
        gate8 = _read_json(paths["k1by8_gate"])
        validation8 = _read_json(paths["k1by8_validation"])
        gate11 = _read_json(paths["k1by11_gate"])
        validation11 = _read_json(paths["k1by11_validation"])
        config8 = k1by8.load_and_validate_config(paths["k1by8_config"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        gate8 = {}
        validation8 = {}
        gate11 = {}
        validation11 = {}
        config8 = {}
    checks["k1by8_expected_access_locus"] = (
        gate8.get("status") == "pass"
        and gate8.get("decision")
        == "innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss"
        and validation8.get("status") == "pass"
        and validation8.get("optimizer_steps") == 0
        and config8.get("run_id") == k1by8.RUN_ID
    )
    checks["k1by11_closed_input_modulation"] = (
        gate11.get("status") == "pass"
        and gate11.get("research_gate_passed") is False
        and gate11.get("decision")
        == "innovation1_runtime_spn_k1by11_input_modulation_not_supported"
        and validation11.get("status") == "pass"
        and validation11.get("optimizer_steps") == 0
    )
    return checks


def build_models(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, torch.nn.Module], Mapping[str, Any], dict[str, Any]]:
    config8 = k1by8.load_and_validate_config(ROOT / config["sources"]["k1by8_config"])
    config7 = k1by7.load_and_validate_config(ROOT / config8["sources"]["k1by7_config"])
    source_models, source_rows = k1by7.load_models_and_source_rows(config7, seed=seed)
    source_model = source_models["correct"]
    task = k1by6.task_map(k1by6.read_tasks())[seed]
    models: dict[str, torch.nn.Module] = {}
    for condition in CONDITIONS:
        representation, runtime = split_condition_key(condition)
        model_key = (
            k1by6.AFFINE_MODEL if runtime == "affine_runtime" else k1by6.CORRECT_MODEL
        )
        residual_mode = (
            POST_EXPERT_RESIDUAL_NONE
            if representation == "anchor_local"
            else POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN
        )
        candidate_task = dict(task)
        candidate_task["model_options"] = {
            **dict(task["model_options"]),
            "linear_histogram_mode": "local",
            "post_expert_residual_mode": residual_mode,
        }
        model = k1by6.build_model_for_task(candidate_task, model_key=model_key)
        k1by8.copy_named_parameters(model, source_model)
        if runtime == "correct_state_shuffled_edges":
            source_cells = model.conditioner.post_expert_edge_source_cells
            source_cells.copy_(
                torch.remainder(
                    SHUFFLE_MULTIPLIER * source_cells + SHUFFLE_OFFSET,
                    model.conditioner.program.cells,
                )
            )
        model.eval()
        models[condition] = model

    metadata = {
        "parameter_fingerprints": {
            condition: k1by8.learned_parameter_fingerprint(model)
            for condition, model in models.items()
        },
        "source_parameter_fingerprint": k1by8.learned_parameter_fingerprint(
            source_model
        ),
        "runtime_fingerprints": {
            condition: _tensor_mapping_fingerprint(dict(model.named_buffers()))
            for condition, model in models.items()
        },
        "runtime_buffer_names": {
            condition: sorted(name for name, _value in model.named_buffers())
            for condition, model in models.items()
        },
        "linear_histogram_modes": {
            condition: model.linear_histogram_mode
            for condition, model in models.items()
        },
        "post_expert_residual_modes": {
            condition: model.post_expert_residual_mode
            for condition, model in models.items()
        },
        "program_semantic_sha256": {
            condition: model.compiled_program_semantic_sha256
            for condition, model in models.items()
        },
        "edge_source_cell_fingerprints": {
            condition: _tensor_fingerprint(
                model.conditioner.post_expert_edge_source_cells
            )
            for condition, model in models.items()
            if representation_from_condition(condition) == "candidate_post_expert"
        },
    }
    return models, source_rows["correct"], metadata


def capture_taps(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    post_expert: list[torch.Tensor] = []
    fusion: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    classifier: list[torch.Tensor] = []
    handles = [
        model.conditioner.post_expert_structural_residual.register_forward_hook(
            lambda _module, _inputs, output: post_expert.append(output.detach())
        ),
        model.conditioner.cell_fusion.register_forward_hook(
            lambda _module, _inputs, output: fusion.append(output.detach())
        ),
        model.conditioner.stage_projection.register_forward_hook(
            lambda _module, inputs, _output: pooled.append(inputs[0].detach())
        ),
        model.backbone.classifier.register_forward_hook(
            lambda _module, inputs, _output: classifier.append(inputs[0].detach())
        ),
    ]
    try:
        with torch.inference_mode():
            model(features)
    finally:
        for handle in handles:
            handle.remove()
    if not (
        len(post_expert) == 2
        and len(fusion) == 2
        and len(pooled) == 2
        and len(classifier) == 1
    ):
        raise ValueError("K1-BY12 hook call geometry drifted")
    return {
        "post_expert_structural_residual": torch.stack(post_expert, dim=1),
        "cell_fusion": torch.stack(fusion, dim=1),
        "pooled_stage_summary": torch.stack(pooled, dim=1),
        "pre_classifier_representation": classifier[0],
    }


def build_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol_checks = {
        **source_binding_checks(config),
        "config_exact": load_and_validate_config() == config,
        "local_cpu_execution_frozen": config["audit"]["device"] == "cpu",
        "zero_optimizer_steps_frozen": config["audit"]["optimizer_steps"] == 0,
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            seed = EXPECTED_SEEDS[0]
            models, _source_row, metadata = build_models(config, seed=seed)
            fixture = torch.as_tensor(
                np.array(
                    np.load(k1by7.validation_dataset_paths()[seed][0], mmap_mode="r")[:8],
                    copy=True,
                ),
                dtype=torch.float32,
            )
            captures = {
                condition: capture_taps(model, fixture)
                for condition, model in models.items()
            }
            with torch.inference_mode():
                outputs = {
                    condition: model(fixture) for condition, model in models.items()
                }
            references, _rows, _metadata = k1by8.build_swapped_models(
                k1by8.load_and_validate_config(
                    ROOT / config["sources"]["k1by8_config"]
                ),
                seed=seed,
            )
            candidates = tuple(
                condition
                for condition in CONDITIONS
                if representation_from_condition(condition) == "candidate_post_expert"
            )
            residual_maxima = {
                condition: _max_actual_residual(models[condition], fixture)
                for condition in candidates
            }
            relabel_errors = {
                condition: _joint_relabel_error(models[condition])
                for condition in candidates
            }
            edge_fingerprints = metadata["edge_source_cell_fingerprints"]
            source_parameter = metadata["source_parameter_fingerprint"]
            no_op = PostExpertStructuralResidual(POST_EXPERT_RESIDUAL_NONE)
            no_op_fixture = torch.randn(3, 16, 32)
            no_op_gate = torch.randn(16, 32)
            no_op_mask = torch.ones(16, 1)
            evidence_checks = {
                "five_frozen_conditions_exact": set(models) == set(CONDITIONS),
                "all_parameters_identical_to_correct_source": (
                    set(metadata["parameter_fingerprints"].values())
                    == {source_parameter}
                ),
                "all_linear_histograms_remain_local": (
                    set(metadata["linear_histogram_modes"].values()) == {"local"}
                ),
                "residual_modes_exact": all(
                    metadata["post_expert_residual_modes"][condition]
                    == (
                        POST_EXPERT_RESIDUAL_NONE
                        if representation_from_condition(condition) == "anchor_local"
                        else POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN
                    )
                    for condition in CONDITIONS
                ),
                "candidate_adds_no_named_parameters": all(
                    len(tuple(models[condition].conditioner.post_expert_structural_residual.named_parameters()))
                    == 0
                    for condition in CONDITIONS
                ),
                "anchor_fixture_outputs_replay_k1by8": all(
                    torch.equal(
                        outputs[condition_key("anchor_local", runtime)],
                        references[k1by8.condition_key("correct_weights", runtime)](
                            fixture
                        ),
                    )
                    for runtime in ("correct_runtime", "affine_runtime")
                ),
                "all_outputs_and_taps_finite": all(
                    output.shape == (len(fixture), 1) and torch.isfinite(output).all()
                    for output in outputs.values()
                )
                and all(
                    tuple(values) == TAPS
                    and all(torch.isfinite(value).all() for value in values.values())
                    for values in captures.values()
                ),
                "candidate_residual_coordinate_bound": all(
                    value <= 1.0 + 1e-6 for value in residual_maxima.values()
                ),
                "disabled_residual_is_exact_identity": torch.equal(
                    no_op(
                        no_op_fixture,
                        no_op_gate,
                        None,
                        no_op_mask,
                    ),
                    no_op_fixture,
                ),
                "candidate_joint_cell_relabel_equivariant": all(
                    value
                    <= float(config["audit"]["joint_cell_relabel_tolerance"])
                    for value in relabel_errors.values()
                ),
                "correct_affine_shuffled_edge_bindings_distinct": (
                    len(set(edge_fingerprints.values())) == 3
                ),
                "shuffled_control_preserves_correct_program": (
                    metadata["program_semantic_sha256"][
                        condition_key(
                            "candidate_post_expert",
                            "correct_state_shuffled_edges",
                        )
                    ]
                    == metadata["program_semantic_sha256"][
                        condition_key("candidate_post_expert", "correct_runtime")
                    ]
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "tap_shapes": {
                    condition: {tap: list(values[tap].shape) for tap in TAPS}
                    for condition, values in captures.items()
                },
                "residual_maxima": residual_maxima,
                "joint_relabel_errors": relabel_errors,
                **metadata,
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
        "execution_authorized": status == "pass",
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "evidence_metrics": evidence_metrics,
        "errors": errors,
    }


def evaluate(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    batch_size = int(config["audit"]["batch_size"])
    epsilon = float(config["audit"]["probe_epsilon"])
    gate8 = _read_json(source_artifact_paths(config)["k1by8_gate"])
    result_rows: list[dict[str, Any]] = []
    final_evaluation: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        features_path, labels_path = k1by7.validation_dataset_paths()[seed]
        features = np.load(features_path, mmap_mode="r")
        labels = np.asarray(np.load(labels_path, mmap_mode="r"), dtype=np.uint8)
        models, _source_row, metadata = build_models(config, seed=seed)
        model_metadata[str(seed)] = metadata
        representations = {condition: {tap: [] for tap in TAPS} for condition in models}
        probabilities: dict[str, list[np.ndarray]] = {
            condition: [] for condition in models
        }
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch = torch.as_tensor(
                np.array(features[start:stop], copy=True),
                dtype=torch.float32,
            )
            for condition, model in models.items():
                captured = capture_taps(model, batch)
                for tap in TAPS:
                    representations[condition][tap].append(
                        captured[tap].numpy(force=True)
                    )
                with torch.inference_mode():
                    probabilities[condition].append(
                        torch.sigmoid(model(batch).flatten()).numpy(force=True)
                    )

        condition_auc = {
            condition: float(binary_auc(labels, np.concatenate(chunks)))
            for condition, chunks in probabilities.items()
        }
        anchor_replay = {}
        for runtime in ("correct_runtime", "affine_runtime"):
            condition = condition_key("anchor_local", runtime)
            source_condition = k1by8.condition_key("correct_weights", runtime)
            source_auc = float(
                gate8["final_evaluation"][str(seed)]["condition_auc"][source_condition]
            )
            replayed_auc = condition_auc[condition]
            anchor_replay[condition] = {
                "source_auc": source_auc,
                "replayed_auc": replayed_auc,
                "absolute_error": abs(replayed_auc - source_auc),
            }
        final_evaluation[str(seed)] = {
            "condition_auc": condition_auc,
            "anchor_replay": anchor_replay,
        }

        for condition, taps in representations.items():
            representation, runtime_program = split_condition_key(condition)
            for tap_index, tap in enumerate(TAPS):
                values = np.concatenate(taps[tap])
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "condition": condition,
                        "representation": representation,
                        "runtime_program": runtime_program,
                        "tap": tap,
                        "tap_index": tap_index,
                        "representation_shape": list(values.shape),
                        **k1by7.mean_difference_probe(values, labels, epsilon=epsilon),
                    }
                )
    return result_rows, final_evaluation, model_metadata


def adjudicate(
    config: Mapping[str, Any],
    *,
    result_rows: Sequence[Mapping[str, Any]],
    final_evaluation: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    readiness: Mapping[str, Any],
    sources_unchanged: bool,
) -> dict[str, Any]:
    mapped = {
        (int(row["seed"]), str(row["condition"]), str(row["tap"])): row
        for row in result_rows
    }
    expected_keys = {
        (seed, condition, tap)
        for seed in EXPECTED_SEEDS
        for condition in CONDITIONS
        for tap in TAPS
    }
    replay_tolerance = float(config["audit"]["anchor_replay_tolerance"])
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("execution_authorized") is True
            and readiness.get("training_authorized") is False
            and readiness.get("optimizer_steps_authorized") == 0
        ),
        "source_bindings_still_exact": all(source_binding_checks(config).values()),
        "source_artifacts_unchanged": sources_unchanged,
        "forty_internal_probe_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS and set(mapped) == expected_keys
        ),
        "probe_rows_finite_and_balanced": all(
            math.isfinite(float(row.get("probe_auc", math.nan)))
            and int(row.get("discovery_positive_rows", -1)) == 512
            and int(row.get("discovery_negative_rows", -1)) == 512
            and int(row.get("evaluation_positive_rows", -1)) == 512
            and int(row.get("evaluation_negative_rows", -1)) == 512
            for row in result_rows
        ),
        "all_anchor_auc_cells_replay": all(
            float(values["absolute_error"]) <= replay_tolerance
            for seed_values in final_evaluation.values()
            for values in seed_values["anchor_replay"].values()
        ),
        "all_five_cells_have_finite_final_auc": all(
            set(seed_values["condition_auc"]) == set(CONDITIONS)
            and all(
                math.isfinite(float(value))
                for value in seed_values["condition_auc"].values()
            )
            for seed_values in final_evaluation.values()
        ),
        "model_metadata_frozen_for_both_seeds": (
            set(model_metadata) == {"2", "3"}
            and all(model_metadata_frozen(values) for values in model_metadata.values())
        ),
    }

    correct_condition = condition_key("candidate_post_expert", "correct_runtime")
    affine_condition = condition_key("candidate_post_expert", "affine_runtime")
    shuffled_condition = condition_key(
        "candidate_post_expert", "correct_state_shuffled_edges"
    )
    anchor_condition = condition_key("anchor_local", "correct_runtime")
    seed_results: dict[str, Any] = {}
    primary_passes: list[bool] = []
    for seed in EXPECTED_SEEDS:
        taps: dict[str, Any] = {}
        first_loss: str | None = None
        for tap in TAPS:
            correct = float(mapped[(seed, correct_condition, tap)]["probe_auc"])
            affine = float(mapped[(seed, affine_condition, tap)]["probe_auc"])
            shuffled = float(mapped[(seed, shuffled_condition, tap)]["probe_auc"])
            affine_margin = correct - affine
            shuffled_margin = correct - shuffled
            passed = affine_margin >= MARGIN_FLOOR and shuffled_margin >= MARGIN_FLOOR
            if first_loss is None and not passed:
                first_loss = tap
            taps[tap] = {
                "correct_runtime_probe_auc": correct,
                "affine_runtime_probe_auc": affine,
                "shuffled_edges_probe_auc": shuffled,
                "correct_minus_affine_probe_auc": affine_margin,
                "correct_minus_shuffled_probe_auc": shuffled_margin,
                "margin_pass": passed,
            }
        aucs = final_evaluation[str(seed)]["condition_auc"]
        correct_final = float(aucs[correct_condition])
        affine_final = float(aucs[affine_condition])
        shuffled_final = float(aucs[shuffled_condition])
        anchor_final = float(aucs[anchor_condition])
        affine_final_margin = correct_final - affine_final
        shuffled_final_margin = correct_final - shuffled_final
        retention = correct_final - anchor_final
        final_margin_pass = (
            affine_final_margin >= MARGIN_FLOOR
            and shuffled_final_margin >= MARGIN_FLOOR
        )
        if first_loss is None and not final_margin_pass:
            first_loss = "final_output"
        retention_pass = retention >= RETENTION_FLOOR
        if first_loss is None and not retention_pass:
            first_loss = "final_retention"
        all_taps_pass = all(bool(values["margin_pass"]) for values in taps.values())
        primary_pass = all_taps_pass and final_margin_pass and retention_pass
        primary_passes.append(primary_pass)
        seed_results[str(seed)] = {
            "taps": taps,
            "anchor_correct_final_auc": anchor_final,
            "candidate_correct_final_auc": correct_final,
            "candidate_affine_final_auc": affine_final,
            "candidate_shuffled_final_auc": shuffled_final,
            "correct_minus_affine_final_auc": affine_final_margin,
            "correct_minus_shuffled_final_auc": shuffled_final_margin,
            "candidate_correct_minus_anchor_correct_final_auc": retention,
            "all_tap_margins_pass": all_taps_pass,
            "final_margins_pass": final_margin_pass,
            "retention_pass": retention_pass,
            "first_margin_loss": first_loss,
            "primary_gate_pass": primary_pass,
        }

    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by12_protocol_invalid"
        next_action = (
            "Repair only the failed frozen source, parameter transfer, residual, "
            "edge binding, relabeling, hook, probe or artifact invariant."
        )
    elif all(primary_passes):
        status = "pass"
        decision = "innovation1_runtime_spn_k1by12_post_expert_locus_supported"
        next_action = (
            "Preregister one local-GPU same-budget trainable post-expert adapter "
            "comparison with zero initial gate, K1-BY8 anchor, affine and shuffled "
            "controls. Do not increase scale."
        )
    else:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by12_deterministic_interventions_exhausted"
        )
        next_action = (
            "Stop frozen-checkpoint deterministic interventions. Preregister a "
            "trainable post-expert adapter with exact zero initialization and the "
            "unchanged K1-BY8 model as same-budget anchor."
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "method_status": "hold",
        "decision": decision,
        "research_gate_passed": not failed_protocol and all(primary_passes),
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "thresholds": {
            "candidate_correct_minus_affine_probe_auc_min_each_tap": MARGIN_FLOOR,
            "candidate_correct_minus_shuffled_probe_auc_min_each_tap": MARGIN_FLOOR,
            "candidate_correct_minus_affine_final_auc_min": MARGIN_FLOOR,
            "candidate_correct_minus_shuffled_final_auc_min": MARGIN_FLOOR,
            "candidate_correct_final_auc_min_relative_to_anchor": RETENTION_FLOOR,
        },
        "final_evaluation": final_evaluation,
        "seed_results": seed_results,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training PRESENT r7 post-expert structural intervention on frozen "
            "K1-BY3 checkpoints and validation caches; probes are local mechanism "
            "diagnosis, not formal-scale, transfer or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        for tap_index, tap in enumerate(TAPS):
            rows.append(
                {
                    "seed": int(seed),
                    "tap_index": tap_index,
                    "tap": tap,
                    **values["taps"][tap],
                    "first_margin_loss": values["first_margin_loss"],
                }
            )
        rows.append(
            {
                "seed": int(seed),
                "tap_index": len(TAPS),
                "tap": "final_output",
                "correct_runtime_probe_auc": values["candidate_correct_final_auc"],
                "affine_runtime_probe_auc": values["candidate_affine_final_auc"],
                "shuffled_edges_probe_auc": values["candidate_shuffled_final_auc"],
                "correct_minus_affine_probe_auc": values[
                    "correct_minus_affine_final_auc"
                ],
                "correct_minus_shuffled_probe_auc": values[
                    "correct_minus_shuffled_final_auc"
                ],
                "margin_pass": values["final_margins_pass"],
                "first_margin_loss": values["first_margin_loss"],
            }
        )
    return rows


def condition_key(representation: str, runtime_program: str) -> str:
    value = f"{representation}__{runtime_program}"
    if value not in CONDITIONS:
        raise ValueError("unknown K1-BY12 condition")
    return value


def split_condition_key(value: str) -> tuple[str, str]:
    if value not in CONDITIONS:
        raise ValueError("invalid K1-BY12 condition key")
    return tuple(value.split("__", 1))  # type: ignore[return-value]


def representation_from_condition(value: str) -> str:
    return split_condition_key(value)[0]


def model_metadata_frozen(metadata: Mapping[str, Any]) -> bool:
    parameters = metadata.get("parameter_fingerprints", {})
    modes = metadata.get("post_expert_residual_modes", {})
    histograms = metadata.get("linear_histogram_modes", {})
    edge_cells = metadata.get("edge_source_cell_fingerprints", {})
    candidates = {
        condition
        for condition in CONDITIONS
        if representation_from_condition(condition) == "candidate_post_expert"
    }
    return (
        set(parameters) == set(CONDITIONS)
        and set(parameters.values()) == {metadata.get("source_parameter_fingerprint")}
        and set(histograms.values()) == {"local"}
        and all(
            modes[condition]
            == (
                POST_EXPERT_RESIDUAL_NONE
                if representation_from_condition(condition) == "anchor_local"
                else POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN
            )
            for condition in CONDITIONS
        )
        and set(edge_cells) == candidates
        and len(set(edge_cells.values())) == 3
    )


def authority_digests(config: Mapping[str, Any]) -> dict[str, str]:
    paths = source_artifact_paths(config)
    for seed, (features, labels) in k1by7.validation_dataset_paths().items():
        paths[f"validation_seed{seed}_features"] = features
        paths[f"validation_seed{seed}_labels"] = labels
    return {name: _file_sha256(path) for name, path in paths.items()}


def source_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    root8 = ROOT / sources["k1by8_root"]
    root11 = ROOT / sources["k1by11_root"]
    return {
        "k1by8_config": ROOT / sources["k1by8_config"],
        "k1by8_preflight": root8 / "preflight.json",
        "k1by8_results": root8 / "results.jsonl",
        "k1by8_gate": root8 / "gate.json",
        "k1by8_validation": root8 / "validation.json",
        "k1by8_summary": root8 / "summary.json",
        "k1by8_model_metadata": root8 / "model_metadata.json",
        "k1by11_config": ROOT / sources["k1by11_config"],
        "k1by11_preflight": root11 / "preflight.json",
        "k1by11_results": root11 / "results.jsonl",
        "k1by11_gate": root11 / "gate.json",
        "k1by11_validation": root11 / "validation.json",
        "k1by11_summary": root11 / "summary.json",
        "k1by11_model_metadata": root11 / "model_metadata.json",
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    values8 = sources["k1by8_digests"]
    values11 = sources["k1by11_digests"]
    result = {}
    for prefix, values in (("k1by8", values8), ("k1by11", values11)):
        result[f"{prefix}_config"] = values["config"]
        for filename in (
            "preflight.json",
            "results.jsonl",
            "gate.json",
            "validation.json",
            "summary.json",
            "model_metadata.json",
        ):
            result[f"{prefix}_{filename.rsplit('.', 1)[0]}"] = values[filename]
    return result


def _max_actual_residual(model: torch.nn.Module, fixture: torch.Tensor) -> float:
    maxima: list[float] = []

    def record(_module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        maxima.append(float((output - inputs[0]).abs().max()))

    handle = model.conditioner.post_expert_structural_residual.register_forward_hook(
        record
    )
    try:
        with torch.inference_mode():
            model(fixture)
    finally:
        handle.remove()
    if len(maxima) != 2:
        raise ValueError("K1-BY12 residual-bound hook geometry drifted")
    return max(maxima)


def _joint_relabel_error(model: torch.nn.Module) -> float:
    generator = torch.Generator().manual_seed(20260801)
    expert = torch.randn(5, 16, model.conditioner.spec.hidden_dim, generator=generator)
    gate = torch.randn(16, model.conditioner.spec.hidden_dim, generator=generator)
    cells = model.conditioner.post_expert_edge_source_cells[0]
    masks = model.conditioner.edge_masks[0]
    original = post_expert_edge_gated_laplacian(expert, gate, cells, masks)
    order = torch.roll(torch.arange(16), shifts=3)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(16)
    relabeled = post_expert_edge_gated_laplacian(
        expert[:, order],
        gate[order],
        inverse[cells[order]],
        masks[order],
    )
    return float((relabeled - original[:, order]).abs().max())


def _tensor_mapping_fingerprint(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_fingerprint(value).encode("ascii"))
    return digest.hexdigest()


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CONFIG_PATH",
    "CONDITIONS",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SEEDS",
    "RUN_ID",
    "TAPS",
    "adjudicate",
    "authority_digests",
    "build_models",
    "build_readiness",
    "capture_taps",
    "comparison_rows",
    "condition_key",
    "evaluate",
    "load_and_validate_config",
    "model_metadata_frozen",
    "source_binding_checks",
    "split_condition_key",
]
