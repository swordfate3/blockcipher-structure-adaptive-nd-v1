from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.gf2_boolean_view import apply_gf2_operator
from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE,
    LINEAR_HISTOGRAM_LOCAL,
    edge_context_covariance_histogram,
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
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_source_bundle_collision_k1by10 as k1by10,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_edge_context_covariance_k1by11_present_r7_"
    "seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_edge_context_covariance_k1by11_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "bffe9635a0fbc6828c93f83a978018cd887c0b78fc3b99f0e57b4f0704a48805"
)
EXPECTED_SEEDS = (2, 3)
TAPS = k1by7.TAPS
CONDITIONS = (
    "anchor_local__correct_runtime",
    "anchor_local__affine_runtime",
    "candidate_edge_covariance__correct_runtime",
    "candidate_edge_covariance__affine_runtime",
    "candidate_edge_covariance__correct_state_shuffled_edges",
)
EXPECTED_RESULT_ROWS = len(EXPECTED_SEEDS) * len(CONDITIONS) * len(TAPS)
MARGIN_FLOOR = 0.005
RETENTION_FLOOR = -0.005
SHUFFLE_MULTIPLIER = 7
SHUFFLE_OFFSET = 3
ANCHOR_BUFFER_NAMES = set(k1by8.RUNTIME_BUFFER_NAMES)
CANDIDATE_BUFFER_NAMES = ANCHOR_BUFFER_NAMES | {
    "conditioner.linear_edge_source_cells",
    "conditioner.linear_edge_source_roles",
}


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    audit = config.get("audit", {})
    gates = config.get("gates", {})
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY11 config digest drifted")
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_edge_context_covariance_k1by11"
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
        or audit.get("candidate_scope") != "linear_histogram_only"
        or audit.get("sbox_histogram_mode") != "local_unchanged"
        or audit.get("edge_context_source") != "compiled_ordered_edge_triples"
        or audit.get("edge_context_aggregation")
        != "masked_mean_over_incoming_edges_per_target_cell"
        or audit.get("edge_context_residual")
        != "bin_indicator_edge_context_covariance"
        or audit.get("candidate_formula")
        != "local_histogram_plus_zero_sum_edge_context_covariance"
        or audit.get("shuffled_source_cell_formula")
        != "(7*source_cell+3)%16"
        or audit.get("shuffled_control_changes_inverse_state") is not False
        or audit.get("uses_cipher_identity") is not False
        or audit.get("uses_absolute_cell_or_bit_identity_as_feature") is not False
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or float(audit.get("anchor_replay_tolerance", math.nan)) != 1e-6
        or float(audit.get("mass_preservation_tolerance", math.nan)) != 1e-6
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
        raise ValueError("K1-BY11 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file() and _file_sha256(path) == expected[name]
        for name, path in paths.items()
    }
    try:
        k1by8_gate = _read_json(paths["k1by8_gate"])
        k1by8_validation = _read_json(paths["k1by8_validation"])
        k1by10_gate = _read_json(paths["k1by10_gate"])
        k1by10_validation = _read_json(paths["k1by10_validation"])
        source8 = k1by8.load_and_validate_config(paths["k1by8_config"])
        source10 = k1by10.load_and_validate_config(paths["k1by10_config"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        k1by8_gate = {}
        k1by8_validation = {}
        k1by10_gate = {}
        k1by10_validation = {}
        source8 = {}
        source10 = {}
    checks["k1by8_expected_histogram_access_loss"] = (
        k1by8_gate.get("status") == "pass"
        and k1by8_gate.get("decision")
        == "innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss"
        and k1by8_validation.get("status") == "pass"
        and k1by8_validation.get("optimizer_steps") == 0
    )
    checks["k1by10_expected_no_stable_locus"] = (
        k1by10_gate.get("status") == "pass"
        and k1by10_gate.get("research_gate_passed") is False
        and k1by10_gate.get("decision")
        == "innovation1_runtime_spn_k1by10_no_stable_partition_locus_identified"
        and k1by10_validation.get("status") == "pass"
        and k1by10_validation.get("optimizer_steps") == 0
    )
    checks["k1by8_source_config_exact"] = source8.get("run_id") == k1by8.RUN_ID
    checks["k1by10_source_config_exact"] = source10.get("run_id") == k1by10.RUN_ID
    return checks


def build_models(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, torch.nn.Module], Mapping[str, Any], dict[str, Any]]:
    k1by8_config = k1by8.load_and_validate_config(
        ROOT / config["sources"]["k1by8_config"]
    )
    k1by7_config = k1by7.load_and_validate_config(
        ROOT / k1by8_config["sources"]["k1by7_config"]
    )
    source_models, source_rows = k1by7.load_models_and_source_rows(
        k1by7_config,
        seed=seed,
    )
    source_model = source_models["correct"]
    task = k1by6.task_map(k1by6.read_tasks())[seed]
    models: dict[str, torch.nn.Module] = {}
    for condition in CONDITIONS:
        representation, runtime = split_condition_key(condition)
        mode = (
            LINEAR_HISTOGRAM_LOCAL
            if representation == "anchor_local"
            else LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE
        )
        model_key = (
            k1by6.AFFINE_MODEL if runtime == "affine_runtime" else k1by6.CORRECT_MODEL
        )
        candidate_task = dict(task)
        candidate_task["model_options"] = {
            **dict(task["model_options"]),
            "linear_histogram_mode": mode,
        }
        model = k1by6.build_model_for_task(candidate_task, model_key=model_key)
        k1by8.copy_named_parameters(model, source_model)
        if runtime == "correct_state_shuffled_edges":
            source_cells = model.conditioner.linear_edge_source_cells
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
        "program_semantic_sha256": {
            condition: model.compiled_program_semantic_sha256
            for condition, model in models.items()
        },
        "edge_source_cell_fingerprints": {
            condition: _tensor_fingerprint(
                model.conditioner.linear_edge_source_cells
            )
            for condition, model in models.items()
            if representation_from_condition(condition) == "candidate_edge_covariance"
        },
        "edge_source_role_fingerprints": {
            condition: _tensor_fingerprint(
                model.conditioner.linear_edge_source_roles
            )
            for condition, model in models.items()
            if representation_from_condition(condition) == "candidate_edge_covariance"
        },
    }
    return models, source_rows["correct"], metadata


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
                    np.load(k1by7.validation_dataset_paths()[seed][0], mmap_mode="r")[
                        :8
                    ],
                    copy=True,
                ),
                dtype=torch.float32,
            )
            captures = {
                condition: k1by7.capture_taps(model, fixture)
                for condition, model in models.items()
            }
            with torch.inference_mode():
                outputs = {
                    condition: model(fixture) for condition, model in models.items()
                }
            reference, _rows, _metadata = k1by8.build_swapped_models(
                k1by8.load_and_validate_config(
                    ROOT / config["sources"]["k1by8_config"]
                ),
                seed=seed,
            )
            candidate_conditions = tuple(
                condition
                for condition in CONDITIONS
                if representation_from_condition(condition)
                == "candidate_edge_covariance"
            )
            mass_errors: dict[str, float] = {}
            relabel_errors: dict[str, float] = {}
            changed_from_local: dict[str, bool] = {}
            for condition in candidate_conditions:
                model = models[condition]
                local_runtime = (
                    "affine_runtime"
                    if runtime_from_condition(condition) == "affine_runtime"
                    else "correct_runtime"
                )
                anchor = models[condition_key("anchor_local", local_runtime)]
                candidate_histograms = _stage_histograms(model, fixture)
                local_histograms = _stage_histograms(anchor, fixture)
                mass_errors[condition] = max(
                    float((values.sum(dim=-1) - 1.0).abs().max())
                    for values in candidate_histograms.values()
                )
                changed_from_local[condition] = any(
                    not torch.equal(candidate_histograms[stage], local_histograms[stage])
                    for stage in candidate_histograms
                )
                relabel_errors[condition] = _max_joint_relabel_error(model, fixture)
            source_parameter = metadata["source_parameter_fingerprint"]
            edge_fingerprints = metadata["edge_source_cell_fingerprints"]
            evidence_checks = {
                "five_frozen_conditions_exact": set(models) == set(CONDITIONS),
                "all_parameters_identical_to_correct_source": (
                    set(metadata["parameter_fingerprints"].values())
                    == {source_parameter}
                ),
                "anchor_buffers_unchanged": all(
                    set(metadata["runtime_buffer_names"][condition])
                    == ANCHOR_BUFFER_NAMES
                    for condition in CONDITIONS[:2]
                ),
                "candidate_adds_only_edge_context_buffers": all(
                    set(metadata["runtime_buffer_names"][condition])
                    == CANDIDATE_BUFFER_NAMES
                    for condition in candidate_conditions
                ),
                "histogram_modes_exact": all(
                    metadata["linear_histogram_modes"][condition]
                    == (
                        LINEAR_HISTOGRAM_LOCAL
                        if representation_from_condition(condition) == "anchor_local"
                        else LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE
                    )
                    for condition in CONDITIONS
                ),
                "anchor_fixture_outputs_replay_k1by8": all(
                    torch.equal(
                        outputs[condition_key("anchor_local", runtime)],
                        reference[k1by8.condition_key("correct_weights", runtime)](
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
                "candidate_changes_linear_histogram": all(
                    changed_from_local.values()
                ),
                "candidate_preserves_histogram_mass": all(
                    value <= float(config["audit"]["mass_preservation_tolerance"])
                    for value in mass_errors.values()
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
                            "candidate_edge_covariance",
                            "correct_state_shuffled_edges",
                        )
                    ]
                    == metadata["program_semantic_sha256"][
                        condition_key("candidate_edge_covariance", "correct_runtime")
                    ]
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "tap_shapes": {
                    condition: {tap: list(values[tap].shape) for tap in TAPS}
                    for condition, values in captures.items()
                },
                "mass_errors": mass_errors,
                "joint_relabel_errors": relabel_errors,
                "changed_from_local": changed_from_local,
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
    k1by8_gate = _read_json(source_artifact_paths(config)["k1by8_gate"])
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
                captured = k1by7.capture_taps(model, batch)
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
                k1by8_gate["final_evaluation"][str(seed)]["condition_auc"][
                    source_condition
                ]
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
        "fifty_internal_probe_rows_exact": (
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

    seed_results: dict[str, Any] = {}
    primary_passes: list[bool] = []
    correct_condition = condition_key("candidate_edge_covariance", "correct_runtime")
    affine_condition = condition_key("candidate_edge_covariance", "affine_runtime")
    shuffled_condition = condition_key(
        "candidate_edge_covariance", "correct_state_shuffled_edges"
    )
    anchor_condition = condition_key("anchor_local", "correct_runtime")
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
        decision = "innovation1_runtime_spn_k1by11_protocol_invalid"
        next_action = (
            "Repair only the failed source, edge buffer, covariance, relabeling, "
            "parameter-transfer, hook, probe or artifact invariant and rerun unchanged."
        )
    elif all(primary_passes):
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by11_edge_context_covariance_supported"
        )
        next_action = (
            "Preregister one same-budget local training confirmation comparing the "
            "edge-context covariance representation with local, affine and shuffled "
            "controls. Keep PRESENT r7 data, seeds, pairs and optimizer fixed."
        )
    else:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by11_input_modulation_not_supported"
        )
        next_action = (
            "Close deterministic input modulation and place one bounded structure "
            "intervention after the frozen linear primitive expert, where K1-BY8 "
            "already shows positive access on both seeds. Do not tune, train or scale."
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
            "Zero-training PRESENT r7 edge-context representation intervention on "
            "frozen K1-BY3 checkpoints and validation caches; internal probes are "
            "diagnostic and not formal-scale, transfer, attack or SOTA evidence."
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
        raise ValueError("unknown K1-BY11 condition")
    return value


def split_condition_key(value: str) -> tuple[str, str]:
    if value not in CONDITIONS:
        raise ValueError("invalid K1-BY11 condition key")
    representation, runtime = value.split("__", 1)
    return representation, runtime


def representation_from_condition(value: str) -> str:
    return split_condition_key(value)[0]


def runtime_from_condition(value: str) -> str:
    return split_condition_key(value)[1]


def model_metadata_frozen(metadata: Mapping[str, Any]) -> bool:
    parameters = metadata.get("parameter_fingerprints", {})
    buffers = metadata.get("runtime_buffer_names", {})
    modes = metadata.get("linear_histogram_modes", {})
    edge_cells = metadata.get("edge_source_cell_fingerprints", {})
    edge_roles = metadata.get("edge_source_role_fingerprints", {})
    candidates = {
        condition
        for condition in CONDITIONS
        if representation_from_condition(condition) == "candidate_edge_covariance"
    }
    return (
        set(parameters) == set(CONDITIONS)
        and set(parameters.values()) == {metadata.get("source_parameter_fingerprint")}
        and all(
            set(buffers[condition])
            == (
                ANCHOR_BUFFER_NAMES
                if representation_from_condition(condition) == "anchor_local"
                else CANDIDATE_BUFFER_NAMES
            )
            for condition in CONDITIONS
        )
        and all(
            modes[condition]
            == (
                LINEAR_HISTOGRAM_LOCAL
                if representation_from_condition(condition) == "anchor_local"
                else LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE
            )
            for condition in CONDITIONS
        )
        and set(edge_cells) == candidates
        and len(set(edge_cells.values())) == 3
        and set(edge_roles) == candidates
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
    root10 = ROOT / sources["k1by10_root"]
    return {
        "k1by8_config": ROOT / sources["k1by8_config"],
        "k1by8_preflight": root8 / "preflight.json",
        "k1by8_results": root8 / "results.jsonl",
        "k1by8_gate": root8 / "gate.json",
        "k1by8_validation": root8 / "validation.json",
        "k1by8_summary": root8 / "summary.json",
        "k1by8_model_metadata": root8 / "model_metadata.json",
        "k1by10_config": ROOT / sources["k1by10_config"],
        "k1by10_preflight": root10 / "preflight.json",
        "k1by10_results": root10 / "results.jsonl",
        "k1by10_gate": root10 / "gate.json",
        "k1by10_validation": root10 / "validation.json",
        "k1by10_summary": root10 / "summary.json",
        "k1by10_partition_summary": root10 / "partition_summary.json",
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    values8 = sources["k1by8_digests"]
    values10 = sources["k1by10_digests"]
    return {
        "k1by8_config": values8["config"],
        "k1by8_preflight": values8["preflight.json"],
        "k1by8_results": values8["results.jsonl"],
        "k1by8_gate": values8["gate.json"],
        "k1by8_validation": values8["validation.json"],
        "k1by8_summary": values8["summary.json"],
        "k1by8_model_metadata": values8["model_metadata.json"],
        "k1by10_config": values10["config"],
        "k1by10_preflight": values10["preflight.json"],
        "k1by10_results": values10["results.jsonl"],
        "k1by10_gate": values10["gate.json"],
        "k1by10_validation": values10["validation.json"],
        "k1by10_summary": values10["summary.json"],
        "k1by10_partition_summary": values10["partition_summary.json"],
    }


def _stage_histograms(
    model: torch.nn.Module,
    fixture: torch.Tensor,
) -> dict[int, torch.Tensor]:
    conditioner = model.conditioner
    current = fixture.reshape(fixture.shape[0], -1, 2, 64).flip(-1)
    rows: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for stage_index in reversed(range(conditioner.program.rounds)):
            triplet = torch.stack(
                (
                    current[:, :, 0],
                    current[:, :, 1],
                    torch.remainder(current[:, :, 0] + current[:, :, 1], 2.0),
                ),
                dim=-1,
            )
            linear_triplet = apply_gf2_operator(
                triplet,
                conditioner.inverse_linear_matrices[stage_index],
            )
            linear_state = linear_triplet[..., :2].permute(0, 1, 3, 2)
            rows[stage_index] = conditioner._difference_histogram(
                linear_state,
                stage_index=stage_index,
                source_values=current,
            )
            left = conditioner.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 0], stage_index
            )
            right = conditioner.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 1], stage_index
            )
            current = torch.stack((left, right), dim=2)
    return rows


def _max_joint_relabel_error(
    model: torch.nn.Module,
    fixture: torch.Tensor,
) -> float:
    conditioner = model.conditioner
    current = fixture.reshape(fixture.shape[0], -1, 2, 64).flip(-1)
    errors = []
    with torch.inference_mode():
        for stage_index in reversed(range(conditioner.program.rounds)):
            triplet = torch.stack(
                (
                    current[:, :, 0],
                    current[:, :, 1],
                    torch.remainder(current[:, :, 0] + current[:, :, 1], 2.0),
                ),
                dim=-1,
            )
            linear_triplet = apply_gf2_operator(
                triplet,
                conditioner.inverse_linear_matrices[stage_index],
            )
            linear_state = linear_triplet[..., :2].permute(0, 1, 3, 2)
            target_difference = torch.remainder(
                linear_state[:, :, 0] + linear_state[:, :, 1], 2.0
            )
            target_bits = target_difference[
                ..., conditioner.semantic_cell_bits.to(fixture.device)
            ]
            weights = torch.tensor((8, 4, 2, 1), device=fixture.device)
            target_values = torch.sum(target_bits * weights, dim=-1).to(torch.long)
            source_difference = torch.remainder(
                current[:, :, 0] + current[:, :, 1], 2.0
            )
            source_bits = source_difference[
                ..., conditioner.semantic_cell_bits.to(fixture.device)
            ]
            edge_cells = conditioner.linear_edge_source_cells[stage_index]
            edge_roles = conditioner.linear_edge_source_roles[stage_index]
            masks = conditioner.edge_masks[stage_index]
            original = edge_context_covariance_histogram(
                target_values,
                source_bits,
                edge_cells,
                edge_roles,
                masks,
            )
            order = torch.roll(torch.arange(16), shifts=3)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(16)
            relabeled = edge_context_covariance_histogram(
                target_values[..., order],
                source_bits[..., order, :],
                inverse[edge_cells[order]],
                edge_roles[order],
                masks[order],
            )
            errors.append(float((relabeled - original[..., order, :]).abs().max()))
            left = conditioner.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 0], stage_index
            )
            right = conditioner.runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 1], stage_index
            )
            current = torch.stack((left, right), dim=2)
    return max(errors)


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
    "comparison_rows",
    "condition_key",
    "evaluate",
    "load_and_validate_config",
    "model_metadata_frozen",
    "source_binding_checks",
    "split_condition_key",
]
