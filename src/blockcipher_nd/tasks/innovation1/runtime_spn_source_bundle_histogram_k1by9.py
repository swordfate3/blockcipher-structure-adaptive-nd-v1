from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    LINEAR_HISTOGRAM_LOCAL,
    LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN,
    source_bundle_equivalence_matrices,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    CompiledSpnProgram,
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
RUN_ID = "i1_runtime_spn_source_bundle_histogram_k1by9_present_r7_seed2_seed3_20260801"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_source_bundle_histogram_k1by9_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "046d843cf219c20978c53d6e3cd346df3239342fe3d238bb0f7618e29252d4a0"
)
REPRESENTATIONS = ("anchor_local", "candidate_source_bundle_mean")
RUNTIME_PROGRAMS = ("correct_runtime", "affine_runtime")
EXPECTED_SEEDS = (2, 3)
TAPS = k1by7.TAPS
MARGIN_FLOOR = 0.005
RETENTION_FLOOR = -0.005
EXPECTED_RESULT_ROWS = (
    len(EXPECTED_SEEDS) * len(REPRESENTATIONS) * len(RUNTIME_PROGRAMS) * len(TAPS)
)
ANCHOR_BUFFER_NAMES = set(k1by8.RUNTIME_BUFFER_NAMES)
CANDIDATE_BUFFER_NAMES = ANCHOR_BUFFER_NAMES | {
    "conditioner.linear_source_bundle_equivalence"
}


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    audit = config.get("audit", {})
    gates = config.get("gates", {})
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY9 config digest drifted")
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_source_bundle_histogram_k1by9"
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
        or tuple(audit.get("representations", ())) != REPRESENTATIONS
        or tuple(audit.get("runtime_programs", ())) != RUNTIME_PROGRAMS
        or audit.get("parameter_transfer") != "named_parameters_only"
        or audit.get("runtime_buffers_preserved_from_target_model") is not True
        or audit.get("candidate_scope") != "linear_histogram_only"
        or audit.get("sbox_histogram_mode") != "local_unchanged"
        or audit.get("source_bundle_signature") != "unordered_unique_source_cell_set"
        or audit.get("source_bundle_context") != "equivalence_class_mean"
        or float(audit.get("local_histogram_weight", math.nan)) != 0.5
        or float(audit.get("source_bundle_mean_weight", math.nan)) != 0.5
        or audit.get("uses_cipher_identity") is not False
        or audit.get("uses_absolute_cell_or_bit_identity") is not False
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or float(audit.get("anchor_replay_tolerance", math.nan)) != 1e-6
        or float(
            gates.get("candidate_correct_minus_affine_probe_auc_min_each_tap", math.nan)
        )
        != MARGIN_FLOOR
        or float(gates.get("candidate_correct_minus_affine_final_auc_min", math.nan))
        != MARGIN_FLOOR
        or float(
            gates.get("candidate_correct_final_auc_min_relative_to_anchor", math.nan)
        )
        != RETENTION_FLOOR
        or gates.get("require_both_seeds") is not True
        or gates.get("remote_scale") != "no"
    ):
        raise ValueError("K1-BY9 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file() and _file_sha256(path) == expected[name]
        for name, path in paths.items()
    }
    try:
        gate = _read_json(paths["k1by8_gate"])
        validation = _read_json(paths["k1by8_validation"])
        metadata = _read_json(paths["k1by8_model_metadata"])
        source_config = k1by8.load_and_validate_config(paths["k1by8_config"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        gate = {}
        validation = {}
        metadata = {}
        source_config = {}
    checks["k1by8_exact_completed_histogram_loss"] = (
        gate.get("status") == "pass"
        and gate.get("method_status") == "hold"
        and gate.get("research_gate_passed") is False
        and gate.get("decision")
        == "innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss"
        and validation.get("status") == "pass"
        and validation.get("result_rows") == 40
        and validation.get("optimizer_steps") == 0
    )
    checks["k1by8_source_config_exact"] = source_config.get(
        "run_id"
    ) == k1by8.RUN_ID and all(k1by8.source_binding_checks(source_config).values())
    checks["k1by8_metadata_has_both_seeds"] = set(metadata) == {"2", "3"}
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
    modes = {
        "anchor_local": LINEAR_HISTOGRAM_LOCAL,
        "candidate_source_bundle_mean": LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN,
    }
    model_keys = {
        "correct_runtime": k1by6.CORRECT_MODEL,
        "affine_runtime": k1by6.AFFINE_MODEL,
    }
    models: dict[str, torch.nn.Module] = {}
    for representation, mode in modes.items():
        candidate_task = dict(task)
        candidate_task["model_options"] = {
            **dict(task["model_options"]),
            "linear_histogram_mode": mode,
        }
        for runtime_program, model_key in model_keys.items():
            model = k1by6.build_model_for_task(candidate_task, model_key=model_key)
            k1by8.copy_named_parameters(model, source_model)
            model.eval()
            models[condition_key(representation, runtime_program)] = model

    metadata = {
        "parameter_fingerprints": {
            condition: k1by8.learned_parameter_fingerprint(model)
            for condition, model in models.items()
        },
        "source_parameter_fingerprint": k1by8.learned_parameter_fingerprint(
            source_model
        ),
        "runtime_fingerprints": {
            condition: (
                k1by8.runtime_buffer_fingerprint(model)
                if representation_from_condition(condition) == "anchor_local"
                else _tensor_mapping_fingerprint(dict(model.named_buffers()))
            )
            for condition, model in models.items()
        },
        "runtime_buffer_names": {
            condition: sorted(name for name, _value in model.named_buffers())
            for condition, model in models.items()
        },
        "runtime_program_semantic_sha256": {
            condition: model.compiled_program_semantic_sha256
            for condition, model in models.items()
        },
        "linear_histogram_modes": {
            condition: model.linear_histogram_mode
            for condition, model in models.items()
        },
        "source_bundle_matrix_fingerprints": {
            condition: _tensor_fingerprint(
                model.conditioner.linear_source_bundle_equivalence
            )
            for condition, model in models.items()
            if representation_from_condition(condition)
            == "candidate_source_bundle_mean"
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
            reference_models, _rows, reference_metadata = k1by8.build_swapped_models(
                k1by8.load_and_validate_config(
                    ROOT / config["sources"]["k1by8_config"]
                ),
                seed=seed,
            )
            runtime_fixture = fixture.reshape(fixture.shape[0], -1, 2, 64).flip(-1)
            sbox_histograms = {
                condition: model.conditioner._difference_histogram(runtime_fixture)
                for condition, model in models.items()
            }
            candidate_models = {
                runtime: models[condition_key("candidate_source_bundle_mean", runtime)]
                for runtime in RUNTIME_PROGRAMS
            }
            matrix_checks = {
                runtime: equivalence_matrix_checks(
                    model.conditioner.linear_source_bundle_equivalence
                )
                for runtime, model in candidate_models.items()
            }
            renamed_matrix_equal = {
                runtime: torch.equal(
                    model.conditioner.linear_source_bundle_equivalence,
                    source_bundle_equivalence_matrices(
                        rename_program_source_cells(model.conditioner.program)
                    ),
                )
                for runtime, model in candidate_models.items()
            }
            source_parameter = metadata["source_parameter_fingerprint"]
            evidence_checks = {
                "four_representation_runtime_cells_exact": (
                    set(models) == expected_conditions()
                ),
                "all_parameters_identical_to_correct_source": (
                    set(metadata["parameter_fingerprints"].values())
                    == {source_parameter}
                ),
                "anchor_buffer_contract_unchanged": all(
                    set(metadata["runtime_buffer_names"][condition])
                    == ANCHOR_BUFFER_NAMES
                    for condition in models
                    if representation_from_condition(condition) == "anchor_local"
                ),
                "candidate_adds_only_bundle_matrix_buffer": all(
                    set(metadata["runtime_buffer_names"][condition])
                    == CANDIDATE_BUFFER_NAMES
                    for condition in models
                    if representation_from_condition(condition)
                    == "candidate_source_bundle_mean"
                ),
                "histogram_modes_exact": (
                    set(
                        mode
                        for condition, mode in metadata[
                            "linear_histogram_modes"
                        ].items()
                        if representation_from_condition(condition) == "anchor_local"
                    )
                    == {LINEAR_HISTOGRAM_LOCAL}
                    and set(
                        mode
                        for condition, mode in metadata[
                            "linear_histogram_modes"
                        ].items()
                        if representation_from_condition(condition)
                        == "candidate_source_bundle_mean"
                    )
                    == {LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN}
                ),
                "anchor_fixture_outputs_replay_k1by8": all(
                    torch.equal(
                        outputs[condition_key("anchor_local", runtime)],
                        reference_models[
                            k1by8.condition_key("correct_weights", runtime)
                        ](fixture),
                    )
                    for runtime in RUNTIME_PROGRAMS
                ),
                "anchor_runtime_fingerprints_replay_k1by8": all(
                    metadata["runtime_fingerprints"][
                        condition_key("anchor_local", runtime)
                    ]
                    == reference_metadata["runtime_fingerprints"][
                        k1by8.condition_key("correct_weights", runtime)
                    ]
                    for runtime in RUNTIME_PROGRAMS
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
                "candidate_changes_linear_histogram_both_runtimes": all(
                    not torch.equal(
                        captures[condition_key("anchor_local", runtime)][
                            "linear_histogram"
                        ],
                        captures[
                            condition_key("candidate_source_bundle_mean", runtime)
                        ]["linear_histogram"],
                    )
                    for runtime in RUNTIME_PROGRAMS
                ),
                "sbox_histogram_path_unchanged": all(
                    torch.equal(
                        sbox_histograms[condition_key("anchor_local", runtime)],
                        sbox_histograms[
                            condition_key("candidate_source_bundle_mean", runtime)
                        ],
                    )
                    for runtime in RUNTIME_PROGRAMS
                ),
                "candidate_matrices_are_valid_equivalence_means": all(
                    all(checks.values()) for checks in matrix_checks.values()
                ),
                "source_cell_renaming_leaves_matrices_exact": all(
                    renamed_matrix_equal.values()
                ),
                "correct_and_affine_bundle_matrices_distinct": (
                    not torch.equal(
                        candidate_models[
                            "correct_runtime"
                        ].conditioner.linear_source_bundle_equivalence,
                        candidate_models[
                            "affine_runtime"
                        ].conditioner.linear_source_bundle_equivalence,
                    )
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
                "tap_shapes": {
                    condition: {tap: list(values[tap].shape) for tap in TAPS}
                    for condition, values in captures.items()
                },
                "matrix_checks": matrix_checks,
                "source_cell_renaming_matrix_equal": renamed_matrix_equal,
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
                    values = torch.sigmoid(model(batch).flatten())
                    probabilities[condition].append(values.numpy(force=True))

        condition_auc = {
            condition: float(binary_auc(labels, np.concatenate(chunks)))
            for condition, chunks in probabilities.items()
        }
        anchor_replay = {}
        for runtime in RUNTIME_PROGRAMS:
            source_condition = k1by8.condition_key("correct_weights", runtime)
            condition = condition_key("anchor_local", runtime)
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
                        **k1by7.mean_difference_probe(
                            values,
                            labels,
                            epsilon=epsilon,
                        ),
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
        (
            int(row["seed"]),
            str(row["representation"]),
            str(row["runtime_program"]),
            str(row["tap"]),
        ): row
        for row in result_rows
    }
    expected_keys = {
        (seed, representation, runtime_program, tap)
        for seed in EXPECTED_SEEDS
        for representation in REPRESENTATIONS
        for runtime_program in RUNTIME_PROGRAMS
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
        "all_four_cells_have_finite_final_auc": all(
            set(seed_values["condition_auc"]) == expected_conditions()
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
    for seed in EXPECTED_SEEDS:
        representations: dict[str, Any] = {}
        for representation in REPRESENTATIONS:
            taps: dict[str, Any] = {}
            first_loss: str | None = None
            for tap in TAPS:
                correct = float(
                    mapped[(seed, representation, "correct_runtime", tap)]["probe_auc"]
                )
                affine = float(
                    mapped[(seed, representation, "affine_runtime", tap)]["probe_auc"]
                )
                margin = correct - affine
                passed = margin >= MARGIN_FLOOR
                if first_loss is None and not passed:
                    first_loss = tap
                taps[tap] = {
                    "correct_runtime_probe_auc": correct,
                    "affine_runtime_probe_auc": affine,
                    "correct_minus_affine_runtime_probe_auc": margin,
                    "margin_pass": passed,
                }
            aucs = final_evaluation[str(seed)]["condition_auc"]
            correct_final = float(
                aucs[condition_key(representation, "correct_runtime")]
            )
            affine_final = float(aucs[condition_key(representation, "affine_runtime")])
            final_margin = correct_final - affine_final
            if first_loss is None and final_margin < MARGIN_FLOOR:
                first_loss = "final_output"
            representations[representation] = {
                "taps": taps,
                "correct_runtime_final_auc": correct_final,
                "affine_runtime_final_auc": affine_final,
                "correct_minus_affine_runtime_final_auc": final_margin,
                "final_margin_pass": final_margin >= MARGIN_FLOOR,
                "first_margin_loss": first_loss,
            }
        anchor = representations["anchor_local"]
        candidate = representations["candidate_source_bundle_mean"]
        retention = (
            candidate["correct_runtime_final_auc"] - anchor["correct_runtime_final_auc"]
        )
        all_taps_pass = all(
            bool(values["margin_pass"]) for values in candidate["taps"].values()
        )
        retention_pass = retention >= RETENTION_FLOOR
        primary_pass = (
            all_taps_pass and candidate["final_margin_pass"] and retention_pass
        )
        primary_passes.append(primary_pass)
        seed_results[str(seed)] = {
            "representations": representations,
            "candidate_correct_final_minus_anchor_correct_final_auc": retention,
            "candidate_all_tap_margins_pass": all_taps_pass,
            "candidate_final_margin_pass": candidate["final_margin_pass"],
            "candidate_retention_pass": retention_pass,
            "primary_gate_pass": primary_pass,
        }

    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by9_protocol_invalid"
        next_action = (
            "Repair only the failed source, equivalence matrix, representation mode, "
            "parameter-transfer, runtime-buffer, hook, probe or artifact invariant "
            "and rerun unchanged."
        )
    elif all(primary_passes):
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by9_source_bundle_histogram_repair_supported"
        )
        next_action = (
            "Preregister one same-budget local training confirmation comparing the "
            "source-bundle candidate with the local-histogram anchor using the same "
            "PRESENT r7 data, seeds, controls and optimizer. Do not scale or add a "
            "cipher before that attribution gate."
        )
    else:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by9_source_bundle_histogram_repair_not_supported"
        )
        next_action = (
            "Discard the fixed 1:1 source-bundle mean. Audit correct/affine equality-"
            "partition collisions before proposing one different representation; do "
            "not tune the blend, train, scale or add ciphers."
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
            "candidate_correct_minus_affine_final_auc_min": MARGIN_FLOOR,
            "candidate_correct_final_auc_min_relative_to_anchor": RETENTION_FLOOR,
        },
        "final_evaluation": final_evaluation,
        "seed_results": seed_results,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training PRESENT r7 representation intervention on frozen K1-BY3 "
            "correct checkpoints and validation caches; internal probes diagnose "
            "runtime access and are not formal-scale, transfer, attack or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, seed_values in sorted(gate.get("seed_results", {}).items()):
        for representation in REPRESENTATIONS:
            values = seed_values["representations"][representation]
            for tap_index, tap in enumerate(TAPS):
                rows.append(
                    {
                        "seed": int(seed),
                        "representation": representation,
                        "tap_index": tap_index,
                        "tap": tap,
                        **values["taps"][tap],
                        "first_margin_loss": values["first_margin_loss"],
                    }
                )
            rows.append(
                {
                    "seed": int(seed),
                    "representation": representation,
                    "tap_index": len(TAPS),
                    "tap": "final_output",
                    "correct_runtime_probe_auc": values["correct_runtime_final_auc"],
                    "affine_runtime_probe_auc": values["affine_runtime_final_auc"],
                    "correct_minus_affine_runtime_probe_auc": values[
                        "correct_minus_affine_runtime_final_auc"
                    ],
                    "margin_pass": values["final_margin_pass"],
                    "first_margin_loss": values["first_margin_loss"],
                }
            )
    return rows


def condition_key(representation: str, runtime_program: str) -> str:
    if representation not in REPRESENTATIONS or runtime_program not in RUNTIME_PROGRAMS:
        raise ValueError("unknown K1-BY9 intervention cell")
    return f"{representation}__{runtime_program}"


def split_condition_key(value: str) -> tuple[str, str]:
    parts = value.split("__")
    if len(parts) != 2:
        raise ValueError("invalid K1-BY9 condition key")
    representation, runtime_program = parts
    condition_key(representation, runtime_program)
    return representation, runtime_program


def representation_from_condition(value: str) -> str:
    return split_condition_key(value)[0]


def expected_conditions() -> set[str]:
    return {
        condition_key(representation, runtime_program)
        for representation in REPRESENTATIONS
        for runtime_program in RUNTIME_PROGRAMS
    }


def equivalence_matrix_checks(matrix: torch.Tensor) -> dict[str, bool]:
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return {
        "shape_exact": tuple(matrix.shape) == (2, 16, 16),
        "finite": bool(torch.isfinite(matrix).all()),
        "row_stochastic": bool(
            torch.allclose(matrix.sum(dim=-1), torch.ones_like(matrix.sum(dim=-1)))
        ),
        "symmetric": bool(torch.allclose(matrix, matrix.transpose(-1, -2))),
        "idempotent": bool(torch.allclose(matrix @ matrix, matrix)),
        "non_identity_context": bool(
            not torch.equal(matrix, identity.expand_as(matrix))
        ),
    }


def rename_program_source_cells(program: CompiledSpnProgram) -> CompiledSpnProgram:
    permutation = tuple((5 * cell + 1) % program.cells for cell in range(program.cells))
    stages = tuple(
        replace(
            stage,
            linear_cells=tuple(
                replace(
                    cell,
                    edges=tuple(
                        sorted(
                            (
                                target_role,
                                permutation[source_cell],
                                source_role,
                            )
                            for target_role, source_cell, source_role in cell.edges
                        )
                    ),
                )
                for cell in stage.linear_cells
            ),
        )
        for stage in program.stages
    )
    return replace(program, stages=stages)


def model_metadata_frozen(metadata: Mapping[str, Any]) -> bool:
    source = metadata.get("source_parameter_fingerprint")
    parameters = metadata.get("parameter_fingerprints", {})
    names = metadata.get("runtime_buffer_names", {})
    modes = metadata.get("linear_histogram_modes", {})
    return (
        set(parameters) == expected_conditions()
        and set(parameters.values()) == {source}
        and all(
            set(names[condition])
            == (
                ANCHOR_BUFFER_NAMES
                if representation_from_condition(condition) == "anchor_local"
                else CANDIDATE_BUFFER_NAMES
            )
            for condition in expected_conditions()
        )
        and all(
            modes[condition]
            == (
                LINEAR_HISTOGRAM_LOCAL
                if representation_from_condition(condition) == "anchor_local"
                else LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN
            )
            for condition in expected_conditions()
        )
    )


def authority_digests(config: Mapping[str, Any]) -> dict[str, str]:
    paths = source_artifact_paths(config)
    paths.update(
        {
            f"validation_seed{seed}_features": feature_path
            for seed, (
                feature_path,
                _label_path,
            ) in k1by7.validation_dataset_paths().items()
        }
    )
    paths.update(
        {
            f"validation_seed{seed}_labels": label_path
            for seed, (
                _feature_path,
                label_path,
            ) in k1by7.validation_dataset_paths().items()
        }
    )
    k1by8_config = k1by8.load_and_validate_config(
        ROOT / config["sources"]["k1by8_config"]
    )
    paths.update(
        {
            f"k1by8_authority_{name}": path
            for name, path in _k1by8_authority_paths(k1by8_config).items()
        }
    )
    return {name: _file_sha256(path) for name, path in paths.items()}


def source_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    root = ROOT / sources["k1by8_root"]
    return {
        "k1by8_config": ROOT / sources["k1by8_config"],
        "k1by8_preflight": root / "preflight.json",
        "k1by8_results": root / "results.jsonl",
        "k1by8_gate": root / "gate.json",
        "k1by8_validation": root / "validation.json",
        "k1by8_summary": root / "summary.json",
        "k1by8_model_metadata": root / "model_metadata.json",
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    values = config["sources"]["k1by8_digests"]
    return {
        "k1by8_config": values["config"],
        "k1by8_preflight": values["preflight.json"],
        "k1by8_results": values["results.jsonl"],
        "k1by8_gate": values["gate.json"],
        "k1by8_validation": values["validation.json"],
        "k1by8_summary": values["summary.json"],
        "k1by8_model_metadata": values["model_metadata.json"],
    }


def _k1by8_authority_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = k1by8.source_artifact_paths(config)
    k1by7_config = k1by7.load_and_validate_config(
        ROOT / config["sources"]["k1by7_config"]
    )
    paths.update(k1by7.source_artifact_paths(k1by7_config))
    return paths


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
    "EXPECTED_RESULT_ROWS",
    "REPRESENTATIONS",
    "RUN_ID",
    "RUNTIME_PROGRAMS",
    "TAPS",
    "adjudicate",
    "authority_digests",
    "build_models",
    "build_readiness",
    "comparison_rows",
    "condition_key",
    "equivalence_matrix_checks",
    "evaluate",
    "expected_conditions",
    "load_and_validate_config",
    "model_metadata_frozen",
    "rename_program_source_cells",
    "source_binding_checks",
    "split_condition_key",
]
