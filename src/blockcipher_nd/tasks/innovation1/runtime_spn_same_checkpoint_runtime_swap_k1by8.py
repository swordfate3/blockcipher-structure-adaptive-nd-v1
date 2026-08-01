from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_learned_access_audit_k1by7 as k1by7,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_same_checkpoint_runtime_swap_k1by8_present_r7_"
    "seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_same_checkpoint_runtime_swap_k1by8_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "59f16610a530fd2e903cf74b25f4055f4282aaba39b50ce1da488f26a97d302f"
)
WEIGHT_SOURCES = ("correct_weights", "affine_weights")
RUNTIME_PROGRAMS = ("correct_runtime", "affine_runtime")
EXPECTED_SEEDS = (2, 3)
TAPS = k1by7.TAPS
MARGIN_FLOOR = 0.005
EXPECTED_RESULT_ROWS = (
    len(EXPECTED_SEEDS) * len(WEIGHT_SOURCES) * len(RUNTIME_PROGRAMS) * len(TAPS)
)
RUNTIME_BUFFER_NAMES = (
    "conditioner.edge_masks",
    "conditioner.edge_tokens",
    "conditioner.inverse_linear_matrices",
    "conditioner.linear_expert_types",
    "conditioner.sbox_truth_bits",
    "conditioner.semantic_cell_bits",
)


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    audit = config.get("audit", {})
    gates = config.get("gates", {})
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY8 config digest drifted")
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_same_checkpoint_runtime_swap_k1by8"
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
        or tuple(audit.get("weight_sources", ())) != WEIGHT_SOURCES
        or tuple(audit.get("runtime_programs", ())) != RUNTIME_PROGRAMS
        or audit.get("parameter_transfer") != "named_parameters_only"
        or audit.get("runtime_buffers_preserved_from_target_model") is not True
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or float(audit.get("diagonal_source_replay_tolerance", math.nan)) != 1e-6
        or float(
            gates.get(
                "correct_weight_correct_minus_affine_runtime_final_auc_min",
                math.nan,
            )
        )
        != MARGIN_FLOOR
        or float(
            gates.get(
                "correct_weight_correct_minus_affine_runtime_"
                "linear_histogram_probe_auc_min",
                math.nan,
            )
        )
        != MARGIN_FLOOR
        or gates.get("require_both_seeds") is not True
    ):
        raise ValueError("K1-BY8 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file()
        and _file_sha256(path) == expected[name]
        for name, path in paths.items()
    }
    try:
        gate = _read_json(paths["k1by7_gate"])
        validation = _read_json(paths["k1by7_validation"])
        source_config = k1by7.load_and_validate_config(paths["k1by7_config"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        gate = {}
        validation = {}
        source_config = {}
    checks["k1by7_exact_completed_audit"] = (
        gate.get("status") == "pass"
        and gate.get("method_status") == "hold"
        and gate.get("decision")
        == "innovation1_runtime_spn_k1by7_first_loss_linear_histogram_identified"
        and validation.get("status") == "pass"
        and validation.get("optimizer_steps") == 0
        and validation.get("result_rows") == 20
    )
    checks["k1by7_source_config_exact"] = (
        source_config.get("run_id") == k1by7.RUN_ID
        and all(k1by7.source_binding_checks(source_config).values())
    )
    return checks


def build_swapped_models(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, torch.nn.Module], dict[str, Mapping[str, Any]], dict[str, Any]]:
    source_config = k1by7.load_and_validate_config(
        ROOT / config["sources"]["k1by7_config"]
    )
    source_models, source_rows = k1by7.load_models_and_source_rows(
        source_config,
        seed=seed,
    )
    task = k1by6.task_map(k1by6.read_tasks())[seed]
    runtime_model_keys = {
        "correct_runtime": k1by6.CORRECT_MODEL,
        "affine_runtime": k1by6.AFFINE_MODEL,
    }
    source_conditions = {
        "correct_weights": "correct",
        "affine_weights": "affine_wrong_endpoint",
    }
    models: dict[str, torch.nn.Module] = {}
    for weight_source, source_condition in source_conditions.items():
        for runtime_program, model_key in runtime_model_keys.items():
            model = k1by6.build_model_for_task(task, model_key=model_key)
            copy_named_parameters(model, source_models[source_condition])
            model.eval()
            models[condition_key(weight_source, runtime_program)] = model

    parameter_fingerprints = {
        condition: learned_parameter_fingerprint(model)
        for condition, model in models.items()
    }
    runtime_fingerprints = {
        condition: runtime_buffer_fingerprint(model)
        for condition, model in models.items()
    }
    source_parameter_fingerprints = {
        weight_source: learned_parameter_fingerprint(
            source_models[source_condition]
        )
        for weight_source, source_condition in source_conditions.items()
    }
    metadata = {
        "parameter_fingerprints": parameter_fingerprints,
        "runtime_fingerprints": runtime_fingerprints,
        "source_parameter_fingerprints": source_parameter_fingerprints,
        "runtime_program_semantic_sha256": {
            condition: model.compiled_program_semantic_sha256
            for condition, model in models.items()
        },
        "runtime_buffer_names": {
            condition: sorted(name for name, _value in model.named_buffers())
            for condition, model in models.items()
        },
    }
    return models, source_rows, metadata


def copy_named_parameters(
    target: torch.nn.Module,
    source: torch.nn.Module,
) -> None:
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if tuple(target_parameters) != tuple(source_parameters):
        raise ValueError("K1-BY8 learned parameter names differ")
    with torch.no_grad():
        for name, target_parameter in target_parameters.items():
            source_parameter = source_parameters[name]
            if target_parameter.shape != source_parameter.shape:
                raise ValueError(f"K1-BY8 parameter shape differs: {name}")
            target_parameter.copy_(source_parameter)


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
            models, _source_rows, metadata = build_swapped_models(
                config,
                seed=2,
            )
            fixture = torch.as_tensor(
                np.array(
                    np.load(k1by7.validation_dataset_paths()[2][0], mmap_mode="r")[:4],
                    copy=True,
                ),
                dtype=torch.float32,
            )
            outputs = {}
            captures = {}
            for condition, model in models.items():
                with torch.inference_mode():
                    outputs[condition] = model(fixture)
                captures[condition] = k1by7.capture_taps(model, fixture)
            parameter_hashes = metadata["parameter_fingerprints"]
            runtime_hashes = metadata["runtime_fingerprints"]
            source_hashes = metadata["source_parameter_fingerprints"]
            expected_buffers = set(RUNTIME_BUFFER_NAMES)
            evidence_checks = {
                "four_swap_cells_exact": set(models) == expected_conditions(),
                "all_outputs_finite": all(
                    output.shape == (4, 1) and torch.isfinite(output).all()
                    for output in outputs.values()
                ),
                "all_taps_captured": all(
                    tuple(values) == TAPS for values in captures.values()
                ),
                "runtime_buffer_names_exact": all(
                    set(names) == expected_buffers
                    for names in metadata["runtime_buffer_names"].values()
                ),
                "correct_weight_parameters_identical_across_runtimes": (
                    parameter_hashes[
                        condition_key("correct_weights", "correct_runtime")
                    ]
                    == parameter_hashes[
                        condition_key("correct_weights", "affine_runtime")
                    ]
                    == source_hashes["correct_weights"]
                ),
                "affine_weight_parameters_identical_across_runtimes": (
                    parameter_hashes[
                        condition_key("affine_weights", "correct_runtime")
                    ]
                    == parameter_hashes[
                        condition_key("affine_weights", "affine_runtime")
                    ]
                    == source_hashes["affine_weights"]
                ),
                "correct_runtime_buffers_identical_across_weights": (
                    runtime_hashes[
                        condition_key("correct_weights", "correct_runtime")
                    ]
                    == runtime_hashes[
                        condition_key("affine_weights", "correct_runtime")
                    ]
                ),
                "affine_runtime_buffers_identical_across_weights": (
                    runtime_hashes[
                        condition_key("correct_weights", "affine_runtime")
                    ]
                    == runtime_hashes[
                        condition_key("affine_weights", "affine_runtime")
                    ]
                ),
                "runtime_fingerprints_distinct": (
                    runtime_hashes[
                        condition_key("correct_weights", "correct_runtime")
                    ]
                    != runtime_hashes[
                        condition_key("correct_weights", "affine_runtime")
                    ]
                ),
                "diagonal_fixture_outputs_match_sources": (
                    torch.equal(
                        outputs[condition_key("correct_weights", "correct_runtime")],
                        source_forward(config, seed=2, condition="correct", features=fixture),
                    )
                    and torch.equal(
                        outputs[condition_key("affine_weights", "affine_runtime")],
                        source_forward(
                            config,
                            seed=2,
                            condition="affine_wrong_endpoint",
                            features=fixture,
                        ),
                    )
                ),
                "off_diagonal_runtime_changes_outputs": (
                    not torch.equal(
                        outputs[condition_key("correct_weights", "correct_runtime")],
                        outputs[condition_key("correct_weights", "affine_runtime")],
                    )
                    and not torch.equal(
                        outputs[condition_key("affine_weights", "correct_runtime")],
                        outputs[condition_key("affine_weights", "affine_runtime")],
                    )
                ),
            }
            evidence_metrics = {
                "fixture_shape": list(fixture.shape),
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


def source_forward(
    config: Mapping[str, Any],
    *,
    seed: int,
    condition: str,
    features: torch.Tensor,
) -> torch.Tensor:
    source_config = k1by7.load_and_validate_config(
        ROOT / config["sources"]["k1by7_config"]
    )
    models, _rows = k1by7.load_models_and_source_rows(source_config, seed=seed)
    with torch.inference_mode():
        return models[condition](features)


def evaluate(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    batch_size = int(config["audit"]["batch_size"])
    epsilon = float(config["audit"]["probe_epsilon"])
    result_rows: list[dict[str, Any]] = []
    final_evaluation: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        features_path, labels_path = k1by7.validation_dataset_paths()[seed]
        features = np.load(features_path, mmap_mode="r")
        labels = np.asarray(np.load(labels_path, mmap_mode="r"), dtype=np.uint8)
        models, source_rows, metadata = build_swapped_models(config, seed=seed)
        model_metadata[str(seed)] = metadata
        representations = {
            condition: {tap: [] for tap in TAPS} for condition in models
        }
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
        diagonal_replay = {
            "correct_weights__correct_runtime": {
                "source_auc": float(source_rows["correct"]["metrics"]["auc"]),
                "replayed_auc": condition_auc[
                    condition_key("correct_weights", "correct_runtime")
                ],
            },
            "affine_weights__affine_runtime": {
                "source_auc": float(
                    source_rows["affine_wrong_endpoint"]["metrics"]["auc"]
                ),
                "replayed_auc": condition_auc[
                    condition_key("affine_weights", "affine_runtime")
                ],
            },
        }
        for values in diagonal_replay.values():
            values["absolute_error"] = abs(
                float(values["replayed_auc"]) - float(values["source_auc"])
            )
        final_evaluation[str(seed)] = {
            "condition_auc": condition_auc,
            "diagonal_replay": diagonal_replay,
        }

        for condition, taps in representations.items():
            weight_source, runtime_program = split_condition_key(condition)
            for tap_index, tap in enumerate(TAPS):
                values = np.concatenate(taps[tap])
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "condition": condition,
                        "weight_source": weight_source,
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
            str(row["weight_source"]),
            str(row["runtime_program"]),
            str(row["tap"]),
        ): row
        for row in result_rows
    }
    expected_keys = {
        (seed, weight_source, runtime_program, tap)
        for seed in EXPECTED_SEEDS
        for weight_source in WEIGHT_SOURCES
        for runtime_program in RUNTIME_PROGRAMS
        for tap in TAPS
    }
    replay_tolerance = float(config["audit"]["diagonal_source_replay_tolerance"])
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
        "all_diagonal_source_auc_replay": all(
            float(values["absolute_error"]) <= replay_tolerance
            for seed_values in final_evaluation.values()
            for values in seed_values["diagonal_replay"].values()
        ),
        "all_four_cells_have_finite_final_auc": all(
            set(seed_values["condition_auc"]) == expected_conditions()
            and all(
                math.isfinite(float(value))
                for value in seed_values["condition_auc"].values()
            )
            for seed_values in final_evaluation.values()
        ),
        "model_metadata_present_for_both_seeds": (
            set(model_metadata) == {"2", "3"}
        ),
    }

    seed_results: dict[str, Any] = {}
    primary_passes: list[bool] = []
    any_histogram_loss = False
    for seed in EXPECTED_SEEDS:
        weights: dict[str, Any] = {}
        for weight_source in WEIGHT_SOURCES:
            taps: dict[str, Any] = {}
            first_loss: str | None = None
            for tap in TAPS:
                correct = float(
                    mapped[(seed, weight_source, "correct_runtime", tap)]["probe_auc"]
                )
                affine = float(
                    mapped[(seed, weight_source, "affine_runtime", tap)]["probe_auc"]
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
                aucs[condition_key(weight_source, "correct_runtime")]
            )
            affine_final = float(
                aucs[condition_key(weight_source, "affine_runtime")]
            )
            final_margin = correct_final - affine_final
            if first_loss is None and final_margin < MARGIN_FLOOR:
                first_loss = "final_output"
            weights[weight_source] = {
                "taps": taps,
                "correct_runtime_final_auc": correct_final,
                "affine_runtime_final_auc": affine_final,
                "correct_minus_affine_runtime_final_auc": final_margin,
                "final_margin_pass": final_margin >= MARGIN_FLOOR,
                "first_margin_loss": first_loss,
            }
        primary = weights["correct_weights"]
        histogram_pass = bool(primary["taps"]["linear_histogram"]["margin_pass"])
        final_pass = bool(primary["final_margin_pass"])
        primary_passes.append(histogram_pass and final_pass)
        any_histogram_loss = any_histogram_loss or not histogram_pass
        seed_results[str(seed)] = {
            "weights": weights,
            "primary_histogram_margin_pass": histogram_pass,
            "primary_final_margin_pass": final_pass,
            "primary_gate_pass": histogram_pass and final_pass,
        }

    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by8_protocol_invalid"
        next_action = (
            "Repair only the failed source, parameter-transfer, runtime-buffer, hook, "
            "probe or artifact invariant and rerun unchanged."
        )
    elif all(primary_passes):
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by8_independent_training_variance_identified"
        )
        next_action = (
            "Retain same-checkpoint runtime interventions as mandatory attribution "
            "controls. Do not redesign the histogram from K1-BY7 alone; next test the "
            "same frozen causal protocol on the uKNIT GF(2) expert before cross-cipher "
            "claims."
        )
    elif any_histogram_loss:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss"
        )
        next_action = (
            "Change exactly the state-to-histogram representation to retain relative "
            "source-bundle incidence, then rerun the same two-seed PRESENT budget and "
            "same-checkpoint runtime controls."
        )
    else:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by8_same_checkpoint_downstream_access_loss"
        )
        next_action = (
            "Use the first failed same-checkpoint tap per seed to choose one downstream "
            "interface change; preserve the histogram, data, weights and controls."
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
            "correct_weight_correct_minus_affine_runtime_final_auc_min": MARGIN_FLOOR,
            "correct_weight_correct_minus_affine_runtime_"
            "linear_histogram_probe_auc_min": MARGIN_FLOOR,
        },
        "final_evaluation": final_evaluation,
        "seed_results": seed_results,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Zero-training same-checkpoint runtime-program intervention on frozen "
            "PRESENT r7 checkpoints and K1-BY3 validation rows; internal probes are "
            "mechanism diagnosis, not formal-scale, transfer or SOTA evidence."
        ),
    }


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed, seed_values in sorted(gate.get("seed_results", {}).items()):
        for weight_source in WEIGHT_SOURCES:
            values = seed_values["weights"][weight_source]
            for tap_index, tap in enumerate(TAPS):
                rows.append(
                    {
                        "seed": int(seed),
                        "weight_source": weight_source,
                        "tap_index": tap_index,
                        "tap": tap,
                        **values["taps"][tap],
                        "first_margin_loss": values["first_margin_loss"],
                    }
                )
            rows.append(
                {
                    "seed": int(seed),
                    "weight_source": weight_source,
                    "tap_index": len(TAPS),
                    "tap": "final_output",
                    "correct_runtime_probe_auc": values[
                        "correct_runtime_final_auc"
                    ],
                    "affine_runtime_probe_auc": values[
                        "affine_runtime_final_auc"
                    ],
                    "correct_minus_affine_runtime_probe_auc": values[
                        "correct_minus_affine_runtime_final_auc"
                    ],
                    "margin_pass": values["final_margin_pass"],
                    "first_margin_loss": values["first_margin_loss"],
                }
            )
    return rows


def condition_key(weight_source: str, runtime_program: str) -> str:
    if weight_source not in WEIGHT_SOURCES or runtime_program not in RUNTIME_PROGRAMS:
        raise ValueError("unknown K1-BY8 intervention cell")
    return f"{weight_source}__{runtime_program}"


def split_condition_key(value: str) -> tuple[str, str]:
    parts = value.split("__")
    if len(parts) != 2:
        raise ValueError("invalid K1-BY8 condition key")
    weight_source, runtime_program = parts
    condition_key(weight_source, runtime_program)
    return weight_source, runtime_program


def expected_conditions() -> set[str]:
    return {
        condition_key(weight_source, runtime_program)
        for weight_source in WEIGHT_SOURCES
        for runtime_program in RUNTIME_PROGRAMS
    }


def learned_parameter_fingerprint(model: torch.nn.Module) -> str:
    return _tensor_mapping_fingerprint(dict(model.named_parameters()))


def runtime_buffer_fingerprint(model: torch.nn.Module) -> str:
    buffers = dict(model.named_buffers())
    if set(buffers) != set(RUNTIME_BUFFER_NAMES):
        raise ValueError("K1-BY8 runtime buffer set drifted")
    return _tensor_mapping_fingerprint(buffers)


def authority_digests(config: Mapping[str, Any]) -> dict[str, str]:
    paths = source_artifact_paths(config)
    paths.update(
        {
            f"validation_seed{seed}_features": feature_path
            for seed, (feature_path, _label_path) in k1by7.validation_dataset_paths().items()
        }
    )
    paths.update(
        {
            f"validation_seed{seed}_labels": label_path
            for seed, (_feature_path, label_path) in k1by7.validation_dataset_paths().items()
        }
    )
    k1by7_config = k1by7.load_and_validate_config(
        ROOT / config["sources"]["k1by7_config"]
    )
    paths.update(k1by7.source_artifact_paths(k1by7_config))
    return {name: _file_sha256(path) for name, path in paths.items()}


def source_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    root = ROOT / sources["k1by7_root"]
    return {
        "k1by7_config": ROOT / sources["k1by7_config"],
        "k1by7_preflight": root / "preflight.json",
        "k1by7_results": root / "results.jsonl",
        "k1by7_gate": root / "gate.json",
        "k1by7_validation": root / "validation.json",
        "k1by7_summary": root / "summary.json",
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    values = config["sources"]["k1by7_digests"]
    return {
        "k1by7_config": values["config"],
        "k1by7_preflight": values["preflight.json"],
        "k1by7_results": values["results.jsonl"],
        "k1by7_gate": values["gate.json"],
        "k1by7_validation": values["validation.json"],
        "k1by7_summary": values["summary.json"],
    }


def _tensor_mapping_fingerprint(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
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
    "RUN_ID",
    "RUNTIME_PROGRAMS",
    "TAPS",
    "WEIGHT_SOURCES",
    "adjudicate",
    "authority_digests",
    "build_readiness",
    "build_swapped_models",
    "comparison_rows",
    "condition_key",
    "copy_named_parameters",
    "evaluate",
    "expected_conditions",
    "learned_parameter_fingerprint",
    "load_and_validate_config",
    "runtime_buffer_fingerprint",
    "source_binding_checks",
    "split_condition_key",
]
