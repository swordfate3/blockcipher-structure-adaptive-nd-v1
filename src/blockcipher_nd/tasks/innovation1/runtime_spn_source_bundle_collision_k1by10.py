from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_learned_access_audit_k1by7 as k1by7,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_source_bundle_histogram_k1by9 as k1by9,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_source_bundle_collision_k1by10_present_r7_"
    "seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_source_bundle_collision_k1by10_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "6e4be10367a62498fb4d29efdc3713e482d0ac59f719ded19bca2cbe47e0311e"
)
EXPECTED_SEEDS = (2, 3)
REPRESENTATIONS = k1by9.REPRESENTATIONS
RUNTIME_PROGRAMS = k1by9.RUNTIME_PROGRAMS
STAGES = (0, 1)
TARGET_CELLS = 16
CELL_TAPS = (
    "linear_histogram",
    "linear_primitive_expert",
    "cell_fusion",
)
REQUIRED_TAPS = ("linear_primitive_expert", "cell_fusion")
EXPECTED_RESULT_ROWS = (
    len(EXPECTED_SEEDS)
    * len(REPRESENTATIONS)
    * len(RUNTIME_PROGRAMS)
    * len(STAGES)
    * TARGET_CELLS
    * len(CELL_TAPS)
)
SEED2_EFFECT_MAX = -0.005
SEED3_EFFECT_MIN = 0.0


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    audit = config.get("audit", {})
    gates = config.get("gates", {})
    if _file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY10 config digest drifted")
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_source_bundle_collision_k1by10"
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
        or tuple(audit.get("stages", ())) != STAGES
        or audit.get("stage_order") != "forward_capture_reverse_program"
        or audit.get("target_cells") != TARGET_CELLS
        or tuple(audit.get("cell_taps", ())) != CELL_TAPS
        or audit.get("discovery_rows") != "even_validation_indices"
        or audit.get("evaluation_rows") != "odd_validation_indices"
        or audit.get("discovery_rows_per_class") != 512
        or audit.get("evaluation_rows_per_class") != 512
        or audit.get("probe") != "variance_normalized_class_mean_difference"
        or float(audit.get("probe_epsilon", math.nan)) != 1e-6
        or audit.get("uses_cell_identity_for_model_input") is not False
        or audit.get("uses_probe_for_model_selection") is not False
        or float(gates.get("seed2_candidate_minus_anchor_margin_max", math.nan))
        != SEED2_EFFECT_MAX
        or float(gates.get("seed3_candidate_minus_anchor_margin_min", math.nan))
        != SEED3_EFFECT_MIN
        or tuple(gates.get("required_taps", ())) != REQUIRED_TAPS
        or gates.get("require_same_target_cell_both_stages") is not True
        or tuple(gates.get("require_correct_affine_class_intersection_sizes", ()))
        != (1, 2)
        or gates.get("require_changed_peer_count_min") != 4
        or gates.get("remote_scale") != "no"
    ):
        raise ValueError("K1-BY10 frozen config contract drifted")
    return config


def source_binding_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    paths = source_artifact_paths(config)
    expected = source_expected_digests(config)
    checks = {
        f"{name}_digest_exact": path.is_file() and _file_sha256(path) == expected[name]
        for name, path in paths.items()
    }
    try:
        source_config = k1by9.load_and_validate_config(paths["k1by9_config"])
        gate = _read_json(paths["k1by9_gate"])
        validation = _read_json(paths["k1by9_validation"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        source_config = {}
        gate = {}
        validation = {}
    checks["k1by9_source_bindings_exact"] = (
        source_config.get("run_id") == k1by9.RUN_ID
        and all(k1by9.source_binding_checks(source_config).values())
    )
    checks["k1by9_expected_valid_miss"] = (
        gate.get("status") == "pass"
        and gate.get("method_status") == "hold"
        and gate.get("research_gate_passed") is False
        and gate.get("decision")
        == "innovation1_runtime_spn_k1by9_source_bundle_histogram_repair_not_supported"
        and validation.get("status") == "pass"
        and validation.get("result_rows") == k1by9.EXPECTED_RESULT_ROWS
        and validation.get("optimizer_steps") == 0
    )
    return checks


def partition_rows(models: Mapping[str, torch.nn.Module]) -> list[dict[str, Any]]:
    matrices = {
        runtime: models[
            k1by9.condition_key("candidate_source_bundle_mean", runtime)
        ].conditioner.linear_source_bundle_equivalence.detach().cpu()
        for runtime in RUNTIME_PROGRAMS
    }
    rows: list[dict[str, Any]] = []
    for tap_stage in STAGES:
        program_stage = len(STAGES) - 1 - tap_stage
        for cell in range(TARGET_CELLS):
            correct = _matrix_peers(matrices["correct_runtime"], program_stage, cell)
            affine = _matrix_peers(matrices["affine_runtime"], program_stage, cell)
            intersection = sorted(correct & affine)
            removed = sorted(correct - affine)
            added = sorted(affine - correct)
            rows.append(
                {
                    "tap_stage": tap_stage,
                    "program_stage": program_stage,
                    "target_cell": cell,
                    "correct_peers": sorted(correct),
                    "affine_peers": sorted(affine),
                    "intersection_peers": intersection,
                    "intersection_size": len(intersection),
                    "removed_peers": removed,
                    "added_peers": added,
                    "changed_peer_count": len(removed) + len(added),
                }
            )
    return rows


def partition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_values: dict[str, Any] = {}
    for tap_stage in STAGES:
        selected = [row for row in rows if int(row["tap_stage"]) == tap_stage]
        correct_pairs = _peer_pairs(selected, field="correct_peers")
        affine_pairs = _peer_pairs(selected, field="affine_peers")
        stage_values[str(tap_stage)] = {
            "program_stage": len(STAGES) - 1 - tap_stage,
            "correct_peer_pairs": len(correct_pairs),
            "affine_peer_pairs": len(affine_pairs),
            "shared_peer_pairs": len(correct_pairs & affine_pairs),
            "changed_peer_pairs": len(correct_pairs ^ affine_pairs),
            "intersection_sizes": sorted(
                {int(row["intersection_size"]) for row in selected}
            ),
            "changed_peer_counts": sorted(
                {int(row["changed_peer_count"]) for row in selected}
            ),
        }
    return {
        "stages": stage_values,
        "stage_partitions_identical": _stage_partition_fingerprint(rows, 0)
        == _stage_partition_fingerprint(rows, 1),
    }


def build_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol_checks = {
        **source_binding_checks(config),
        "config_exact": load_and_validate_config() == config,
        "zero_training_frozen": (
            config["audit"]["neural_training_performed"] is False
            and config["audit"]["optimizer_steps"] == 0
        ),
        "local_cpu_execution_frozen": config["audit"]["device"] == "cpu",
    }
    evidence_checks: dict[str, bool] = {}
    evidence_metrics: dict[str, Any] = {}
    errors: list[str] = []
    if all(protocol_checks.values()):
        try:
            all_rows: dict[int, list[dict[str, Any]]] = {}
            metadata: dict[int, Mapping[str, Any]] = {}
            shape_maps: dict[int, dict[str, Any]] = {}
            for seed in EXPECTED_SEEDS:
                models, _source_row, values = k1by9.build_models(
                    k1by9.load_and_validate_config(
                        ROOT / config["sources"]["k1by9_config"]
                    ),
                    seed=seed,
                )
                feature_path, _label_path = k1by7.validation_dataset_paths()[seed]
                fixture = torch.as_tensor(
                    np.array(np.load(feature_path, mmap_mode="r")[:8], copy=True),
                    dtype=torch.float32,
                )
                captures = {
                    condition: k1by7.capture_taps(model, fixture)
                    for condition, model in models.items()
                }
                shape_maps[seed] = {
                    condition: {
                        tap: list(captures[condition][tap].shape)
                        for tap in CELL_TAPS
                    }
                    for condition in models
                }
                all_rows[seed] = partition_rows(models)
                metadata[seed] = values
            evidence_checks = {
                "four_frozen_models_per_seed": all(
                    set(values["parameter_fingerprints"])
                    == k1by9.expected_conditions()
                    for values in metadata.values()
                ),
                "model_metadata_replays_k1by9": all(
                    k1by9.model_metadata_frozen(values)
                    for values in metadata.values()
                ),
                "cell_tap_shapes_exact_and_finite": all(
                    shape[0] == 8
                    and shape[1] == len(STAGES)
                    and shape[2] == TARGET_CELLS
                    for shapes in shape_maps.values()
                    for taps in shapes.values()
                    for shape in taps.values()
                ),
                "partition_metadata_identical_across_seeds": all_rows[2]
                == all_rows[3],
                "partition_rows_exact": len(all_rows[2])
                == len(STAGES) * TARGET_CELLS,
                "partition_overlap_is_bounded_and_changed_both_stages": all(
                    int(row["intersection_size"]) in {1, 2}
                    and int(row["changed_peer_count"]) >= 4
                    for row in all_rows[2]
                ),
                "partition_is_stage_stable": partition_summary(all_rows[2])[
                    "stage_partitions_identical"
                ],
            }
            evidence_metrics = {
                "tap_shapes": shape_maps[2],
                "partition_summary": partition_summary(all_rows[2]),
                "partition_fingerprint": _json_fingerprint(all_rows[2]),
            }
        except Exception as exc:  # pragma: no cover - readiness artifact captures it
            errors.append(f"{type(exc).__name__}: {exc}")
    failed = sorted(
        name
        for name, passed in {**protocol_checks, **evidence_checks}.items()
        if not passed
    )
    return {
        "run_id": RUN_ID,
        "status": "pass" if not failed and not errors else "fail",
        "execution_authorized": not failed and not errors,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "evidence_metrics": evidence_metrics,
        "failed_checks": failed,
        "errors": errors,
    }


def evaluate(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_size = int(config["audit"]["batch_size"])
    epsilon = float(config["audit"]["probe_epsilon"])
    source_config = k1by9.load_and_validate_config(
        ROOT / config["sources"]["k1by9_config"]
    )
    result_rows: list[dict[str, Any]] = []
    partition: list[dict[str, Any]] | None = None
    model_metadata: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        feature_path, label_path = k1by7.validation_dataset_paths()[seed]
        features = np.load(feature_path, mmap_mode="r")
        labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=np.uint8)
        models, _source_row, metadata = k1by9.build_models(source_config, seed=seed)
        model_metadata[str(seed)] = metadata
        captured_values = {
            condition: {tap: [] for tap in CELL_TAPS} for condition in models
        }
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            batch = torch.as_tensor(
                np.array(features[start:stop], copy=True),
                dtype=torch.float32,
            )
            for condition, model in models.items():
                captured = k1by7.capture_taps(model, batch)
                for tap in CELL_TAPS:
                    captured_values[condition][tap].append(
                        captured[tap].numpy(force=True)
                    )
        current_partition = partition_rows(models)
        if partition is None:
            partition = current_partition
        elif partition != current_partition:
            raise ValueError("K1-BY10 partition metadata changed across seeds")

        for condition, taps in captured_values.items():
            representation, runtime_program = k1by9.split_condition_key(condition)
            for tap_index, tap in enumerate(CELL_TAPS):
                values = np.concatenate(taps[tap])
                for tap_stage in STAGES:
                    for target_cell in range(TARGET_CELLS):
                        cell_values = values[:, tap_stage, target_cell]
                        result_rows.append(
                            {
                                "run_id": RUN_ID,
                                "seed": seed,
                                "condition": condition,
                                "representation": representation,
                                "runtime_program": runtime_program,
                                "tap": tap,
                                "tap_index": tap_index,
                                "tap_stage": tap_stage,
                                "program_stage": len(STAGES) - 1 - tap_stage,
                                "target_cell": target_cell,
                                "representation_shape": list(cell_values.shape),
                                **k1by7.mean_difference_probe(
                                    cell_values,
                                    labels,
                                    epsilon=epsilon,
                                ),
                            }
                        )
    if partition is None:  # pragma: no cover - frozen seeds make this unreachable
        raise ValueError("K1-BY10 produced no partition metadata")
    return result_rows, partition, model_metadata


def effect_rows(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mapped = {
        (
            int(row["seed"]),
            str(row["representation"]),
            str(row["runtime_program"]),
            int(row["tap_stage"]),
            int(row["target_cell"]),
            str(row["tap"]),
        ): float(row["probe_auc"])
        for row in result_rows
    }
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for tap_stage in STAGES:
            for target_cell in range(TARGET_CELLS):
                for tap_index, tap in enumerate(CELL_TAPS):
                    anchor_correct = mapped[
                        (seed, "anchor_local", "correct_runtime", tap_stage, target_cell, tap)
                    ]
                    anchor_affine = mapped[
                        (seed, "anchor_local", "affine_runtime", tap_stage, target_cell, tap)
                    ]
                    candidate_correct = mapped[
                        (
                            seed,
                            "candidate_source_bundle_mean",
                            "correct_runtime",
                            tap_stage,
                            target_cell,
                            tap,
                        )
                    ]
                    candidate_affine = mapped[
                        (
                            seed,
                            "candidate_source_bundle_mean",
                            "affine_runtime",
                            tap_stage,
                            target_cell,
                            tap,
                        )
                    ]
                    anchor_margin = anchor_correct - anchor_affine
                    candidate_margin = candidate_correct - candidate_affine
                    rows.append(
                        {
                            "seed": seed,
                            "tap_stage": tap_stage,
                            "program_stage": len(STAGES) - 1 - tap_stage,
                            "target_cell": target_cell,
                            "tap": tap,
                            "tap_index": tap_index,
                            "anchor_correct_auc": anchor_correct,
                            "anchor_affine_auc": anchor_affine,
                            "anchor_runtime_margin": anchor_margin,
                            "candidate_correct_auc": candidate_correct,
                            "candidate_affine_auc": candidate_affine,
                            "candidate_runtime_margin": candidate_margin,
                            "candidate_minus_anchor_margin": (
                                candidate_margin - anchor_margin
                            ),
                        }
                    )
    return rows


def adjudicate(
    config: Mapping[str, Any],
    *,
    result_rows: Sequence[Mapping[str, Any]],
    partition: Sequence[Mapping[str, Any]],
    model_metadata: Mapping[str, Any],
    readiness: Mapping[str, Any],
    sources_unchanged: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effects = effect_rows(result_rows)
    result_keys = {
        (
            int(row["seed"]),
            str(row["representation"]),
            str(row["runtime_program"]),
            int(row["tap_stage"]),
            int(row["target_cell"]),
            str(row["tap"]),
        )
        for row in result_rows
    }
    expected_keys = {
        (seed, representation, runtime, stage, cell, tap)
        for seed in EXPECTED_SEEDS
        for representation in REPRESENTATIONS
        for runtime in RUNTIME_PROGRAMS
        for stage in STAGES
        for cell in range(TARGET_CELLS)
        for tap in CELL_TAPS
    }
    partition_map = {
        (int(row["tap_stage"]), int(row["target_cell"])): row for row in partition
    }
    effect_map = {
        (
            int(row["seed"]),
            int(row["tap_stage"]),
            int(row["target_cell"]),
            str(row["tap"]),
        ): row
        for row in effects
    }
    protocol_checks = {
        "readiness_exact_pass": (
            readiness.get("status") == "pass"
            and readiness.get("execution_authorized") is True
            and readiness.get("training_authorized") is False
            and readiness.get("optimizer_steps_authorized") == 0
        ),
        "source_bindings_still_exact": all(source_binding_checks(config).values()),
        "source_artifacts_unchanged": sources_unchanged,
        "seven_hundred_sixty_eight_probe_rows_exact": (
            len(result_rows) == EXPECTED_RESULT_ROWS and result_keys == expected_keys
        ),
        "all_probe_rows_finite_and_balanced": all(
            math.isfinite(float(row.get("probe_auc", math.nan)))
            and int(row.get("discovery_positive_rows", -1)) == 512
            and int(row.get("discovery_negative_rows", -1)) == 512
            and int(row.get("evaluation_positive_rows", -1)) == 512
            and int(row.get("evaluation_negative_rows", -1)) == 512
            for row in result_rows
        ),
        "partition_rows_exact": len(partition_map) == len(STAGES) * TARGET_CELLS,
        "partition_overlap_bounded_and_changed": all(
            int(row["intersection_size"]) in {1, 2}
            and int(row["changed_peer_count"]) >= 4
            for row in partition
        ),
        "partition_stable_across_stages": partition_summary(partition)[
            "stage_partitions_identical"
        ],
        "model_metadata_frozen_for_both_seeds": (
            set(model_metadata) == {"2", "3"}
            and all(
                k1by9.model_metadata_frozen(values)
                for values in model_metadata.values()
            )
        ),
        "effect_rows_exact_and_finite": (
            len(effects)
            == len(EXPECTED_SEEDS) * len(STAGES) * TARGET_CELLS * len(CELL_TAPS)
            and all(
                math.isfinite(float(row["candidate_minus_anchor_margin"]))
                for row in effects
            )
        ),
    }

    locus_results: list[dict[str, Any]] = []
    supported_cells: list[int] = []
    for cell in range(TARGET_CELLS):
        clauses: dict[str, bool] = {}
        measurements: dict[str, Any] = {}
        for stage in STAGES:
            partition_value = partition_map[(stage, cell)]
            clauses[f"stage{stage}_partition_intersection_allowed"] = (
                int(partition_value["intersection_size"]) in {1, 2}
            )
            clauses[f"stage{stage}_peer_change_present"] = (
                int(partition_value["changed_peer_count"]) >= 4
            )
            for tap in REQUIRED_TAPS:
                seed2 = float(
                    effect_map[(2, stage, cell, tap)][
                        "candidate_minus_anchor_margin"
                    ]
                )
                seed3 = float(
                    effect_map[(3, stage, cell, tap)][
                        "candidate_minus_anchor_margin"
                    ]
                )
                measurements[f"stage{stage}_{tap}_seed2_effect"] = seed2
                measurements[f"stage{stage}_{tap}_seed3_effect"] = seed3
                clauses[f"stage{stage}_{tap}_seed2_loss"] = (
                    seed2 <= SEED2_EFFECT_MAX
                )
                clauses[f"stage{stage}_{tap}_seed3_nonnegative"] = (
                    seed3 >= SEED3_EFFECT_MIN
                )
        passed = all(clauses.values())
        if passed:
            supported_cells.append(cell)
        locus_results.append(
            {
                "target_cell": cell,
                "supported_locus": passed,
                "clauses": clauses,
                "measurements": measurements,
            }
        )

    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    research_passed = not failed_protocol and bool(supported_cells)
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by10_protocol_invalid"
        next_action = (
            "Repair only the failed K1-BY9 source, tap shape, equality partition, "
            "probe split or artifact binding and rerun K1-BY10 unchanged."
        )
    elif research_passed:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by10_source_bundle_oversmoothing_locus_identified"
        )
        next_action = (
            "Preregister one zero-training non-averaging source-bundle deviation "
            "residual. Preserve every local cell histogram and expose signed "
            "local-minus-bundle context; compare against the local anchor before "
            "training, scaling or adding a cipher."
        )
    else:
        status = "pass"
        decision = (
            "innovation1_runtime_spn_k1by10_no_stable_partition_locus_identified"
        )
        next_action = (
            "Close equality-partition pooling and return to an edge-conditioned "
            "residual that preserves individual cells. Do not reuse bundle means, "
            "tune the K1-BY9 blend, train or scale."
        )
    return (
        {
            "run_id": RUN_ID,
            "status": status,
            "method_status": "hold",
            "decision": decision,
            "research_gate_passed": research_passed,
            "remote_scale": "no",
            "protocol_checks": protocol_checks,
            "failed_protocol_checks": failed_protocol,
            "thresholds": {
                "seed2_candidate_minus_anchor_margin_max": SEED2_EFFECT_MAX,
                "seed3_candidate_minus_anchor_margin_min": SEED3_EFFECT_MIN,
                "required_taps": list(REQUIRED_TAPS),
                "same_target_cell_required_both_stages": True,
                "correct_affine_class_intersection_sizes": [1, 2],
                "changed_peer_count_min": 4,
            },
            "partition_summary": partition_summary(partition),
            "supported_target_cells": supported_cells,
            "locus_results": locus_results,
            "extreme_effects": _extreme_effects(effects),
            "next_action": next_action,
            "blocked_actions": list(config["blocked_actions"]),
            "claim_scope": (
                "Zero-training per-cell decomposition of frozen K1-BY9 PRESENT r7 "
                "representations. Internal probe effects are mechanism diagnostics, "
                "not distinguisher, transfer, attack, formal-scale or SOTA evidence."
            ),
        },
        effects,
    )


def authority_digests(config: Mapping[str, Any]) -> dict[str, str]:
    paths = source_artifact_paths(config)
    source_config = k1by9.load_and_validate_config(paths["k1by9_config"])
    paths.update(
        {
            f"k1by9_authority_{name}": path
            for name, path in k1by9.source_artifact_paths(source_config).items()
        }
    )
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
    return {name: _file_sha256(path) for name, path in paths.items()}


def source_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sources = config["sources"]
    root = ROOT / sources["k1by9_root"]
    return {
        "k1by9_config": ROOT / sources["k1by9_config"],
        "k1by9_preflight": root / "preflight.json",
        "k1by9_results": root / "results.jsonl",
        "k1by9_gate": root / "gate.json",
        "k1by9_validation": root / "validation.json",
        "k1by9_summary": root / "summary.json",
        "k1by9_model_metadata": root / "model_metadata.json",
    }


def source_expected_digests(config: Mapping[str, Any]) -> dict[str, str]:
    values = config["sources"]["k1by9_digests"]
    return {
        "k1by9_config": values["config"],
        "k1by9_preflight": values["preflight.json"],
        "k1by9_results": values["results.jsonl"],
        "k1by9_gate": values["gate.json"],
        "k1by9_validation": values["validation.json"],
        "k1by9_summary": values["summary.json"],
        "k1by9_model_metadata": values["model_metadata.json"],
    }


def _matrix_peers(matrix: torch.Tensor, stage: int, cell: int) -> set[int]:
    return {
        peer
        for peer, value in enumerate(matrix[stage, cell].tolist())
        if float(value) > 0.0
    }


def _peer_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for row in rows:
        peers = [int(value) for value in row[field]]
        for left_index, left in enumerate(peers):
            for right in peers[left_index + 1 :]:
                pairs.add((min(left, right), max(left, right)))
    return pairs


def _stage_partition_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    stage: int,
) -> str:
    selected = [
        {
            key: value
            for key, value in row.items()
            if key not in {"tap_stage", "program_stage"}
        }
        for row in rows
        if int(row["tap_stage"]) == stage
    ]
    return _json_fingerprint(selected)


def _extreme_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        for stage in STAGES:
            for tap in REQUIRED_TAPS:
                selected = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed
                    and int(row["tap_stage"]) == stage
                    and str(row["tap"]) == tap
                ]
                minimum = min(
                    selected,
                    key=lambda row: float(row["candidate_minus_anchor_margin"]),
                )
                maximum = max(
                    selected,
                    key=lambda row: float(row["candidate_minus_anchor_margin"]),
                )
                result[f"seed{seed}_stage{stage}_{tap}"] = {
                    "minimum_cell": int(minimum["target_cell"]),
                    "minimum_effect": float(
                        minimum["candidate_minus_anchor_margin"]
                    ),
                    "maximum_cell": int(maximum["target_cell"]),
                    "maximum_effect": float(
                        maximum["candidate_minus_anchor_margin"]
                    ),
                }
    return result


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    "CELL_TAPS",
    "CONFIG_PATH",
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SEEDS",
    "REQUIRED_TAPS",
    "RUN_ID",
    "STAGES",
    "TARGET_CELLS",
    "adjudicate",
    "authority_digests",
    "build_readiness",
    "effect_rows",
    "evaluate",
    "load_and_validate_config",
    "partition_rows",
    "partition_summary",
    "source_binding_checks",
]
