from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.gf2_boolean_view import apply_gf2_operator
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    CompiledSpnProgram,
    compile_ordered_primitive_program,
    materialize_ordered_primitive_payload,
    permute_program_source_roles,
    permute_program_target_bindings,
    replay_ordered_primitive_program,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_permutation_control_identifiability_k1by4_"
    "present_r7_seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_permutation_control_identifiability_"
    "k1by4_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "3b7b552dd98dabb19f69c7d42579f9996b6165ba8fdbbbf6779f331b33bb408b"
)
CURRENT_CONTROL = "wrong_target_binding_seed11"
SOURCE_ROLE_CONTROL = "source_role_permutation_1_3_0_2"
CONTROLS = (CURRENT_CONTROL, SOURCE_ROLE_CONTROL)
TAPS = ("inverse_linear", "post_inverse_sbox")
EXPECTED_SEEDS = (2, 3)
EXPECTED_STAGE_INDICES = (1, 0)
EXPECTED_RESULT_ROWS = 16
EXPECTED_EXPERT_USAGE = {
    "sbox4_table": 32,
    "linear_permutation": 32,
    "linear_gf2": 0,
}


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY4 config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BY4 identity drifted")
    if config.get("experiment") != (
        "innovation1_runtime_spn_permutation_control_identifiability_k1by4"
    ):
        raise ValueError("K1-BY4 experiment name drifted")
    audit = config.get("audit", {})
    if (
        audit.get("cipher") != "PRESENT-80"
        or audit.get("rounds") != 7
        or tuple(audit.get("seeds", ())) != EXPECTED_SEEDS
        or audit.get("pairs_per_sample") != 16
        or audit.get("validation_samples_total") != 2048
        or audit.get("runtime_rounds") != 2
        or audit.get("runtime_round_start") != 0
        or tuple(audit.get("controls", ())) != CONTROLS
        or tuple(audit.get("source_role_permutation", ())) != (1, 3, 0, 2)
        or tuple(audit.get("execution_stage_indices", ()))
        != EXPECTED_STAGE_INDICES
        or tuple(audit.get("taps", ())) != TAPS
        or audit.get("expected_result_rows") != EXPECTED_RESULT_ROWS
        or audit.get("neural_training_performed") is not False
        or audit.get("optimizer_steps") != 0
        or audit.get("data_generation") is not False
        or audit.get("device") != "cpu"
        or audit.get("execution") != "local_audit"
    ):
        raise ValueError("K1-BY4 audit contract drifted")
    gates = config.get("gates", {})
    if gates != {
        "identifiable_multiset_equal_rate_max": 0.95,
        "identifiable_pooled_summary_l1_min": 0.0001,
        "source_role_multiset_equal_rate_improvement_min": 0.01,
        "source_role_pooled_summary_l1_improvement_min": 0.0001,
        "require_every_seed_stage_tap": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BY4 gate contract drifted")
    caches = config.get("source", {}).get("validation_caches", [])
    if len(caches) != 2 or {int(item.get("seed", -1)) for item in caches} != set(
        EXPECTED_SEEDS
    ):
        raise ValueError("K1-BY4 cache contract drifted")
    return config


def build_programs(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> dict[str, CompiledSpnProgram]:
    source = config["source"]
    descriptor = load_runtime_spn_descriptor(
        project_root / str(source["descriptor"]),
        rounds=int(config["audit"]["runtime_rounds"]),
        round_start=int(config["audit"]["runtime_round_start"]),
    )
    correct = compile_ordered_primitive_program(descriptor.structure)
    return {
        "correct": correct,
        CURRENT_CONTROL: permute_program_target_bindings(correct, seed=11),
        SOURCE_ROLE_CONTROL: permute_program_source_roles(
            correct,
            role_permutation=config["audit"]["source_role_permutation"],
        ),
    }


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Path],
    dict[str, bool],
    dict[str, CompiledSpnProgram],
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    bound_paths: dict[str, Path] = {
        name: source_root / name for name in source["digests"]
    }
    bound_paths["source_plan"] = project_root / str(source["plan"])
    bound_paths["descriptor"] = project_root / str(source["descriptor"])
    for item in source["validation_caches"]:
        seed = int(item["seed"])
        cache_root = source_root / str(item["path"])
        bound_paths[f"seed{seed}_features"] = cache_root / "features.npy"
        bound_paths[f"seed{seed}_labels"] = cache_root / "labels.npy"
        bound_paths[f"seed{seed}_metadata"] = cache_root / "metadata.json"

    expected_digests = dict(source["digests"])
    expected_digests["source_plan"] = source["plan_sha256"]
    expected_digests["descriptor"] = source["descriptor_sha256"]
    for item in source["validation_caches"]:
        seed = int(item["seed"])
        expected_digests[f"seed{seed}_features"] = item["features_sha256"]
        expected_digests[f"seed{seed}_labels"] = item["labels_sha256"]
        expected_digests[f"seed{seed}_metadata"] = item["metadata_sha256"]

    checks = {
        f"{name}_digest_exact": path.is_file()
        and file_sha256(path) == expected_digests[name]
        for name, path in bound_paths.items()
    }
    try:
        source_gate = _read_json(source_root / "gate.json")
        source_validation = _read_json(source_root / "validation.json")
        source_preflight = _read_json(source_root / "preflight.json")
    except (OSError, ValueError, json.JSONDecodeError):
        source_gate = {}
        source_validation = {}
        source_preflight = {}
    checks["k1by3_required_hold_exact"] = (
        source_gate.get("status") == source["required_status"]
        and source_gate.get("decision") == source["required_decision"]
        and source_gate.get("failed_protocol_checks") == []
    )
    checks["k1by3_validation_passed"] = (
        source_validation.get("status") == "pass"
        and source_validation.get("errors") == []
    )
    checks["k1by3_preflight_passed"] = (
        source_preflight.get("status") == "pass"
        and source_preflight.get("optimizer_step_authorized") is True
    )

    datasets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for item in source["validation_caches"]:
        seed = int(item["seed"])
        try:
            features = np.load(bound_paths[f"seed{seed}_features"], mmap_mode="r")
            labels = np.load(bound_paths[f"seed{seed}_labels"], mmap_mode="r")
            metadata = _read_json(bound_paths[f"seed{seed}_metadata"])
            datasets[seed] = (features, labels)
            checks[f"seed{seed}_cache_geometry_exact"] = (
                features.shape == (2048, 2048)
                and features.dtype == np.uint8
                and labels.shape == (2048,)
                and labels.dtype == np.uint8
                and set(np.unique(labels).tolist()) == {0, 1}
                and int(labels.sum()) == 1024
                and metadata.get("cipher") == "PRESENT-80"
                and metadata.get("rounds") == 7
                and metadata.get("input_bits") == 2048
                and metadata.get("pairs_per_sample") == 16
                and metadata.get("samples_total") == 2048
                and metadata.get("samples_per_class") == 1024
                and metadata.get("key_schedule") == "per_pair_random"
                and metadata.get("negative_mode") == "encrypted_random_plaintexts"
            )
        except (OSError, ValueError, json.JSONDecodeError):
            checks[f"seed{seed}_cache_geometry_exact"] = False

    programs: dict[str, CompiledSpnProgram] = {}
    try:
        programs = build_programs(config, project_root=project_root)
        _truth, source_role_inverse = materialize_ordered_primitive_payload(
            programs[SOURCE_ROLE_CONTROL]
        )
        checks["three_programs_have_expected_usage"] = all(
            program.expert_usage == EXPECTED_EXPERT_USAGE
            for program in programs.values()
        )
        checks["three_program_semantics_are_distinct"] = (
            len({program.semantic_sha256 for program in programs.values()}) == 3
        )
        checks["source_role_control_preserves_permutation_geometry"] = bool(
            torch.all(source_role_inverse.sum(dim=-1) == 1)
            and torch.all(source_role_inverse.sum(dim=-2) == 1)
        )
        checks["source_role_control_label_exact"] = (
            programs[SOURCE_ROLE_CONTROL].control == SOURCE_ROLE_CONTROL
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        checks["program_construction_succeeded"] = False
    return datasets, bound_paths, checks, programs


def extract_histogram_taps(
    features: np.ndarray,
    program: CompiledSpnProgram,
    *,
    batch_size: int = 256,
) -> dict[tuple[int, int, str], np.ndarray]:
    if features.ndim != 2 or features.shape[1] != 16 * 2 * program.block_bits:
        raise ValueError("K1-BY4 feature geometry drifted")
    if batch_size <= 0:
        raise ValueError("K1-BY4 batch size must be positive")
    _truth, inverse = materialize_ordered_primitive_payload(program)
    runtime_structure = replay_ordered_primitive_program(program)
    semantic_cell_bits = torch.tensor(
        program.semantic_cell_to_native_bits,
        dtype=torch.long,
    )
    chunks: dict[tuple[int, int, str], list[np.ndarray]] = {
        (step, stage, tap): []
        for step, stage in enumerate(reversed(range(program.rounds)))
        for tap in TAPS
    }
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        current = (
            torch.from_numpy(np.array(features[start:stop], copy=True))
            .to(torch.float32)
            .reshape(stop - start, 16, 2, program.block_bits)
            .flip(-1)
        )
        for execution_step, stage_index in enumerate(reversed(range(program.rounds))):
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
                inverse[stage_index].to(torch.float32),
            )
            linear_state = linear_triplet[..., :2].permute(0, 1, 3, 2)
            chunks[(execution_step, stage_index, "inverse_linear")].append(
                _integer_difference_histogram(linear_state, semantic_cell_bits)
            )
            left = runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 0], stage_index
            )
            right = runtime_structure.apply_inverse_sboxes(
                linear_state[:, :, 1], stage_index
            )
            current = torch.stack((left, right), dim=2)
            chunks[(execution_step, stage_index, "post_inverse_sbox")].append(
                _integer_difference_histogram(current, semantic_cell_bits)
            )
    return {
        key: np.concatenate(parts, axis=0) for key, parts in chunks.items()
    }


def compare_histogram_taps(
    correct: Mapping[tuple[int, int, str], np.ndarray],
    control: Mapping[tuple[int, int, str], np.ndarray],
) -> list[dict[str, Any]]:
    if set(correct) != set(control):
        raise ValueError("K1-BY4 tap sets differ")
    rows = []
    for execution_step, stage_index, tap in sorted(correct):
        reference = correct[(execution_step, stage_index, tap)]
        candidate = control[(execution_step, stage_index, tap)]
        if reference.shape != candidate.shape or reference.ndim != 3:
            raise ValueError("K1-BY4 histogram geometry differs")
        canonical_reference = _canonical_cell_multiset(reference)
        canonical_candidate = _canonical_cell_multiset(candidate)
        multiset_equal = np.all(
            canonical_reference == canonical_candidate,
            axis=(1, 2),
        )
        reference_summary = _invariant_summary(reference)
        candidate_summary = _invariant_summary(candidate)
        pairs = float(reference.sum(axis=-1).max())
        cells = reference.shape[1]
        rows.append(
            {
                "execution_step": execution_step,
                "source_stage_index": stage_index,
                "tap": tap,
                "samples_total": int(reference.shape[0]),
                "cells": int(cells),
                "pairs_per_sample": int(pairs),
                "multiset_equal_rate": float(multiset_equal.mean()),
                "pooled_summary_l1": float(
                    np.abs(reference_summary - candidate_summary).mean() / pairs
                ),
                "ordered_histogram_l1": float(
                    np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
                    .sum(axis=(1, 2))
                    .mean()
                    / (2.0 * cells * pairs)
                ),
            }
        )
    return rows


def evaluate(
    *,
    config: Mapping[str, Any],
    datasets: Mapping[int, tuple[np.ndarray, np.ndarray]],
    programs: Mapping[str, CompiledSpnProgram],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        features, _labels = datasets[seed]
        reference = extract_histogram_taps(features, programs["correct"])
        for control_name in CONTROLS:
            observed = extract_histogram_taps(features, programs[control_name])
            for row in compare_histogram_taps(reference, observed):
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher": config["audit"]["cipher"],
                        "rounds": config["audit"]["rounds"],
                        "seed": seed,
                        "condition": control_name,
                        "neural_training_performed": False,
                        "optimizer_steps": 0,
                        **row,
                    }
                )
    return rows


def adjudicate(
    *,
    config: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
    source_unchanged: bool,
) -> dict[str, Any]:
    mapped = _result_map(result_rows, fail_closed=False)
    protocol_checks = {
        "config_digest_exact": file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256,
        "all_source_and_geometry_checks_pass": bool(source_checks)
        and all(source_checks.values()),
        "source_cache_unchanged_after_audit": source_unchanged,
        "sixteen_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS
        and set(mapped) == _expected_result_keys(),
        "histogram_metrics_finite_and_bounded": bool(mapped)
        and all(
            0.0 <= float(row["multiset_equal_rate"]) <= 1.0
            and 0.0 <= float(row["pooled_summary_l1"]) <= 1.0
            and 0.0 <= float(row["ordered_histogram_l1"]) <= 1.0
            and all(
                math.isfinite(float(row[name]))
                for name in (
                    "multiset_equal_rate",
                    "pooled_summary_l1",
                    "ordered_histogram_l1",
                )
            )
            and int(row["samples_total"]) == 2048
            and int(row["cells"]) == 16
            and int(row["pairs_per_sample"]) == 16
            and row.get("neural_training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            for row in mapped.values()
        ),
    }
    gates = config["gates"]
    research_checks: dict[str, bool] = {}
    for key in sorted(_expected_group_keys()):
        seed, execution_step, stage_index, tap = key
        current = mapped.get((seed, CURRENT_CONTROL, execution_step, stage_index, tap))
        source_role = mapped.get(
            (seed, SOURCE_ROLE_CONTROL, execution_step, stage_index, tap)
        )
        prefix = f"seed{seed}_step{execution_step}_stage{stage_index}_{tap}"
        research_checks[f"{prefix}_current_identifiable"] = (
            current is not None and _identifiable(current, gates)
        )
        research_checks[f"{prefix}_source_role_identifiable"] = (
            source_role is not None and _identifiable(source_role, gates)
        )
        research_checks[f"{prefix}_source_role_dominates"] = (
            current is not None
            and source_role is not None
            and float(current["multiset_equal_rate"])
            - float(source_role["multiset_equal_rate"])
            >= float(gates["source_role_multiset_equal_rate_improvement_min"])
            and float(source_role["pooled_summary_l1"])
            - float(current["pooled_summary_l1"])
            >= float(gates["source_role_pooled_summary_l1_improvement_min"])
        )

    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    current_identifiable_all = all(
        value
        for name, value in research_checks.items()
        if name.endswith("_current_identifiable")
    )
    source_identifiable_all = all(
        value
        for name, value in research_checks.items()
        if name.endswith("_source_role_identifiable")
    )
    source_dominates_all = all(
        value
        for name, value in research_checks.items()
        if name.endswith("_source_role_dominates")
    )
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1by4_protocol_invalid"
        next_action = (
            "Repair only the failed source digest, program geometry, histogram or "
            "artifact invariant and rerun the frozen zero-training audit."
        )
    elif current_identifiable_all:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by4_learned_pooling_audit_required"
        next_action = (
            "Replay the frozen K1-BY3 correct and wrong-binding checkpoints at the "
            "linear histogram, primitive expert, cell fusion, invariant pooling and "
            "final taps; redesign only the first learned tap that erases the already "
            "identifiable deterministic difference."
        )
    elif source_identifiable_all and source_dominates_all:
        status = "pass"
        decision = "innovation1_runtime_spn_k1by4_source_role_control_preferred"
        next_action = (
            "Preregister K1-BY5 at the identical PRESENT r7 2048/class, 16-pair, "
            "seed2/3 budget. Train only the frozen source-role corruption row per seed "
            "and compare it with the existing K1-BY3 correct/no-conditioner anchors."
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_k1by4_permutation_expert_hold"
        next_action = (
            "Hold permutation-expert neural training and remote scale. Design a new "
            "equal-geometry corruption that changes the invariant deterministic taps "
            "before another optimizer run."
        )
    panels = _comparison_panels(mapped)
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "thresholds": dict(gates),
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": failed_protocol,
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "route_summary": {
            "current_wrong_binding_identifiable_everywhere": current_identifiable_all,
            "source_role_identifiable_everywhere": source_identifiable_all,
            "source_role_uniformly_dominates": source_dominates_all,
        },
        "panels": panels,
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Local zero-training deterministic representation-identifiability audit "
            "on frozen PRESENT-80 r7 K1-BY3 validation caches; not neural performance, "
            "formal scale, transfer, attack or SOTA evidence."
        ),
    }


def run_audit(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("K1-BY4 is a frozen local CPU audit")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"K1-BY4 output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    datasets, bound_paths, source_checks, programs = load_authority(
        config,
        project_root=project_root,
    )
    if not source_checks or not all(source_checks.values()):
        raise ValueError(f"K1-BY4 source binding failed: {source_checks}")
    source_digests_before = {
        name: file_sha256(path) for name, path in bound_paths.items()
    }
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "program_semantic_sha256": {
            name: program.semantic_sha256 for name, program in programs.items()
        },
        "program_expert_usage": {
            name: program.expert_usage for name, program in programs.items()
        },
        "device": device,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    _append_progress(
        output_root / "progress.jsonl",
        "deterministic_tap_audit_start",
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    result_rows = evaluate(config=config, datasets=datasets, programs=programs)
    source_digests_after = {
        name: file_sha256(path) for name, path in bound_paths.items()
    }
    source_unchanged = source_digests_before == source_digests_after
    gate = adjudicate(
        config=config,
        result_rows=result_rows,
        source_checks=source_checks,
        source_unchanged=source_unchanged,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "source_digests_before": source_digests_before,
        "source_digests_after": source_digests_after,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "remote_scale": gate["remote_scale"],
        "route_summary": gate["route_summary"],
        "panels": gate["panels"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
        "result_rows": len(result_rows),
        "optimizer_steps": 0,
    }
    _write_jsonl(output_root / "results.jsonl", result_rows)
    _write_comparison_csv(output_root / "condition_comparison.csv", result_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
    )
    return {
        "preflight": preflight,
        "results": result_rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _integer_difference_histogram(
    values: torch.Tensor,
    semantic_cell_bits: torch.Tensor,
) -> np.ndarray:
    difference = torch.remainder(values[:, :, 0] + values[:, :, 1], 2.0)
    bits = difference[..., semantic_cell_bits]
    weights = torch.tensor((8, 4, 2, 1), dtype=torch.long)
    cell_values = torch.sum(bits.to(torch.long) * weights, dim=-1)
    counts = F.one_hot(cell_values, num_classes=16).sum(dim=1)
    if not torch.all(counts.sum(dim=-1) == values.shape[1]):
        raise ValueError("K1-BY4 histogram count invariant failed")
    return counts.to(torch.uint8).cpu().numpy()


def _canonical_cell_multiset(histograms: np.ndarray) -> np.ndarray:
    canonical = np.empty_like(histograms)
    for index, sample in enumerate(histograms):
        order = np.lexsort(tuple(sample[:, column] for column in range(15, -1, -1)))
        canonical[index] = sample[order]
    return canonical


def _invariant_summary(histograms: np.ndarray) -> np.ndarray:
    values = histograms.astype(np.float64)
    return np.concatenate((values.mean(axis=1), values.max(axis=1)), axis=1)


def _identifiable(row: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    return (
        float(row["multiset_equal_rate"])
        <= float(gates["identifiable_multiset_equal_rate_max"])
        and float(row["pooled_summary_l1"])
        >= float(gates["identifiable_pooled_summary_l1_min"])
    )


def _expected_group_keys() -> set[tuple[int, int, int, str]]:
    return {
        (seed, execution_step, stage_index, tap)
        for seed in EXPECTED_SEEDS
        for execution_step, stage_index in enumerate(EXPECTED_STAGE_INDICES)
        for tap in TAPS
    }


def _expected_result_keys() -> set[tuple[int, str, int, int, str]]:
    return {
        (seed, control, execution_step, stage_index, tap)
        for seed, execution_step, stage_index, tap in _expected_group_keys()
        for control in CONTROLS
    }


def _result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, str, int, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["condition"]),
            int(row["execution_step"]),
            int(row["source_stage_index"]),
            str(row["tap"]),
        )
        if key in mapped:
            raise ValueError(f"duplicate K1-BY4 result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != _expected_result_keys():
        raise ValueError("K1-BY4 result matrix is incomplete")
    return mapped


def _comparison_panels(
    mapped: Mapping[tuple[int, str, int, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    panels = []
    for seed, execution_step, stage_index, tap in sorted(_expected_group_keys()):
        current = mapped.get((seed, CURRENT_CONTROL, execution_step, stage_index, tap))
        source_role = mapped.get(
            (seed, SOURCE_ROLE_CONTROL, execution_step, stage_index, tap)
        )
        if current is None or source_role is None:
            continue
        panels.append(
            {
                "seed": seed,
                "execution_step": execution_step,
                "source_stage_index": stage_index,
                "tap": tap,
                "current_multiset_equal_rate": float(
                    current["multiset_equal_rate"]
                ),
                "source_role_multiset_equal_rate": float(
                    source_role["multiset_equal_rate"]
                ),
                "multiset_equal_rate_improvement": float(
                    current["multiset_equal_rate"]
                    - source_role["multiset_equal_rate"]
                ),
                "current_pooled_summary_l1": float(current["pooled_summary_l1"]),
                "source_role_pooled_summary_l1": float(
                    source_role["pooled_summary_l1"]
                ),
                "pooled_summary_l1_improvement": float(
                    source_role["pooled_summary_l1"]
                    - current["pooled_summary_l1"]
                ),
                "current_ordered_histogram_l1": float(
                    current["ordered_histogram_l1"]
                ),
                "source_role_ordered_histogram_l1": float(
                    source_role["ordered_histogram_l1"]
                ),
            }
        )
    return panels


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        "condition",
        "execution_step",
        "source_stage_index",
        "tap",
        "samples_total",
        "cells",
        "pairs_per_sample",
        "multiset_equal_rate",
        "pooled_summary_l1",
        "ordered_histogram_l1",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    row = {"run_id": RUN_ID, "event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "CONFIG_PATH",
    "CONTROLS",
    "CURRENT_CONTROL",
    "EXPECTED_RESULT_ROWS",
    "ROOT",
    "RUN_ID",
    "SOURCE_ROLE_CONTROL",
    "TAPS",
    "adjudicate",
    "build_programs",
    "compare_histogram_taps",
    "evaluate",
    "extract_histogram_taps",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
