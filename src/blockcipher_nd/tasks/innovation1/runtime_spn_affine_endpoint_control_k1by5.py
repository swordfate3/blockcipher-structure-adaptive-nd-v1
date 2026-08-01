from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    CompiledSpnProgram,
    materialize_ordered_primitive_payload,
    permute_program_source_endpoints_affine,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_permutation_control_identifiability_k1by4 as k1by4,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_runtime_spn_affine_endpoint_control_k1by5_"
    "present_r7_seed2_seed3_20260801"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_affine_endpoint_control_k1by5_20260801.json"
)
EXPECTED_CONFIG_SHA256 = (
    "447b81724f6c4d2824b26bff52b2cbe1d804cb5a30b906a0857ccf3c65b0f071"
)
CONTROL = "source_endpoint_affine_m5_b1_mod64"
EXPECTED_SEEDS = (2, 3)
EXPECTED_STAGE_INDICES = (1, 0)
EXPECTED_RESULT_ROWS = 8


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BY5 config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BY5 identity drifted")
    if config.get("experiment") != (
        "innovation1_runtime_spn_affine_endpoint_control_k1by5"
    ):
        raise ValueError("K1-BY5 experiment name drifted")
    audit = config.get("audit", {})
    if (
        audit.get("cipher") != "PRESENT-80"
        or audit.get("rounds") != 7
        or tuple(audit.get("seeds", ())) != EXPECTED_SEEDS
        or audit.get("pairs_per_sample") != 16
        or audit.get("validation_samples_total") != 2048
        or audit.get("runtime_rounds") != 2
        or audit.get("runtime_round_start") != 0
        or audit.get("control") != CONTROL
        or audit.get("source_endpoint_flattening")
        != "4 * source_cell + source_role"
        or audit.get("affine_multiplier") != 5
        or audit.get("affine_offset") != 1
        or audit.get("affine_modulus") != 64
        or tuple(audit.get("execution_stage_indices", ()))
        != EXPECTED_STAGE_INDICES
        or tuple(audit.get("taps", ())) != k1by4.TAPS
        or audit.get("expected_result_rows") != EXPECTED_RESULT_ROWS
        or audit.get("neural_training_performed") is not False
        or audit.get("optimizer_steps") != 0
        or audit.get("data_generation") is not False
        or audit.get("device") != "cpu"
        or audit.get("execution") != "local_audit"
    ):
        raise ValueError("K1-BY5 audit contract drifted")
    if config.get("gates") != {
        "identifiable_multiset_equal_rate_max": 0.95,
        "identifiable_pooled_summary_l1_min": 0.0001,
        "require_every_seed_stage_tap": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BY5 gate contract drifted")
    if config.get("decisions") != {
        "pass": "innovation1_runtime_spn_k1by5_affine_endpoint_control_ready",
        "hold": (
            "innovation1_runtime_spn_k1by5_affine_endpoint_control_not_identifiable"
        ),
        "invalid": "innovation1_runtime_spn_k1by5_protocol_invalid",
    }:
        raise ValueError("K1-BY5 decision contract drifted")
    return config


def build_affine_program(
    correct: CompiledSpnProgram,
    config: Mapping[str, Any],
) -> CompiledSpnProgram:
    return permute_program_source_endpoints_affine(
        correct,
        multiplier=int(config["audit"]["affine_multiplier"]),
        offset=int(config["audit"]["affine_offset"]),
    )


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Path],
    dict[str, bool],
    CompiledSpnProgram,
    CompiledSpnProgram,
]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    source_config_path = project_root / str(source["config"])
    checks: dict[str, bool] = {
        "k1by4_config_digest_exact": source_config_path.is_file()
        and file_sha256(source_config_path) == source["config_sha256"]
    }
    bound_paths = {
        f"k1by4_{name}": source_root / name for name in source["digests"]
    }
    checks.update(
        {
            f"k1by4_{name}_digest_exact": path.is_file()
            and file_sha256(path) == source["digests"][name]
            for name, path in (
                (name.removeprefix("k1by4_"), path)
                for name, path in bound_paths.items()
            )
        }
    )
    try:
        source_gate = _read_json(source_root / "gate.json")
        source_validation = _read_json(source_root / "validation.json")
        source_preflight = _read_json(source_root / "preflight.json")
        k1by4_config = k1by4.load_and_validate_config(source_config_path)
    except (OSError, ValueError, json.JSONDecodeError):
        source_gate = {}
        source_validation = {}
        source_preflight = {}
        k1by4_config = {}
    checks["k1by4_required_hold_exact"] = (
        source_gate.get("status") == source["required_status"]
        and source_gate.get("decision") == source["required_decision"]
        and source_gate.get("failed_protocol_checks") == []
    )
    checks["k1by4_validation_passed"] = (
        source_validation.get("status") == "pass"
        and source_validation.get("errors") == []
        and source_validation.get("result_rows") == 16
    )
    checks["k1by4_preflight_zero_training"] = (
        source_preflight.get("status") == "pass"
        and source_preflight.get("training_authorized") is False
        and source_preflight.get("optimizer_steps") == 0
    )

    datasets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    correct: CompiledSpnProgram | None = None
    affine: CompiledSpnProgram | None = None
    if k1by4_config:
        (
            datasets,
            inherited_paths,
            inherited_checks,
            programs,
        ) = k1by4.load_authority(k1by4_config, project_root=project_root)
        bound_paths.update(
            {f"k1by4_authority_{name}": path for name, path in inherited_paths.items()}
        )
        checks.update(
            {f"k1by4_authority_{name}": passed for name, passed in inherited_checks.items()}
        )
        correct = programs.get("correct")
    if correct is not None:
        try:
            affine = build_affine_program(correct, config)
            _truth, inverse = materialize_ordered_primitive_payload(affine)
            endpoint_map = tuple((5 * endpoint + 1) % 64 for endpoint in range(64))
            split_counts = tuple(
                len({endpoint_map[4 * cell + role] // 4 for role in range(4)})
                for cell in range(16)
            )
            checks["affine_endpoint_map_is_bijective"] = (
                sorted(endpoint_map) == list(range(64))
            )
            checks["affine_endpoint_map_splits_every_source_cell"] = all(
                count >= 2 for count in split_counts
            )
            checks["affine_program_control_label_exact"] = affine.control == CONTROL
            checks["affine_program_semantics_differ"] = (
                affine.semantic_sha256 != correct.semantic_sha256
            )
            checks["affine_program_routes_only_permutation_expert"] = (
                affine.expert_usage == k1by4.EXPECTED_EXPERT_USAGE
            )
            checks["affine_inverse_is_one_to_one"] = bool(
                torch.all(inverse.sum(dim=-1) == 1)
                and torch.all(inverse.sum(dim=-2) == 1)
            )
        except (ValueError, RuntimeError):
            checks["affine_program_construction_succeeded"] = False
    else:
        checks["affine_program_construction_succeeded"] = False
    if correct is None or affine is None:
        raise ValueError(f"K1-BY5 program authority failed: {checks}")
    return datasets, bound_paths, checks, correct, affine


def evaluate(
    *,
    config: Mapping[str, Any],
    datasets: Mapping[int, tuple[np.ndarray, np.ndarray]],
    correct: CompiledSpnProgram,
    affine: CompiledSpnProgram,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        features, _labels = datasets[seed]
        reference_taps = k1by4.extract_histogram_taps(features, correct)
        affine_taps = k1by4.extract_histogram_taps(features, affine)
        for row in k1by4.compare_histogram_taps(reference_taps, affine_taps):
            rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher": config["audit"]["cipher"],
                    "rounds": config["audit"]["rounds"],
                    "seed": seed,
                    "condition": CONTROL,
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
        "all_source_artifacts_unchanged_after_audit": source_unchanged,
        "eight_result_rows_exact": len(result_rows) == EXPECTED_RESULT_ROWS
        and set(mapped) == _expected_keys(),
        "metrics_finite_and_bounded": bool(mapped)
        and all(_row_valid(row) for row in mapped.values()),
    }
    gates = config["gates"]
    research_checks = {
        f"seed{seed}_step{step}_stage{stage}_{tap}_identifiable": (
            (row := mapped.get((seed, step, stage, tap))) is not None
            and float(row["multiset_equal_rate"])
            <= float(gates["identifiable_multiset_equal_rate_max"])
            and float(row["pooled_summary_l1"])
            >= float(gates["identifiable_pooled_summary_l1_min"])
        )
        for seed, step, stage, tap in sorted(_expected_keys())
    }
    failed_protocol = sorted(
        name for name, passed in protocol_checks.items() if not passed
    )
    all_identifiable = bool(research_checks) and all(research_checks.values())
    if failed_protocol:
        status = "invalid"
        decision = config["decisions"]["invalid"]
        next_action = (
            "Repair only the failed K1-BY4 binding, affine bijection, program geometry, "
            "cache, histogram or artifact invariant and rerun unchanged."
        )
    elif all_identifiable:
        status = "pass"
        decision = config["decisions"]["pass"]
        next_action = (
            "Preregister K1-BY6 at the identical PRESENT r7 2048/class, 16-pair, "
            "seed2/3 and 10-epoch budget. Train only the affine wrong-control row per "
            "seed and compare with frozen K1-BY3 correct/no-conditioner anchors."
        )
    else:
        status = "hold"
        decision = config["decisions"]["hold"]
        next_action = (
            "Stop endpoint-permutation search and neural scaling. Audit whether source "
            "cell identity must be retained before invariant aggregation while keeping "
            "the shared width-independent parameter geometry."
        )
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
        "all_taps_identifiable": all_identifiable,
        "panels": _panels(mapped),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
        "claim_scope": (
            "Local zero-training deterministic affine P-layer control audit on frozen "
            "PRESENT-80 r7 validation caches; not neural performance, formal scale, "
            "transfer, attack or SOTA evidence."
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
        raise ValueError("K1-BY5 is a frozen local CPU audit")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"K1-BY5 output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _append_progress(output_root / "progress.jsonl", "run_start")
    datasets, bound_paths, source_checks, correct, affine = load_authority(
        config,
        project_root=project_root,
    )
    if not source_checks or not all(source_checks.values()):
        raise ValueError(f"K1-BY5 source binding failed: {source_checks}")
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
        "correct_program_semantic_sha256": correct.semantic_sha256,
        "affine_program_semantic_sha256": affine.semantic_sha256,
        "affine_program_expert_usage": affine.expert_usage,
        "device": device,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    _write_json(output_root / "preflight.json", preflight)
    _append_progress(
        output_root / "progress.jsonl",
        "affine_endpoint_audit_start",
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    result_rows = evaluate(
        config=config,
        datasets=datasets,
        correct=correct,
        affine=affine,
    )
    source_digests_after = {
        name: file_sha256(path) for name, path in bound_paths.items()
    }
    gate = adjudicate(
        config=config,
        result_rows=result_rows,
        source_checks=source_checks,
        source_unchanged=source_digests_before == source_digests_after,
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
        "all_taps_identifiable": gate["all_taps_identifiable"],
        "panels": gate["panels"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
        "result_rows": len(result_rows),
        "optimizer_steps": 0,
    }
    _write_jsonl(output_root / "results.jsonl", result_rows)
    _write_csv(output_root / "condition_comparison.csv", result_rows)
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


def _expected_keys() -> set[tuple[int, int, int, str]]:
    return {
        (seed, step, stage, tap)
        for seed in EXPECTED_SEEDS
        for step, stage in enumerate(EXPECTED_STAGE_INDICES)
        for tap in k1by4.TAPS
    }


def _result_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    fail_closed: bool = True,
) -> dict[tuple[int, int, int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["execution_step"]),
            int(row["source_stage_index"]),
            str(row["tap"]),
        )
        if row.get("condition") != CONTROL or key in mapped:
            raise ValueError(f"invalid or duplicate K1-BY5 result row: {key}")
        mapped[key] = row
    if fail_closed and set(mapped) != _expected_keys():
        raise ValueError("K1-BY5 result matrix is incomplete")
    return mapped


def _row_valid(row: Mapping[str, Any]) -> bool:
    metrics = (
        float(row["multiset_equal_rate"]),
        float(row["pooled_summary_l1"]),
        float(row["ordered_histogram_l1"]),
    )
    return (
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics)
        and int(row["samples_total"]) == 2048
        and int(row["cells"]) == 16
        and int(row["pairs_per_sample"]) == 16
        and row.get("neural_training_performed") is False
        and int(row.get("optimizer_steps", -1)) == 0
    )


def _panels(
    mapped: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "execution_step": step,
            "source_stage_index": stage,
            "tap": tap,
            "multiset_equal_rate": float(row["multiset_equal_rate"]),
            "multiset_change_rate": 1.0 - float(row["multiset_equal_rate"]),
            "pooled_summary_l1": float(row["pooled_summary_l1"]),
            "ordered_histogram_l1": float(row["ordered_histogram_l1"]),
        }
        for seed, step, stage, tap in sorted(_expected_keys())
        if (row := mapped.get((seed, step, stage, tap))) is not None
    ]


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    "CONTROL",
    "EXPECTED_RESULT_ROWS",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "build_affine_program",
    "evaluate",
    "load_and_validate_config",
    "load_authority",
    "run_audit",
]
