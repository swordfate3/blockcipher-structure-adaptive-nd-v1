from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    ControlledInterruption,
    ResumableSearchConfig,
    run_resumable_integral_property_search,
)
from blockcipher_nd.tasks.innovation2.present_sbox4_real_atm_runner_compatibility import (
    ATM_COMMIT,
    SOURCE_HASHES,
    ManagerCountingOracle,
    RealAtmModelBundle,
    _clauses,
    _internal_snapshot,
    _model_contract,
    _real_oracle,
    build_bitset_extension,
    canonical_relations,
    audit_relation_spaces,
    import_real_atm_runtime,
    sha256,
)


RUN_ID = "i2_present_r9_atm_split333_resumable_generation_20260720"
E103_GATE_SHA256 = "8a651da1f16824482ae7653229757d2a3dfde25772a6ded8babe69691f5116e3"
E103_DECISION = "innovation2_present_sbox4_real_atm_compatibility_passed"
SPLIT = (3, 3, 3)
ROUNDS = sum(SPLIT)
LIMIT = 1 << 10
WORKERS = 36
MINIMUM_FREE_BYTES = 500 * 1024**3
PRESENT_SBOX = (0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2)
PRESENT_BIT_PERMUTATION = tuple(
    (16 * bit) % 63 if bit < 63 else 63 for bit in range(64)
)


def build_present_r9_split333_bundle() -> RealAtmModelBundle:
    components = importlib.import_module("Construction.Components")
    compound = importlib.import_module("Construction.CompoundFunction")
    iterated = importlib.import_module("Construction.IteratedCipher")
    formulas = importlib.import_module("pysat.formula")
    present_sbox = components.SBox(4, 4, list(PRESENT_SBOX))
    round_function = compound.CompoundFunction(64, 64)
    sbox_ids = [round_function.add_component(present_sbox) for _ in range(16)]
    for bit in range(64):
        round_function.connect_components(
            compound.INPUT_ID, bit, sbox_ids[bit // 4], bit % 4
        )
        round_function.connect_components(
            sbox_ids[bit // 4],
            bit % 4,
            compound.OUTPUT_ID,
            PRESENT_BIT_PERMUTATION[bit],
        )
    all_key_bits = (1 << 64) - 1
    f1 = iterated.construct_iterated_cipher(
        [round_function] * SPLIT[0], [all_key_bits] * SPLIT[0] + [0]
    )
    f2 = iterated.construct_iterated_cipher(
        [round_function] * SPLIT[1], [all_key_bits] * (SPLIT[1] + 1)
    )
    f3 = iterated.construct_iterated_cipher(
        [round_function] * SPLIT[2], [0] + [all_key_bits] * SPLIT[2]
    )
    pool = formulas.IDPool()
    unified_model, input_vars, intermediate_vars1, _ = (
        f1.optimized_for_nonzero_trail_detection().to_model(pool)
    )
    next_model, _, intermediate_vars2, _ = (
        f2.optimized_for_nonzero_trail_detection().to_model(
            pool, input_vars=intermediate_vars1
        )
    )
    unified_model += next_model
    next_model, _, output_vars, _ = f3.optimized_for_nonzero_trail_detection().to_model(
        pool, input_vars=intermediate_vars2
    )
    unified_model += next_model
    f1_parts = f1.to_model()
    f2_parts = f2.to_model()
    f3_parts = f3.to_model()
    return RealAtmModelBundle(
        unified_model=_clauses(unified_model),
        input_vars=tuple(input_vars),
        intermediate_vars1=tuple(intermediate_vars1),
        intermediate_vars2=tuple(intermediate_vars2),
        output_vars=tuple(output_vars),
        f1_model=_clauses(f1_parts[0]),
        f1_input_vars=tuple(f1_parts[1]),
        f1_output_vars=tuple(f1_parts[2]),
        f1_key_vars=tuple(f1_parts[3]),
        f2_model=_clauses(f2_parts[0]),
        f2_input_vars=tuple(f2_parts[1]),
        f2_output_vars=tuple(f2_parts[2]),
        f2_key_vars=tuple(f2_parts[3]),
        f3_model=_clauses(f3_parts[0]),
        f3_input_vars=tuple(f3_parts[1]),
        f3_output_vars=tuple(f3_parts[2]),
        f3_key_vars=tuple(f3_parts[3]),
    )


def search_config() -> ResumableSearchConfig:
    return ResumableSearchConfig(
        run_id=RUN_ID,
        input_size=64,
        output_size=64,
        is_permutation=True,
        num_workers=WORKERS,
        oracle_id="official_atm_present_r9_split_3_3_3_limit1024",
        source_commit=ATM_COMMIT,
        search_source_sha256=SOURCE_HASHES["Modelling/Search.py"],
        oracle_parameters=(
            ("rounds", str(ROUNDS)),
            ("split", ",".join(map(str, SPLIT))),
            ("limit", str(LIMIT)),
            ("state_bits", "64"),
            ("key_model", "independent_64bit_round_keys"),
            ("qmc_shim", "single_process_cp_sat"),
        ),
    )


def execute_phase(
    output_root: Path,
    *,
    atm_root: Path,
    e103_gate: dict[str, Any],
    e103_gate_sha256: str,
    actual_atm_commit: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"readiness", "probe", "search"}:
        raise ValueError("mode must be readiness, probe, or search")
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "phase_progress.jsonl"
    started = time.perf_counter()
    _append_progress(progress, "phase_start", mode=mode)
    hashes = {relative: sha256(atm_root / relative) for relative in SOURCE_HASHES}
    source_checks = {
        "e103_gate_hash_matches": e103_gate_sha256 == E103_GATE_SHA256,
        "e103_status_pass": e103_gate.get("status") == "pass",
        "e103_decision_matches": e103_gate.get("decision") == E103_DECISION,
        "atm_commit_matches": actual_atm_commit == ATM_COMMIT,
        "all_source_hashes_match": hashes == SOURCE_HASHES,
    }
    if not all(source_checks.values()):
        return _phase_failure(
            output_root,
            mode=mode,
            decision="innovation2_present_r9_split333_source_protocol_invalid",
            source_checks=source_checks,
            next_action="repair frozen source replay only",
        )
    free_bytes = shutil.disk_usage(output_root).free
    build = build_bitset_extension(atm_root)
    runtime = import_real_atm_runtime(atm_root)
    model_started = time.perf_counter()
    bundle = build_present_r9_split333_bundle()
    model_seconds = time.perf_counter() - model_started
    model_contract = _model_contract(bundle)
    environment_checks = {
        "disk_free_at_least_500gib": free_bytes >= MINIMUM_FREE_BYTES,
        "compiled_extension_exists": Path(build["extension_path"]).is_file(),
        "compiled_extension_imported": Path(runtime["bitset_module"]).resolve()
        == Path(build["extension_path"]).resolve(),
        "python_extension_suffix_matches": Path(build["extension_path"]).name.endswith(
            str(build["extension_suffix"])
        ),
        "dependency_versions_recorded": len(runtime["dependency_versions"]) == 5,
        "qmc_shim_installed": runtime["qmc_compatibility_shim"]
        == "single_process_cp_sat",
        "full_present_state_width": len(bundle.input_vars)
        == len(bundle.output_vars)
        == 64,
        "split_key_bits_match": (
            len(bundle.f1_key_vars),
            len(bundle.f2_key_vars),
            len(bundle.f3_key_vars),
        )
        == (192, 256, 192),
        "worker_count_frozen": search_config().num_workers == 36,
        "logical_cpu_count_at_least_workers": (os.cpu_count() or 0) >= WORKERS,
    }
    readiness = {
        "run_id": RUN_ID,
        "mode": mode,
        "source_checks": source_checks,
        "environment_checks": environment_checks,
        "source_hashes": hashes,
        "build": build,
        "runtime": runtime,
        "model_contract": model_contract,
        "model_build_seconds": model_seconds,
        "disk_free_bytes": free_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "source_hashes.json", hashes)
    _write_json(output_root / "bitset_build.json", build)
    _write_json(output_root / "runtime_environment.json", runtime)
    _write_json(output_root / "model_contract.json", model_contract)
    _write_json(output_root / "readiness.json", readiness)
    if not all(environment_checks.values()):
        return _phase_failure(
            output_root,
            mode=mode,
            decision="innovation2_present_r9_split333_environment_hold",
            source_checks=source_checks,
            environment_checks=environment_checks,
            next_action="repair Windows ABI, dependencies, disk, model, or worker readiness",
        )
    _write_marker(
        output_root / "readiness_passed.marker",
        {"model_sha256": model_contract["sha256"], "parameter_hash": search_config().parameter_hash()},
    )
    _append_progress(
        progress,
        "readiness_passed",
        model_sha256=model_contract["sha256"],
        model_build_seconds=model_seconds,
    )
    if mode == "readiness":
        gate = {
            "run_id": RUN_ID,
            "status": "pass",
            "decision": "innovation2_present_r9_split333_phase_a_ready",
            "source_checks": source_checks,
            "environment_checks": environment_checks,
            "next_action": {
                "action": "run the ten-minute one-new-candidate probe",
                "probe_open": True,
                "long_search_open": False,
            },
        }
        _write_json(output_root / "phase_a_gate.json", gate)
        _append_progress(progress, "phase_done", mode=mode, status="pass")
        return {"gate": gate, "readiness": readiness}

    config = search_config()
    before_candidates = _candidate_files(output_root / "search_state")
    with multiprocessing.Manager() as manager:
        oracle, internal = _real_oracle(bundle, manager, limit=LIMIT)
        calls = manager.dict()
        counted = ManagerCountingOracle(oracle, calls, manager.Lock())
        phase_started = time.perf_counter()
        controlled = False
        result: dict[str, Any] | None = None
        try:
            result = run_resumable_integral_property_search(
                counted,
                config=config,
                output_root=output_root / "search_state",
                interrupt_after_new_candidates=1 if mode == "probe" else None,
            )
        except ControlledInterruption:
            controlled = True
        phase_seconds = time.perf_counter() - phase_started
        calls_snapshot = _call_rows(dict(calls))
        internal_snapshot = _internal_snapshot(internal)
    after_candidates = _candidate_files(output_root / "search_state")
    reused_events = sum(
        event == "candidate_reused"
        for event in _progress_events(output_root / "search_state/progress.jsonl")
    )
    phase_summary = {
        "run_id": RUN_ID,
        "mode": mode,
        "parameter_hash": config.parameter_hash(),
        "controlled_interruption": controlled,
        "candidate_files_before": len(before_candidates),
        "candidate_files_after": len(after_candidates),
        "new_durable_candidates": len(after_candidates) - len(before_candidates),
        "candidate_bytes": sum(path.stat().st_size for path in after_candidates),
        "candidate_reuse_events_total": reused_events,
        "candidate_calls": calls_snapshot,
        "candidate_call_sum": sum(row["calls"] for row in calls_snapshot),
        "internal": internal_snapshot,
        "phase_seconds": phase_seconds,
        "model_build_seconds": model_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if mode == "probe":
        probe_pass = (
            controlled
            and len(after_candidates) == len(before_candidates) + 1
            and internal_snapshot["oracle_call_sum"] > 0
            and (len(before_candidates) == 0 or reused_events > 0)
        )
        probe_index = len(after_candidates)
        decision = (
            "innovation2_present_r9_split333_probe_candidate_completed"
            if probe_pass
            else "innovation2_present_r9_split333_candidate_boundary_invalid"
        )
        gate = {
            "run_id": RUN_ID,
            "status": "pass" if probe_pass else "hold",
            "decision": decision,
            "metrics": phase_summary,
            "next_action": {
                "action": (
                    "run a fresh-process reuse probe"
                    if probe_pass and len(before_candidates) == 0
                    else (
                        "launch the staged single-split search"
                        if probe_pass
                        else "repair candidate persistence before long search"
                    )
                ),
                "reuse_probe_open": probe_pass and len(before_candidates) == 0,
                "long_search_open": probe_pass and len(before_candidates) > 0,
            },
        }
        _write_json(output_root / f"probe_{probe_index:03d}.json", phase_summary)
        _write_json(output_root / "probe_gate.json", gate)
        if probe_pass:
            _write_marker(
                output_root / f"probe_{probe_index:03d}_passed.marker",
                {
                    "parameter_hash": config.parameter_hash(),
                    "candidate_files": len(after_candidates),
                },
            )
        _append_progress(
            progress,
            "phase_done",
            mode=mode,
            status=gate["status"],
            decision=decision,
        )
        return {"gate": gate, "readiness": readiness, "phase": phase_summary}

    if result is None:
        raise RuntimeError("search mode exited without result or interruption")
    relations = canonical_relations(result["relations"])
    relation_rank = audit_relation_spaces(relations, relations)["official_rank"]
    result_summary = {
        **phase_summary,
        "relations": len(relations),
        "relation_rank": relation_rank,
        "completed_result_reused": result["completed_result_reused"],
    }
    gate = {
        "run_id": RUN_ID,
        "status": "pass",
        "decision": "innovation2_present_r9_split333_generation_passed",
        "metrics": result_summary,
        "claim_scope": (
            "locally generated confirmation relations for the official independent-round-key "
            "PRESENT r9 ATM split (3,3,3); not a published result, PRESENT-80 key schedule, "
            "neural result, distinguisher, attack, or SOTA claim"
        ),
        "next_action": {
            "action": "retrieve and validate the generated relation set before any E105 evaluation",
            "e105_source_heldout_evaluation_open": True,
            "r10_search_open": False,
            "training": False,
        },
    }
    _write_json(output_root / "summary.json", result_summary)
    _write_json(output_root / "gate.json", gate)
    _write_json(
        output_root / "relations.json",
        {
            "relations": [
                [[left, right] for left, right in relation] for relation in relations
            ]
        },
    )
    _write_marker(
        output_root / "generation_passed.marker",
        {
            "parameter_hash": config.parameter_hash(),
            "relations": len(relations),
        },
    )
    _append_progress(progress, "phase_done", mode=mode, status="pass")
    return {"gate": gate, "readiness": readiness, "phase": result_summary}


def _phase_failure(
    output_root: Path,
    *,
    mode: str,
    decision: str,
    source_checks: dict[str, bool],
    next_action: str,
    environment_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    gate = {
        "run_id": RUN_ID,
        "status": "fail" if "source" in decision else "hold",
        "decision": decision,
        "source_checks": source_checks,
        "environment_checks": environment_checks or {},
        "next_action": {"action": next_action, "long_search_open": False},
    }
    _write_json(output_root / f"{mode}_gate.json", gate)
    return {"gate": gate}


def _candidate_files(search_root: Path) -> tuple[Path, ...]:
    return tuple(sorted((search_root / "candidate_results").glob("*.json")))


def _progress_events(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _call_rows(calls: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in calls.items():
        left, right = key.split(":")
        rows.append({"u": int(left), "v": int(right), "calls": int(count)})
    return sorted(rows, key=lambda row: (row["u"], row["v"]))


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    record = {"timestamp": time.time(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
