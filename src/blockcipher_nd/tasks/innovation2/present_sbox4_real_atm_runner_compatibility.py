from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import pybind11

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    install_single_process_qmc_compatibility_shim,
)
from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    ControlledInterruption,
    ResumableSearchConfig,
    run_resumable_integral_property_search,
)


RUN_ID = "i2_present_sbox4_r3_real_atm_runner_compatibility_20260720"
E102_GATE_SHA256 = "91188f35d5c2fa24c26fcefb4cde1045356f60121ba0e4f5a32c7eef3fa4b738"
E102_DECISION = "innovation2_present_atm_resumable_runner_fixture_passed"
ATM_COMMIT = "b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b"
SOURCE_HASHES = {
    "Modelling/Search.py": "5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d",
    "Tools/AvecImplementations.py": "c0a9dd19f4c92e8ceeccde01b6e13de9599e05f2531ffb0c3ce7c31a9aacccae",
    "Construction/Components.py": "28e17729b5c74ee752c2c105b92af952c6215d34d6e4b248eff9d3fc77b06caf",
    "Construction/CompoundFunction.py": "b64fe879e707ac89d5eb8a68face50f143ff2f53f636e469b821a35c43fbdee6",
    "Construction/IteratedCipher.py": "3bfaa5a9f76cb78b604884028b06c87ae643a2dd1c8a8c53fffcf5adc1a73bdb",
    "Modelling/Trails.py": "9852c1facd19b35b5edcef2165dab7c43873d25e1fdfa1e7c0999e3f98634346",
    "bitarrays/src/bitset.cpp": "5bb16abc1102fa4bfe0973c7a95737efffd2f2aec7ff1c75334c37b8d3992094",
    "bitarrays/src/bitset.hpp": "6f5a758a6acbda04784fe6160e212406f0f43e0a078e410aa1ea7f199a107d8d",
}
PRESENT_SBOX = (0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2)

Coordinate = tuple[int, int]
Relation = tuple[Coordinate, ...]


@dataclass(frozen=True)
class RealAtmModelBundle:
    unified_model: tuple[tuple[int, ...], ...]
    input_vars: tuple[int, ...]
    intermediate_vars1: tuple[int, ...]
    intermediate_vars2: tuple[int, ...]
    output_vars: tuple[int, ...]
    f1_model: tuple[tuple[int, ...], ...]
    f1_input_vars: tuple[int, ...]
    f1_output_vars: tuple[int, ...]
    f1_key_vars: tuple[int, ...]
    f2_model: tuple[tuple[int, ...], ...]
    f2_input_vars: tuple[int, ...]
    f2_output_vars: tuple[int, ...]
    f2_key_vars: tuple[int, ...]
    f3_model: tuple[tuple[int, ...], ...]
    f3_input_vars: tuple[int, ...]
    f3_output_vars: tuple[int, ...]
    f3_key_vars: tuple[int, ...]


class ManagerCountingOracle:
    def __init__(self, delegate: Any, calls: Any, lock: Any) -> None:
        self.delegate = delegate
        self.calls = calls
        self.lock = lock

    def __call__(self, coordinate: Coordinate) -> Any:
        key = f"{coordinate[0]}:{coordinate[1]}"
        with self.lock:
            self.calls[key] = int(self.calls.get(key, 0)) + 1
        return self.delegate(coordinate)


def build_bitset_extension(
    atm_root: Path,
    *,
    compiler: str | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    source = atm_root / "bitarrays/src/bitset.cpp"
    header = atm_root / "bitarrays/src/bitset.hpp"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python EXT_SUFFIX is unavailable")
    output = atm_root / "bitarrays" / f"bitset{extension_suffix}"
    platform_name = platform_name or os.name
    compiler = compiler or ("cl.exe" if platform_name == "nt" else "g++")
    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}{output.suffix}"
    )
    includes = tuple(
        dict.fromkeys(
            value
            for value in (
                sysconfig.get_path("include"),
                pybind11.get_include(),
                pybind11.get_include(user=True),
            )
            if value
        )
    )
    command, auxiliary_paths = _bitset_build_command(
        platform_name=platform_name,
        compiler=compiler,
        source=source,
        output=temporary,
        includes=includes,
        build_root=atm_root / "bitarrays/.build",
    )
    output_encoding = "mbcs" if platform_name == "nt" else "utf-8"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding=output_encoding,
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "bitset extension compilation failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    version_command = [compiler] if platform_name == "nt" else [compiler, "--version"]
    version_result = subprocess.run(
        version_command,
        check=False,
        capture_output=True,
        text=True,
        encoding=output_encoding,
        errors="replace",
    )
    version_lines = [
        line.strip()
        for line in (version_result.stdout + "\n" + version_result.stderr).splitlines()
        if line.strip()
    ]
    if not version_lines:
        raise RuntimeError("compiler version output is empty")
    compiler_version = _compiler_version_line(version_lines)
    if shutil.which("file"):
        file_type = subprocess.run(
            ["file", str(output)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        file_type = "Windows CPython extension (.pyd)"
    return {
        "python_version": sys.version.splitlines()[0],
        "python_cache_tag": sys.implementation.cache_tag,
        "extension_suffix": extension_suffix,
        "compiler_version": compiler_version,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "seconds": time.perf_counter() - started,
        "source_sha256": sha256(source),
        "header_sha256": sha256(header),
        "source_normalized_lf_sha256": normalized_lf_sha256(source),
        "header_normalized_lf_sha256": normalized_lf_sha256(header),
        "extension_path": str(output),
        "extension_sha256": sha256(output),
        "extension_bytes": output.stat().st_size,
        "file_type": file_type,
        "platform_name": platform_name,
        "auxiliary_paths": [str(path) for path in auxiliary_paths],
    }


def _bitset_build_command(
    *,
    platform_name: str,
    compiler: str,
    source: Path,
    output: Path,
    includes: Sequence[str],
    build_root: Path,
) -> tuple[list[str], tuple[Path, ...]]:
    if platform_name != "nt":
        return (
            [
                compiler,
                "-O3",
                "-Wall",
                "-shared",
                "-std=c++2a",
                "-DNDEBUG",
                "-funroll-loops",
                "-fvisibility=hidden",
                "-fPIC",
                *(item for include in includes for item in ("-I", include)),
                str(source),
                "-o",
                str(output),
            ],
            (),
        )
    build_root.mkdir(parents=True, exist_ok=True)
    python_root = Path(sysconfig.get_config_var("installed_base") or sys.prefix)
    library_name = sysconfig.get_config_var("LDLIBRARY") or (
        f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    )
    object_path = build_root / "bitset.obj"
    pdb_path = build_root / "bitset.pdb"
    import_library = build_root / "bitset.lib"
    command = [
        compiler,
        "/nologo",
        "/O2",
        "/LD",
        "/std:c++20",
        "/EHsc",
        "/DNDEBUG",
        *(f"/I{include}" for include in includes),
        f"/Fo{object_path}",
        f"/Fd{pdb_path}",
        str(source),
        "/link",
        f"/OUT:{output}",
        f"/IMPLIB:{import_library}",
        f"/LIBPATH:{python_root / 'libs'}",
        library_name,
    ]
    return command, (object_path, pdb_path, import_library)


def _compiler_version_line(lines: Sequence[str]) -> str:
    return next(
        (
            line
            for line in lines
            if "Microsoft" in line and "C/C++" in line
        ),
        lines[0],
    )


def import_real_atm_runtime(atm_root: Path) -> dict[str, Any]:
    root_text = str(atm_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    bitset = importlib.import_module("bitarrays.bitset")
    install_single_process_qmc_compatibility_shim()
    distributions = ("galois", "ortools", "python-sat", "pybind11", "numpy")
    return {
        "bitset_module": str(Path(bitset.__file__).resolve()),
        "dependency_versions": {
            distribution: importlib.metadata.version(distribution)
            for distribution in distributions
        },
        "qmc_compatibility_shim": "single_process_cp_sat",
    }


def build_present_sbox4_model_bundle() -> RealAtmModelBundle:
    components = importlib.import_module("Construction.Components")
    compound = importlib.import_module("Construction.CompoundFunction")
    iterated = importlib.import_module("Construction.IteratedCipher")
    formulas = importlib.import_module("pysat.formula")
    sbox = components.SBox(4, 4, list(PRESENT_SBOX))
    round_function = compound.CompoundFunction(4, 4)
    sbox_id = round_function.add_component(sbox)
    for bit in range(4):
        round_function.connect_components(compound.INPUT_ID, bit, sbox_id, bit)
        round_function.connect_components(sbox_id, bit, compound.OUTPUT_ID, bit)
    all_key_bits = (1 << 4) - 1
    f1 = iterated.construct_iterated_cipher([round_function], [all_key_bits, 0])
    f2 = iterated.construct_iterated_cipher(
        [round_function], [all_key_bits, all_key_bits]
    )
    f3 = iterated.construct_iterated_cipher([round_function], [0, all_key_bits])

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


def execute_real_atm_runner_compatibility(
    output_root: Path,
    *,
    atm_root: Path,
    actual_atm_commit: str,
    e102_gate: dict[str, Any],
    e102_gate_sha256: str,
    hard_cap_seconds: int = 180,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"E103 output root must be fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes = {relative: sha256(atm_root / relative) for relative in SOURCE_HASHES}
    build = build_bitset_extension(atm_root)
    runtime = import_real_atm_runtime(atm_root)
    bundle_started = time.perf_counter()
    bundle = build_present_sbox4_model_bundle()
    model_seconds = time.perf_counter() - bundle_started
    model_contract = _model_contract(bundle)

    official_search = importlib.import_module("Modelling.Search").search_integral_properties
    with multiprocessing.Manager() as manager:
        anchor_oracle, anchor_internal = _real_oracle(bundle, manager, limit=1 << 6)
        anchor_calls = manager.dict()
        anchor_counted = ManagerCountingOracle(anchor_oracle, anchor_calls, manager.Lock())
        anchor_started = time.perf_counter()
        anchor_raw = official_search(anchor_counted, 4, 4, True, 2)
        anchor_seconds = time.perf_counter() - anchor_started
        anchor_call_snapshot = dict(anchor_calls)
        anchor_internal_snapshot = _internal_snapshot(anchor_internal)

    config = ResumableSearchConfig(
        run_id=RUN_ID,
        input_size=4,
        output_size=4,
        is_permutation=True,
        num_workers=2,
        oracle_id="official_atm_present_sbox4_r3_split_1_1_1_limit64",
        source_commit=ATM_COMMIT,
        search_source_sha256=SOURCE_HASHES["Modelling/Search.py"],
        oracle_parameters=(
            ("rounds", "3"),
            ("split", "1,1,1"),
            ("limit", "64"),
            ("state_bits", "4"),
            ("qmc_shim", "single_process_cp_sat"),
        ),
    )
    with multiprocessing.Manager() as manager:
        runner_oracle, runner_internal = _real_oracle(bundle, manager, limit=1 << 6)
        runner_calls = manager.dict()
        counted_runner = ManagerCountingOracle(
            runner_oracle, runner_calls, manager.Lock()
        )
        interrupted = False
        try:
            run_resumable_integral_property_search(
                counted_runner,
                config=config,
                output_root=output_root / "resumed_runner",
                interrupt_after_new_candidates=1,
            )
        except ControlledInterruption:
            interrupted = True
        first_candidate = _first_candidate(output_root / "resumed_runner")
        calls_after_interrupt = dict(runner_calls)
        runner_started = time.perf_counter()
        runner = run_resumable_integral_property_search(
            counted_runner,
            config=config,
            output_root=output_root / "resumed_runner",
        )
        runner_resume_seconds = time.perf_counter() - runner_started
        runner_call_snapshot = dict(runner_calls)
        runner_internal_snapshot = _internal_snapshot(runner_internal)

    anchor_relations = canonical_relations(anchor_raw)
    runner_relations = canonical_relations(runner["relations"])
    relation_audit = audit_relation_spaces(anchor_relations, runner_relations)
    progress_events = _progress_events(
        output_root / "resumed_runner/progress.jsonl"
    )
    source_checks = {
        "e102_gate_hash_matches": e102_gate_sha256 == E102_GATE_SHA256,
        "e102_status_pass": e102_gate.get("status") == "pass",
        "e102_decision_matches": e102_gate.get("decision") == E102_DECISION,
        "atm_commit_matches": actual_atm_commit == ATM_COMMIT,
        "all_source_hashes_match": hashes == SOURCE_HASHES,
    }
    environment_checks = {
        "bitset_source_hash_matches": build["source_sha256"]
        == SOURCE_HASHES["bitarrays/src/bitset.cpp"],
        "bitset_header_hash_matches": build["header_sha256"]
        == SOURCE_HASHES["bitarrays/src/bitset.hpp"],
        "compiled_extension_exists": Path(build["extension_path"]).is_file(),
        "compiled_extension_hash_recorded": len(build["extension_sha256"]) == 64,
        "compiled_extension_imported": Path(runtime["bitset_module"]).resolve()
        == Path(build["extension_path"]).resolve(),
        "dependency_versions_recorded": len(runtime["dependency_versions"]) == 5,
        "qmc_shim_installed": runtime["qmc_compatibility_shim"]
        == "single_process_cp_sat",
    }
    runtime_checks = {
        "controlled_interrupt_observed": interrupted,
        "resume_event_recorded": "resume_start" in progress_events,
        "candidate_reused": runner["reused_candidate_results"] >= 1,
        "completed_candidate_not_recalled": (
            first_candidate is not None
            and calls_after_interrupt.get(_coordinate_key(first_candidate), 0)
            == runner_call_snapshot.get(_coordinate_key(first_candidate), 0)
        ),
        "two_worker_runner_used": config.num_workers == 2,
        "official_manager_calls_nonzero": sum(anchor_call_snapshot.values()) > 0,
        "runner_manager_calls_nonzero": sum(runner_call_snapshot.values()) > 0,
        "official_internal_oracle_activity_nonzero": anchor_internal_snapshot[
            "oracle_call_sum"
        ]
        > 0,
        "runner_internal_oracle_activity_nonzero": runner_internal_snapshot[
            "oracle_call_sum"
        ]
        > 0,
        "relation_space_rank_equal": relation_audit["rank_equal"],
        "anchor_span_in_runner": relation_audit["anchor_span_in_runner"],
        "runner_span_in_anchor": relation_audit["runner_span_in_anchor"],
        "singleton_relations_equal": relation_audit["singleton_relations_equal"],
        "within_hard_cap": time.perf_counter() - started < hard_cap_seconds,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_present_sbox4_real_atm_source_protocol_invalid"
        action = "repair the frozen E102 or ATM source replay"
    elif not all(environment_checks.values()):
        status = "hold"
        decision = "innovation2_present_sbox4_real_atm_environment_incompatible"
        action = "repair the reproducible bitset, dependency, or QMC runtime"
    elif not runtime_checks["within_hard_cap"]:
        status = "hold"
        decision = "innovation2_present_sbox4_real_atm_resource_cap_hit"
        action = "reduce only the real-ATM compatibility fixture before any R9 work"
    elif not all(runtime_checks.values()):
        status = "hold"
        decision = "innovation2_present_sbox4_real_atm_runner_mismatch"
        action = "repair real-oracle multiprocessing, resume, or relation-space equivalence"
    else:
        status = "pass"
        decision = "innovation2_present_sbox4_real_atm_compatibility_passed"
        action = (
            "preregister an E104 R9 (3,3,3) single-split generation plan with target-machine "
            "ABI, disk, cache-capacity, timeout, cancellation, and monitor gates"
        )
    total_seconds = time.perf_counter() - started
    metrics = {
        "bitset_build_seconds": build["seconds"],
        "model_build_seconds": model_seconds,
        "official_anchor_seconds": anchor_seconds,
        "runner_resume_seconds": runner_resume_seconds,
        "total_seconds": total_seconds,
        "hard_cap_seconds": hard_cap_seconds,
        "official_candidate_calls": sum(anchor_call_snapshot.values()),
        "runner_candidate_calls": sum(runner_call_snapshot.values()),
        "runner_calls_at_interrupt": sum(calls_after_interrupt.values()),
        "runner_reused_candidates": runner["reused_candidate_results"],
        "official_internal_oracle_call_sum": anchor_internal_snapshot[
            "oracle_call_sum"
        ],
        "runner_internal_oracle_call_sum": runner_internal_snapshot[
            "oracle_call_sum"
        ],
        "official_relations": len(anchor_relations),
        "runner_relations": len(runner_relations),
        **relation_audit,
    }
    gate = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "environment_checks": environment_checks,
        "runtime_checks": runtime_checks,
        "metrics": metrics,
        "claim_scope": (
            "local runtime compatibility evidence on a 4-bit three-round PRESENT S-box slice "
            "using independent 4-bit round keys; not the 64-bit PRESENT round function, P-layer, "
            "PRESENT-80 key schedule, r9/r10 relation generation, distinguisher, attack, remote "
            "evidence, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "r9_single_split_plan_open": status == "pass",
            "r9_search_started": False,
            "r10_search_open": False,
            "remote_scale": False,
            "training": False,
        },
    }
    return {
        "run_id": RUN_ID,
        "gate": gate,
        "source_hashes": hashes,
        "build": build,
        "runtime": runtime,
        "model_contract": model_contract,
        "relation_audit": relation_audit,
        "anchor_relations": _serializable_relations(anchor_relations),
        "runner_relations": _serializable_relations(runner_relations),
        "anchor_candidate_calls": _call_rows(anchor_call_snapshot),
        "runner_candidate_calls": _call_rows(runner_call_snapshot),
        "anchor_internal": anchor_internal_snapshot,
        "runner_internal": runner_internal_snapshot,
    }


def audit_relation_spaces(
    anchor: Sequence[Relation],
    runner: Sequence[Relation],
) -> dict[str, Any]:
    coordinates = tuple(
        sorted(
            {
                coordinate
                for relation in (*tuple(anchor), *tuple(runner))
                for coordinate in relation
            }
        )
    )
    index = {coordinate: position for position, coordinate in enumerate(coordinates)}
    anchor_rows = tuple(_relation_word(relation, index) for relation in anchor)
    runner_rows = tuple(_relation_word(relation, index) for relation in runner)
    anchor_pivots = _gf2_pivots(anchor_rows)
    runner_pivots = _gf2_pivots(runner_rows)
    anchor_in_runner = all(_reduce_word(row, runner_pivots) == 0 for row in anchor_rows)
    runner_in_anchor = all(_reduce_word(row, anchor_pivots) == 0 for row in runner_rows)
    anchor_singletons = {relation for relation in anchor if len(relation) == 1}
    runner_singletons = {relation for relation in runner if len(relation) == 1}
    return {
        "support_coordinates": len(coordinates),
        "official_rank": len(anchor_pivots),
        "runner_rank": len(runner_pivots),
        "rank_equal": len(anchor_pivots) == len(runner_pivots),
        "anchor_span_in_runner": anchor_in_runner,
        "runner_span_in_anchor": runner_in_anchor,
        "singleton_relations_equal": anchor_singletons == runner_singletons,
        "official_singletons": len(anchor_singletons),
        "runner_singletons": len(runner_singletons),
    }


def canonical_relations(relations: Iterable[Iterable[Coordinate]]) -> tuple[Relation, ...]:
    canonical = {
        tuple(sorted((int(left), int(right)) for left, right in relation))
        for relation in relations
    }
    return tuple(sorted(canonical, key=lambda relation: (len(relation), relation)))


def result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    gate = summary["gate"]
    common = {
        "run_id": RUN_ID,
        "task": "innovation2_present_sbox4_real_atm_runner_compatibility",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
        "search_scope": "present_sbox4_three_round_real_atm_compatibility_only",
    }
    rows: list[dict[str, Any]] = []
    for kind in ("source_checks", "environment_checks", "runtime_checks"):
        rows.extend(
            {
                **common,
                "result_kind": kind.removesuffix("s"),
                "check": name,
                "passed": passed,
            }
            for name, passed in gate[kind].items()
        )
    return rows


def _real_oracle(
    bundle: RealAtmModelBundle,
    manager: Any,
    *,
    limit: int,
) -> tuple[Any, dict[str, Any]]:
    implementation = importlib.import_module("Tools.AvecImplementations")
    caches = [manager.dict() for _ in range(6)]
    oracle_call_counts = manager.list()
    oracle = partial(
        implementation.Avec_unified_model_with_partial_trail_counting_constant,
        *caches,
        oracle_call_counts,
        bundle.unified_model,
        bundle.input_vars,
        bundle.intermediate_vars1,
        bundle.intermediate_vars2,
        bundle.output_vars,
        bundle.f1_model,
        bundle.f1_input_vars,
        bundle.f1_output_vars,
        bundle.f1_key_vars,
        bundle.f2_model,
        bundle.f2_input_vars,
        bundle.f2_output_vars,
        bundle.f2_key_vars,
        bundle.f3_model,
        bundle.f3_input_vars,
        bundle.f3_output_vars,
        bundle.f3_key_vars,
        limit=limit,
    )
    return oracle, {"caches": caches, "oracle_call_counts": oracle_call_counts}


def _internal_snapshot(internal: dict[str, Any]) -> dict[str, Any]:
    counts = list(internal["oracle_call_counts"])
    return {
        "cache_sizes": [len(cache) for cache in internal["caches"]],
        "oracle_call_entries": len(counts),
        "oracle_call_sum": sum(int(value) for value in counts),
    }


def _model_contract(bundle: RealAtmModelBundle) -> dict[str, Any]:
    payload = {
        "unified_model": bundle.unified_model,
        "input_vars": bundle.input_vars,
        "intermediate_vars1": bundle.intermediate_vars1,
        "intermediate_vars2": bundle.intermediate_vars2,
        "output_vars": bundle.output_vars,
        "f1_model": bundle.f1_model,
        "f2_model": bundle.f2_model,
        "f3_model": bundle.f3_model,
    }
    return {
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "unified_clauses": len(bundle.unified_model),
        "unified_variables": max(
            abs(literal) for clause in bundle.unified_model for literal in clause
        ),
        "f1_clauses": len(bundle.f1_model),
        "f2_clauses": len(bundle.f2_model),
        "f3_clauses": len(bundle.f3_model),
        "f1_key_bits": len(bundle.f1_key_vars),
        "f2_key_bits": len(bundle.f2_key_vars),
        "f3_key_bits": len(bundle.f3_key_vars),
    }


def _first_candidate(root: Path) -> Coordinate | None:
    candidates = sorted((root / "candidate_results").glob("*.json"))
    if not candidates:
        return None
    envelope = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload = envelope["payload"]
    return int(payload["u"]), int(payload["v"])


def _progress_events(path: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _coordinate_key(coordinate: Coordinate) -> str:
    return f"{coordinate[0]}:{coordinate[1]}"


def _call_rows(calls: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, count in calls.items():
        left, right = key.split(":")
        rows.append({"u": int(left), "v": int(right), "calls": int(count)})
    return sorted(rows, key=lambda row: (row["u"], row["v"]))


def _serializable_relations(relations: Sequence[Relation]) -> list[list[list[int]]]:
    return [
        [[left, right] for left, right in relation] for relation in relations
    ]


def _relation_word(relation: Relation, index: dict[Coordinate, int]) -> int:
    return sum(1 << index[coordinate] for coordinate in relation)


def _gf2_pivots(rows: Iterable[int]) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for row in rows:
        value = int(row)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return pivots


def _reduce_word(word: int, pivots: dict[int, int]) -> int:
    value = int(word)
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            return value
        value ^= pivots[pivot]
    return 0


def _clauses(values: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(literal) for literal in clause) for clause in values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
