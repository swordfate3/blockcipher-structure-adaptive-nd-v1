from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    ArtifactIntegrityError,
    ControlledInterruption,
    ParameterMismatchError,
    ResumableSearchConfig,
    run_resumable_integral_property_search,
)


RUN_ID = "i2_present_atm_resumable_search_runner_fixture_20260720"
E101_GATE_SHA256 = "056366dcbe692306c830284348b0a28fbbd4dc685c9dc671237c7bf1a5519933"
E101_DECISION = "innovation2_present_high_round_resumable_runner_required"
ATM_COMMIT = "b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b"
SEARCH_SHA256 = "5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d"


class CountingAtmFixtureOracle:
    def __init__(self) -> None:
        self.calls: Counter[tuple[int, int]] = Counter()

    def __call__(self, coordinate: tuple[int, int]) -> tuple[bool, set[tuple[int, int]]]:
        self.calls[coordinate] += 1
        if coordinate in {(3, 2), (3, 4)}:
            return False, {(1, 1)}
        if coordinate == (5, 1):
            return True, set()
        input_weight = (coordinate[0] ^ 0b111).bit_count()
        if input_weight + coordinate[1].bit_count() == 2:
            return False, set()
        return True, set()


def fixture_config(*, output_size: int = 3) -> ResumableSearchConfig:
    return ResumableSearchConfig(
        run_id=RUN_ID,
        input_size=3,
        output_size=output_size,
        is_permutation=True,
        num_workers=1,
        oracle_id="deterministic_three_bit_atm_fixture_v1",
        source_commit=ATM_COMMIT,
        search_source_sha256=SEARCH_SHA256,
        oracle_parameters=(("fixture", "basis+wuv+key-dependent"),),
    )


def execute_resumable_runner_fixture(
    output_root: Path,
    *,
    actual_atm_commit: str,
    actual_search_sha256: str,
    e101_gate: dict[str, Any],
    e101_gate_sha256: str,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"E102 output root must be fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config = fixture_config()

    anchor_oracle = CountingAtmFixtureOracle()
    anchor = run_resumable_integral_property_search(
        anchor_oracle,
        config=config,
        output_root=output_root / "anchor",
    )

    resumed_oracle = CountingAtmFixtureOracle()
    interrupted = False
    try:
        run_resumable_integral_property_search(
            resumed_oracle,
            config=config,
            output_root=output_root / "resumed",
            interrupt_after_new_candidates=1,
        )
    except ControlledInterruption:
        interrupted = True
    calls_at_interrupt = Counter(resumed_oracle.calls)
    first_completed = next(iter(calls_at_interrupt), None)
    resumed = run_resumable_integral_property_search(
        resumed_oracle,
        config=config,
        output_root=output_root / "resumed",
    )

    parameter_mismatch_rejected = False
    try:
        run_resumable_integral_property_search(
            CountingAtmFixtureOracle(),
            config=replace(config, output_size=4),
            output_root=output_root / "resumed",
        )
    except ParameterMismatchError:
        parameter_mismatch_rejected = True

    corrupt_oracle = CountingAtmFixtureOracle()
    try:
        run_resumable_integral_property_search(
            corrupt_oracle,
            config=config,
            output_root=output_root / "corrupt_recovery",
            interrupt_after_new_candidates=1,
        )
    except ControlledInterruption:
        pass
    corrupt_coordinate = next(iter(corrupt_oracle.calls), None)
    candidate_files = sorted(
        (output_root / "corrupt_recovery/candidate_results").glob("*.json")
    )
    if len(candidate_files) != 1:
        raise RuntimeError("corruption fixture expected exactly one durable candidate")
    candidate_files[0].write_text('{"payload":', encoding="utf-8")
    incomplete = output_root / "corrupt_recovery/candidate_results/.unfinished.tmp"
    incomplete.write_text("partial", encoding="utf-8")
    corrupt_recovery = run_resumable_integral_property_search(
        corrupt_oracle,
        config=config,
        output_root=output_root / "corrupt_recovery",
    )
    incomplete_ignored = incomplete.exists()
    incomplete.unlink()

    completed_reuse_oracle = CountingAtmFixtureOracle()
    completed_reuse = run_resumable_integral_property_search(
        completed_reuse_oracle,
        config=config,
        output_root=output_root / "resumed",
    )

    anchor_result = output_root / "anchor/result.json"
    resumed_result = output_root / "resumed/result.json"
    result_bytes_equal = anchor_result.read_bytes() == resumed_result.read_bytes()
    call_rows = _call_rows(
        anchor_oracle.calls,
        resumed_oracle.calls,
        corrupt_oracle.calls,
    )
    artifact_rows = _artifact_contract_rows(output_root)
    fixture_checks = {
        "controlled_interrupt_observed": interrupted,
        "resume_event_recorded": "resume_start" in _progress_events(
            output_root / "resumed/progress.jsonl"
        ),
        "canonical_result_bytes_equal": result_bytes_equal,
        "mathematical_relations_equal": resumed["relations"] == anchor["relations"],
        "completed_candidate_not_recalled": (
            first_completed is not None
            and calls_at_interrupt[first_completed] == 1
            and resumed_oracle.calls[first_completed] == 1
        ),
        "total_oracle_calls_match_anchor": (
            sum(resumed_oracle.calls.values()) == sum(anchor_oracle.calls.values())
        ),
        "parameter_mismatch_rejected": parameter_mismatch_rejected,
        "corrupt_candidate_recomputed": (
            corrupt_coordinate is not None
            and corrupt_oracle.calls[corrupt_coordinate] == 2
            and corrupt_recovery["rejected_candidate_artifacts"] == 1
        ),
        "incomplete_temporary_candidate_ignored": incomplete_ignored,
        "completed_result_hash_reuse_zero_calls": (
            completed_reuse["completed_result_reused"]
            and sum(completed_reuse_oracle.calls.values()) == 0
        ),
        "basis_path_covered": resumed["basis_candidates"] > 0,
        "wuv_path_covered": resumed["wuv_candidates"] > 0,
        "key_dependent_path_covered": resumed["key_dependent_candidates"] > 0,
        "all_artifact_contract_rows_pass": all(row["passed"] for row in artifact_rows),
        "result_precedes_complete_marker": all(
            (output_root / variant / "result.json").stat().st_mtime_ns
            <= (output_root / variant / "complete.marker").stat().st_mtime_ns
            for variant in ("anchor", "resumed", "corrupt_recovery")
        ),
    }
    source_checks = {
        "e101_gate_hash_matches": e101_gate_sha256 == E101_GATE_SHA256,
        "e101_status_hold": e101_gate.get("status") == "hold",
        "e101_decision_matches": e101_gate.get("decision") == E101_DECISION,
        "atm_commit_matches": actual_atm_commit == ATM_COMMIT,
        "official_search_hash_matches": actual_search_sha256 == SEARCH_SHA256,
    }
    coverage_checks = {
        name: fixture_checks[name]
        for name in (
            "basis_path_covered",
            "wuv_path_covered",
            "key_dependent_path_covered",
        )
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_present_atm_resumable_runner_protocol_invalid"
        action = "repair E101 or frozen ATM source replay before rerunning the fixture"
    elif not all(coverage_checks.values()):
        status = "hold"
        decision = "innovation2_present_atm_resumable_runner_fixture_insufficient"
        action = "expand the local fixture to cover basis, WUV, and key-dependent paths"
    elif not all(fixture_checks.values()):
        status = "hold"
        decision = "innovation2_present_atm_resumable_runner_protocol_invalid"
        action = "repair persistence, resume, integrity, or equivalence behavior only"
    else:
        status = "pass"
        decision = "innovation2_present_atm_resumable_runner_fixture_passed"
        action = (
            "preregister an E103 bounded real-ATM low-round compatibility gate; keep R9/R10 "
            "long search and remote execution closed"
        )
    metrics = {
        "anchor_oracle_calls": sum(anchor_oracle.calls.values()),
        "resumed_oracle_calls": sum(resumed_oracle.calls.values()),
        "calls_durable_at_interrupt": sum(calls_at_interrupt.values()),
        "resumed_reused_candidates": resumed["reused_candidate_results"],
        "corrupt_recovery_oracle_calls": sum(corrupt_oracle.calls.values()),
        "corrupt_artifacts_rejected": corrupt_recovery[
            "rejected_candidate_artifacts"
        ],
        "relations": len(anchor["relations"]),
        "basis_candidates": resumed["basis_candidates"],
        "wuv_candidates": resumed["wuv_candidates"],
        "key_dependent_candidates": resumed["key_dependent_candidates"],
        "fixture_checks": len(fixture_checks),
        "fixture_checks_passed": sum(fixture_checks.values()),
        "source_checks": len(source_checks),
        "source_checks_passed": sum(source_checks.values()),
        "artifact_checks": len(artifact_rows),
        "artifact_checks_passed": sum(row["passed"] for row in artifact_rows),
    }
    gate = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "fixture_checks": fixture_checks,
        "metrics": metrics,
        "claim_scope": (
            "local deterministic fixture evidence for route-owned incremental ATM search "
            "persistence, resume, and integrity only; no PRESENT r9/r10 model or search, no "
            "new relation, training, independent-round-key result, PRESENT-80 distinguisher, "
            "attack, remote evidence, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "real_atm_bounded_compatibility_open": status == "pass",
            "r9_missing_split_search_open": False,
            "r10_search_open": False,
            "remote_scale": False,
            "training": False,
        },
    }
    return {
        "run_id": RUN_ID,
        "config": config.parameter_payload(),
        "gate": gate,
        "source_checks": source_checks,
        "fixture_checks": fixture_checks,
        "call_rows": call_rows,
        "artifact_rows": artifact_rows,
        "anchor": _serializable_run(anchor),
        "resumed": _serializable_run(resumed),
        "corrupt_recovery": _serializable_run(corrupt_recovery),
    }


def result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    gate = summary["gate"]
    common = {
        "run_id": RUN_ID,
        "task": "innovation2_present_atm_resumable_search_runner_fixture",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_performed": False,
        "search_scope": "deterministic_three_bit_fixture_only",
    }
    return [
        {
            **common,
            "result_kind": "source_check",
            "check": name,
            "passed": passed,
        }
        for name, passed in summary["source_checks"].items()
    ] + [
        {
            **common,
            "result_kind": "fixture_check",
            "check": name,
            "passed": passed,
        }
        for name, passed in summary["fixture_checks"].items()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_rows(
    anchor: Counter[tuple[int, int]],
    resumed: Counter[tuple[int, int]],
    corrupt: Counter[tuple[int, int]],
) -> list[dict[str, Any]]:
    coordinates = sorted(set(anchor) | set(resumed) | set(corrupt))
    return [
        {
            "u": coordinate[0],
            "v": coordinate[1],
            "anchor_calls": anchor[coordinate],
            "resumed_calls": resumed[coordinate],
            "corrupt_recovery_calls": corrupt[coordinate],
        }
        for coordinate in coordinates
    ]


def _artifact_contract_rows(output_root: Path) -> list[dict[str, Any]]:
    required = (
        "metadata.json",
        "started.marker",
        "progress.jsonl",
        "result.json",
        "complete.marker",
    )
    rows = [
        {
            "variant": variant,
            "artifact": artifact,
            "passed": (output_root / variant / artifact).is_file()
            and (output_root / variant / artifact).stat().st_size > 0,
        }
        for variant in ("anchor", "resumed", "corrupt_recovery")
        for artifact in required
    ]
    rows.extend(
        {
            "variant": variant,
            "artifact": "candidate_results/*.json",
            "passed": any((output_root / variant / "candidate_results").glob("*.json")),
        }
        for variant in ("anchor", "resumed", "corrupt_recovery")
    )
    return rows


def _progress_events(path: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _serializable_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        **run,
        "relations": [
            [[left, right] for left, right in relation]
            for relation in run["relations"]
        ],
    }
