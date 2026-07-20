from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from blockcipher_nd.tasks.innovation2.atm_resumable_search_runner import (
    ArtifactIntegrityError,
    ControlledInterruption,
    ParameterMismatchError,
    ResumableSearchConfig,
    _gf2_nullspace,
    run_resumable_integral_property_search,
)


ATM_COMMIT = "b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b"
SEARCH_SHA256 = "5d9a5c117d7f0940473c15dded0e3243dc06a6182fe584733e0badc8928f459d"


class CountingFixtureOracle:
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


def _config(*, workers: int = 1) -> ResumableSearchConfig:
    return ResumableSearchConfig(
        run_id="i2_present_atm_resumable_search_runner_fixture_20260720",
        input_size=3,
        output_size=3,
        is_permutation=True,
        num_workers=workers,
        oracle_id="deterministic_three_bit_atm_fixture_v1",
        source_commit=ATM_COMMIT,
        search_source_sha256=SEARCH_SHA256,
        oracle_parameters=(("fixture", "basis+wuv+key-dependent"),),
    )


def _events(path: Path) -> list[str]:
    return [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_resume_matches_uninterrupted_and_skips_durable_candidate(tmp_path: Path) -> None:
    config = _config()
    anchor_oracle = CountingFixtureOracle()
    anchor = run_resumable_integral_property_search(
        anchor_oracle,
        config=config,
        output_root=tmp_path / "anchor",
    )
    resumed_oracle = CountingFixtureOracle()
    with pytest.raises(ControlledInterruption):
        run_resumable_integral_property_search(
            resumed_oracle,
            config=config,
            output_root=tmp_path / "resumed",
            interrupt_after_new_candidates=1,
        )
    first_coordinate = next(iter(resumed_oracle.calls))
    assert resumed_oracle.calls[first_coordinate] == 1
    resumed = run_resumable_integral_property_search(
        resumed_oracle,
        config=config,
        output_root=tmp_path / "resumed",
    )
    assert resumed_oracle.calls[first_coordinate] == 1
    assert resumed["relations"] == anchor["relations"]
    assert ((3, 2), (3, 4)) in resumed["relations"]
    assert (tmp_path / "anchor/result.json").read_bytes() == (
        tmp_path / "resumed/result.json"
    ).read_bytes()
    assert sum(anchor_oracle.calls.values()) == sum(resumed_oracle.calls.values())
    assert resumed["reused_candidate_results"] == 1
    assert resumed["basis_candidates"] > 0
    assert resumed["wuv_candidates"] > 0
    assert resumed["key_dependent_candidates"] > 0
    events = _events(tmp_path / "resumed/progress.jsonl")
    assert "controlled_interrupt" in events
    assert "resume_start" in events
    assert events[-1] == "run_complete"
    assert (tmp_path / "resumed/started.marker").is_file()
    assert (tmp_path / "resumed/complete.marker").is_file()
    assert (tmp_path / "resumed/result.json").stat().st_mtime_ns <= (
        tmp_path / "resumed/complete.marker"
    ).stat().st_mtime_ns


def test_parameter_mismatch_is_rejected_before_reuse(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(ControlledInterruption):
        run_resumable_integral_property_search(
            CountingFixtureOracle(),
            config=_config(),
            output_root=output,
            interrupt_after_new_candidates=1,
        )
    with pytest.raises(ParameterMismatchError, match="does not match"):
        run_resumable_integral_property_search(
            CountingFixtureOracle(),
            config=replace(_config(), output_size=4),
            output_root=output,
        )


def test_corrupt_candidate_is_recomputed_and_temporary_file_is_ignored(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    oracle = CountingFixtureOracle()
    with pytest.raises(ControlledInterruption):
        run_resumable_integral_property_search(
            oracle,
            config=_config(),
            output_root=output,
            interrupt_after_new_candidates=1,
        )
    candidate = next((output / "candidate_results").glob("*.json"))
    coordinate = next(iter(oracle.calls))
    candidate.write_text('{"payload":', encoding="utf-8")
    (output / "candidate_results/.unfinished.tmp").write_text("partial", encoding="utf-8")
    result = run_resumable_integral_property_search(
        oracle,
        config=_config(),
        output_root=output,
    )
    assert oracle.calls[coordinate] == 2
    assert result["rejected_candidate_artifacts"] == 1
    assert "candidate_artifact_rejected" in _events(output / "progress.jsonl")


def test_completed_result_hash_is_checked_before_reuse(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_resumable_integral_property_search(
        CountingFixtureOracle(),
        config=_config(),
        output_root=output,
    )
    (output / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="result file hash"):
        run_resumable_integral_property_search(
            CountingFixtureOracle(),
            config=_config(),
            output_root=output,
        )


def test_two_worker_incremental_runner_preserves_math_result(tmp_path: Path) -> None:
    single = run_resumable_integral_property_search(
        CountingFixtureOracle(),
        config=_config(),
        output_root=tmp_path / "single",
    )
    parallel = run_resumable_integral_property_search(
        CountingFixtureOracle(),
        config=_config(workers=2),
        output_root=tmp_path / "parallel",
    )
    assert parallel["relations"] == single["relations"]


def test_gf2_nullspace_returns_relation_between_duplicate_columns() -> None:
    assert _gf2_nullspace((0b11,), width=2) == (0b11,)
