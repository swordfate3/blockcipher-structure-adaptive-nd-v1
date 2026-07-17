from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.tasks.innovation2 import speck_hwang_positions as positions
from blockcipher_nd.tasks.innovation2.speck_hwang_phase_c import make_phase_c_keys


def _cache_roles(selected: tuple[int, ...]) -> tuple[dict[str, bool], dict[str, int]]:
    roles = {
        f"position{start:02d}_screen"
        for start in positions.POSITION_STARTS
        if start not in {positions.ANCHOR_START, positions.CONTROL_START}
    } | {f"position{start:02d}_validation" for start in selected}
    return ({role: True for role in roles}, {role: 0 for role in roles})


def _evaluate(candidate_starts: tuple[int, ...]):
    config = positions.SpeckHwangPositionConfig(run_id="e27-test")
    keys = make_phase_c_keys(config.phase_c_config())
    unbalanced = np.uint32(positions.hwang_speck_basis_masks(7)[0] & -positions.hwang_speck_basis_masks(7)[0])
    screen = np.full((30, 8), unbalanced, dtype=np.uint32)
    screen[positions.POSITION_STARTS.index(positions.ANCHOR_START)] = 0
    for start in candidate_starts:
        screen[positions.POSITION_STARTS.index(start)] = 0
    selected = positions.select_validation_candidates(candidate_starts)
    validation = np.zeros((len(selected), 56), dtype=np.uint32)
    completed, resumed = _cache_roles(selected)
    timing_rows = 28 * 8 + len(selected) * 56
    return positions.evaluate_position_family(
        config,
        keys=keys,
        anchor_words=np.zeros(64, dtype=np.uint32),
        control_words=np.full(64, unbalanced, dtype=np.uint32),
        screen_parity_rows=screen,
        screen_candidates=candidate_starts,
        selected_candidates=selected,
        validation_parity_rows=validation,
        baseline_valid=True,
        caches_completed=completed,
        resume_rows_generated=resumed,
        mapping_fixture_valid=True,
        cuda_available=True,
        device_count=1,
        timing_rows=timing_rows,
    )


def test_position_family_and_mapping_fixture() -> None:
    assert positions.POSITION_STARTS == tuple(range(15)) + tuple(range(16, 31))
    assert positions.active_bits_for_pair(5) == tuple(
        bit for bit in range(32) if bit not in {5, 6}
    )
    assert positions.verify_position_mapping_fixture() is True
    with pytest.raises(ValueError, match="unsupported"):
        positions.active_bits_for_pair(15)


def test_frozen_candidate_selection_balances_words_and_caps_at_eight() -> None:
    candidates = (1, 2, 3, 4, 6, 7, 8, 16, 17, 18, 19, 20, 21)
    assert positions.select_validation_candidates(candidates) == (
        1, 2, 3, 4, 16, 17, 18, 19
    )
    with pytest.raises(ValueError, match="anchor and control"):
        positions.select_validation_candidates((5, 16))


def test_advance_gate_requires_four_stable_positions_and_both_words() -> None:
    result = _evaluate((1, 16, 17))
    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_speck_hwang_position_family_advance"
    )
    assert result["gate"]["metrics"]["stable_positive_positions"] == [
        1, 5, 16, 17
    ]
    assert result["gate"]["metrics"]["sampled_negative_count"] >= 8
    assert all(result["gate"]["readiness_checks"].values())


def test_narrow_and_anchor_only_branches_are_distinct() -> None:
    narrow = _evaluate((1,))
    assert narrow["gate"]["decision"] == (
        "innovation2_speck_hwang_position_family_narrow"
    )
    anchor_only = _evaluate(())
    assert anchor_only["gate"]["decision"] == (
        "innovation2_speck_hwang_position_family_anchor_only"
    )


def test_collector_screens_all_positions_then_validates_only_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = positions.SpeckHwangPositionConfig(run_id="e27-collect")
    keys = make_phase_c_keys(config.phase_c_config())
    unbalanced = np.uint32(positions.hwang_speck_basis_masks(7)[0] & -positions.hwang_speck_basis_masks(7)[0])
    candidates = {1, 16}
    baseline = {
        "keys": keys,
        "anchor": {"parity_rows": np.zeros((2, 64), dtype=np.uint32)},
        "control": {"parity_rows": np.full((1, 64), unbalanced, dtype=np.uint32)},
    }
    monkeypatch.setattr(positions, "load_phase_c_position_baselines", lambda *args, **kwargs: baseline)
    calls: list[tuple[int, int]] = []

    def fake_cached(cache_config, *, cache_root, progress_callback=None):
        fixed = set(range(32)) - set(cache_config.active_bits)
        start = min(fixed)
        calls.append((start, len(cache_config.keys)))
        value = np.uint32(0 if start in candidates else unbalanced)
        array = np.full((1, len(cache_config.keys)), value, dtype=np.uint32)
        return {
            "parity_rows": array,
            "completed": np.ones(array.shape, dtype=np.bool_),
            "metadata": {"start": start, "keys": len(cache_config.keys)},
            "cache_status": "created",
            "rows_generated": len(cache_config.keys) if calls.count((start, len(cache_config.keys))) == 1 else 0,
        }

    monkeypatch.setattr(positions, "run_cached_speck_parity_rows", fake_cached)
    collected = positions.collect_position_parity_rows(
        config,
        phase_c_root=tmp_path / "phase_c",
        cache_root=tmp_path / "cache",
    )
    assert collected["screen_candidates"] == (1, 16)
    assert collected["selected_candidates"] == (1, 16)
    assert collected["screen_parity_rows"].shape == (30, 8)
    assert collected["validation_parity_rows"].shape == (2, 56)
    assert sum(length == 8 for _, length in calls) == 56
    assert sum(length == 56 for _, length in calls) == 4
    assert all(value == 0 for value in collected["resume_rows_generated"].values())


def test_remote_scripts_obey_windows_and_monitor_rules() -> None:
    run_script = Path(
        "configs/remote/generated/run_i2_speck32_hwang_positions_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    launch_script = Path(
        "configs/remote/generated/launch_i2_speck32_hwang_positions_gpu0_20260717.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = Path(
        "configs/remote/generated/monitor_i2_speck32_hwang_positions_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    postprocess = Path(
        "configs/remote/generated/postprocess_i2_speck32_hwang_positions_gpu0_20260717.sh"
    ).read_text(encoding="utf-8")
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in launch_script
    assert "EnableDelayedExpansion" not in run_script
    assert "monitor_remote_results.py" not in monitor_script
    assert "retrieved_from_verified_result_branch.marker" in monitor_script
    assert "validate-innovation2-speck-hwang-positions" in postprocess
    assert "plot-innovation2-speck-hwang-positions" in postprocess
    assert "visual_qa_pending.marker" in postprocess
