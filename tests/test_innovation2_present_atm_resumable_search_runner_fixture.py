from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_atm_resumable_search_runner_fixture import (
    render_resumable_runner_fixture,
)
from blockcipher_nd.tasks.innovation2.present_atm_resumable_search_runner_fixture import (
    ATM_COMMIT,
    E101_DECISION,
    E101_GATE_SHA256,
    SEARCH_SHA256,
    execute_resumable_runner_fixture,
    result_rows,
)


def _execute(output: Path, *, source_hash: str = SEARCH_SHA256) -> dict[str, object]:
    return execute_resumable_runner_fixture(
        output,
        actual_atm_commit=ATM_COMMIT,
        actual_search_sha256=source_hash,
        e101_gate={"status": "hold", "decision": E101_DECISION},
        e101_gate_sha256=E101_GATE_SHA256,
    )


def test_e102_fixture_passes_resume_integrity_and_path_gates(tmp_path: Path) -> None:
    summary = _execute(tmp_path / "run")
    gate = summary["gate"]
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_atm_resumable_runner_fixture_passed"
    )
    assert all(gate["source_checks"].values())
    assert all(gate["fixture_checks"].values())
    assert gate["metrics"]["anchor_oracle_calls"] == gate["metrics"][
        "resumed_oracle_calls"
    ]
    assert gate["metrics"]["calls_durable_at_interrupt"] == 1
    assert gate["metrics"]["resumed_reused_candidates"] == 1
    assert gate["metrics"]["corrupt_artifacts_rejected"] == 1
    assert gate["next_action"]["real_atm_bounded_compatibility_open"] is True
    assert gate["next_action"]["r9_missing_split_search_open"] is False
    assert gate["next_action"]["r10_search_open"] is False
    assert gate["next_action"]["remote_scale"] is False
    assert len(result_rows(summary)) == len(gate["source_checks"]) + len(
        gate["fixture_checks"]
    )


def test_e102_source_drift_fails_before_opening_real_atm_gate(tmp_path: Path) -> None:
    summary = _execute(tmp_path / "run", source_hash="0" * 64)
    gate = summary["gate"]
    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation2_present_atm_resumable_runner_protocol_invalid"
    )
    assert gate["next_action"]["real_atm_bounded_compatibility_open"] is False


def test_e102_plot_explains_resume_equivalence_and_scope(tmp_path: Path) -> None:
    summary = _execute(tmp_path / "run")
    output = tmp_path / "curves.svg"
    render_resumable_runner_fixture(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "ATM逐候选断点恢复一致性门" in svg
    assert "正常恢复没有增加oracle调用" in svg
    assert "恢复与完整性契约逐项通过" in svg
    assert "官方搜索的三条结果路径均被覆盖" in svg
    assert "不是九/十轮新关系" in svg
