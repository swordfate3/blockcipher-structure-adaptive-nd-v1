from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_sbox4_real_atm_runner_compatibility import (
    render_real_atm_compatibility,
)
from blockcipher_nd.tasks.innovation2.present_sbox4_real_atm_runner_compatibility import (
    _bitset_build_command,
    audit_relation_spaces,
    canonical_relations,
)


def test_relation_space_audit_accepts_different_bases_of_same_space() -> None:
    a = (1, 1)
    b = (2, 2)
    c = (3, 3)
    anchor = (((a, b)), ((b, c)))
    runner = (((a, c)), ((b, c)))
    audit = audit_relation_spaces(anchor, runner)
    assert audit["official_rank"] == 2
    assert audit["runner_rank"] == 2
    assert audit["rank_equal"] is True
    assert audit["anchor_span_in_runner"] is True
    assert audit["runner_span_in_anchor"] is True
    assert audit["singleton_relations_equal"] is True


def test_relation_space_audit_rejects_missing_singleton_and_span() -> None:
    a = (1, 1)
    b = (2, 2)
    audit = audit_relation_spaces(((a,), (b,)), ((a,),))
    assert audit["rank_equal"] is False
    assert audit["anchor_span_in_runner"] is False
    assert audit["runner_span_in_anchor"] is True
    assert audit["singleton_relations_equal"] is False


def test_canonical_relations_removes_duplicates_and_orders_terms() -> None:
    relations = [frozenset({(2, 2), (1, 1)}), frozenset({(1, 1), (2, 2)})]
    assert canonical_relations(relations) == (((1, 1), (2, 2)),)


def test_e103_plot_states_empty_space_and_claim_boundary(tmp_path: Path) -> None:
    summary = {
        "gate": {
            "decision": "innovation2_present_sbox4_real_atm_compatibility_passed",
            "metrics": {
                "bitset_build_seconds": 6.8,
                "model_build_seconds": 0.02,
                "official_anchor_seconds": 0.64,
                "runner_resume_seconds": 0.19,
                "total_seconds": 8.7,
                "hard_cap_seconds": 180,
                "official_candidate_calls": 16,
                "runner_candidate_calls": 18,
                "runner_calls_at_interrupt": 3,
                "runner_reused_candidates": 1,
                "official_relations": 0,
                "runner_relations": 0,
                "official_rank": 0,
                "runner_rank": 0,
                "official_internal_oracle_call_sum": 14_234,
                "runner_internal_oracle_call_sum": 14_992,
            },
        }
    }
    output = tmp_path / "curves.svg"
    render_real_atm_compatibility(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "真实ATM运行时与断点恢复兼容性门" in svg
    assert "多进程预取透明计数" in svg
    assert "两边同为空空间：只证明兼容，不是发现" in svg
    assert "尚未启动R9或R10搜索" in svg
    assert "不是64-bit PRESENT" in svg


def test_windows_bitset_build_command_keeps_intermediates_in_build_root(
    tmp_path: Path,
) -> None:
    build_root = tmp_path / "atm/bitarrays/.build"
    command, auxiliary = _bitset_build_command(
        platform_name="nt",
        compiler="cl.exe",
        source=tmp_path / "atm/bitarrays/src/bitset.cpp",
        output=tmp_path / "atm/bitarrays/bitset.cp310-win_amd64.pyd",
        includes=(r"G:\run\venv\Include", r"G:\run\venv\pybind11"),
        build_root=build_root,
    )
    assert command[0] == "cl.exe"
    assert "/LD" in command
    assert "/std:c++20" in command
    assert any(item.startswith("/OUT:") and item.endswith(".pyd") for item in command)
    assert any(item.startswith("/LIBPATH:") for item in command)
    assert auxiliary
    assert all(path.is_relative_to(build_root) for path in auxiliary)
