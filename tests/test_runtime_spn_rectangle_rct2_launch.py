from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_rct2_launch import (
    _plans_match_scale_only,
    adjudicate_runtime_spn_rectangle_rct2_launch,
)


ROOT = Path(__file__).resolve().parents[1]
RCT1_PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_noncontiguous_attribution_"
    "rct1_2048_seed0_seed1.csv"
)
RCT2_PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv"
)
SUCCESSOR = ROOT / "configs/remote/generated/monitor_i1_rct2_after_rtg3a_20260725.sh"


def _authority() -> dict[str, object]:
    return {
        "gate_identity_exact": True,
        "gate_recomputed_exact": True,
        "validation_exact": True,
        "visual_qa_passed": True,
        "results_sha256": "a" * 64,
        "gate_sha256": "b" * 64,
    }


def _gate(
    *,
    sessions: int = 0,
    published: bool = True,
    readiness: str = "pass",
    authority: dict[str, object] | None = None,
):
    return adjudicate_runtime_spn_rectangle_rct2_launch(
        source_commit="c" * 40,
        upstream_ref="origin/main",
        rct1_authority=authority or _authority(),
        readiness_status=readiness,
        rtg3_session_count=sessions,
        plans_match_scale_only=True,
        source_commit_valid=True,
        source_commit_exists=True,
        source_commit_published=published,
        source_assets_committed=True,
        source_assets_match=True,
        protected_paths_unchanged=True,
        protected_worktree_clean=True,
    )


def test_rct2_launch_gate_authorizes_only_released_published_lane() -> None:
    gate = _gate()

    assert gate["status"] == "pass"
    assert gate["decision"] == ("innovation1_rct2_rectangle_remote_launch_authorized")
    assert gate["remote_config_readiness"] == "pass"
    assert gate["rct1_authority"]["gate_recomputed_exact"] is True
    assert gate["rtg3_session_count"] == 0
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is True


def test_rct2_launch_gate_holds_without_ssh_while_rtg3_lane_is_busy() -> None:
    gate = _gate(sessions=2)

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_rct2_rectangle_waiting_for_rtg3_remote_lane"
    )
    assert gate["rtg3_session_count"] == 2
    assert gate["should_ssh"] is False
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is False


def test_rct2_launch_gate_holds_unpublished_source_without_overlay_escape() -> None:
    gate = _gate(published=False)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_rct2_rectangle_source_not_published"
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is False
    assert gate["launch_authorized"] is False
    assert "scp or dirty-overlay source publication" in gate["blocked_actions"]


def test_rct2_launch_gate_fails_closed_on_authority_or_session_drift() -> None:
    bad_authority = _authority()
    bad_authority["gate_recomputed_exact"] = False

    invalid_authority = _gate(authority=bad_authority)
    invalid_count = _gate(sessions=-1)
    invalid_readiness = _gate(readiness="fail")

    assert invalid_authority["status"] == "fail"
    assert invalid_authority["should_ssh"] is False
    assert invalid_count["status"] == "fail"
    assert invalid_count["lane_checks"]["rtg3_session_count_nonnegative"] is False
    assert invalid_readiness["status"] == "fail"
    assert (
        invalid_readiness["readiness_checks"]["remote_config_readiness_pass"] is False
    )


def test_real_rct2_plan_is_rct1_seed0_with_scale_only_change() -> None:
    assert _plans_match_scale_only(RCT1_PLAN, RCT2_PLAN) is True


def test_rct2_successor_is_fail_closed_clean_clone_handoff() -> None:
    script = SUCCESSOR.read_text(encoding="utf-8")

    assert "^i1_rtg3a" in script
    assert "--rtg3-session-count 0" in script
    assert "innovation1_rct2_rectangle_remote_launch_authorized" in script
    assert "g.get('should_ssh') is True" in script
    assert "g.get('ssh_allowed') is True" in script
    assert "g.get('launch_authorized') is True" in script
    assert "clone --no-checkout" in script
    assert "blockcipher-structure-adaptive-nd-runs" in script
    assert 'if exist \\"${REMOTE_RUN_ROOT}\\" (exit /b 3)' in script
    assert "status --porcelain" in script
    assert "cmd.exe /c" in script
    assert "cmd.exe /k" not in script
    assert "scp " not in script
    assert "for attempt in $(seq 1 30)" in script
    assert "bounded_start_confirmation_passed" in script
    assert "i1_rct2_rectangle80_medium_monitor" in script
    assert "rct2_result_monitor_started.marker" in script
