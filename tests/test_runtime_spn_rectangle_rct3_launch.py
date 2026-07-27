from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_rct3_launch import (
    _plans_match_scale_only,
    adjudicate_runtime_spn_rectangle_rct3_launch,
)


ROOT = Path(__file__).resolve().parents[1]
RCT2_PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_medium_rct2_65536_seed0.csv"
)
RCT3_PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_spn_rectangle80_runtime_e4_scale_rct3_262144_seed0.csv"
)
SUCCESSOR = ROOT / "configs/remote/generated/monitor_i1_rct3_after_rct2_20260727.sh"
REMOTE_CONFIG = (
    ROOT / "configs/remote/innovation1_rct3_rectangle80_runtime_e4_scale_262144_"
    "seed0_gpu1_20260727.json"
)


def _authority() -> dict[str, object]:
    return {
        "gate_identity_exact": True,
        "gate_recomputed_exact": True,
        "protocol_checks_pass": True,
        "research_checks_pass": True,
        "plan_validation_pass": True,
        "verified_result_branch_retrieval": True,
        "visual_qa_passed": True,
        "results_sha256": "a" * 64,
        "gate_sha256": "b" * 64,
    }


def _gate(
    *,
    published: bool = True,
    readiness: str = "pass",
    authority: dict[str, object] | None = None,
):
    return adjudicate_runtime_spn_rectangle_rct3_launch(
        source_commit="c" * 40,
        upstream_ref="origin/main",
        rct2_authority=authority or _authority(),
        readiness_status=readiness,
        plans_match_scale_only=True,
        source_commit_valid=True,
        source_commit_exists=True,
        source_commit_published=published,
        source_assets_committed=True,
        source_assets_match=True,
        protected_paths_unchanged=True,
        protected_worktree_clean=True,
    )


def test_rct3_launch_gate_authorizes_only_complete_published_authority() -> None:
    gate = _gate()

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rct3_rectangle_remote_launch_authorized"
    assert gate["remote_config_readiness"] == "pass"
    assert gate["rct2_authority"]["gate_recomputed_exact"] is True
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is True
    assert "GPU1" in gate["next_action"]


def test_rct3_launch_gate_fails_closed_without_visual_or_result_branch() -> None:
    no_visual = _authority()
    no_visual["visual_qa_passed"] = False
    no_branch = _authority()
    no_branch["verified_result_branch_retrieval"] = False

    visual_gate = _gate(authority=no_visual)
    branch_gate = _gate(authority=no_branch)

    assert visual_gate["status"] == "fail"
    assert visual_gate["should_ssh"] is False
    assert visual_gate["launch_authorized"] is False
    assert branch_gate["status"] == "fail"
    assert branch_gate["should_ssh"] is False


def test_rct3_launch_gate_holds_unpublished_source_without_overlay() -> None:
    gate = _gate(published=False)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_rct3_rectangle_source_not_published"
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is False
    assert gate["launch_authorized"] is False
    assert "scp or dirty-overlay source publication" in gate["blocked_actions"]


def test_real_rct3_plan_is_rct2_with_scale_only_change() -> None:
    assert _plans_match_scale_only(RCT2_PLAN, RCT3_PLAN) is True


def test_rct3_successor_waits_on_local_authority_before_first_ssh() -> None:
    script = SUCCESSOR.read_text(encoding="utf-8")

    assert "gate.local.json" in script
    assert "validation.local.json" in script
    assert "retrieved_from_verified_result_branch.marker" in script
    assert "visual_qa_passed.marker" in script
    assert "rct2_gate_${status}" in script
    assert "innovation1_rct3_rectangle_remote_launch_authorized" in script
    assert "g.get('should_ssh') is True" in script
    assert "g.get('ssh_allowed') is True" in script
    assert "g.get('launch_authorized') is True" in script
    assert script.index(
        "scripts/check-runtime-spn-rectangle-rct3-launch"
    ) < script.index("ssh -o BatchMode=yes")
    assert "clone --no-checkout" in script
    assert 'if exist \\"${REMOTE_RUN_ROOT}\\" (exit /b 3)' in script
    assert "status --porcelain" in script
    assert "cmd.exe /c" in script
    assert "cmd.exe /k" not in script
    assert "scp " not in script
    assert "for attempt in $(seq 1 30)" in script
    assert "bounded_start_confirmation_passed" in script
    assert "REMOTE_SCHEDULE_DISABLED_MARKER" in script
    assert "REMOTE_RUN_LOCK" in script
    assert "run.lock\\\\NUL" in script
    assert "i1_rct3_rectangle80_scale_monitor" in script
    assert "rct3_result_monitor_started.marker" in script


def test_rct3_remote_config_reserves_gpu1() -> None:
    config = json.loads(REMOTE_CONFIG.read_text(encoding="utf-8"))

    assert config["physical_gpu"] == 1
    assert config["validation_label"].endswith("_gpu1")
    assert "GPU1" in config["launch_policy"]
    assert "visual QA" in config["launch_policy"]
