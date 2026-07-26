from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.runtime_spn_present_rtg3b_launch import (
    BASE_PLAN,
    FORMAL_PLAN,
    REMOTE_CONFIG,
    REQUIRED_SOURCE_ASSETS,
    adjudicate_runtime_spn_present_rtg3b_launch,
    plans_match_scale_only,
)


ROOT = Path(__file__).resolve().parents[1]


def _c2_gate() -> dict:
    return {
        "run_id": "i1_runtime_spn_method_boundary_c2_20260726",
        "status": "pass",
        "decision": "innovation1_runtime_spn_method_boundary_frozen",
        "requirement_status": {"R5": "supported"},
        "universal_runtime_spn_supported": False,
    }


def _t1_gate(seed: int) -> dict:
    return {
        "seed": seed,
        "status": "pass",
        "decision": f"innovation1_runtime_spn_present_transfer_seed{seed}_supported",
        "protocol_checks": {"protocol": True},
        "research_checks": {"research": True},
    }


def _adjudicate(*, remote_sha: str | None) -> dict:
    commit = "a" * 40
    return adjudicate_runtime_spn_present_rtg3b_launch(
        source_commit=commit,
        remote="origin",
        branch="main",
        remote_sha=remote_sha,
        c2_gate=_c2_gate(),
        c2_validation={"status": "pass"},
        t1_gates=(_t1_gate(0), _t1_gate(1)),
        readiness={
            "status": "pass",
            "errors": [],
            "max_samples_per_class": 1_000_000,
            "expected_rows": 3,
            "plan_rows": 3,
        },
        source_commit_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        plans_match_scale_only=True,
    )


def test_formal_present_plan_changes_only_scale_and_identity() -> None:
    assert plans_match_scale_only(ROOT / BASE_PLAN, ROOT / FORMAL_PLAN)


def test_rtg3b_remote_config_and_source_assets_are_ready() -> None:
    report = remote_readiness_report(ROOT / REMOTE_CONFIG)

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["plan_rows"] == 3
    assert report["max_samples_per_class"] == 1_000_000
    assert all((ROOT / path).is_file() for path in REQUIRED_SOURCE_ASSETS)


def test_rtg3b_generated_scripts_preserve_remote_path_and_cache_policy() -> None:
    generated = ROOT / "configs/remote/generated"
    run_script = (
        generated
        / "run_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        generated
        / "launch_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = (
        generated
        / "monitor_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_20260726.sh"
    ).read_text(encoding="utf-8")

    assert "--dataset-cache-root" in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--progress-output" in run_script
    assert "--samples-per-class 1000000" in run_script
    assert "--phase rtg3b" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in launch_script
    assert "cmd.exe /k" not in monitor_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch_script
    assert "live_remote_sha" in monitor_script


def test_rtg3b_launch_requires_exact_live_remote_sha() -> None:
    commit = "a" * 40
    held = _adjudicate(remote_sha=None)
    wrong = _adjudicate(remote_sha="b" * 40)
    passed = _adjudicate(remote_sha=commit)

    assert held["status"] == "hold"
    assert held["should_ssh"] is True
    assert held["ssh_allowed"] is False
    assert held["launch_authorized"] is False
    assert wrong["status"] == "hold"
    assert passed["status"] == "pass"
    assert passed["should_ssh"] is True
    assert passed["ssh_allowed"] is True
    assert passed["launch_authorized"] is True


def test_rtg3b_launch_fails_closed_when_c2_does_not_support_r5() -> None:
    gate = _c2_gate()
    gate["requirement_status"]["R5"] = "partial"
    commit = "a" * 40

    result = adjudicate_runtime_spn_present_rtg3b_launch(
        source_commit=commit,
        remote="origin",
        branch="main",
        remote_sha=commit,
        c2_gate=gate,
        c2_validation={"status": "pass"},
        t1_gates=(_t1_gate(0), _t1_gate(1)),
        readiness={
            "status": "pass",
            "errors": [],
            "max_samples_per_class": 1_000_000,
            "expected_rows": 3,
            "plan_rows": 3,
        },
        source_commit_valid=True,
        source_commit_exists=True,
        source_assets_committed=True,
        source_assets_match=True,
        plans_match_scale_only=True,
    )

    assert result["status"] == "fail"
    assert result["should_ssh"] is False
    assert result["ssh_allowed"] is True
    assert result["launch_authorized"] is False
