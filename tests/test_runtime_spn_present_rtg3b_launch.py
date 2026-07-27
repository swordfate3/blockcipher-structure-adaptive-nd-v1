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
        / "run_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        generated
        / "launch_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = (
        generated
        / "monitor_i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727.sh"
    ).read_text(encoding="utf-8")

    assert "--dataset-cache-root" in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--progress-output" in run_script
    assert "--samples-per-class 1000000" in run_script
    assert "--phase rtg3b" in run_script
    assert (
        "set REMOTE_CONFIG=configs\\remote\\"
        "innovation1_rtg3b_present80_one_to_one_formal_1000000_"
        "seed0_retry1_gpu0_20260727.json"
    ) in run_script
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run_script
    assert "set RUN_LOCK=%RUN_ROOT%\\run.lock" in run_script
    assert '2> nul mkdir "%RUN_LOCK%" || goto duplicate_instance' in run_script
    assert "goto existing_run_evidence" in run_script
    assert "_cache_reuse_audit.txt" in run_script
    assert "_cache_reuse_audit_stderr.txt" in run_script
    assert "seed0_20260726\\cache" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in launch_script
    assert 'schtasks /Change /TN "%TASK_NAME%" /DISABLE' in launch_script
    assert "_schedule_disabled.marker" in launch_script
    assert ":schedule_disable_failed" in launch_script
    assert 'schtasks /End /TN "%TASK_NAME%"' in launch_script
    assert 'schtasks /Delete /TN "%TASK_NAME%" /F' in launch_script
    assert launch_script.index("schtasks /Run") < launch_script.index(
        "schtasks /Change"
    )
    assert "cmd.exe /k" not in monitor_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch_script
    assert "live_remote_sha" in monitor_script
    assert "_schedule_disabled.marker" in monitor_script
    assert "run.lock\\\\NUL" in monitor_script


def test_rtg3b_postprocess_recovery_reuses_results_without_training() -> None:
    path = (
        ROOT
        / "configs/remote/generated/"
        "recover_i1_rtg3b_present80_one_to_one_formal_1000000_"
        "seed0_retry1_20260727.cmd"
    )
    text = path.read_text(encoding="utf-8")

    assert "EnableExtensions DisableDelayedExpansion" in text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in text
    assert "cmd.exe /k" not in text
    assert "!" not in text
    assert "scripts\\train" not in text
    assert "_gate_stderr.txt" in text
    assert "No module named 'matplotlib'" in text
    assert "scripts\\validate-results" in text
    assert "scripts\\gate-runtime-spn-present-transfer" in text
    assert "--no-plot" in text
    assert "_verify_checkpoint_payloads" in text
    assert "recovered_without_retraining.marker" in text
    assert "SHA256SUMS" in text
    assert "results/%RUN_ID%" in text
    assert "%RUN_ID%_result_branch_pushed.marker" in text


def test_all_rtg3b_batch_assets_fail_closed_against_duplicate_writers() -> None:
    generated = ROOT / "configs/remote/generated"
    run_paths = sorted(generated.glob("run_i1_rtg3b_present80_*.cmd"))
    launch_paths = sorted(generated.glob("launch_i1_rtg3b_present80_*.cmd"))

    assert len(run_paths) == 4
    assert len(launch_paths) == 4
    for path in run_paths:
        text = path.read_text(encoding="utf-8")
        assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in text
        assert "set RUN_LOCK=%RUN_ROOT%\\run.lock" in text
        assert '2> nul mkdir "%RUN_LOCK%" || goto duplicate_instance' in text
        assert text.index('mkdir "%RUN_LOCK%"') < text.index(
            'if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"'
        )
        assert "goto existing_run_evidence" in text
    for path in launch_paths:
        text = path.read_text(encoding="utf-8")
        assert "cmd.exe /c" in text
        assert "cmd.exe /k" not in text
        assert 'schtasks /Change /TN "%TASK_NAME%" /DISABLE' in text
        assert "_schedule_disabled.marker" in text
        assert ":schedule_disable_failed" in text
        assert 'schtasks /End /TN "%TASK_NAME%"' in text
        assert 'schtasks /Delete /TN "%TASK_NAME%" /F' in text
        assert text.index("schtasks /Run") < text.index("schtasks /Change")
        assert text.index("schtasks /Change") < text.index("schtasks /Query")


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
