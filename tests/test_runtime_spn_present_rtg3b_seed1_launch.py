from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.runtime_spn_present_rtg3b_seed1_launch import (
    REQUIRED_SOURCE_ASSETS,
    SEED0_DECISION,
    SEED0_RUN_ID,
    SEED1_PLAN,
    SEED1_REMOTE_CONFIG,
    adjudicate_runtime_spn_present_rtg3b_seed1_launch,
    plans_match_seed_only,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_present_rtg3b_launch import (
    FORMAL_PLAN as SEED0_PLAN,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _seed0_gate() -> dict:
    return {
        "run_id": SEED0_RUN_ID,
        "phase": "rtg3b",
        "seed": 0,
        "samples_per_class": 1_000_000,
        "status": "pass",
        "decision": SEED0_DECISION,
        "protocol_checks": {"protocol": True},
        "research_checks": {"research": True},
        "aucs": {"true": 0.74, "corrupted": 0.60, "independent": 0.59},
        "margins": {
            "true_minus_corrupted": 0.14,
            "true_minus_independent": 0.15,
        },
    }


def _results() -> list[dict]:
    return [
        {
            "model": model,
            "seed": 0,
            "samples_per_class": 1_000_000,
        }
        for model in (
            "present_runtime_e4_equivariant_true",
            "present_runtime_e4_equivariant_corrupted",
            "present_runtime_e4_equivariant_independent",
        )
    ]


def _history() -> list[dict[str, str]]:
    return [
        {"model": model, "epoch": str(epoch)}
        for model in (
            "present_runtime_e4_equivariant_true",
            "present_runtime_e4_equivariant_corrupted",
            "present_runtime_e4_equivariant_independent",
        )
        for epoch in range(1, 6)
    ]


def _adjudicate(*, live_remote_sha: str | None = COMMIT, seed0_pass: bool = True):
    gate = _seed0_gate()
    if not seed0_pass:
        gate["status"] = "hold"
    return adjudicate_runtime_spn_present_rtg3b_seed1_launch(
        source_commit=COMMIT,
        remote="origin",
        branch="main",
        live_remote_sha=live_remote_sha,
        artifact_names={
            "curves.svg",
            "gate.local.json",
            "history.csv",
            "results.jsonl",
            "retrieved_from_verified_result_branch.marker",
            "validation.local.json",
            "visual_qa_passed.marker",
        },
        seed0_gate=gate,
        seed0_validation={
            "status": "pass",
            "expected_rows": 3,
            "result_rows": 3,
            "errors": [],
        },
        results=_results(),
        history=_history(),
        readiness_status="pass",
        source_commit_valid=True,
        source_commit_exists=True,
        training_commit_exists=True,
        protected_changes=[],
        source_assets_committed=True,
        source_assets_match=True,
        plans_match_seed_only=True,
    )


def test_seed1_plan_changes_only_seed_and_descriptive_identity() -> None:
    assert plans_match_seed_only(ROOT / SEED0_PLAN, ROOT / SEED1_PLAN)


def test_seed1_remote_package_is_disk_backed_and_complete() -> None:
    report = remote_readiness_report(ROOT / SEED1_REMOTE_CONFIG)
    run_script = (
        ROOT / "configs/remote/generated/"
        "run_i1_rtg3b_present80_one_to_one_formal_1000000_"
        "seed1_retry1_20260727.cmd"
    ).read_text(encoding="utf-8")

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["plan_rows"] == 3
    assert report["max_samples_per_class"] == 1_000_000
    assert all((ROOT / path).is_file() for path in REQUIRED_SOURCE_ASSETS)
    assert (
        "set REMOTE_CONFIG=configs\\remote\\"
        "innovation1_rtg3b_present80_one_to_one_formal_1000000_"
        "seed1_retry1_gpu0_20260727.json"
    ) in run_script


def test_seed1_launch_requires_complete_seed0_and_exact_live_sha() -> None:
    passed = _adjudicate()
    missing_live_sha = _adjudicate(live_remote_sha=None)
    held_seed0 = _adjudicate(seed0_pass=False)

    assert passed["status"] == "pass"
    assert passed["launch_authorized"] is True
    assert all(passed["evidence_checks"].values())
    assert all(passed["readiness_checks"].values())
    assert all(passed["publication_checks"].values())
    assert missing_live_sha["status"] == "hold"
    assert missing_live_sha["should_ssh"] is True
    assert missing_live_sha["ssh_allowed"] is False
    assert held_seed0["status"] == "fail"
    assert held_seed0["should_ssh"] is False


def test_seed1_generated_scripts_preserve_remote_and_control_policy() -> None:
    generated = ROOT / "configs/remote/generated"
    run_script = (
        generated
        / "run_i1_rtg3b_present80_one_to_one_formal_1000000_seed1_retry1_20260727.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        generated
        / "launch_i1_rtg3b_present80_one_to_one_formal_1000000_seed1_retry1_20260727.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = (
        generated
        / "monitor_i1_rtg3b_present80_one_to_one_formal_1000000_seed1_retry1_20260727.sh"
    ).read_text(encoding="utf-8")
    successor_script = (
        generated / "monitor_i1_rtg3b_seed1_after_seed0_retry1_20260727.sh"
    ).read_text(encoding="utf-8")

    assert "--dataset-cache-root" in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--seed 1" in run_script
    assert "--samples-per-class 1000000" in run_script
    assert "--phase rtg3b" in run_script
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src" in run_script
    assert "set RUN_LOCK=%RUN_ROOT%\\run.lock" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in launch_script
    assert 'schtasks /Change /TN "%TASK_NAME%" /DISABLE' in launch_script
    assert "cmd.exe /k" not in monitor_script
    assert "cmd.exe /k" not in successor_script
    assert "_schedule_disabled.marker" in monitor_script
    assert "run.lock\\\\NUL" in monitor_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run_script
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch_script
    assert "innovation1_runtime_spn_present_formal_seed0_supported" in launch_script
    assert "visual_qa_passed.marker" in successor_script
    assert successor_script.index(
        'if [[ -f "${SEED0_ROOT}/gate.local.json" ]]'
    ) < successor_script.index(
        'if [[ -f "${SEED0_MONITOR}/remote_failed.marker" ]]'
    )
