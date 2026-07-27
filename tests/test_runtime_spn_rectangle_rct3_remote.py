from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report


ROOT = Path(__file__).resolve().parents[1]
REMOTE_CONFIG = (
    ROOT / "configs/remote/innovation1_rct3_rectangle80_runtime_e4_scale_262144_"
    "seed0_gpu1_20260727.json"
)
RUN_SCRIPT = (
    ROOT / "configs/remote/generated/run_i1_rct3_rectangle80_runtime_e4_scale_"
    "262144_seed0_20260727.cmd"
)
LAUNCH_SCRIPT = (
    ROOT / "configs/remote/generated/launch_i1_rct3_rectangle80_runtime_e4_scale_"
    "262144_seed0_20260727.cmd"
)
MONITOR_SCRIPT = (
    ROOT / "configs/remote/generated/monitor_i1_rct3_rectangle80_runtime_e4_scale_"
    "262144_seed0_20260727.sh"
)


def test_rectangle_rct3_remote_config_passes_scale_cache_readiness() -> None:
    report = remote_readiness_report(REMOTE_CONFIG)
    config = json.loads(REMOTE_CONFIG.read_text(encoding="utf-8"))

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["expected_rows"] == report["plan_rows"] == 3
    assert report["max_samples_per_class"] == 262_144
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert config["dataset_cache"] is True
    assert config["physical_gpu"] == 1
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )


def test_rectangle_rct3_windows_scripts_are_clean_clone_and_cache_safe() -> None:
    run = RUN_SCRIPT.read_text(encoding="utf-8")
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")

    assert "cmd.exe /k" not in run + launch
    assert "cmd.exe /c" in launch
    assert "setlocal EnableDelayedExpansion" not in run + launch
    assert "!" not in run + launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launch
    assert '--dataset-cache-root "%CACHE_ROOT%"' in run
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert '--progress-output "%LOG_DIR%\\progress.jsonl"' in run
    assert "scripts\\gate-runtime-spn-rectangle-medium" in run
    assert "--phase rct3" in run
    assert "visual_qa_pending.marker" in run
    assert "fc /b" in run
    assert "git status --porcelain" in run + launch
    assert 'git checkout --detach "%EXPECTED_COMMIT%"' in launch
    assert "source_expected_commit.txt" in run + launch
    assert "set RUN_LOCK=%RUN_ROOT%\\run.lock" in run
    assert '2> nul mkdir "%RUN_LOCK%" || goto duplicate_instance' in run
    assert "goto existing_run_evidence" in run
    assert run.index('mkdir "%RUN_LOCK%"') < run.index(
        'if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"'
    )
    assert 'schtasks /Change /TN "%TASK_NAME%" /DISABLE' in launch
    assert "_schedule_disabled.marker" in launch
    assert ":schedule_disable_failed" in launch
    assert 'schtasks /End /TN "%TASK_NAME%"' in launch
    assert 'schtasks /Delete /TN "%TASK_NAME%" /F' in launch
    assert launch.index("schtasks /Run") < launch.index("schtasks /Change")
    assert launch.index("schtasks /Change") < launch.index("schtasks /Query")


def test_rectangle_rct3_monitor_verifies_and_defers_visual_qa() -> None:
    monitor = MONITOR_SCRIPT.read_text(encoding="utf-8")

    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor
    assert "git_revision.txt" in monitor
    assert '== "${SOURCE_COMMIT}"' in monitor
    assert 'local_gate_root="${DESTINATION}/local_adjudication"' in monitor
    assert '--output-root "${local_gate_root}"' in monitor
    assert "--phase rct3" in monitor
    assert (
        'cp "${local_gate_root}/gate.json" "${DESTINATION}/gate.local.json"' in monitor
    )
    assert 'cp "${local_gate_root}/curves.svg" "${DESTINATION}/curves.svg"' in monitor
    assert "visual_qa_pending.marker" in monitor
    assert "visual_qa_passed.marker" not in monitor
    assert "scripts/index-results" in monitor
    assert "sleep 300" in monitor
