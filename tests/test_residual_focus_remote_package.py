from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plan_residual_focus_remote_package import main as plan_remote_package_main


def test_residual_focus_remote_package_translates_action_plan_to_windows_launcher(tmp_path):
    action_plan = tmp_path / "action_plan.json"
    source_gate = tmp_path / "source_gate.json"
    output_dir = tmp_path / "generated"
    report_path = tmp_path / "remote_package.json"
    source_run = "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706"
    action_plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "artifact_root": "outputs/local_audits/i1_present_r8_residual_focus_262k",
                "seeds": [
                    {
                        "seed": 0,
                        "run_id": source_run,
                        "remote_train_trail_position_checkpoint": (
                            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
                            f"{source_run}\\checkpoints\\row0002_present_trail_position_stats_pairset_seed0.pt"
                        ),
                        "planned_outputs": {
                            "focus10_slice_eval": (
                                "outputs/local_audits/i1_present_r8_residual_focus_262k/"
                                "seed0/residual_focus10_slice_eval.json"
                            )
                        },
                    }
                ],
                "commands": [
                    (
                        "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export-checkpoint-scores "
                        "--checkpoint "
                        f"outputs/remote_results/{source_run}/checkpoints/"
                        "row0002_present_trail_position_stats_pairset_seed0.pt "
                        "--eval-plan configs/experiment/innovation1/"
                        "innovation1_spn_present_r8_trail_position_beamstats_262k_seed0.csv "
                        "--output-dir outputs/local_audits/i1_present_r8_residual_focus_262k/"
                        "seed0/train_trail_position_scores"
                    )
                ],
                "control_commands": [
                    (
                        "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/evaluate-residual-slice-correction "
                        "--validation-corrected-artifact outputs/local_audits/"
                        "i1_present_r8_residual_focus_262k/seed0/residual_focus10_validation_scores "
                        "--output outputs/local_audits/i1_present_r8_residual_focus_262k/"
                        "seed0/residual_focus10_slice_eval.json"
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    source_gate.write_text(
        json.dumps({"status": "pass", "dirty": False, "ahead": 0, "behind": 0}),
        encoding="utf-8",
    )

    status = plan_remote_package_main(
        [
            "--action-plan",
            str(action_plan),
            "--source-gate",
            str(source_gate),
            "--output-dir",
            str(output_dir),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    launcher = Path(report["launcher"]).read_text(encoding="utf-8")
    launch_wrapper = Path(report["launch_wrapper"]).read_text(encoding="utf-8")
    monitor = Path(report["monitor"]).read_text(encoding="utf-8")
    assert status == 0
    assert report["status"] == "pass"
    assert report["launch_allowed"] is True
    assert report["run_id"] == "i1_present_r8_residual_focus_262k"
    assert "cmd.exe /c" in report["launch_policy"]
    assert "cmd.exe /k" not in launcher.lower()
    assert "UV_CACHE_DIR" not in launcher
    assert "outputs/remote_results" not in launcher
    assert "%PYTHON_EXE% scripts\\export-checkpoint-scores" in launcher
    assert (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{source_run}\\checkpoints\\row0002_present_trail_position_stats_pairset_seed0.pt"
    ) in launcher
    assert "%ARTIFACT_ROOT%\\seed0\\train_trail_position_scores" in launcher
    assert "launch_allowed" in launch_wrapper
    assert f'PACKAGE_REPORT="{report_path}"' in launch_wrapper
    assert "source_gate_not_pass" in launch_wrapper
    assert "tmux new-session -d -s monitor_i1_present_r8_residual_focus_262k_20260707" in launch_wrapper
    assert "ssh lxy-a6000" in launch_wrapper
    assert "cmd.exe /c call" in launch_wrapper
    assert "run_i1_present_r8_residual_focus_262k_20260707.cmd" in launch_wrapper
    assert "outputs/local_audits/i1_present_r8_residual_focus_262k" in monitor
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs/i1_present_r8_residual_focus_262k/artifacts" in monitor


def test_residual_focus_remote_package_blocks_launch_when_source_gate_fails(tmp_path):
    action_plan = tmp_path / "action_plan.json"
    source_gate = tmp_path / "source_gate.json"
    output_dir = tmp_path / "generated"
    report_path = tmp_path / "remote_package.json"
    action_plan.write_text(
        json.dumps({"status": "pass", "artifact_root": "outputs/local_audits/i1_present_r8_residual_focus_262k"}),
        encoding="utf-8",
    )
    source_gate.write_text(
        json.dumps({"status": "fail", "errors": ["unpushed_commits"], "ahead": 88}),
        encoding="utf-8",
    )

    status = plan_remote_package_main(
        [
            "--action-plan",
            str(action_plan),
            "--source-gate",
            str(source_gate),
            "--output-dir",
            str(output_dir),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["launch_allowed"] is False
    assert "source_gate_not_pass" in report["blockers"]
    assert report["source_gate_errors"] == ["unpushed_commits"]
