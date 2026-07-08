from __future__ import annotations

import json
import stat
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
                "source_selection_commands": [
                    (
                        "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/analyze-residual-bucket-axis-spectrum "
                        "--output outputs/local_audits/i1_present_r8_residual_focus_262k/"
                        "seed0/train_residual_loss_axis_spectrum.json"
                    )
                ],
                "source_selection_summary_command": (
                    "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/summarize-residual-axis-spectrum "
                    "--output outputs/local_audits/i1_present_r8_residual_focus_262k/"
                    "residual_axis_spectrum_summary.json"
                ),
                "source_selected_commands": [
                    (
                        "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/fit-residual-correction-feature-expert "
                        "--include-feature-prefixes-from-summary outputs/local_audits/"
                        "i1_present_r8_residual_focus_262k/residual_axis_spectrum_summary.json "
                        "--output-validation-dir outputs/local_audits/"
                        "i1_present_r8_residual_focus_262k/seed0/"
                        "residual_focus10_source_selected_validation_scores "
                        "--output-report outputs/local_audits/i1_present_r8_residual_focus_262k/"
                        "seed0/residual_focus10_source_selected_report.json"
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
    launch_wrapper_path = Path(report["launch_wrapper"])
    monitor_path = Path(report["monitor"])
    launch_wrapper = launch_wrapper_path.read_text(encoding="utf-8")
    monitor = monitor_path.read_text(encoding="utf-8")
    assert status == 0
    assert report["status"] == "pass"
    assert report["launch_allowed"] is True
    assert report["run_id"] == "i1_present_r8_residual_focus_262k"
    assert "cmd.exe /c" in report["launch_policy"]
    assert "cmd.exe /k" not in launcher.lower()
    assert "UV_CACHE_DIR" not in launcher
    assert "outputs/remote_results" not in launcher
    assert "set PYTHONPATH=%SOURCE_ROOT%\\src;%PYTHONPATH%" in launcher
    assert launcher.index("set PYTHONPATH=%SOURCE_ROOT%\\src;%PYTHONPATH%") < launcher.index("echo command_0>")
    assert "%PYTHON_EXE% scripts\\export-checkpoint-scores" in launcher
    assert "%PYTHON_EXE% scripts\\analyze-residual-bucket-axis-spectrum" in launcher
    assert "%PYTHON_EXE% scripts\\summarize-residual-axis-spectrum" in launcher
    assert "%PYTHON_EXE% scripts\\fit-residual-correction-feature-expert" in launcher
    assert (
        launcher.index("scripts\\evaluate-residual-slice-correction")
        < launcher.index("scripts\\analyze-residual-bucket-axis-spectrum")
        < launcher.index("scripts\\summarize-residual-axis-spectrum")
        < launcher.index("scripts\\fit-residual-correction-feature-expert")
    )
    assert "%ARTIFACT_ROOT%\\residual_axis_spectrum_summary.json" in launcher
    assert "%ARTIFACT_ROOT%\\seed0\\residual_focus10_source_selected_validation_scores" in launcher
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
    assert "REMOTE_SOURCE_ROOT=" in launch_wrapper
    assert "git clone --branch main" in launch_wrapper
    assert "git fetch origin main" in launch_wrapper
    assert "cmd.exe /c if not exist" in launch_wrapper
    assert "&& call" in launch_wrapper
    assert launch_wrapper.index("git clone --branch main") < launch_wrapper.index("&& call")
    assert "run_i1_present_r8_residual_focus_262k_20260707.cmd" in launch_wrapper
    assert "outputs/local_audits/i1_present_r8_residual_focus_262k" in monitor
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs/i1_present_r8_residual_focus_262k/artifacts" in monitor
    assert launch_wrapper_path.stat().st_mode & stat.S_IXUSR
    assert monitor_path.stat().st_mode & stat.S_IXUSR


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


def test_residual_focus_remote_package_accepts_isolated_retry_run_id(tmp_path):
    action_plan = tmp_path / "action_plan.json"
    source_gate = tmp_path / "source_gate.json"
    output_dir = tmp_path / "generated"
    report_path = tmp_path / "remote_package.json"
    retry_run_id = "i1_present_r8_residual_focus_262k_retry1_20260707"
    action_plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "artifact_root": "outputs/local_audits/i1_present_r8_residual_focus_262k",
                "commands": [
                    "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export-bit-sensitivity-features "
                    "--output-dir outputs/local_audits/i1_present_r8_residual_focus_262k/seed0/features"
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
            "--run-id",
            retry_run_id,
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    launcher = Path(report["launcher"]).read_text(encoding="utf-8")
    launch_wrapper = Path(report["launch_wrapper"]).read_text(encoding="utf-8")
    monitor = Path(report["monitor"]).read_text(encoding="utf-8")
    assert status == 0
    assert report["run_id"] == retry_run_id
    assert f"run_{retry_run_id}_20260707.cmd" in report["launcher"]
    assert f"RUN_ID=\"{retry_run_id}\"" in launch_wrapper
    assert f"G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\{retry_run_id}" in launch_wrapper
    assert f"RUN_ID={retry_run_id}" in launcher
    assert f"G:/lxy/blockcipher-structure-adaptive-nd-runs/{retry_run_id}/artifacts" in monitor
    assert "outputs/local_audits/i1_present_r8_residual_focus_262k" in monitor
