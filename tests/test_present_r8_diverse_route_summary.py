from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blockcipher_nd.cli.summarize_present_r8_diverse_route import (
    main as summarize_route_main,
)


def test_present_r8_diverse_route_summary_waits_for_running_residual_focus(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "running",
            "gate_status": "pending",
            "gate_decision": "wait_for_residual_focus_262k_outputs",
            "missing_output_count": 18,
            "repair_active": False,
            "repair_stale_reason": "source_summary_not_current_context",
            "repair_context_current": False,
            "source_selection_summary_status": "missing",
            "source_selection_summary_decision": "wait_for_train_axis_spectrum_summary",
            "source_selection_report_count": 4,
            "source_selection_existing_report_count": 0,
            "source_selection_missing_report_count": 4,
            "source_selection_recommended_feature_prefixes": [],
            "source_selection_selected_groups": [],
            "latest_monitor_event": "running missing=18",
            "progress_summary": {
                "event": "cache_positive_chunk",
                "stage": "dataset_cache",
                "seed": 0,
                "split": "train",
                "class_progress_fraction": 0.5625,
                "total_progress_fraction": 0.28125,
            },
            "progress_by_seed_split": [
                {
                    "event": "cache_positive_chunk",
                    "stage": "dataset_cache",
                    "seed": 0,
                    "split": "train",
                    "class_progress_fraction": 0.5625,
                    "total_progress_fraction": 0.28125,
                }
            ],
            "planned_output_count": 18,
            "existing_planned_output_count": 0,
            "next_action": {"branch": "wait_for_residual_focus_outputs"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(
        tmp_path / "state_token_plan.json",
        {
            "status": "pending",
            "state_token_control_status": "hold",
            "state_token_failing_seed_count": 2,
        },
    )
    output = tmp_path / "summary.json"
    action_plan = _write_json(
        tmp_path / "action_plan.json",
        {
            "status": "pass",
            "should_run": True,
            "candidates": ["focus05", "focus10"],
            "commands": ["cmd0", "cmd1", "cmd2"],
            "control_commands": ["control0"],
            "source_selection_commands": ["source0", "source1"],
            "seeds": [
                {
                    "seed": 0,
                    "planned_outputs": {
                        "raw117_report": "seed0/raw117_report.json",
                        "focus10_report": "seed0/focus10_report.json",
                    },
                    "source_selection_outputs": {
                        "train_residual_loss_axis_spectrum": "seed0/loss.json",
                    },
                },
                {
                    "seed": 1,
                    "planned_outputs": {
                        "raw117_report": "seed1/raw117_report.json",
                    },
                    "source_selection_outputs": {
                        "train_residual_loss_axis_spectrum": "seed1/loss.json",
                        "train_hard_error_axis_spectrum": "seed1/hard.json",
                    },
                },
            ],
        },
    )

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--residual-action-plan",
            str(action_plan),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["selected_next_action"]["branch"] == "wait_for_residual_focus_outputs"
    assert report["residual_focus"]["repair_active"] is False
    assert report["residual_focus"]["repair_stale_reason"] == "source_summary_not_current_context"
    assert report["residual_focus"]["repair_context_current"] is False
    assert report["residual_focus"]["source_selection_summary_status"] == "missing"
    assert report["residual_focus"]["source_selection_summary_decision"] == "wait_for_train_axis_spectrum_summary"
    assert report["residual_focus"]["source_selection_missing_report_count"] == 4
    assert report["residual_focus"]["source_selection_recommended_feature_prefixes"] == []
    assert report["residual_focus"]["source_selection_selected_groups"] == []
    assert report["residual_focus"]["latest_monitor_event"] == "running missing=18"
    assert report["residual_focus"]["progress_summary"]["class_progress_fraction"] == 0.5625
    assert report["residual_focus"]["progress_by_seed_split"][0]["split"] == "train"
    assert report["residual_focus"]["planned_output_count"] == 18
    assert report["residual_focus"]["existing_planned_output_count"] == 0
    blocker_codes = [blocker["code"] for blocker in report["residual_focus"]["blockers"]]
    assert blocker_codes == [
        "residual_focus_outputs_missing",
        "residual_focus_gate_pending",
        "train_source_selection_reports_missing",
    ]
    assert report["residual_focus"]["blockers"][0]["missing_count"] == 18
    assert report["residual_focus"]["blockers"][0]["planned_count"] == 18
    assert report["residual_focus"]["blockers"][0]["existing_count"] == 0
    assert report["residual_focus"]["blockers"][2]["missing_count"] == 4
    assert report["residual_focus"]["action_plan_summary"] == {
        "exists": True,
        "status": "pass",
        "should_run": True,
        "seed_count": 2,
        "candidate_count": 2,
        "candidates": ["focus05", "focus10"],
        "command_count": 3,
        "control_command_count": 1,
        "source_selection_command_count": 2,
        "planned_output_count": 3,
        "source_selection_output_count": 3,
    }
    assert report["residual_focus"]["execution_interpretation"] == {
        "observed_progress_stream_count": 1,
        "active_workload_estimate": 1,
        "parallel_competition_likelihood": "low",
        "planned_stage_command_count": 6,
        "current_stage": "dataset_cache",
        "current_event": "cache_positive_chunk",
        "current_split": "train",
        "current_seed": 0,
        "interpretation": "single_heavy_dataset_cache_stage",
        "reason": "one progress stream is currently observed even though multiple stage commands are planned",
        "workload_message": "one active dataset-cache stream observed; planned commands are sequential work items, not evidence of many parallel training jobs",
    }
    assert "scripts/monitor-health" in report["residual_focus"]["monitor_health_command"]
    assert "--run-id i1_present_r8_residual_focus_262k_retry1" in report["residual_focus"]["monitor_health_command"]
    assert "--progress-root outputs/local_audits/i1_present_r8_residual_focus_262k" in report["residual_focus"]["monitor_health_command"]
    assert "scripts/advance-residual-focus-results" in report["residual_focus"]["advance_command"]
    assert f"--action-plan {action_plan}" in report["residual_focus"]["advance_command"]
    assert "--gate-output outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json" in report["residual_focus"]["advance_command"]
    assert "ssh" not in report["residual_focus"]["advance_command"]
    assert "scripts/watch-residual-focus-results" in report["residual_focus"]["watch_command"]
    assert f"--action-plan {action_plan}" in report["residual_focus"]["watch_command"]
    assert f"--pool-output {pool_plan}" in report["residual_focus"]["watch_command"]
    assert "ssh" not in report["residual_focus"]["watch_command"]
    assert report["candidate_routes"]["state_token_residual"]["status"] == "blocked_by_controls"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "blocked_by_residual_focus"
    linear_combo = report["candidate_routes"]["linear_combo_integral_residual"]
    assert linear_combo["status"] == "blocked_by_residual_focus"
    assert linear_combo["decision"] == "wait_for_residual_focus_outputs_before_linear_combo_integral_expert"
    assert linear_combo["selection_split"] == "train_only"
    assert linear_combo["requires_source_selection"] is True
    assert linear_combo["next_action"]["branch"] == "wait_for_residual_focus_outputs"
    assert report["post_retrieval_sequence"] == [
        {
            "step": "watch_until_residual_focus_outputs_exist",
            "command_field": "residual_focus.watch_command",
            "condition": "residual_focus.missing_output_count > 0",
            "remote_action": False,
        },
        {
            "step": "advance_residual_focus_gate_and_pool_handoff",
            "command_field": "residual_focus.advance_command",
            "condition": "residual_focus.missing_output_count == 0 or residual_focus.status == outputs_ready_gate_needed",
            "remote_action": False,
        },
        {
            "step": "follow_selected_local_route_or_repair_branch",
            "command_field": "selected_next_action",
            "condition": "advance command emits gate, pool, or repair decision",
            "remote_action": False,
        },
    ]
    safety = report["local_command_safety"]
    assert safety["status"] == "pass"
    assert safety["forbidden_tokens"] == ["ssh", "scp", "cmd.exe", "G:\\lxy"]
    assert safety["checked_fields"] == [
        "residual_focus.monitor_health_command",
        "residual_focus.advance_command",
        "residual_focus.watch_command",
    ]
    assert safety["findings"] == []
    assert report["should_launch_remote"] is False


def test_present_r8_diverse_route_summary_can_embed_monitor_health_eta(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "running",
            "gate_status": "pending",
            "gate_decision": "wait_for_residual_focus_262k_outputs",
            "missing_output_count": 18,
            "next_action": {"branch": "wait_for_residual_focus_outputs"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    run_id = "i1_present_r8_residual_focus_262k_retry1"
    run_root = tmp_path / "remote_results" / run_id
    monitor_dir = run_root / "monitor"
    monitor_dir.mkdir(parents=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True)
    old_head = _git_output("rev-list", "--max-count=1", "HEAD~1")
    current_head = _git_output("rev-parse", "HEAD")
    logs_dir.joinpath(f"{run_id}_git_revision.txt").write_text(old_head + "\n", encoding="utf-8")
    logs_dir.joinpath(f"{run_id}_command_0.marker").write_text("command_0\n", encoding="utf-8")
    logs_dir.joinpath(f"{run_id}_command_3.marker").write_text("command_3\n", encoding="utf-8")
    monitor_dir.joinpath("monitor.log").write_text(
        "2026-07-08T08:41:41+08:00 running missing=18\n",
        encoding="utf-8",
    )
    progress_root = tmp_path / "residual_focus_262k"
    progress_path = progress_root / "seed0" / "dataset_cache" / "seed0_train_feature_export_progress.jsonl"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "time": 100.0,
                        "event": "cache_positive_chunk",
                        "stage": "dataset_cache",
                        "seed": 0,
                        "split": "train",
                        "samples_per_class": 262144,
                        "rows_done": 262144,
                        "total_rows": 524288,
                        "class_rows_done": 262144,
                        "class_total": 262144,
                        "chunk_rows": 8192,
                        "model": "present_trail_position_stats_pairset",
                    }
                ),
                json.dumps(
                    {
                        "time": 200.0,
                        "event": "cache_negative_chunk",
                        "stage": "dataset_cache",
                        "seed": 0,
                        "split": "train",
                        "samples_per_class": 262144,
                        "rows_done": 393216,
                        "total_rows": 524288,
                        "class_rows_done": 131072,
                        "class_total": 262144,
                        "chunk_rows": 8192,
                        "model": "present_trail_position_stats_pairset",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--include-monitor-health",
            "--monitor-health-root",
            str(tmp_path / "remote_results"),
            "--monitor-health-progress-root",
            str(progress_root),
            "--monitor-health-stale-after-seconds",
            "999999999",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    monitor = report["residual_focus"]["monitor_health_summary"]
    progress = monitor["progress_summary"]
    assert status == 0
    assert monitor["status"] == "running"
    assert monitor["needs_main_thread_intervention"] is False
    assert monitor["results_jsonl_exists"] is False
    assert monitor["source_revision"]["launched_commit"] == old_head
    assert monitor["source_revision"]["current_head"] == current_head
    assert monitor["source_revision"]["revision_lag"] == {
        "status": "behind_current_head",
        "commits_behind": 1,
    }
    assert monitor["command_markers"] == {
        "marker_count": 2,
        "command_indices": [0, 3],
        "latest_command_index": 3,
        "latest_marker": f"logs/{run_id}_command_3.marker",
    }
    assert progress["source_kind"] == "external_progress_jsonl"
    assert progress["cache_total_progress_percent"] == 75.0
    assert progress["cache_negative_class_progress_percent"] == 50.0
    assert progress["cache_eta_seconds"] == 100
    assert report["residual_focus"]["wait_diagnosis"] == {
        "status": "continue_monitoring",
        "main_thread_intervention_required": False,
        "results_ready": False,
        "cache_eta_seconds": 100,
        "cache_eta_hours": 0.028,
        "monitor_status": "running",
        "latest_command_index": 3,
        "latest_command_marker": f"logs/{run_id}_command_3.marker",
    }


def test_present_r8_diverse_route_summary_can_refresh_residual_status_before_reading(tmp_path: Path):
    stale_residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "running",
            "gate_status": "pending",
            "gate_decision": "stale_wait",
            "missing_output_count": 0,
            "source_selection_report_count": 0,
            "source_selection_missing_report_count": 0,
            "next_action": {"branch": "stale_branch"},
        },
    )
    planned_output = tmp_path / "artifacts" / "seed0" / "residual_focus10_report.json"
    source_loss = tmp_path / "artifacts" / "seed0" / "train_residual_loss_axis_spectrum.json"
    source_hard = tmp_path / "artifacts" / "seed0" / "train_hard_error_axis_spectrum.json"
    action_plan = _write_json(
        tmp_path / "action_plan.json",
        {
            "seeds": [
                {
                    "seed": 0,
                    "planned_outputs": {"focus10": str(planned_output)},
                    "source_selection_outputs": {
                        "train_residual_loss_axis_spectrum": str(source_loss),
                        "train_hard_error_axis_spectrum": str(source_hard),
                    },
                }
            ],
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"},
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    repair = _write_json(tmp_path / "repair.json", {"status": "pending"})
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    monitor_dir.joinpath("monitor.log").write_text(
        "2026-07-07T21:05:31+08:00 running missing=1\n",
        encoding="utf-8",
    )
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--refresh-residual-status",
            "--residual-status",
            str(stale_residual_status),
            "--residual-action-plan",
            str(action_plan),
            "--residual-gate",
            str(gate),
            "--residual-repair-plan",
            str(repair),
            "--residual-monitor-dir",
            str(monitor_dir),
            "--residual-artifact-root",
            str(tmp_path / "artifacts"),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    refreshed_status = json.loads(stale_residual_status.read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["residual_focus"]["gate_decision"] == "wait_for_residual_focus_262k_outputs"
    assert report["residual_focus"]["missing_output_count"] == 1
    assert report["residual_focus"]["source_selection_report_count"] == 2
    assert report["residual_focus"]["source_selection_missing_report_count"] == 2
    assert refreshed_status["latest_monitor_event"] == "running missing=1"


def test_present_r8_diverse_route_summary_prefers_pool3_after_residual_pool_ready(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "pool_ready",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
            "pool_status": "pass",
            "should_run_pool": True,
            "next_action": {"branch": "instantiate_residual_guided_pool3_fixed_fusion"},
        },
    )
    pool_plan = _write_json(
        tmp_path / "pool_plan.json",
        {
            "status": "pass",
            "decision": "residual_guided_diverse_pool_ready",
            "selected_residual_candidate": "focus10",
            "should_run_pool": True,
            "expert_families": [
                "trail_position_anchor",
                "compressed_span_structural",
                "residual_focus_aux_word",
                "residual_focus_source_selected_aux",
            ],
            "planned_fixed_fusions": [
                "best_single",
                "trail_position + raw117",
                "trail_position + raw117 + residual_focus",
                "trail_position + raw117 + source_selected_residual_focus",
            ],
            "source_selection_status": "pass",
            "source_selection_decision": "residual_axis_spectrum_stable_groups_selected",
            "source_selection_recommended_feature_prefixes": ["aux_depth_word_", "aux_word_"],
            "source_selection_selected_groups": [
                "aux_depth_word_global_mean",
                "aux_word_global_mean",
            ],
            "next_action": {"branch": "instantiate_residual_guided_pool3_fixed_fusion"},
        },
    )
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(
        tmp_path / "state_token_plan.json",
        {
            "status": "hold",
            "decision": "repair_state_token_controls_before_pool",
            "state_token_control_status": "hold",
            "state_token_failing_seed_count": 2,
        },
    )
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "ready"
    assert report["decision"] == "instantiate_residual_guided_pool3_fixed_fusion"
    assert report["selected_next_action"]["branch"] == "instantiate_residual_guided_pool3_fixed_fusion"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "ready"
    assert report["candidate_routes"]["pool3_residual_guided"]["selected_residual_candidate"] == "focus10"
    assert "residual_focus_source_selected_aux" in report["candidate_routes"]["pool3_residual_guided"]["expert_families"]
    assert "trail_position + raw117 + source_selected_residual_focus" in report["candidate_routes"]["pool3_residual_guided"]["planned_fixed_fusions"]
    assert report["candidate_routes"]["pool3_residual_guided"]["source_selection_status"] == "pass"
    assert report["candidate_routes"]["pool3_residual_guided"]["source_selection_recommended_feature_prefixes"] == [
        "aux_depth_word_",
        "aux_word_",
    ]
    assert report["candidate_routes"]["pool3_residual_guided"]["source_selection_selected_groups"] == [
        "aux_depth_word_global_mean",
        "aux_word_global_mean",
    ]
    assert report["candidate_routes"]["state_token_residual"]["status"] == "blocked_by_controls"
    readiness = report["post_gate_route_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["decision"] == "pool3_ready_linear_combo_tracked"
    assert readiness["primary_route"] == "pool3_residual_guided"
    assert readiness["selected_next_route"] == "trail_position + raw117 + source_selected_residual_focus"
    assert readiness["route_priority_order"] == [
        "pool3_residual_guided",
        "linear_combo_integral_residual",
        "bucket_conditioned_residual",
        "state_token_residual",
    ]
    assert readiness["priority_reason"] == (
        "prefer Pool3 when residual-focus and controls are ready; keep linear/integral as train-selected "
        "backup, bucket-conditioned as migration candidate, and state-token held by controls"
    )
    assert readiness["routes"]["pool3_residual_guided"] == {
        "status": "ready",
        "ready": True,
        "reason": "residual_guided_pool3_plan_ready",
        "selected_residual_candidate": "focus10",
        "source_selection_status": "pass",
        "planned_fixed_fusions": [
            "best_single",
            "trail_position + raw117",
            "trail_position + raw117 + residual_focus",
            "trail_position + raw117 + source_selected_residual_focus",
        ],
    }
    assert readiness["routes"]["linear_combo_integral_residual"] == {
        "status": "waiting_for_train_source_selection",
        "ready": False,
        "reason": "train_source_selection_required",
        "source_selection_summary_status": "",
        "selected_groups": [],
    }
    assert readiness["routes"]["state_token_residual"]["ready"] is False
    assert readiness["routes"]["state_token_residual"]["reason"] == "controls_hold"
    assert readiness["routes"]["bucket_conditioned_residual"]["ready"] is False


def test_present_r8_diverse_route_summary_runs_pool_planner_after_gate_pass(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "gate_passed_pool_plan_needed",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
            "next_action": {"branch": "run_residual_guided_pool_planner"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "run_residual_guided_pool_planner"
    assert report["selected_next_action"]["branch"] == "run_residual_guided_pool_planner"
    assert report["post_gate_route_readiness"]["route_priority_order"] == [
        "pool3_residual_guided",
        "linear_combo_integral_residual",
        "bucket_conditioned_residual",
        "state_token_residual",
    ]
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "waiting_for_pool_plan"
    linear_combo = report["candidate_routes"]["linear_combo_integral_residual"]
    assert linear_combo["status"] == "waiting_for_train_source_selection"
    assert linear_combo["decision"] == "wait_for_train_axis_spectrum_before_linear_combo_integral_expert"


def test_present_r8_diverse_route_summary_tracks_linear_combo_after_source_selection(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "gate_passed_pool_plan_needed",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
            "source_selection_summary_status": "pass",
            "source_selection_summary_decision": "residual_axis_spectrum_stable_groups_selected",
            "source_selection_recommended_feature_prefixes": ["aux_depth_word_", "aux_word_"],
            "source_selection_selected_groups": ["aux_depth_word_global_mean"],
            "next_action": {"branch": "run_residual_guided_pool_planner"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    route = report["candidate_routes"]["linear_combo_integral_residual"]
    assert status == 0
    assert route["status"] == "planned_after_source_selection"
    assert route["decision"] == "plan_linear_combo_integral_residual_expert"
    assert route["recommended_feature_prefixes"] == ["aux_depth_word_", "aux_word_"]
    assert route["selected_groups"] == ["aux_depth_word_global_mean"]
    assert route["next_action"]["should_launch_remote"] is False


def test_present_r8_diverse_route_summary_runs_residual_gate_after_outputs_ready(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "outputs_ready_gate_needed",
            "gate_status": "missing",
            "gate_decision": "",
            "missing_output_count": 0,
            "next_action": {"branch": "run_residual_focus_gate"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "run_residual_focus_gate"
    assert report["selected_next_action"]["branch"] == "run_residual_focus_gate"
    assert report["residual_focus"]["missing_output_count"] == 0
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "blocked_by_residual_focus"


def test_present_r8_diverse_route_summary_repairs_failed_residual_focus_first(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "repair_ready",
            "gate_status": "fail",
            "gate_decision": "hold_residual_focus_262k_controls_failed",
            "next_action": {"branch": "separate_focus_from_uniform_residual_objective"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "hold"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "ready"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "separate_focus_from_uniform_residual_objective"
    assert report["selected_next_action"]["branch"] == "separate_focus_from_uniform_residual_objective"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "blocked_by_residual_focus"


def test_present_r8_diverse_route_summary_uses_residual_repair_branch_as_decision(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "gate_failed",
            "gate_status": "fail",
            "gate_decision": "hold_residual_focus_262k_controls_failed",
            "next_action": {"branch": "repair_residual_focus_controls_before_scaleup"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "ready"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_focus_controls_before_scaleup"
    assert report["selected_next_action"]["branch"] == "repair_residual_focus_controls_before_scaleup"


def test_present_r8_diverse_route_summary_documents_evaluated_pool3_support(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "pool_evaluated",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
        },
    )
    pool_plan = _write_json(
        tmp_path / "pool_plan.json",
        {
            "status": "pass",
            "decision": "residual_guided_diverse_pool_ready",
            "should_run_pool": True,
            "selected_residual_candidate": "focus10",
        },
    )
    pool_eval = _write_json(
        tmp_path / "pool_eval.json",
        {
            "status": "pass",
            "decision": "support_residual_guided_pool3_fixed_fusion",
            "next_action": {"branch": "document_residual_guided_pool3_fixed_fusion"},
        },
    )
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "hold"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "document_residual_guided_pool3_fixed_fusion"
    assert report["selected_next_action"]["branch"] == "document_residual_guided_pool3_fixed_fusion"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "evaluated"
    assert report["candidate_routes"]["pool3_residual_guided"]["decision"] == "support_residual_guided_pool3_fixed_fusion"


def test_present_r8_diverse_route_summary_repairs_pool3_control_hold(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "pool3_control_hold",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
        },
    )
    pool_plan = _write_json(
        tmp_path / "pool_plan.json",
        {
            "status": "pass",
            "decision": "residual_guided_diverse_pool_ready",
            "should_run_pool": True,
            "selected_residual_candidate": "focus10",
        },
    )
    pool_eval = _write_json(
        tmp_path / "pool_eval.json",
        {
            "status": "hold",
            "decision": "residual_guided_pool3_fixed_fusion_diagnostic_only",
            "next_action": {"branch": "repair_residual_guided_pool3_before_scaleup"},
        },
    )
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "ready"})
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_guided_pool3_before_scaleup"
    assert report["selected_next_action"]["branch"] == "repair_residual_guided_pool3_before_scaleup"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "control_hold"
    assert report["candidate_routes"]["pool3_residual_guided"]["decision"] == (
        "residual_guided_pool3_fixed_fusion_diagnostic_only"
    )


def test_present_r8_diverse_route_summary_reports_bucket_residual_pending_migration(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "running",
            "gate_status": "pending",
            "gate_decision": "wait_for_residual_focus_262k_outputs",
            "next_action": {"branch": "wait_for_residual_focus_outputs"},
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    bucket_plan = _write_json(
        tmp_path / "bucket_plan.json",
        {
            "status": "pending",
            "decision": "wait_for_trail_position_262k_score_artifacts",
            "should_run": False,
            "reason": "trail_position_262k_postprocess_not_ready",
            "source_status": "pass",
            "source_decision": "hold_trail_position_score_residual_mixed_runs",
            "missing": ["validation_trail_position_scores"],
            "next_action": "Let the local tmux watchers retrieve and verify the 262k score artifacts first.",
        },
    )
    bucket_gate = _write_json(
        tmp_path / "bucket_gate.json",
        {
            "status": "pass",
            "decision": "bucket_conditioned_residual_controls_pass_local_diagnostic",
            "seed_count": 2,
            "min_three_vs_two_auc_delta": 0.000005,
            "min_bucket_vs_nobucket_auc_delta": 0.000028,
            "next_action": {"branch": "wait_for_262k_trail_position_artifacts_then_run_v16_planner"},
        },
    )
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--bucket-residual-plan",
            str(bucket_plan),
            "--bucket-residual-control-gate",
            str(bucket_gate),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    route = report["candidate_routes"]["bucket_conditioned_residual"]
    assert status == 0
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert route["status"] == "pending_262k_artifacts"
    assert route["control_status"] == "pass"
    assert route["plan_status"] == "pending"
    assert route["plan_decision"] == "wait_for_trail_position_262k_score_artifacts"
    assert route["reason"] == "trail_position_262k_postprocess_not_ready"
    assert route["source_status"] == "pass"
    assert route["source_decision"] == "hold_trail_position_score_residual_mixed_runs"
    assert route["missing"] == ["validation_trail_position_scores"]
    assert route["min_three_vs_two_auc_delta"] == 0.000005


def test_present_r8_diverse_route_summary_blocks_bucket_residual_failed_controls(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "running",
            "gate_status": "pending",
            "gate_decision": "wait_for_residual_focus_262k_outputs",
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    bucket_plan = _write_json(
        tmp_path / "bucket_plan.json",
        {
            "status": "pass",
            "decision": "bucket_residual_262k_action_plan_ready",
            "should_run": True,
            "next_action": "Run commands after postprocess remains pass.",
        },
    )
    bucket_gate = _write_json(
        tmp_path / "bucket_gate.json",
        {
            "status": "fail",
            "decision": "hold_bucket_conditioned_residual_controls_failed",
            "errors": ["seed1: validation_bucket_shuffle_three_score_not_below_two_score"],
            "next_action": {"branch": "repair_bucket_conditioned_residual_controls"},
        },
    )
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--bucket-residual-plan",
            str(bucket_plan),
            "--bucket-residual-control-gate",
            str(bucket_gate),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    route = report["candidate_routes"]["bucket_conditioned_residual"]
    assert status == 0
    assert route["status"] == "blocked_by_controls"
    assert route["control_status"] == "fail"
    assert route["errors"] == ["seed1: validation_bucket_shuffle_three_score_not_below_two_score"]
    assert route["next_action"]["branch"] == "repair_bucket_conditioned_residual_controls"


def test_present_r8_diverse_route_summary_reports_bucket_residual_ready(tmp_path: Path):
    residual_status = _write_json(
        tmp_path / "residual_status.json",
        {
            "status": "pool_ready",
            "gate_status": "pass",
            "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
        },
    )
    pool_plan = _write_json(tmp_path / "pool_plan.json", {"status": "pending"})
    pool_eval = _write_json(tmp_path / "pool_eval.json", {"status": "pending"})
    state_token_plan = _write_json(tmp_path / "state_token_plan.json", {"status": "pending"})
    bucket_plan = _write_json(
        tmp_path / "bucket_plan.json",
        {
            "status": "pass",
            "decision": "bucket_residual_262k_action_plan_ready",
            "should_run": True,
            "gate_output": "bucket_residual_controls_gate.json",
            "next_action": "Run these commands only after the 262k trail-position postprocess remains pass.",
        },
    )
    bucket_gate = _write_json(
        tmp_path / "bucket_gate.json",
        {
            "status": "pass",
            "decision": "bucket_conditioned_residual_controls_pass_local_diagnostic",
            "seed_count": 2,
            "min_three_vs_two_auc_delta": 0.000005,
            "next_action": {"branch": "wait_for_262k_trail_position_artifacts_then_run_v16_planner"},
        },
    )
    output = tmp_path / "summary.json"

    status = summarize_route_main(
        [
            "--residual-status",
            str(residual_status),
            "--pool-plan",
            str(pool_plan),
            "--pool-eval",
            str(pool_eval),
            "--state-token-plan",
            str(state_token_plan),
            "--bucket-residual-plan",
            str(bucket_plan),
            "--bucket-residual-control-gate",
            str(bucket_gate),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    route = report["candidate_routes"]["bucket_conditioned_residual"]
    assert status == 0
    assert route["status"] == "ready_262k_migration_plan"
    assert route["plan_status"] == "pass"
    assert route["control_status"] == "pass"
    assert route["should_run"] is True
    assert route["next_action"]["branch"] == "wait_for_262k_trail_position_artifacts_then_run_v16_planner"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
