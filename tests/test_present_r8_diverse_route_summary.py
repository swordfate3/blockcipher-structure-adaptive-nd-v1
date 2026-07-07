from __future__ import annotations

import json
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
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["selected_next_action"]["branch"] == "wait_for_residual_focus_outputs"
    assert report["candidate_routes"]["state_token_residual"]["status"] == "blocked_by_controls"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "blocked_by_residual_focus"
    assert report["should_launch_remote"] is False


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
    assert report["candidate_routes"]["state_token_residual"]["status"] == "blocked_by_controls"


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
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "waiting_for_pool_plan"


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
    assert report["decision"] == "repair_residual_focus_before_diverse_pool"
    assert report["selected_next_action"]["branch"] == "separate_focus_from_uniform_residual_objective"
    assert report["candidate_routes"]["pool3_residual_guided"]["status"] == "blocked_by_residual_focus"


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
