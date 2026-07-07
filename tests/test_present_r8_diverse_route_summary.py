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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
