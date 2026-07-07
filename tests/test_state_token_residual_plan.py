from __future__ import annotations

import json

from blockcipher_nd.cli.plan_state_token_residual_expert import (
    main as plan_state_token_main,
)


def test_state_token_residual_plan_waits_for_pending_residual_focus(tmp_path):
    status = tmp_path / "residual_status.json"
    output = tmp_path / "state_token_plan.json"
    status.write_text(
        json.dumps(
            {
                "status": "running",
                "gate_status": "pending",
                "gate_decision": "wait_for_residual_focus_262k_outputs",
                "missing_output_count": 18,
                "next_action": {"branch": "wait_for_residual_focus_outputs"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = plan_state_token_main(["--status", str(status), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs_before_state_token_expert"
    assert report["route"] == "present_r8_state_token_residual_expert"
    assert report["should_launch_remote"] is False
    assert report["missing_output_count"] == 18
    assert "launch_state_token_remote" in report["forbidden_actions"]
    assert report["allowed_local_actions"] == [
        "finalize_state_token_experiment_plan",
        "prepare_local_smoke_tests_only",
    ]


def test_state_token_residual_plan_keeps_pending_residual_focus_ahead_of_control_summary(tmp_path):
    status = tmp_path / "residual_status.json"
    control_summary = tmp_path / "state_token_control_summary.json"
    output = tmp_path / "state_token_plan.json"
    status.write_text(
        json.dumps(
            {
                "status": "running",
                "gate_status": "pending",
                "gate_decision": "wait_for_residual_focus_262k_outputs",
                "missing_output_count": 18,
                "next_action": {"branch": "wait_for_residual_focus_outputs"},
            }
        ),
        encoding="utf-8",
    )
    control_summary.write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "hold_state_token_coordinate_controls",
                "failing_seed_count": 2,
                "failing_control_event_count": 6,
            }
        ),
        encoding="utf-8",
    )

    exit_code = plan_state_token_main(
        [
            "--status",
            str(status),
            "--control-summary",
            str(control_summary),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs_before_state_token_expert"
    assert report["state_token_control_status"] == "hold"
    assert report["state_token_failing_seed_count"] == 2


def test_state_token_residual_plan_ready_after_residual_focus_pass(tmp_path):
    status = tmp_path / "residual_status.json"
    output = tmp_path / "state_token_plan.json"
    status.write_text(
        json.dumps(
            {
                "status": "ready",
                "gate_status": "pass",
                "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
                "pool_eval_status": "pending",
                "next_action": {"branch": "run_residual_guided_pool3"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = plan_state_token_main(["--status", str(status), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "ready"
    assert report["decision"] == "state_token_residual_expert_local_plan_ready"
    assert report["should_launch_remote"] is False
    assert report["activation_gate"]["gate_status"] == "pass"
    assert report["candidate"]["model_family"] == "state_token_residual_graph"
    assert report["candidate"]["objective"] == "frozen_base_residual_correction"
    assert report["candidate"]["primary_tokens"] == [
        "depth_word_cell_span",
        "depth_cell_span",
        "word_span",
        "depth_word_span",
        "cell_span",
    ]
    assert report["required_controls"] == [
        "same_input_global_control",
        "uniform_residual_control",
        "label_shuffle_control",
        "token_coordinate_shuffle_control",
        "token_coordinate_drop_control",
        "train_only_selection_control",
    ]


def test_state_token_residual_plan_holds_when_control_summary_holds(tmp_path):
    status = tmp_path / "residual_status.json"
    control_summary = tmp_path / "state_token_control_summary.json"
    output = tmp_path / "state_token_plan.json"
    status.write_text(
        json.dumps(
            {
                "status": "ready",
                "gate_status": "pass",
                "gate_decision": "keep_residual_focus_262k_hard_slice_candidate",
                "pool_eval_status": "pending",
                "next_action": {"branch": "run_residual_guided_pool3"},
            }
        ),
        encoding="utf-8",
    )
    control_summary.write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "hold_state_token_coordinate_controls",
                "failing_seed_count": 2,
                "failing_control_event_count": 6,
                "next_action": {"branch": "do_not_promote_state_token_coordinate_route"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = plan_state_token_main(
        [
            "--status",
            str(status),
            "--control-summary",
            str(control_summary),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_state_token_controls_before_pool"
    assert report["should_launch_remote"] is False
    assert report["state_token_control_summary"] == str(control_summary)
    assert report["state_token_control_status"] == "hold"
    assert report["state_token_control_decision"] == "hold_state_token_coordinate_controls"
    assert report["state_token_failing_seed_count"] == 2
    assert report["allowed_local_actions"] == ["repair_state_token_coordinate_or_value_only_controls"]


def test_state_token_residual_plan_repairs_when_residual_focus_holds(tmp_path):
    status = tmp_path / "residual_status.json"
    output = tmp_path / "state_token_plan.json"
    status.write_text(
        json.dumps(
            {
                "status": "running",
                "gate_status": "fail",
                "gate_decision": "hold_residual_focus_262k_controls_failed",
                "repair_hints": ["candidate_not_better_than_uniform_control"],
                "next_action": {"branch": "repair_residual_focus_controls_before_scaleup"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = plan_state_token_main(["--status", str(status), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_focus_before_state_token_expert"
    assert report["repair_hints"] == ["candidate_not_better_than_uniform_control"]
    assert report["should_launch_remote"] is False
    assert report["allowed_local_actions"] == ["repair_residual_focus_source_or_objective"]
