from __future__ import annotations

import json

from blockcipher_nd.cli.plan_residual_focus_repair import main as repair_main


def test_plan_residual_focus_repair_maps_control_hints_to_repair_branches(tmp_path):
    gate = tmp_path / "gate.json"
    output = tmp_path / "repair_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "fail",
                "decision": "hold_residual_focus_262k_controls_failed",
                "repair_hints": [
                    "candidate_not_better_than_uniform_control",
                    "label_shuffle_control_failed",
                ],
                "errors": [
                    "seed0: focus10_not_better_than_uniform_focus_loss",
                    "seed0: label_shuffle_did_not_worsen_focus_loss",
                ],
            }
        ),
        encoding="utf-8",
    )

    status = repair_main(["--summary", str(gate), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "ready"
    assert report["decision"] == "repair_residual_focus_before_pool_or_scaleup"
    assert report["should_launch_remote"] is False
    assert report["primary_repair_branch"] == "separate_focus_from_uniform_residual_objective"
    assert report["repair_hints"] == [
        "candidate_not_better_than_uniform_control",
        "label_shuffle_control_failed",
    ]
    assert [row["branch"] for row in report["repair_branches"]] == [
        "separate_focus_from_uniform_residual_objective",
        "repair_label_shuffle_attribution_control",
    ]
    first_branch = report["repair_branches"][0]
    assert [command["name"] for command in first_branch["command_templates"]] == [
        "inspect_status",
        "rerun_repair_plan",
    ]
    assert all(command["remote"] is False for command in first_branch["command_templates"])
    assert "scripts/residual-focus-status" in first_branch["command_templates"][0]["command"]
    assert "scripts/plan-residual-focus-repair" in first_branch["command_templates"][1]["command"]
    assert first_branch["implementation_notes"] == [
        "Compare residual-focused and uniform correction on identical train-derived hard slices.",
        "Do not change labels, negative_mode, sample_structure, or validation split.",
    ]
    assert "launch_residual_guided_pool3" in report["forbidden_actions"]
    assert "scale_residual_focus_to_1m" in report["forbidden_actions"]


def test_plan_residual_focus_repair_waits_when_gate_is_pending(tmp_path):
    gate = tmp_path / "gate.json"
    output = tmp_path / "repair_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_residual_focus_262k_outputs",
                "missing_outputs": ["seed0/focus10_slice_eval.json"],
            }
        ),
        encoding="utf-8",
    )

    status = repair_main(["--summary", str(gate), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_gate_before_repair_plan"
    assert report["repair_branches"] == []
    assert report["missing_outputs"] == ["seed0/focus10_slice_eval.json"]


def test_plan_residual_focus_repair_handles_pool3_control_hold_without_hints(tmp_path):
    pool_eval = tmp_path / "pool_eval.json"
    output = tmp_path / "repair_plan.json"
    pool_eval.write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "residual_guided_pool3_fixed_fusion_mixed_or_controlled",
                "selected_residual_candidate": "focus10",
            }
        ),
        encoding="utf-8",
    )

    status = repair_main(["--summary", str(pool_eval), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "ready"
    assert report["primary_repair_branch"] == "repair_residual_guided_pool3_controls"
    assert report["repair_branches"][0]["success_gate"] == (
        "residual-focus fusion strictly beats trail+raw117 and both residual controls"
    )
    assert report["repair_branches"][0]["implementation_notes"] == [
        "Inspect per-seed Pool 3 fixed-fusion reports before changing any model.",
        "Do not launch 1M/class until residual-focus fusion beats both controls.",
    ]
