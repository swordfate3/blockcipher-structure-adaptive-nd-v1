from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plan_residual_guided_diverse_pool import main as plan_pool_main


def test_plan_residual_guided_pool_waits_for_pending_residual_focus_gate(tmp_path):
    gate = tmp_path / "gate.json"
    output = tmp_path / "pool_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_residual_focus_262k_outputs",
                "passing_candidates": [],
                "next_action": {"branch": "finish_residual_focus_262k_outputs"},
            }
        ),
        encoding="utf-8",
    )

    status = plan_pool_main(["--residual-focus-gate", str(gate), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_gate"
    assert report["should_run_pool"] is False
    assert report["next_action"]["branch"] == "finish_residual_focus_262k_outputs"


def test_plan_residual_guided_pool_promotes_passing_focus_candidate(tmp_path):
    gate = tmp_path / "gate.json"
    output = tmp_path / "pool_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "keep_residual_focus_262k_hard_slice_candidate",
                "passing_candidates": ["focus05", "focus10"],
                "min_focus05_vs_uniform_loss_margin": -0.002,
                "min_focus10_vs_uniform_loss_margin": -0.004,
                "min_shuffle_focus_loss_delta": 0.03,
                "claim_scope": "262144/class residual-focused hard-slice gate only",
            }
        ),
        encoding="utf-8",
    )

    status = plan_pool_main(["--residual-focus-gate", str(gate), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "residual_guided_diverse_pool_ready"
    assert report["should_run_pool"] is True
    assert report["selected_residual_candidate"] == "focus10"
    assert report["expert_families"] == [
        "trail_position_anchor",
        "compressed_span_structural",
        "residual_focus_aux_word",
    ]
    assert "uniform_residual_control" in report["control_families"]
    assert "labelshuffle_residual_control" in report["control_families"]
    assert "trail_position + raw117 + residual_focus" in report["planned_fixed_fusions"]
    assert report["claim_scope"].startswith("application-level medium diagnostic")


def test_plan_residual_guided_pool_includes_source_selected_residual_expert(tmp_path):
    gate = tmp_path / "gate.json"
    source_summary = tmp_path / "residual_axis_spectrum_summary.json"
    output = tmp_path / "pool_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "keep_residual_focus_262k_hard_slice_candidate",
                "passing_candidates": ["focus10"],
                "min_focus10_vs_uniform_loss_margin": -0.004,
            }
        ),
        encoding="utf-8",
    )
    source_summary.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "residual_axis_spectrum_stable_groups_selected",
                "recommended_feature_prefixes": ["aux_depth_word_", "aux_word_"],
                "selected_groups": ["aux_depth_word_global_mean", "aux_word_global_mean"],
            }
        ),
        encoding="utf-8",
    )

    status = plan_pool_main(
        [
            "--residual-focus-gate",
            str(gate),
            "--source-selection-summary",
            str(source_summary),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert "residual_focus_source_selected_aux" in report["expert_families"]
    assert "trail_position + raw117 + source_selected_residual_focus" in report["planned_fixed_fusions"]
    assert report["source_selection_status"] == "pass"
    assert report["source_selection_decision"] == "residual_axis_spectrum_stable_groups_selected"
    assert report["source_selection_recommended_feature_prefixes"] == ["aux_depth_word_", "aux_word_"]
    assert report["source_selection_selected_groups"] == [
        "aux_depth_word_global_mean",
        "aux_word_global_mean",
    ]


def test_plan_residual_guided_pool_holds_failed_residual_focus_gate(tmp_path):
    gate = tmp_path / "gate.json"
    output = tmp_path / "pool_plan.json"
    gate.write_text(
        json.dumps(
            {
                "status": "fail",
                "decision": "hold_residual_focus_262k_controls_failed",
                "passing_candidates": [],
                "errors": ["seed0: label_shuffle_did_not_worsen_focus_loss"],
                "repair_hints": ["label_shuffle_control_failed"],
                "next_action": {"branch": "repair_residual_focus_controls_before_scaleup"},
            }
        ),
        encoding="utf-8",
    )

    status = plan_pool_main(["--residual-focus-gate", str(gate), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_focus_before_pool"
    assert report["should_run_pool"] is False
    assert report["repair_hints"] == ["label_shuffle_control_failed"]
    assert report["next_action"]["branch"] == "repair_residual_focus_controls_before_scaleup"
