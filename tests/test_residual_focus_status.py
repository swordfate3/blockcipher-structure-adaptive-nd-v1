from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.residual_focus_status import main as status_main


def test_residual_focus_status_reports_running_with_progress_and_missing_outputs(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    progress = tmp_path / "artifacts" / "seed0" / "dataset_cache" / "progress.jsonl"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    progress.parent.mkdir(parents=True)
    monitor_dir.joinpath("monitor.log").write_text(
        "2026-07-07T14:17:56+08:00 sync\n"
        "2026-07-07T14:17:57+08:00 running missing=18\n",
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"event": "cache_start", "total_rows": 524288, "samples_per_class": 262144}) + "\n",
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_residual_focus_262k_outputs",
                "missing_outputs": ["missing0.json", "missing1.json"],
            }
        ),
        encoding="utf-8",
    )
    pool.write_text(json.dumps({"status": "pending", "should_run_pool": False}), encoding="utf-8")

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "running"
    assert report["gate_status"] == "pending"
    assert report["planned_output_count"] == 2
    assert report["existing_planned_output_count"] == 0
    assert report["missing_output_count"] == 2
    assert report["latest_monitor_event"] == "running missing=18"
    assert report["latest_progress"]["event"] == "cache_start"
    assert report["next_action"]["branch"] == "wait_for_residual_focus_outputs"


def test_residual_focus_status_reports_outputs_ready_before_gate(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    monitor_dir.joinpath("monitor.log").write_text("2026-07-07T14:31:57+08:00 sync\n", encoding="utf-8")
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )
    pool.write_text(json.dumps({"status": "pending", "should_run_pool": False}), encoding="utf-8")

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "outputs_ready_gate_needed"
    assert report["existing_planned_output_count"] == 2
    assert report["missing_output_count"] == 0
    assert report["next_action"]["branch"] == "run_residual_focus_gate"


def test_residual_focus_status_reports_pool_ready_after_gate(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    gate.write_text(
        json.dumps({"status": "pass", "decision": "keep_residual_focus_262k_hard_slice_candidate"}),
        encoding="utf-8",
    )
    pool.write_text(
        json.dumps({"status": "pass", "decision": "residual_guided_diverse_pool_ready", "should_run_pool": True}),
        encoding="utf-8",
    )

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pool_ready"
    assert report["should_run_pool"] is True
    assert report["next_action"]["branch"] == "instantiate_residual_guided_pool3_fixed_fusion"


def test_residual_focus_status_reports_waiting_for_pool3_score_artifacts(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    gate.write_text(
        json.dumps({"status": "pass", "decision": "keep_residual_focus_262k_hard_slice_candidate"}),
        encoding="utf-8",
    )
    pool.write_text(
        json.dumps({"status": "pass", "decision": "residual_guided_diverse_pool_ready", "should_run_pool": True}),
        encoding="utf-8",
    )
    pool_eval.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_pool3_score_artifacts",
                "missing_score_artifacts": ["seed0/validation_raw117_scores"],
            }
        ),
        encoding="utf-8",
    )

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--pool-eval",
            str(pool_eval),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pool3_scores_pending"
    assert report["pool_eval_status"] == "pending"
    assert report["pool_eval_decision"] == "wait_for_pool3_score_artifacts"
    assert report["missing_pool3_score_artifact_count"] == 1
    assert report["next_action"]["branch"] == "wait_for_pool3_score_artifacts"


def test_residual_focus_status_reports_pool3_control_hold(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    gate.write_text(
        json.dumps({"status": "pass", "decision": "keep_residual_focus_262k_hard_slice_candidate"}),
        encoding="utf-8",
    )
    pool.write_text(
        json.dumps({"status": "pass", "decision": "residual_guided_diverse_pool_ready", "should_run_pool": True}),
        encoding="utf-8",
    )
    pool_eval.write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "residual_guided_pool3_fixed_fusion_mixed_or_controlled",
            }
        ),
        encoding="utf-8",
    )

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--pool-eval",
            str(pool_eval),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pool3_control_hold"
    assert report["pool_eval_status"] == "hold"
    assert report["next_action"]["branch"] == "repair_residual_guided_pool3_before_scaleup"


def test_residual_focus_status_reports_repair_ready_when_repair_plan_exists(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    repair = tmp_path / "repair.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "status": "fail",
                "decision": "hold_residual_focus_262k_controls_failed",
                "repair_hints": ["candidate_not_better_than_uniform_control"],
            }
        ),
        encoding="utf-8",
    )
    pool.write_text(
        json.dumps(
            {
                "status": "hold",
                "decision": "repair_residual_focus_before_pool",
                "should_run_pool": False,
                "repair_hints": ["candidate_not_better_than_uniform_control"],
            }
        ),
        encoding="utf-8",
    )
    repair.write_text(
        json.dumps(
            {
                "status": "ready",
                "decision": "repair_residual_focus_before_pool_or_scaleup",
                "source_summary": str(gate),
                "primary_repair_branch": "separate_focus_from_uniform_residual_objective",
                "repair_hints": ["candidate_not_better_than_uniform_control"],
                "repair_branches": [{"branch": "separate_focus_from_uniform_residual_objective"}],
            }
        ),
        encoding="utf-8",
    )

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--repair-plan",
            str(repair),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "repair_ready"
    assert report["repair_status"] == "ready"
    assert report["repair_primary_branch"] == "separate_focus_from_uniform_residual_objective"
    assert report["repair_hints"] == ["candidate_not_better_than_uniform_control"]
    assert report["next_action"]["branch"] == "separate_focus_from_uniform_residual_objective"


def test_residual_focus_status_ignores_stale_repair_plan_while_gate_pending(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    repair = tmp_path / "repair.json"
    monitor_dir = tmp_path / "remote" / "monitor"
    output = tmp_path / "status.json"
    monitor_dir.mkdir(parents=True)
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )
    pool.write_text(json.dumps({"status": "pending", "should_run_pool": False}), encoding="utf-8")
    repair.write_text(
        json.dumps(
            {
                "status": "ready",
                "decision": "repair_residual_focus_before_pool_or_scaleup",
                "source_summary": str(tmp_path / "other_gate.json"),
                "primary_repair_branch": "separate_focus_from_uniform_residual_objective",
            }
        ),
        encoding="utf-8",
    )

    status = status_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate",
            str(gate),
            "--pool-plan",
            str(pool),
            "--repair-plan",
            str(repair),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "running"
    assert report["repair_status"] == "stale"
    assert report["repair_context_current"] is False
    assert report["next_action"]["branch"] == "wait_for_residual_focus_outputs"


def _write_action_plan(tmp_path: Path, *, create_outputs: bool) -> Path:
    paths = [
        tmp_path / "artifacts" / "seed0" / "residual_focus05_slice_eval.json",
        tmp_path / "artifacts" / "seed1" / "residual_focus05_slice_eval.json",
    ]
    if create_outputs:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    action_plan = tmp_path / "action_plan.json"
    action_plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "seeds": [
                    {
                        "seed": 0,
                        "planned_outputs": {"focus05_slice_eval": str(paths[0])},
                    },
                    {
                        "seed": 1,
                        "planned_outputs": {"focus05_slice_eval": str(paths[1])},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return action_plan
