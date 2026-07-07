from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.watch_residual_focus_results import main as watch_main
from blockcipher_nd.cli.watch_residual_focus_results import watch_residual_focus_results


def test_watch_residual_focus_results_runs_one_pending_iteration(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    advance_output = tmp_path / "advance.json"
    output = tmp_path / "watch.json"
    monitor_dir = tmp_path / "monitor"
    artifact_root = tmp_path / "artifacts"
    progress = artifact_root / "seed0" / "dataset_cache" / "progress.jsonl"
    monitor_dir.mkdir(parents=True)
    progress.parent.mkdir(parents=True)
    monitor_dir.joinpath("monitor.log").write_text("2026-07-07T14:17:57+08:00 running missing=4\n", encoding="utf-8")
    progress.write_text(
        json.dumps(
            {
                "time": 10.0,
                "event": "cache_positive_chunk",
                "stage": "dataset_cache",
                "seed": 0,
                "split": "train",
                "rows_done": 8192,
                "total_rows": 524288,
                "class_rows_done": 8192,
                "class_total": 262144,
                "samples_per_class": 262144,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = watch_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--pool-eval-output",
            str(pool_eval),
            "--status-output",
            str(status_output),
            "--advance-output",
            str(advance_output),
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
            "--max-iterations",
            "1",
            "--sleep-seconds",
            "0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    advance = json.loads(advance_output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["iteration_count"] == 1
    assert report["terminal"] is False
    assert report["planned_output_count"] == 4
    assert report["existing_planned_output_count"] == 0
    assert report["missing_outputs"] == advance["missing_outputs"]
    assert advance["ran_gate"] is False
    assert advance["ran_pool_planner"] is False
    assert report["latest_monitor_event"] == "running missing=4"
    assert report["progress_summary"]["event"] == "cache_positive_chunk"
    assert report["progress_summary"]["class_progress_fraction"] == 0.03125
    assert report["progress_by_seed_split"][0]["seed"] == 0
    assert report["source_selection_report_count"] == 0
    assert report["source_selection_missing_report_count"] == 0


def test_watch_residual_focus_results_passes_repair_output_to_advance(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    repair = tmp_path / "repair.json"
    status_output = tmp_path / "status.json"
    advance_output = tmp_path / "advance.json"
    output = tmp_path / "watch.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = watch_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--pool-eval-output",
            str(pool_eval),
            "--repair-output",
            str(repair),
            "--status-output",
            str(status_output),
            "--advance-output",
            str(advance_output),
            "--output",
            str(output),
            "--max-iterations",
            "1",
            "--sleep-seconds",
            "0",
        ]
    )

    advance = json.loads(advance_output.read_text(encoding="utf-8"))
    report = json.loads(output.read_text(encoding="utf-8"))
    status_report = json.loads(status_output.read_text(encoding="utf-8"))
    assert status == 0
    assert advance["repair_plan"] == str(repair)
    assert status_report["repair_plan"] == str(repair)
    assert report["repair_plan"] == str(repair)
    assert report["repair_context_current"] is False


def test_watch_residual_focus_results_writes_report_each_iteration(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    advance_output = tmp_path / "advance.json"
    output = tmp_path / "watch.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = watch_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--pool-eval-output",
            str(pool_eval),
            "--status-output",
            str(status_output),
            "--advance-output",
            str(advance_output),
            "--output",
            str(output),
            "--max-iterations",
            "1",
            "--sleep-seconds",
            "0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["iteration_count"] == 1
    assert report["terminal"] is False


def test_watch_residual_focus_results_function_writes_iteration_report(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    advance_output = tmp_path / "advance.json"
    output = tmp_path / "watch.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    report = watch_residual_focus_results(
        action_plan=action_plan,
        gate_output=gate,
        pool_output=pool,
        pool_eval_output=pool_eval,
        status_output=status_output,
        advance_output=advance_output,
        monitor_dir=tmp_path / "monitor",
        artifact_root=tmp_path / "artifact_root",
        output=output,
        max_iterations=1,
        sleep_seconds=0,
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pending"
    assert written["status"] == "pending"
    assert written["iteration_count"] == 1


def test_watch_residual_focus_results_waits_for_pool3_score_artifacts(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    advance_output = tmp_path / "advance.json"
    output = tmp_path / "watch.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = watch_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--pool-eval-output",
            str(pool_eval),
            "--status-output",
            str(status_output),
            "--advance-output",
            str(advance_output),
            "--output",
            str(output),
            "--max-iterations",
            "1",
            "--sleep-seconds",
            "0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    advance = json.loads(advance_output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_pool3_score_artifacts"
    assert report["iteration_count"] == 1
    assert report["terminal"] is False
    assert advance["ran_gate"] is True
    assert advance["ran_pool_planner"] is True
    assert report["ran_pool_evaluator"] is False
    assert report["pool_eval_status"] == "pending"
    assert report["pool_eval_decision"] == "wait_for_pool3_score_artifacts"
    assert report["missing_pool3_score_artifact_count"] == advance["missing_pool3_score_artifact_count"]
    assert report["missing_pool3_score_artifacts"] == advance["missing_pool3_score_artifacts"]


def _write_action_plan(tmp_path: Path, *, create_outputs: bool) -> Path:
    outputs = {
        "uniform_slice_eval": tmp_path / "uniform_slice_eval.json",
        "focus10_shuffle_slice_eval": tmp_path / "focus10_shuffle_slice_eval.json",
        "focus05_slice_eval": tmp_path / "focus05_slice_eval.json",
        "focus10_slice_eval": tmp_path / "focus10_slice_eval.json",
    }
    if create_outputs:
        _write_slice(outputs["uniform_slice_eval"], loss_delta=-0.001, auc_delta=0.001)
        _write_slice(outputs["focus10_shuffle_slice_eval"], loss_delta=0.02, auc_delta=-0.1)
        _write_slice(outputs["focus05_slice_eval"], loss_delta=-0.01, auc_delta=0.002)
        _write_slice(outputs["focus10_slice_eval"], loss_delta=-0.02, auc_delta=0.003)
    action_plan = tmp_path / "action_plan.json"
    action_plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "seeds": [
                    {
                        "seed": 0,
                        "planned_outputs": {key: str(path) for key, path in outputs.items()},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return action_plan


def _write_slice(path: Path, *, loss_delta: float, auc_delta: float) -> None:
    path.write_text(
        json.dumps(
            {
                "focus": {"mode": "train_derived_base_residual_loss_threshold"},
                "validation_focus_metrics": {"rows": 16},
                "validation_focus_delta": {
                    "auc": auc_delta,
                    "residual_loss_mean": loss_delta,
                },
            }
        ),
        encoding="utf-8",
    )
