from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import blockcipher_nd.cli.advance_residual_focus_results as advance_cli
from blockcipher_nd.cli.advance_residual_focus_results import main as advance_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


@pytest.fixture(autouse=True)
def _isolate_default_repair_output(tmp_path, monkeypatch):
    monkeypatch.setattr(advance_cli, "DEFAULT_REPAIR_OUTPUT", tmp_path / "isolated_default_repair.json")


def test_advance_residual_focus_results_uses_isolated_default_repair_output(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    status_report = json.loads(status_output.read_text(encoding="utf-8"))
    isolated_repair = tmp_path / "isolated_default_repair.json"
    assert status == 0
    assert report["repair_plan"] == str(isolated_repair)
    assert status_report["repair_plan"] == str(isolated_repair)


def test_advance_residual_focus_results_waits_when_outputs_missing(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    monitor_dir = tmp_path / "monitor"
    artifact_root = tmp_path / "artifacts"
    progress = artifact_root / "seed0" / "dataset_cache" / "progress.jsonl"
    monitor_dir.mkdir(parents=True)
    progress.parent.mkdir(parents=True)
    monitor_dir.joinpath("monitor.log").write_text("2026-07-07T14:17:57+08:00 running missing=8\n", encoding="utf-8")
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

    status = advance_main(
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
            "--monitor-dir",
            str(monitor_dir),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    action_payload = json.loads(action_plan.read_text(encoding="utf-8"))
    expected_missing_outputs = sorted(
        output_path
        for seed_plan in action_payload["seeds"]
        for output_path in seed_plan["planned_outputs"].values()
    )
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["ran_gate"] is False
    assert report["ran_pool_planner"] is False
    assert report["planned_output_count"] == 8
    assert report["existing_planned_output_count"] == 0
    assert report["missing_outputs"] == expected_missing_outputs
    assert report["latest_monitor_event"] == "running missing=8"
    assert report["progress_summary"]["event"] == "cache_positive_chunk"
    assert report["progress_summary"]["class_progress_fraction"] == 0.03125
    assert report["progress_by_seed_split"][0]["seed"] == 0
    assert not pool.exists()


def test_advance_residual_focus_results_marks_stale_repair_plan_when_outputs_missing(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    repair = tmp_path / "repair.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )
    repair.write_text(
        json.dumps(
            {
                "status": "ready",
                "decision": "repair_residual_guided_pool3_before_scaleup",
                "source_summary": str(tmp_path / "old_pool_eval.json"),
                "primary_repair_branch": "repair_residual_guided_pool3_controls",
            }
        ),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    status_report = json.loads(status_output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["repair_status"] == "stale"
    assert report["repair_source_summary"] == str(tmp_path / "old_pool_eval.json")
    assert report["repair_context_current"] is False
    assert report["repair_primary_branch"] == "repair_residual_guided_pool3_controls"
    assert status_report["repair_status"] == "stale"
    assert status_report["next_action"]["branch"] == "wait_for_residual_focus_outputs"


def test_advance_residual_focus_results_runs_gate_and_pool_when_outputs_ready(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    status_report = json.loads(status_output.read_text(encoding="utf-8"))
    gate_report = json.loads(gate.read_text(encoding="utf-8"))
    pool_report = json.loads(pool.read_text(encoding="utf-8"))
    assert status == 0
    assert report["ran_gate"] is True
    assert report["ran_pool_planner"] is True
    assert gate_report["decision"] == "keep_residual_focus_262k_hard_slice_candidate"
    assert pool_report["decision"] == "residual_guided_diverse_pool_ready"
    assert pool_report["selected_residual_candidate"] == "focus10"
    assert report["ran_pool_evaluator"] is False
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_pool3_score_artifacts"
    assert report["pool_eval_status"] == "pending"
    assert report["pool_eval_decision"] == "wait_for_pool3_score_artifacts"
    assert report["missing_pool3_score_artifact_count"] > 0
    assert report["source_selection_report_count"] == 4
    assert report["source_selection_existing_report_count"] == 0
    assert report["source_selection_missing_report_count"] == 4
    assert report["source_selection_missing_reports"] == [
        str(tmp_path / "seed0" / "train_residual_loss_axis_spectrum.json"),
        str(tmp_path / "seed0" / "train_hard_error_axis_spectrum.json"),
        str(tmp_path / "seed1" / "train_residual_loss_axis_spectrum.json"),
        str(tmp_path / "seed1" / "train_hard_error_axis_spectrum.json"),
    ]
    assert status_report["pool_eval"] == str(pool_eval)
    assert status_report["pool_eval_status"] == "pending"
    assert status_report["pool_eval_decision"] == "wait_for_pool3_score_artifacts"


def test_advance_residual_focus_results_runs_source_selection_summary_when_reports_exist(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True, create_source_reports=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    action_payload = json.loads(action_plan.read_text(encoding="utf-8"))
    summary_path = Path(action_payload["source_selection_summary_output"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert report["ran_source_selection_summary"] is True
    assert report["source_selection_summary_status"] == "pass"
    assert report["source_selection_summary_decision"] == "residual_axis_spectrum_stable_groups_selected"
    assert report["source_selection_summary_output"] == str(summary_path)
    assert report["source_selection_report_count"] == 4
    assert report["source_selection_existing_report_count"] == 4
    assert report["source_selection_missing_report_count"] == 0
    assert report["source_selection_missing_reports"] == []
    assert summary["recommended_feature_prefixes"][:2] == ["aux_word_", "aux_depth_word_"]
    assert "validation" not in "\n".join(summary["spectrum_reports"])


def test_advance_residual_focus_results_defaults_source_summary_under_artifact_root_for_old_plan(
    tmp_path,
):
    artifact_root = tmp_path / "residual_focus262k"
    action_plan = _write_action_plan(
        tmp_path,
        create_outputs=True,
        create_source_reports=True,
        include_source_summary_output=False,
    )
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    summary_path = artifact_root / "residual_axis_spectrum_summary.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["ran_source_selection_summary"] is True
    assert report["source_selection_summary_output"] == str(summary_path)
    assert summary_path.exists()


def test_advance_residual_focus_results_runs_pool_evaluator_when_score_artifacts_exist(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True, create_score_artifacts=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    pool_eval_report = json.loads(pool_eval.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "residual_guided_pool_evaluated"
    assert report["ran_pool_evaluator"] is True
    assert report["pool_eval_status"] == "pass"
    assert pool_eval_report["decision"] == "residual_guided_pool3_fixed_fusion_evaluated"
    assert [seed_report["seed"] for seed_report in pool_eval_report["seed_reports"]] == [0, 1]
    assert all(seed_report["decision"] == "support_residual_guided_pool3_fixed_fusion" for seed_report in pool_eval_report["seed_reports"])


def test_advance_residual_focus_results_holds_when_pool_evaluator_controls_fail(tmp_path):
    action_plan = _write_action_plan(
        tmp_path,
        create_outputs=True,
        create_score_artifacts=True,
        bad_score_seed=1,
    )
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    pool_eval_report = json.loads(pool_eval.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_guided_pool3_before_scaleup"
    assert report["pool_eval_status"] == "hold"
    assert pool_eval_report["decision"] == "residual_guided_pool3_fixed_fusion_mixed_or_controlled"
    assert [seed_report["status"] for seed_report in pool_eval_report["seed_reports"]] == ["pass", "hold"]


def test_advance_residual_focus_results_writes_repair_plan_when_gate_fails(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True, strong_uniform=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    pool_eval = tmp_path / "pool_eval.json"
    repair = tmp_path / "repair.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
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
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    repair_report = json.loads(repair.read_text(encoding="utf-8"))
    status_report = json.loads(status_output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "repair_residual_focus_before_pool"
    assert report["repair_status"] == "ready"
    assert report["repair_plan"] == str(repair)
    assert repair_report["primary_repair_branch"] == "separate_focus_from_uniform_residual_objective"
    assert status_report["status"] == "repair_ready"
    assert status_report["repair_plan"] == str(repair)


def _write_action_plan(
    tmp_path: Path,
    *,
    create_outputs: bool,
    create_source_reports: bool = False,
    include_source_summary_output: bool = True,
    create_score_artifacts: bool = False,
    bad_score_seed: int | None = None,
    strong_uniform: bool = False,
) -> Path:
    seeds = []
    for seed in [0, 1]:
        seed_root = tmp_path / f"seed{seed}"
        outputs = {
            "uniform_slice_eval": seed_root / "uniform_slice_eval.json",
            "focus10_shuffle_slice_eval": seed_root / "focus10_shuffle_slice_eval.json",
            "focus05_slice_eval": seed_root / "focus05_slice_eval.json",
            "focus10_slice_eval": seed_root / "focus10_slice_eval.json",
        }
        if create_outputs:
            seed_root.mkdir(parents=True, exist_ok=True)
            _write_slice(outputs["uniform_slice_eval"], loss_delta=-0.03 if strong_uniform else -0.001, auc_delta=0.001)
            _write_slice(outputs["focus10_shuffle_slice_eval"], loss_delta=0.03, auc_delta=-0.1)
            _write_slice(outputs["focus05_slice_eval"], loss_delta=-0.01, auc_delta=0.002)
            _write_slice(outputs["focus10_slice_eval"], loss_delta=-0.02, auc_delta=0.003)
        if create_score_artifacts:
            _write_pool3_score_artifacts(seed_root, seed=seed, bad_scores=seed == bad_score_seed)
        source_selection_outputs = {
            "train_residual_loss_axis_spectrum": str(seed_root / "train_residual_loss_axis_spectrum.json"),
            "train_hard_error_axis_spectrum": str(seed_root / "train_hard_error_axis_spectrum.json"),
        }
        if create_source_reports:
            _write_axis_spectrum_report(
                seed_root / "train_residual_loss_axis_spectrum.json",
                feature_dir=f"outputs/run/seed{seed}/train_span_summary_features",
                target="residual_loss",
                global_groups=[("aux_word_global_mean", 0.07), ("aux_depth_word_global_mean", 0.06)],
                bucket_groups=[("aux_word_mean", 0.08)],
            )
            _write_axis_spectrum_report(
                seed_root / "train_hard_error_axis_spectrum.json",
                feature_dir=f"outputs/run/seed{seed}/train_span_summary_features",
                target="residual_error_at_0_5",
                global_groups=[("aux_cell_global_max", 0.48)],
                bucket_groups=[("aux_cell_mean", 0.47)],
            )
        seeds.append(
            {
                "seed": seed,
                "artifact_root": str(seed_root),
                "validation_trail_position_scores": str(seed_root / "validation_trail_position_scores"),
                "planned_outputs": {key: str(path) for key, path in outputs.items()},
                "source_selection_outputs": source_selection_outputs,
            }
        )
    payload = {
        "status": "pass",
        "source_decision": "hold_trail_position_score_residual_mixed_runs",
        "source_gate_assessment": "score_artifacts_ready_but_trail_position_gate_not_promoted",
        "seeds": seeds,
    }
    if include_source_summary_output:
        payload["source_selection_summary_output"] = str(tmp_path / "residual_axis_spectrum_summary.json")
    action_plan = tmp_path / "action_plan.json"
    action_plan.write_text(json.dumps(payload), encoding="utf-8")
    return action_plan


def _write_axis_spectrum_report(
    path: Path,
    *,
    feature_dir: str,
    target: str,
    global_groups: list[tuple[str, float]],
    bucket_groups: list[tuple[str, float]],
) -> None:
    def group_report(group: str, target_score: float) -> dict[str, object]:
        return {
            "group": group,
            "feature_count": 1,
            "label_auc": 0.5,
            "label_score": 0.0,
            "residual_error_auc": 0.5,
            "residual_error_score": 0.0,
            "target_auc": 0.5 + target_score,
            "target_score": target_score,
        }

    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "residual_bucket_axis_spectrum_ready",
                "feature_dir": feature_dir,
                "target": target,
                "row_count": 16,
                "residual_error_rate_at_0_5": 0.125,
                "global_top_groups": [group_report(group, score) for group, score in global_groups],
                "bucket_reports": [
                    {
                        "bucket": 0,
                        "rows": 8,
                        "top_groups": [group_report(group, score) for group, score in bucket_groups],
                    }
                ],
                "claim_scope": "train-only diagnostic",
            }
        ),
        encoding="utf-8",
    )


def _write_pool3_score_artifacts(seed_root: Path, *, seed: int, bad_scores: bool = False) -> None:
    _write_score_artifact(
        seed_root / "validation_trail_position_scores",
        "trail_position",
        [0.10, 0.60, 0.40, 0.90],
        family="trail_position_anchor",
        seed=seed,
    )
    _write_score_artifact(
        seed_root / "validation_raw117_scores",
        "raw117",
        [0.20, 0.55, 0.45, 0.80],
        family="compressed_span_structural",
        seed=seed,
    )
    _write_score_artifact(
        seed_root / "residual_focus10_validation_scores",
        "residual_focus10",
        [0.90, 0.80, 0.20, 0.10] if bad_scores else [0.45, 0.10, 0.90, 0.55],
        family="residual_focus_aux_word",
        seed=seed,
    )
    _write_score_artifact(
        seed_root / "residual_uniform_validation_scores",
        "residual_uniform",
        [0.50, 0.50, 0.50, 0.50],
        family="uniform_residual_control",
        seed=seed,
    )
    _write_score_artifact(
        seed_root / "residual_focus10_labelshuffle_validation_scores",
        "residual_focus10_labelshuffle",
        [0.90, 0.80, 0.20, 0.10],
        family="labelshuffle_residual_control",
        seed=seed,
    )


def _write_score_artifact(
    path: Path,
    model_key: str,
    probabilities: list[float],
    *,
    family: str,
    seed: int,
) -> None:
    probs = np.array(probabilities, dtype=np.float32)
    logits = np.log(np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0))
    write_score_artifact(
        path,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 1, 1], dtype=np.float32),
            probabilities=probs,
            logits=logits.astype(np.float32),
            sample_ids=np.array(["s0", "s1", "s2", "s3"], dtype=str),
            metadata={
                "cipher": "PRESENT-80",
                "rounds": 8,
                "seed": seed,
                "samples_per_class": 262144,
                "validation_samples_per_class": 262144,
                "pairs_per_sample": 16,
                "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
                "difference_profile": "present_zhang_wang2022_mcnd",
                "difference_member": 0,
                "train_key": "0x00000000000000000000",
                "validation_key": "0x11111111111111111111",
                "model_key": model_key,
                "run_id": f"run_seed{seed}_{model_key}",
                "checkpoint_path": f"/tmp/seed{seed}_{model_key}.pt",
                "expert_family": family,
                "candidate_status": "weak_positive",
                "git_commit": "test",
            },
        ),
    )


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
