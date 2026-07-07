from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.evaluate_residual_guided_diverse_pool import main as evaluate_pool_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


def _metadata(model_key: str, *, family: str) -> dict[str, object]:
    return {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "seed": 0,
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
        "run_id": f"run_{model_key}",
        "checkpoint_path": f"/tmp/{model_key}.pt",
        "expert_family": family,
        "candidate_status": "weak_positive",
        "git_commit": "test",
    }


def _write_artifact(path: Path, model_key: str, probabilities: list[float], *, family: str) -> Path:
    probs = np.array(probabilities, dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    logits = np.log(np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0))
    write_score_artifact(
        path,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=probs,
            logits=logits.astype(np.float32),
            sample_ids=np.array(["s0", "s1", "s2", "s3"], dtype=str),
            metadata=_metadata(model_key, family=family),
        ),
    )
    return path


def test_residual_guided_pool_evaluation_waits_when_pool_plan_pending(tmp_path):
    plan = tmp_path / "pool_plan.json"
    output = tmp_path / "pool_eval.json"
    plan.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_residual_focus_gate",
                "should_run_pool": False,
                "next_action": {"branch": "finish_residual_focus_262k_outputs"},
            }
        ),
        encoding="utf-8",
    )

    status = evaluate_pool_main(["--pool-plan", str(plan), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_guided_pool_plan"
    assert report["should_run_pool"] is False
    assert report["next_action"]["branch"] == "finish_residual_focus_262k_outputs"


def test_residual_guided_pool_evaluation_compares_candidate_to_controls(tmp_path):
    plan = tmp_path / "pool_plan.json"
    output = tmp_path / "pool_eval.json"
    plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "residual_guided_diverse_pool_ready",
                "should_run_pool": True,
                "selected_residual_candidate": "focus10",
                "claim_scope": "application-level medium diagnostic evidence",
            }
        ),
        encoding="utf-8",
    )
    trail = _write_artifact(
        tmp_path / "trail",
        "trail_position",
        [0.10, 0.60, 0.40, 0.90],
        family="trail_position_anchor",
    )
    raw117 = _write_artifact(
        tmp_path / "raw117",
        "raw117",
        [0.20, 0.55, 0.45, 0.80],
        family="compressed_span_structural",
    )
    residual = _write_artifact(
        tmp_path / "residual_focus10",
        "residual_focus10",
        [0.45, 0.10, 0.90, 0.55],
        family="residual_focus_aux_word",
    )
    uniform = _write_artifact(
        tmp_path / "uniform",
        "residual_uniform",
        [0.50, 0.50, 0.50, 0.50],
        family="uniform_residual_control",
    )
    labelshuffle = _write_artifact(
        tmp_path / "labelshuffle",
        "residual_labelshuffle",
        [0.90, 0.80, 0.20, 0.10],
        family="labelshuffle_residual_control",
    )

    status = evaluate_pool_main(
        [
            "--pool-plan",
            str(plan),
            "--trail-position-artifact",
            str(trail),
            "--raw117-artifact",
            str(raw117),
            "--residual-focus-artifact",
            str(residual),
            "--uniform-control-artifact",
            str(uniform),
            "--labelshuffle-control-artifact",
            str(labelshuffle),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "support_residual_guided_pool3_fixed_fusion"
    assert [row["name"] for row in report["comparisons"]] == [
        "trail_position_plus_raw117",
        "trail_position_plus_raw117_plus_residual_focus",
        "trail_position_plus_raw117_plus_uniform_control",
        "trail_position_plus_raw117_plus_labelshuffle_control",
    ]
    assert report["candidate_delta_vs_base_auc"] > 0.0
    assert report["candidate_delta_vs_uniform_control_auc"] >= 0.0
    assert report["candidate_delta_vs_labelshuffle_control_auc"] > 0.0
    assert report["claim_scope"].startswith("application-level medium diagnostic")


def test_residual_guided_pool_evaluation_holds_when_candidate_fails_controls(tmp_path):
    plan = tmp_path / "pool_plan.json"
    output = tmp_path / "pool_eval.json"
    plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "residual_guided_diverse_pool_ready",
                "should_run_pool": True,
                "selected_residual_candidate": "focus10",
            }
        ),
        encoding="utf-8",
    )
    trail = _write_artifact(
        tmp_path / "trail",
        "trail_position",
        [0.10, 0.60, 0.40, 0.90],
        family="trail_position_anchor",
    )
    raw117 = _write_artifact(
        tmp_path / "raw117",
        "raw117",
        [0.20, 0.55, 0.45, 0.80],
        family="compressed_span_structural",
    )
    residual = _write_artifact(
        tmp_path / "residual_focus10",
        "residual_focus10",
        [0.90, 0.80, 0.20, 0.10],
        family="residual_focus_aux_word",
    )
    uniform = _write_artifact(
        tmp_path / "uniform",
        "residual_uniform",
        [0.50, 0.50, 0.50, 0.50],
        family="uniform_residual_control",
    )
    labelshuffle = _write_artifact(
        tmp_path / "labelshuffle",
        "residual_labelshuffle",
        [0.90, 0.80, 0.20, 0.10],
        family="labelshuffle_residual_control",
    )

    status = evaluate_pool_main(
        [
            "--pool-plan",
            str(plan),
            "--trail-position-artifact",
            str(trail),
            "--raw117-artifact",
            str(raw117),
            "--residual-focus-artifact",
            str(residual),
            "--uniform-control-artifact",
            str(uniform),
            "--labelshuffle-control-artifact",
            str(labelshuffle),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "residual_guided_pool3_fixed_fusion_diagnostic_only"
    assert report["candidate_delta_vs_base_auc"] <= 0.0
    assert report["candidate_delta_vs_labelshuffle_control_auc"] == 0.0
    assert report["next_action"]["branch"] == "repair_residual_guided_pool3_before_scaleup"
