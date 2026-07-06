from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.analyze_trail_position_scores import main as analyze_trail_position_scores_main
from blockcipher_nd.cli.apply_bit_sensitivity_projection import main as apply_bit_sensitivity_main
from blockcipher_nd.cli.evaluate_neural_ensemble import main as evaluate_ensemble_main
from blockcipher_nd.cli.export_checkpoint_scores import main as export_scores_main
from blockcipher_nd.cli.export_bit_sensitivity_features import (
    main as export_bit_sensitivity_features_main,
)
from blockcipher_nd.cli.postprocess_bit_sensitivity_projection import (
    main as postprocess_bit_sensitivity_main,
)
from blockcipher_nd.cli.postprocess_neural_ensemble import main as postprocess_ensemble_main
from blockcipher_nd.cli.postprocess_trail_position_result import main as postprocess_trail_position_main
from blockcipher_nd.cli.render_trail_position_report import main as render_trail_position_report_main
from blockcipher_nd.cli.select_bit_sensitivity_projection import main as select_bit_sensitivity_main
from blockcipher_nd.cli.neural_ensemble_status import main as neural_ensemble_status_main
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.cli.verify_score_artifacts import main as verify_score_artifacts_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


def write_tiny_speck_plan(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,validation_key,key_rotation_interval,sample_structure,integral_active_nibble,difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature",
                'SPECK32/64,ARX,Tiny-Speck-MLP,mlp,tiny,0,1,1,0,8,1,ciphertext_pair_bits,encrypted_random_plaintexts,0x1918111009080100,0x1918111009080101,0,independent_pairs,0,,,bce,0.001,adam,0,none,,val_auc,true,0,0.0,"{}","SMOKE only for neural ensemble checkpoint scoring","test"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_two_seed_tiny_speck_plan(path: Path) -> Path:
    header = (
        "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,"
        "samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,"
        "validation_key,key_rotation_interval,sample_structure,integral_active_nibble,"
        "difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,"
        "lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,"
        "early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature"
    )
    rows = [
        'SPECK32/64,ARX,Tiny-Speck-MLP-seed0,mlp,tiny,0,1,1,0,8,1,ciphertext_pair_bits,encrypted_random_plaintexts,0x1918111009080100,0x1918111009080101,0,independent_pairs,0,,,bce,0.001,adam,0,none,,val_auc,true,0,0.0,"{}","SMOKE checkpoint matrix seed0","test"',
        'SPECK32/64,ARX,Tiny-Speck-MLP-seed1,mlp,tiny,1,1,1,1,8,1,ciphertext_pair_bits,encrypted_random_plaintexts,0x1918111009080100,0x1918111009080101,0,independent_pairs,0,,,bce,0.001,adam,0,none,,val_auc,true,0,0.0,"{}","SMOKE checkpoint matrix seed1","test"',
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_train_matrix_checkpoint_output_dir_writes_per_row_checkpoints(tmp_path):
    plan = write_two_seed_tiny_speck_plan(tmp_path / "matrix.csv")
    checkpoint_dir = tmp_path / "checkpoints"
    output = tmp_path / "results.jsonl"

    train_main(
        [
            "--plan",
            str(plan),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "8",
            "--device",
            "cpu",
            "--checkpoint-output-dir",
            str(checkpoint_dir),
            "--output",
            str(output),
        ]
    )

    checkpoints = sorted(checkpoint_dir.glob("row*.pt"))
    result_rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint_outputs = [row["training"]["checkpoint_output"] for row in result_rows]

    assert len(checkpoints) == 2
    assert len(result_rows) == 2
    assert checkpoint_outputs == [str(path) for path in checkpoints]
    assert checkpoints[0].name.startswith("row0001_mlp_seed0")
    assert checkpoints[1].name.startswith("row0002_mlp_seed1")


def test_export_checkpoint_scores_writes_artifact(tmp_path):
    checkpoint = tmp_path / "model.pt"
    train_output = tmp_path / "train.jsonl"
    train_main(
        [
            "--ciphers",
            "speck32",
            "--models",
            "mlp",
            "--rounds",
            "1",
            "--seeds",
            "0",
            "--samples-per-class",
            "8",
            "--pairs-per-sample",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "8",
            "--device",
            "cpu",
            "--checkpoint-output",
            str(checkpoint),
            "--output",
            str(train_output),
        ]
    )
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    artifact_dir = tmp_path / "artifact"

    status = export_scores_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--model-key",
            "mlp",
            "--hidden-bits",
            "8",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--expert-family",
            "raw_mcnd",
            "--candidate-status",
            "weak_positive",
            "--output-dir",
            str(artifact_dir),
        ]
    )

    assert status == 0
    labels = np.load(artifact_dir / "labels.npy")
    probabilities = np.load(artifact_dir / "probabilities.npy")
    logits = np.load(artifact_dir / "logits.npy")
    metadata = json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))
    assert labels.shape == probabilities.shape == logits.shape
    assert labels.shape[0] == 16
    assert metadata["model_key"] == "mlp"
    assert metadata["negative_mode"] == "encrypted_random_plaintexts"
    assert metadata["expert_family"] == "raw_mcnd"
    assert metadata["candidate_status"] == "weak_positive"


def test_export_checkpoint_scores_can_use_dataset_cache_and_progress(tmp_path):
    checkpoint = tmp_path / "model.pt"
    train_output = tmp_path / "train.jsonl"
    train_main(
        [
            "--ciphers",
            "speck32",
            "--models",
            "mlp",
            "--rounds",
            "1",
            "--seeds",
            "0",
            "--samples-per-class",
            "8",
            "--pairs-per-sample",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "8",
            "--device",
            "cpu",
            "--checkpoint-output",
            str(checkpoint),
            "--output",
            str(train_output),
        ]
    )
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    artifact_dir = tmp_path / "artifact"
    cache_root = tmp_path / "dataset_cache"
    progress_output = tmp_path / "score_export_progress.jsonl"

    status = export_scores_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--model-key",
            "mlp",
            "--hidden-bits",
            "8",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--expert-family",
            "raw_mcnd",
            "--candidate-status",
            "weak_positive",
            "--dataset-cache-root",
            str(cache_root),
            "--dataset-cache-chunk-size",
            "5",
            "--dataset-cache-workers",
            "1",
            "--progress-output",
            str(progress_output),
            "--output-dir",
            str(artifact_dir),
        ]
    )

    assert status == 0
    assert (artifact_dir / "labels.npy").exists()
    assert list(cache_root.glob("**/features.npy"))
    progress_events = [
        json.loads(line)["event"]
        for line in progress_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "cache_start" in progress_events
    assert "cache_done" in progress_events
    metadata = json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))
    assert metadata["dataset_cache_enabled"] is True
    assert metadata["dataset_cache_root"] == str(cache_root)


def test_evaluate_neural_ensemble_cli_writes_summary(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_options": {},
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "run_id": "left",
        "checkpoint_path": "/tmp/left.pt",
        "git_commit": "test",
    }
    write_score_artifact(
        left_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 1, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
            logits=np.array([-2.2, -0.8, 0.8, 2.2], dtype=np.float32),
            sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
            metadata={**metadata, "model_key": "left"},
        ),
    )
    write_score_artifact(
        right_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 1, 1], dtype=np.float32),
            probabilities=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            logits=np.array([-1.4, -0.4, 0.4, 1.4], dtype=np.float32),
            sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
            metadata={**metadata, "model_key": "right", "run_id": "right"},
        ),
    )
    output = tmp_path / "ensemble_summary.json"

    status = evaluate_ensemble_main(
        [
            "--artifacts",
            str(left_dir),
            str(right_dir),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["best_single"]["model_key"] == "left"
    assert summary["ensembles"][0]["mode"] == "probability_mean"
    assert summary["claim_scope"].startswith("application-level")


def test_postprocess_trail_position_result_reports_pending_until_artifacts_ready(tmp_path):
    run_root = tmp_path / "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706"
    (run_root / "results").mkdir(parents=True)
    (run_root / "score_artifacts").mkdir(parents=True)
    (run_root / "results" / "train_matrix.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "postprocess.json"

    status = postprocess_trail_position_main(
        [
            "--run-root",
            str(run_root),
            "--expected-score-rows",
            "4",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_trail_position_score_artifacts"
    assert report["pending_run_count"] == 1
    assert report["runs"][0]["train_rows"] == 0
    assert "not formal SPN/PRESENT evidence" in report["claim_scope"]


def test_postprocess_trail_position_result_verifies_and_analyzes_ready_artifacts(tmp_path):
    run_root = tmp_path / "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706"
    (run_root / "results").mkdir(parents=True)
    (run_root / "results" / "train_matrix.jsonl").write_text(
        json.dumps({"model": "present_pairset_global_stats"}) + "\n"
        + json.dumps({"model": "present_trail_position_stats_pairset"}) + "\n",
        encoding="utf-8",
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "cipher_key": "present80",
        "rounds": 8,
        "validation_samples_per_class": 2,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    write_score_artifact(
        run_root / "score_artifacts" / "global_stats_control",
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.2, 0.6, 0.8, 0.4], dtype=np.float32),
            logits=np.array([-1.4, 0.4, 1.4, -0.4], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
                "candidate_status": "near_neighbor_control",
            },
        ),
    )
    write_score_artifact(
        run_root / "score_artifacts" / "trail_position",
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
            logits=np.array([-2.0, -1.4, 1.4, 2.0], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
                "candidate_status": "weak_positive",
            },
        ),
    )
    output = tmp_path / "postprocess.json"

    status = postprocess_trail_position_main(
        [
            "--run-root",
            str(run_root),
            "--expected-score-rows",
            "4",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    run = report["runs"][0]
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "support_trail_position_score_residual_all_runs"
    assert report["ready_run_count"] == 1
    assert run["verification"]["status"] == "pass"
    assert run["analysis"]["decision"] == "support_trail_position_score_residual"
    assert run["analysis"]["rows"] == 4
    assert (run_root / "score_artifacts" / "verification_summary_local.json").exists()
    assert (run_root / "score_artifacts" / "trail_position_score_analysis.json").exists()


def test_render_trail_position_report_keeps_pending_claim_guardrails(tmp_path):
    postprocess = tmp_path / "postprocess.json"
    report_path = tmp_path / "decision.md"
    postprocess.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_trail_position_score_artifacts",
                "action": "let_tmux_watchers_finish_retrieval_before_score_claims",
                "ready_run_count": 0,
                "pending_run_count": 1,
                "failed_run_count": 0,
                "expected_score_rows": 262144,
                "expected_matrix_rows": 2,
                "claim_scope": "PRESENT r8 trail-position score postprocess only; not formal SPN/PRESENT evidence",
                "runs": [
                    {
                        "run_id": "seed0",
                        "status": "pending",
                        "train_rows": 0,
                        "reason": "train_matrix_or_score_artifacts_not_ready",
                        "missing_score_files": ["models.json", "labels.npy"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = render_trail_position_report_main(
        ["--postprocess", str(postprocess), "--output", str(report_path)]
    )

    markdown = report_path.read_text(encoding="utf-8")
    assert status == 0
    assert "Status: `pending`" in markdown
    assert "`train_matrix_or_score_artifacts_not_ready`" in markdown
    assert "no AUC or score-overlap claim is allowed yet" in markdown
    assert "not formal SPN/PRESENT evidence" in markdown


def test_render_trail_position_report_includes_pass_gate_metrics(tmp_path):
    postprocess = tmp_path / "postprocess.json"
    report_path = tmp_path / "decision.md"
    postprocess.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "support_trail_position_score_residual_all_runs",
                "action": "update_experiment_record_and_compare_against_medium_gate_scope",
                "ready_run_count": 1,
                "pending_run_count": 0,
                "failed_run_count": 0,
                "expected_score_rows": 262144,
                "expected_matrix_rows": 2,
                "claim_scope": "medium diagnostic only",
                "runs": [
                    {
                        "run_id": "seed0",
                        "status": "pass",
                        "train_rows": 2,
                        "missing_score_files": [],
                        "analysis": {
                            "decision": "support_trail_position_score_residual",
                            "margins_vs_global_control": {"auc": 0.0123456789},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = render_trail_position_report_main(
        ["--postprocess", str(postprocess), "--output", str(report_path)]
    )

    markdown = report_path.read_text(encoding="utf-8")
    assert status == 0
    assert "Status: `pass`" in markdown
    assert "`support_trail_position_score_residual`" in markdown
    assert "`0.0123456789`" in markdown
    assert "Prepare a `>=1000000/class` multi-seed plan" in markdown


def test_verify_score_artifacts_cli_accepts_aligned_required_models(tmp_path):
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    left_dir = tmp_path / "global_stats_control"
    right_dir = tmp_path / "trail_position"
    write_score_artifact(
        left_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.8, 0.2, 0.7], dtype=np.float32),
            logits=np.array([-2.0, 1.5, -1.5, 1.0], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
                "candidate_status": "near_neighbor_control",
                "cipher_key": "present80",
                "rounds": 8,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
            },
        ),
    )
    write_score_artifact(
        right_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.05, 0.9, 0.1, 0.85], dtype=np.float32),
            logits=np.array([-3.0, 2.2, -2.1, 1.8], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
                "candidate_status": "weak_positive",
                "cipher_key": "present80",
                "rounds": 8,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
            },
        ),
    )
    summary_path = tmp_path / "verification_summary.json"

    status = verify_score_artifacts_main(
        [
            "--artifacts",
            str(left_dir),
            str(right_dir),
            "--expected-rows",
            "4",
            "--require-model",
            "present_pairset_global_stats:trail_position_global_control:near_neighbor_control",
            "--require-model",
            "present_trail_position_stats_pairset:trail_position:weak_positive",
            "--output",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["artifact_count"] == 2
    assert summary["rows"] == 4
    assert summary["alignment"]["labels"] is True
    assert summary["alignment"]["sample_ids"] is True
    assert summary["errors"] == []


def test_verify_score_artifacts_cli_rejects_misaligned_labels(tmp_path):
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    common_metadata = {
        "expert_family": "trail_position",
        "candidate_status": "weak_positive",
    }
    write_score_artifact(
        left_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 1, 0, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.8, 0.2, 0.7], dtype=np.float32),
            logits=np.array([-2.0, 1.5, -1.5, 1.0], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={**common_metadata, "model_key": "left"},
        ),
    )
    write_score_artifact(
        right_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 0, 1], dtype=np.float32),
            probabilities=np.array([0.05, 0.9, 0.1, 0.85], dtype=np.float32),
            logits=np.array([-3.0, 2.2, -2.1, 1.8], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={**common_metadata, "model_key": "right"},
        ),
    )
    summary_path = tmp_path / "verification_summary.json"

    status = verify_score_artifacts_main(
        [
            "--artifacts",
            str(left_dir),
            str(right_dir),
            "--expected-rows",
            "4",
            "--output",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 1
    assert summary["status"] == "fail"
    assert summary["alignment"]["labels"] is False
    assert "labels_mismatch" in summary["errors"]


def test_analyze_trail_position_scores_supports_candidate_residual(tmp_path):
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    common_metadata = {
        "cipher": "PRESENT-80",
        "cipher_key": "present80",
        "rounds": 8,
        "seed": 0,
        "samples_per_class": 16,
        "validation_samples_per_class": 3,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    global_dir = tmp_path / "global_stats_control"
    candidate_dir = tmp_path / "trail_position"
    write_score_artifact(
        global_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.6, 0.2, 0.4, 0.8, 0.9], dtype=np.float32),
            logits=np.array([-2.0, 0.4, -1.4, -0.2, 1.8, 2.2], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                **common_metadata,
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
                "candidate_status": "near_neighbor_control",
            },
        ),
    )
    write_score_artifact(
        candidate_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.05, 0.4, 0.15, 0.7, 0.85, 0.95], dtype=np.float32),
            logits=np.array([-3.0, -0.4, -1.8, 0.9, 2.0, 2.7], dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                **common_metadata,
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
                "candidate_status": "weak_positive",
            },
        ),
    )
    output = tmp_path / "trail_position_score_analysis.json"

    status = analyze_trail_position_scores_main(
        [
            "--global-artifact",
            str(global_dir),
            "--candidate-artifact",
            str(candidate_dir),
            "--output",
            str(output),
            "--improvement-margin",
            "0.0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "support_trail_position_score_residual"
    assert report["margins_vs_global_control"]["auc"] > 0
    assert report["overlap_at_0_5"]["candidate_correct_global_wrong_rate_at_0_5"] > 0
    assert report["overlap_at_0_5"]["global_correct_candidate_wrong_rate_at_0_5"] == 0
    assert "not a diverse ensemble claim" in report["claim_scope"]


def test_analyze_trail_position_scores_holds_when_candidate_does_not_clear_control(tmp_path):
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    common_metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 2,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    global_dir = tmp_path / "global_stats_control"
    candidate_dir = tmp_path / "trail_position"
    for path, model_key, family in [
        (global_dir, "present_pairset_global_stats", "trail_position_global_control"),
        (candidate_dir, "present_trail_position_stats_pairset", "trail_position"),
    ]:
        write_score_artifact(
            path,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
                logits=np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
                sample_ids=sample_ids,
                metadata={
                    **common_metadata,
                    "model_key": model_key,
                    "expert_family": family,
                    "candidate_status": "weak_positive",
                },
            ),
        )
    output = tmp_path / "trail_position_score_analysis.json"

    status = analyze_trail_position_scores_main(
        [
            "--global-artifact",
            str(global_dir),
            "--candidate-artifact",
            str(candidate_dir),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "hold_trail_position_score_residual"
    assert report["margins_vs_global_control"]["auc"] == 0.0
    assert report["action"] == "do_not_promote_score_artifacts_beyond_diagnostic_use"


def test_select_bit_sensitivity_projection_writes_train_only_mask(tmp_path):
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.float32)
    sample_ids = np.array(["s0", "s1", "s2", "s3", "s4", "s5"], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 3,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    control_dir = tmp_path / "control"
    anchor_dir = tmp_path / "anchor"
    write_score_artifact(
        control_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.4, 0.6, 0.4, 0.6, 0.3, 0.4], dtype=np.float32),
            logits=np.zeros_like(labels),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
            },
        ),
    )
    write_score_artifact(
        anchor_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.3, 0.2, 0.8, 0.7, 0.4, 0.9], dtype=np.float32),
            logits=np.ones_like(labels),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
            },
        ),
    )
    features = np.array(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, features)
    mask_path = tmp_path / "mask.json"
    report_path = tmp_path / "report.json"

    status = select_bit_sensitivity_main(
        [
            "--features",
            str(feature_path),
            "--control-artifact",
            str(control_dir),
            "--anchor-artifact",
            str(anchor_dir),
            "--output-mask",
            str(mask_path),
            "--output-report",
            str(report_path),
            "--top-k",
            "2",
        ]
    )

    assert status == 0
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert mask["selection_split"] == "train"
    assert mask["selected_axis_count"] == 2
    assert mask["selected_axes"][0] == 2
    assert mask["axis_scores"][0]["axis"] == 2
    assert mask["axis_scores"][0]["positive_mean"] > mask["axis_scores"][0]["negative_mean"]
    assert report["decision"] == "projection_mask_ready_for_local_screen"
    assert "do_not_select_mask_on_validation" in report["guardrails"]
    assert "not a trained model result" in report["claim_scope"]


def test_apply_bit_sensitivity_projection_writes_score_artifact(tmp_path):
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    sample_ids = np.array(["v0", "v1", "v2", "v3"], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 2,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_key": "present_trail_position_stats_pairset",
        "expert_family": "trail_position",
    }
    reference_dir = tmp_path / "reference"
    write_score_artifact(
        reference_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.2, 0.3, 0.7, 0.8], dtype=np.float32),
            logits=np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32),
            sample_ids=sample_ids,
            metadata=metadata,
        ),
    )
    features = np.array(
        [
            [0.0, 0.2],
            [0.1, 0.4],
            [0.8, 0.1],
            [1.0, 0.3],
        ],
        dtype=np.float32,
    )
    feature_path = tmp_path / "validation_features.npy"
    np.save(feature_path, features)
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(
        json.dumps(
            {
                "selection_split": "train",
                "selected_axes": [0],
                "axis_scores": [
                    {
                        "axis": 0,
                        "positive_mean": 1.0,
                        "negative_mean": 0.0,
                        "score": 2.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "projection_artifact"
    report_path = tmp_path / "projection_report.json"

    status = apply_bit_sensitivity_main(
        [
            "--features",
            str(feature_path),
            "--mask",
            str(mask_path),
            "--reference-artifact",
            str(reference_dir),
            "--output-dir",
            str(output_dir),
            "--output-report",
            str(report_path),
            "--run-id",
            "local_projection_screen",
        ]
    )

    assert status == 0
    artifact = output_dir
    metadata_out = json.loads((artifact / "models.json").read_text(encoding="utf-8"))
    probabilities = np.load(artifact / "probabilities.npy")
    logits = np.load(artifact / "logits.npy")
    assert metadata_out["model_key"] == "present_r8_bit_sensitivity_projection_expert"
    assert metadata_out["expert_family"] == "bit_sensitivity_projection"
    assert metadata_out["candidate_status"] == "projection_screen"
    assert np.array_equal(np.load(artifact / "labels.npy"), labels)
    assert np.array_equal(np.load(artifact / "sample_ids.npy").astype(str), sample_ids)
    assert probabilities[2] > probabilities[0]
    assert logits.shape == probabilities.shape == labels.shape
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "projection_score_artifact_ready_for_local_gate"
    assert "not a trained neural model" in report["claim_scope"]


def test_export_bit_sensitivity_features_matches_reference_artifact(tmp_path):
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    first_output = tmp_path / "first"

    status = export_bit_sensitivity_features_main(
        [
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "validation",
            "--samples-per-class",
            "4",
            "--output-dir",
            str(first_output),
        ]
    )

    assert status == 0
    labels = np.load(first_output / "labels.npy")
    sample_ids = np.load(first_output / "sample_ids.npy").astype(str)
    metadata = json.loads((first_output / "metadata.json").read_text(encoding="utf-8"))
    assert labels.shape[0] == 8
    assert sample_ids.tolist() == [str(index) for index in range(8)]
    assert metadata["split"] == "validation"
    assert metadata["alignment"]["reference_checked"] is False

    reference_dir = tmp_path / "reference_artifact"
    write_score_artifact(
        reference_dir,
        EnsembleScoreArtifact(
            labels=labels.astype(np.float32, copy=False),
            probabilities=np.linspace(0.1, 0.9, len(labels), dtype=np.float32),
            logits=np.linspace(-2.0, 2.0, len(labels), dtype=np.float32),
            sample_ids=sample_ids,
            metadata={
                "cipher": metadata["cipher"],
                "cipher_key": metadata["cipher_key"],
                "rounds": metadata["rounds"],
                "validation_samples_per_class": metadata["samples_per_class"],
                "pairs_per_sample": metadata["pairs_per_sample"],
                "feature_encoding": metadata["feature_encoding"],
                "negative_mode": metadata["negative_mode"],
                "sample_structure": metadata["sample_structure"],
                "difference_profile": metadata["difference_profile"],
                "difference_member": metadata["difference_member"],
                "validation_key": metadata["validation_key"],
                "model_key": "reference",
            },
        ),
    )
    aligned_output = tmp_path / "aligned"

    status = export_bit_sensitivity_features_main(
        [
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "validation",
            "--samples-per-class",
            "4",
            "--reference-artifact",
            str(reference_dir),
            "--output-dir",
            str(aligned_output),
        ]
    )

    aligned_metadata = json.loads((aligned_output / "metadata.json").read_text(encoding="utf-8"))
    assert status == 0
    assert np.array_equal(np.load(aligned_output / "labels.npy"), labels)
    assert np.array_equal(np.load(aligned_output / "sample_ids.npy").astype(str), sample_ids)
    assert aligned_metadata["alignment"]["reference_checked"] is True
    assert aligned_metadata["alignment"]["labels"] is True
    assert aligned_metadata["alignment"]["sample_ids"] is True


def test_export_bit_sensitivity_features_rejects_reference_label_mismatch(tmp_path):
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    reference_dir = tmp_path / "reference_artifact"
    write_score_artifact(
        reference_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32),
            probabilities=np.full((8,), 0.5, dtype=np.float32),
            logits=np.zeros((8,), dtype=np.float32),
            sample_ids=np.array([str(index) for index in range(8)], dtype=str),
            metadata={
                "cipher": "SPECK32/64",
                "cipher_key": "speck32",
                "rounds": 1,
                "validation_samples_per_class": 4,
                "pairs_per_sample": 1,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "difference_profile": "",
                "difference_member": "",
                "validation_key": "0x1918111009080101",
                "model_key": "reference",
            },
        ),
    )

    status = export_bit_sensitivity_features_main(
        [
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "validation",
            "--samples-per-class",
            "4",
            "--reference-artifact",
            str(reference_dir),
            "--output-dir",
            str(tmp_path / "mismatch"),
        ]
    )

    assert status == 1


def test_postprocess_bit_sensitivity_projection_promotes_low_overlap_expert(tmp_path):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([f"v{index}" for index in range(len(labels))], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    global_dir = tmp_path / "global_stats_control"
    anchor_dir = tmp_path / "trail_position"
    projection_dir = tmp_path / "bit_sensitivity_projection"
    write_score_artifact(
        global_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            logits=np.zeros_like(labels),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
                "candidate_status": "near_neighbor_control",
            },
        ),
    )
    write_score_artifact(
        anchor_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.6, 0.4], dtype=np.float32),
            logits=np.ones_like(labels),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
                "candidate_status": "weak_positive",
            },
        ),
    )
    write_score_artifact(
        projection_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.05, 0.7, 0.1, 0.2, 0.95, 0.4, 0.85, 0.9], dtype=np.float32),
            logits=np.full_like(labels, 0.5),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_r8_bit_sensitivity_projection_expert",
                "expert_family": "bit_sensitivity_projection",
                "candidate_status": "projection_screen",
            },
        ),
    )
    output = tmp_path / "bit_sensitivity_gate.json"

    status = postprocess_bit_sensitivity_main(
        [
            "--global-artifact",
            str(global_dir),
            "--anchor-artifact",
            str(anchor_dir),
            "--projection-artifact",
            str(projection_dir),
            "--output",
            str(output),
            "--min-margin-vs-global",
            "0.01",
            "--max-error-jaccard-with-anchor",
            "0.5",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "projection_expert_ready_for_local_screen"
    assert report["projection"]["metrics"]["auc"] > report["global_control"]["metrics"]["auc"]
    assert report["margins_vs_global_control"]["auc"] > 0.01
    assert report["overlap_with_anchor"]["error_jaccard_at_0_5"] <= 0.5
    assert report["next_action"]["branch"] == "bit_sensitivity_projection_local_screen_ready"
    assert report["next_action"]["should_launch_remote"] is False
    assert "not formal SPN/PRESENT evidence" in report["claim_scope"]


def test_postprocess_bit_sensitivity_projection_holds_high_overlap_duplicate(tmp_path):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([f"v{index}" for index in range(len(labels))], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    global_dir = tmp_path / "global_stats_control"
    anchor_dir = tmp_path / "trail_position"
    projection_dir = tmp_path / "bit_sensitivity_projection"
    anchor_probabilities = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.6, 0.4], dtype=np.float32)
    write_score_artifact(
        global_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            logits=np.zeros_like(labels),
            sample_ids=sample_ids,
            metadata={
                **metadata,
                "model_key": "present_pairset_global_stats",
                "expert_family": "trail_position_global_control",
                "candidate_status": "near_neighbor_control",
            },
        ),
    )
    for path, model_key, family in [
        (anchor_dir, "present_trail_position_stats_pairset", "trail_position"),
        (projection_dir, "present_r8_bit_sensitivity_projection_expert", "bit_sensitivity_projection"),
    ]:
        write_score_artifact(
            path,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=anchor_probabilities,
                logits=np.ones_like(labels),
                sample_ids=sample_ids,
                metadata={
                    **metadata,
                    "model_key": model_key,
                    "expert_family": family,
                    "candidate_status": "projection_screen",
                },
            ),
        )
    output = tmp_path / "bit_sensitivity_gate.json"

    status = postprocess_bit_sensitivity_main(
        [
            "--global-artifact",
            str(global_dir),
            "--anchor-artifact",
            str(anchor_dir),
            "--projection-artifact",
            str(projection_dir),
            "--output",
            str(output),
            "--min-margin-vs-global",
            "0.01",
            "--max-error-jaccard-with-anchor",
            "0.5",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "hold_projection_duplicate_or_weak"
    assert "high_error_overlap_with_anchor" in report["hold_reasons"]
    assert report["next_action"]["branch"] == "hold_bit_sensitivity_projection"


def test_postprocess_bit_sensitivity_projection_fails_misaligned_labels(tmp_path):
    sample_ids = np.array(["v0", "v1", "v2", "v3"], dtype=str)
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 2,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    dirs = [tmp_path / "global", tmp_path / "anchor", tmp_path / "projection"]
    labels_by_dir = [
        np.array([0, 0, 1, 1], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.float32),
        np.array([0, 1, 1, 1], dtype=np.float32),
    ]
    for path, labels, model_key, family in zip(
        dirs,
        labels_by_dir,
        [
            "present_pairset_global_stats",
            "present_trail_position_stats_pairset",
            "present_r8_bit_sensitivity_projection_expert",
        ],
        ["trail_position_global_control", "trail_position", "bit_sensitivity_projection"],
    ):
        write_score_artifact(
            path,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
                logits=np.zeros_like(labels),
                sample_ids=sample_ids,
                metadata={**metadata, "model_key": model_key, "expert_family": family},
            ),
        )
    output = tmp_path / "bit_sensitivity_gate.json"

    status = postprocess_bit_sensitivity_main(
        [
            "--global-artifact",
            str(dirs[0]),
            "--anchor-artifact",
            str(dirs[1]),
            "--projection-artifact",
            str(dirs[2]),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert report["status"] == "fail"
    assert report["decision"] == "fail_protocol_alignment"
    assert report["errors"]


def test_neural_ensemble_end_to_end_tiny_smoke(tmp_path):
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    artifact_dirs = []
    for seed in [0, 1]:
        checkpoint = tmp_path / f"model_seed{seed}.pt"
        train_main(
            [
                "--ciphers",
                "speck32",
                "--models",
                "mlp",
                "--rounds",
                "1",
                "--seeds",
                str(seed),
                "--samples-per-class",
                "8",
                "--pairs-per-sample",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--hidden-bits",
                "8",
                "--device",
                "cpu",
                "--checkpoint-output",
                str(checkpoint),
                "--output",
                str(tmp_path / f"train_seed{seed}.jsonl"),
            ]
        )
        artifact_dir = tmp_path / f"artifact_seed{seed}"
        export_scores_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--eval-plan",
                str(plan),
                "--eval-row-index",
                "0",
                "--model-key",
                "mlp",
                "--hidden-bits",
                "8",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--output-dir",
                str(artifact_dir),
            ]
        )
        artifact_dirs.append(artifact_dir)
    output = tmp_path / "ensemble_summary.json"

    status = evaluate_ensemble_main(
        [
            "--artifacts",
            str(artifact_dirs[0]),
            str(artifact_dirs[1]),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert len(summary["models"]) == 2
    assert len(summary["ensembles"]) == 5


def test_postprocess_neural_ensemble_keeps_complementary_positive_route(tmp_path):
    train_results = tmp_path / "train_matrix.jsonl"
    train_results.write_text(
        "\n".join(
            [
                json.dumps({"model": "present_zhang_wang_keras_mcnd", "metrics": {"auc": 0.790}}),
                json.dumps({"model": "present_nibble_invp_only_spn_only", "metrics": {"auc": 0.797}}),
                json.dumps({"model": "present_nibble_ddt_graph", "metrics": {"auc": 0.740}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ensemble_summary = tmp_path / "neural_ensemble_summary.json"
    ensemble_summary.write_text(
        json.dumps(
            {
                "status": "pass",
                "best_single": {
                    "model_key": "present_nibble_invp_only_spn_only",
                    "metrics": {"auc": 0.7970, "calibrated_accuracy": 0.733},
                },
                "best_ensemble": {
                    "mode": "logit_mean",
                    "metrics": {"auc": 0.7992, "calibrated_accuracy": 0.736},
                },
                "delta_best_ensemble_vs_single_auc": 0.0022,
                "models": [
                    {
                        "model_key": "present_zhang_wang_keras_mcnd",
                        "metrics": {"auc": 0.7900, "calibrated_accuracy": 0.724},
                    },
                    {
                        "model_key": "present_nibble_invp_only_spn_only",
                        "metrics": {"auc": 0.7970, "calibrated_accuracy": 0.733},
                    },
                    {
                        "model_key": "present_nibble_ddt_graph",
                        "metrics": {"auc": 0.7400, "calibrated_accuracy": 0.670},
                    },
                ],
                "ensembles": [
                    {"mode": "probability_mean", "metrics": {"auc": 0.7980}},
                    {"mode": "logit_mean", "metrics": {"auc": 0.7992}},
                ],
                "diversity": {
                    "oracle_accuracy_at_0_5": 0.755,
                    "all_models_wrong_rate_at_0_5": 0.245,
                    "pairwise": [
                        {
                            "left": "present_zhang_wang_keras_mcnd",
                            "right": "present_nibble_invp_only_spn_only",
                            "double_fault_rate_at_0_5": 0.18,
                            "error_jaccard_at_0_5": 0.62,
                        },
                        {
                            "left": "present_nibble_invp_only_spn_only",
                            "right": "present_nibble_ddt_graph",
                            "double_fault_rate_at_0_5": 0.20,
                            "error_jaccard_at_0_5": 0.66,
                        },
                    ],
                },
                "claim_scope": "application-level frozen score aggregation diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_doc = tmp_path / "plan.md"
    plan_doc.write_text("# Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    status = postprocess_ensemble_main(
        [
            "--train-results",
            str(train_results),
            "--ensemble-summary",
            str(ensemble_summary),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705",
            "--expected-rows",
            "3",
            "--update-plan-doc",
            str(plan_doc),
        ]
    )

    summary = json.loads(
        (
            output_dir
            / "i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705_postprocess_summary.json"
        ).read_text(encoding="utf-8")
    )
    gate = json.loads(
        (
            output_dir
            / "i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705_neural_ensemble_gate.json"
        ).read_text(encoding="utf-8")
    )
    plan_text = plan_doc.read_text(encoding="utf-8")

    assert status == 0
    assert summary["status"] == "pass"
    assert summary["decision"] == "keep_neural_ensemble_route_prepare_262k_confirmation"
    assert summary["next_action"]["branch"] == "neural_ensemble_262k_confirmation"
    assert gate["max_error_jaccard_at_0_5"] == 0.66
    assert "Retrieved Neural Ensemble Result" in plan_text
    assert "application-level" in plan_text


def test_postprocess_neural_ensemble_does_not_promote_failed_diverse_pool(tmp_path):
    train_results = tmp_path / "train_matrix.jsonl"
    train_results.write_text(
        "\n".join(
            [
                json.dumps({"model": "present_nibble_invp_only_spn_only", "metrics": {"auc": 0.797}}),
                json.dumps({"model": "present_nibble_ddt_graph", "metrics": {"auc": 0.790}}),
                json.dumps({"model": "present_nibble_invp_p_layer_graph_spn_only", "metrics": {"auc": 0.785}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ensemble_summary = tmp_path / "neural_ensemble_summary.json"
    ensemble_summary.write_text(
        json.dumps(
            {
                "status": "pass",
                "best_single": {
                    "model_key": "present_nibble_invp_only_spn_only",
                    "metrics": {"auc": 0.7970, "calibrated_accuracy": 0.733},
                },
                "best_ensemble": {
                    "mode": "logit_mean",
                    "metrics": {"auc": 0.8000, "calibrated_accuracy": 0.737},
                },
                "delta_best_ensemble_vs_single_auc": 0.0030,
                "models": [],
                "ensembles": [{"mode": "logit_mean", "metrics": {"auc": 0.8000}}],
                "diversity": {
                    "pairwise": [
                        {
                            "left": "present_nibble_invp_only_spn_only",
                            "right": "present_nibble_ddt_graph",
                            "double_fault_rate_at_0_5": 0.16,
                            "error_jaccard_at_0_5": 0.54,
                        }
                    ]
                },
                "diverse_expert_pool": {
                    "status": "fail",
                    "decision": "diverse_expert_pool_not_ready",
                    "errors": ["missing_non_neighbor_expert"],
                },
                "claim_scope": "application-level frozen score aggregation diagnostic only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "postprocess"

    status = postprocess_ensemble_main(
        [
            "--train-results",
            str(train_results),
            "--ensemble-summary",
            str(ensemble_summary),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "near_neighbor_positive",
            "--expected-rows",
            "3",
        ]
    )

    summary = json.loads((output_dir / "near_neighbor_positive_postprocess_summary.json").read_text())

    assert status == 0
    assert summary["decision"] == "keep_near_neighbor_ensemble_control_not_diverse_pool"
    assert summary["next_action"]["branch"] == "neural_ensemble_near_neighbor_control"
    assert summary["diverse_expert_pool"]["errors"] == ["missing_non_neighbor_expert"]


def test_neural_ensemble_status_reports_partial_local_artifacts(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "results").mkdir(parents=True)
    (run_root / "checkpoints").mkdir()
    (run_root / "logs").mkdir()
    (run_root / "results" / "train_matrix.jsonl").write_text(
        json.dumps({"model": "present_zhang_wang_keras_mcnd", "metrics": {"auc": 0.7614}}) + "\n",
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "row0001_present_zhang_wang_keras_mcnd_seed0.pt").write_bytes(
        b"checkpoint"
    )
    (run_root / "logs" / "train_matrix_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "validation_start",
                "index": 2,
                "total": 3,
                "model": "present_nibble_invp_only_spn_only",
                "epoch": 11,
                "epochs": 18,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "status.json"

    status = neural_ensemble_status_main(
        [
            "--run-root",
            str(run_root),
            "--expected-rows",
            "3",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "running"
    assert report["train_rows"] == 1
    assert report["checkpoint_count"] == 1
    assert report["score_artifacts_ready"] is False
    assert report["ensemble_summary_ready"] is False
    assert report["latest_progress"]["model"] == "present_nibble_invp_only_spn_only"
    assert "neural_ensemble_summary.json" in report["missing_artifacts"]


def test_neural_ensemble_status_reports_recovered_after_failed_marker(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "results").mkdir(parents=True)
    (run_root / "checkpoints").mkdir()
    (run_root / "logs").mkdir()
    for name in ["zhang_wang", "invp_only", "ddt_graph"]:
        artifact_dir = run_root / "score_artifacts" / name
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "models.json").write_text("{}", encoding="utf-8")
    (run_root / "results" / "neural_ensemble_summary.json").write_text("{}", encoding="utf-8")
    (run_root / "logs" / "run_failed.marker").write_text("export failed before repair\n", encoding="utf-8")
    (run_root / "results" / "train_matrix.jsonl").write_text(
        "\n".join(
            json.dumps({"model": f"model_{index}", "metrics": {"auc": 0.7}})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(3):
        (run_root / "checkpoints" / f"row000{index + 1}_model_{index}_seed0.pt").write_bytes(
            b"checkpoint"
        )
    output = tmp_path / "status.json"

    status = neural_ensemble_status_main(
        [
            "--run-root",
            str(run_root),
            "--expected-rows",
            "3",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "recovered"
    assert report["missing_artifacts"] == []
    assert report["failed_marker_present"] is True
