from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.analyze_trail_position_scores import main as analyze_trail_position_scores_main
from blockcipher_nd.cli.analyze_reliability_residual_buckets import (
    main as analyze_reliability_residual_buckets_main,
)
from blockcipher_nd.cli.analyze_residual_bucket_axis_spectrum import (
    main as analyze_residual_bucket_axis_spectrum_main,
)
from blockcipher_nd.cli.apply_bit_sensitivity_projection import main as apply_bit_sensitivity_main
from blockcipher_nd.cli.evaluate_neural_ensemble import main as evaluate_ensemble_main
from blockcipher_nd.cli.evaluate_stacked_ensemble import main as evaluate_stacked_ensemble_main
from blockcipher_nd.cli.fit_compressed_feature_expert import main as fit_compressed_feature_main
from blockcipher_nd.cli.fit_bucket_conditioned_feature_expert import (
    main as fit_bucket_conditioned_feature_main,
)
from blockcipher_nd.cli.fit_residual_correction_feature_expert import (
    main as fit_residual_correction_feature_main,
)
from blockcipher_nd.cli.evaluate_residual_slice_correction import (
    main as evaluate_residual_slice_correction_main,
)
from blockcipher_nd.cli.fit_compressed_span_grouped_expert import (
    main as fit_compressed_span_grouped_main,
)
from blockcipher_nd.cli.fit_compressed_span_interaction_expert import (
    main as fit_compressed_span_interaction_main,
)
from blockcipher_nd.cli.fit_compressed_span_block_interaction_expert import (
    main as fit_compressed_span_block_interaction_main,
)
from blockcipher_nd.cli.fit_compressed_span_low_rank_interaction_expert import (
    main as fit_compressed_span_low_rank_interaction_main,
)
from blockcipher_nd.cli.fit_compressed_span_learned_low_rank_interaction_expert import (
    main as fit_compressed_span_learned_low_rank_interaction_main,
)
from blockcipher_nd.cli.audit_compressed_feature_sparsity import (
    main as audit_compressed_feature_sparsity_main,
)
from blockcipher_nd.cli.decode_compressed_feature_sparsity import (
    main as decode_compressed_feature_sparsity_main,
)
from blockcipher_nd.cli.audit_compressed_feature_families import (
    main as audit_compressed_feature_families_main,
)
from blockcipher_nd.cli.export_compressed_span_blocks import (
    main as export_compressed_span_blocks_main,
)
from blockcipher_nd.cli.summarize_compressed_feature_expert import (
    main as summarize_compressed_feature_main,
)
from blockcipher_nd.cli.summarize_compressed_span_route import (
    main as summarize_compressed_span_route_main,
)
from blockcipher_nd.cli.summarize_stacked_route import main as summarize_stacked_route_main
from blockcipher_nd.cli.summarize_stacked_selection import main as summarize_stacked_selection_main
from blockcipher_nd.cli.export_checkpoint_scores import main as export_scores_main
from blockcipher_nd.cli.export_bit_sensitivity_features import (
    main as export_bit_sensitivity_features_main,
)
from blockcipher_nd.cli.postprocess_bit_sensitivity_projection import (
    main as postprocess_bit_sensitivity_main,
)
from blockcipher_nd.cli.postprocess_neural_ensemble import main as postprocess_ensemble_main
from blockcipher_nd.cli.postprocess_trail_position_result import main as postprocess_trail_position_main
from blockcipher_nd.cli.gate_bucket_residual_controls import main as gate_bucket_residual_controls_main
from blockcipher_nd.cli.plan_bucket_residual_262k import main as plan_bucket_residual_262k_main
from blockcipher_nd.cli.render_trail_position_report import main as render_trail_position_report_main
from blockcipher_nd.cli.select_bit_sensitivity_projection import main as select_bit_sensitivity_main
from blockcipher_nd.cli.neural_ensemble_status import main as neural_ensemble_status_main
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.cli.verify_score_artifacts import main as verify_score_artifacts_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, load_score_artifact, write_score_artifact
from blockcipher_nd.training.metrics import binary_auc


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


def write_tiny_present_trail_position_plan(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,validation_key,key_rotation_interval,sample_structure,integral_active_nibble,difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature",
                'PRESENT-80,SPN,Tiny-PRESENT-TrailPosition,present_trail_position_stats_pairset,tiny,0,1,8,0,2,2,present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits,encrypted_random_plaintexts,0x00000000000000000000,0x11111111111111111111,0,plaintext_integral_nibble_difference_matched_negative,0,present_zhang_wang2022_mcnd,0,mse,0.0001,adam,0.00001,none,,val_auc,true,0,0.0,"{""trail_depth"":4,""trail_words_per_depth"":9,""stats_hidden_bits"":64}","SMOKE only for bit-sensitivity structural feature view","test"',
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


def test_export_checkpoint_scores_train_split_aligns_with_feature_export(tmp_path):
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
    artifact_dir = tmp_path / "train_artifact"
    features_dir = tmp_path / "train_features"

    status = export_scores_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "train",
            "--samples-per-class",
            "4",
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
    feature_status = export_bit_sensitivity_features_main(
        [
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "train",
            "--samples-per-class",
            "4",
            "--output-dir",
            str(features_dir),
        ]
    )

    assert status == 0
    assert feature_status == 0
    assert np.array_equal(np.load(artifact_dir / "labels.npy"), np.load(features_dir / "labels.npy"))
    assert np.array_equal(
        np.load(artifact_dir / "sample_ids.npy").astype(str),
        np.load(features_dir / "sample_ids.npy").astype(str),
    )
    metadata = json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))
    assert metadata["score_split"] == "train"
    assert metadata["score_samples_per_class"] == 4
    assert metadata["train_samples_per_class"] == 4


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


def test_evaluate_stacked_ensemble_cli_fits_on_train_and_scores_validation(tmp_path):
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "seed": 0,
        "samples_per_class": 4,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_options": {},
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "git_commit": "test",
    }
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    train_left = tmp_path / "train_left"
    train_right = tmp_path / "train_right"
    validation_left = tmp_path / "validation_left"
    validation_right = tmp_path / "validation_right"
    for directory, model_key, probabilities in [
        (train_left, "left", np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.8], dtype=np.float32)),
        (train_right, "right", np.array([0.8, 0.7, 0.2, 0.3, 0.6, 0.7, 0.4, 0.5], dtype=np.float32)),
        (validation_left, "left", np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.8], dtype=np.float32)),
        (validation_right, "right", np.array([0.8, 0.7, 0.2, 0.3, 0.6, 0.7, 0.4, 0.5], dtype=np.float32)),
    ]:
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=probabilities,
                logits=np.log(np.clip(probabilities, 1e-6, 1.0) / np.clip(1.0 - probabilities, 1e-6, 1.0)),
                sample_ids=sample_ids,
                metadata={**metadata, "model_key": model_key, "run_id": model_key},
            ),
        )
    output = tmp_path / "stacked_summary.json"

    status = evaluate_stacked_ensemble_main(
        [
            "--train-artifacts",
            str(train_left),
            str(train_right),
            "--validation-artifacts",
            str(validation_left),
            str(validation_right),
            "--steps",
            "200",
            "--learning-rate",
            "0.1",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["model_order"] == ["left", "right"]
    assert summary["validation_metrics"]["auc"] >= summary["validation_best_single"]["metrics"]["auc"]
    assert "train-fitted validation-evaluated" in summary["claim_scope"]


def test_evaluate_stacked_ensemble_cli_selects_settings_on_train_holdout(tmp_path):
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "seed": 0,
        "samples_per_class": 8,
        "validation_samples_per_class": 8,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_options": {},
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "git_commit": "test",
    }
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    left_probabilities = np.array(
        [0.2, 0.3, 0.25, 0.35, 0.7, 0.75, 0.8, 0.85, 0.8, 0.7, 0.75, 0.65, 0.3, 0.25, 0.2, 0.15],
        dtype=np.float32,
    )
    right_probabilities = 1.0 - left_probabilities
    artifact_specs = [
        (tmp_path / "train_left", "left", left_probabilities),
        (tmp_path / "train_right", "right", right_probabilities),
        (tmp_path / "validation_left", "left", left_probabilities),
        (tmp_path / "validation_right", "right", right_probabilities),
    ]
    for directory, model_key, probabilities in artifact_specs:
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=probabilities,
                logits=np.log(np.clip(probabilities, 1e-6, 1.0) / np.clip(1.0 - probabilities, 1e-6, 1.0)),
                sample_ids=sample_ids,
                metadata={**metadata, "model_key": model_key, "run_id": model_key},
            ),
        )
    output = tmp_path / "stacked_selection_summary.json"

    status = evaluate_stacked_ensemble_main(
        [
            "--train-artifacts",
            str(tmp_path / "train_left"),
            str(tmp_path / "train_right"),
            "--validation-artifacts",
            str(tmp_path / "validation_left"),
            str(tmp_path / "validation_right"),
            "--train-holdout-fraction",
            "0.25",
            "--selection-seed",
            "0",
            "--candidate-feature-spaces",
            "logits",
            "probabilities",
            "--candidate-l2",
            "0.0",
            "0.1",
            "--candidate-standardize",
            "both",
            "--steps",
            "80",
            "--learning-rate",
            "0.1",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["selection"]["mode"] == "train_holdout"
    assert summary["selection"]["candidate_count"] == 8
    assert summary["selection"]["fit_rows"] == 12
    assert summary["selection"]["holdout_rows"] == 4
    assert "train split holdout" in summary["selection"]["claim_scope"]
    assert summary["fit"]["l2"] == summary["selection"]["selected"]["l2"]
    assert summary["feature_space"] == summary["selection"]["selected"]["feature_space"]
    assert "validation_metrics" in summary


def test_analyze_reliability_residual_buckets_cli_reports_train_derived_buckets(tmp_path):
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "seed": 0,
        "samples_per_class": 6,
        "validation_samples_per_class": 6,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_options": {},
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "git_commit": "test",
    }
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    left_probabilities = np.array(
        [0.1, 0.2, 0.8, 0.7, 0.15, 0.25, 0.9, 0.8, 0.2, 0.3, 0.85, 0.75],
        dtype=np.float32,
    )
    right_probabilities = np.array(
        [0.2, 0.8, 0.2, 0.3, 0.25, 0.35, 0.8, 0.2, 0.8, 0.7, 0.75, 0.65],
        dtype=np.float32,
    )
    dirs = {
        "train_left": tmp_path / "train_left",
        "train_right": tmp_path / "train_right",
        "validation_left": tmp_path / "validation_left",
        "validation_right": tmp_path / "validation_right",
    }
    for name, directory, model_key, probabilities in [
        ("train_left", dirs["train_left"], "left", left_probabilities),
        ("train_right", dirs["train_right"], "right", right_probabilities),
        ("validation_left", dirs["validation_left"], "left", left_probabilities),
        ("validation_right", dirs["validation_right"], "right", right_probabilities),
    ]:
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=probabilities,
                logits=np.log(np.clip(probabilities, 1e-6, 1.0) / np.clip(1.0 - probabilities, 1e-6, 1.0)),
                sample_ids=sample_ids,
                metadata={**metadata, "model_key": model_key, "run_id": name},
            ),
        )
    output = tmp_path / "residual_buckets.json"

    status = analyze_reliability_residual_buckets_main(
        [
            "--train-artifacts",
            str(dirs["train_left"]),
            str(dirs["train_right"]),
            "--validation-artifacts",
            str(dirs["validation_left"]),
            str(dirs["validation_right"]),
            "--bucket-count",
            "3",
            "--min-disagreement-rate",
            "0.01",
            "--min-bucket-fraction",
            "0.1",
            "--min-correction-gap",
            "0.01",
            "--min-both-wrong-lift",
            "0.01",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["model_order"] == ["left", "right"]
    assert summary["bucket_count"] == 3
    assert "min_confidence" in summary["bucket_reports"]
    assert len(summary["bucket_reports"]["min_confidence"]["validation"]) == 3
    assert summary["candidate_buckets"]
    assert summary["decision"] == "reliability_residual_bucket_route_candidate_local"
    assert "train-derived bucket edges" in summary["claim_scope"]


def test_fit_bucket_conditioned_feature_expert_writes_score_artifact(tmp_path):
    metadata = {
        "status": "pass",
        "kind": "bit_sensitivity_feature_matrix",
        "cipher": "PRESENT-80",
        "rounds": 8,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x0",
        "validation_key": "0x1",
        "feature_view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_view_metadata": {
            "feature_names": [
                "primary_depth_trailword_mean_depth0_trailword0",
                "aux_depth_cell_global_mean",
                "aux_depth_word_global_mean",
                "aux_word_global_mean",
            ]
        },
    }
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    train_features = np.array(
        [
            [0.1, 0.2, 0.2, 0.1],
            [0.2, 0.1, 0.2, 0.1],
            [0.8, 0.7, 0.8, 0.6],
            [0.7, 0.8, 0.7, 0.6],
            [0.8, 0.9, 0.8, 0.9],
            [0.9, 0.8, 0.7, 0.9],
            [0.2, 0.3, 0.2, 0.4],
            [0.3, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    validation_features = train_features.copy()
    train_feature_dir = tmp_path / "train_features"
    validation_feature_dir = tmp_path / "validation_features"
    for directory, split, features in [
        (train_feature_dir, "train", train_features),
        (validation_feature_dir, "validation", validation_features),
    ]:
        directory.mkdir()
        np.save(directory / "features.npy", features)
        np.save(directory / "labels.npy", labels)
        np.save(directory / "sample_ids.npy", sample_ids)
        (directory / "metadata.json").write_text(
            json.dumps({**metadata, "split": split, "samples_per_class": 4, "total_rows": 8}) + "\n",
            encoding="utf-8",
        )
    bucket_left_probs = np.array([0.1, 0.2, 0.6, 0.7, 0.9, 0.8, 0.4, 0.3], dtype=np.float32)
    bucket_right_probs = np.array([0.2, 0.1, 0.7, 0.6, 0.8, 0.9, 0.3, 0.4], dtype=np.float32)
    bucket_dirs = []
    for name, model_key, probabilities in [
        ("train_left", "left", bucket_left_probs),
        ("train_right", "right", bucket_right_probs),
        ("validation_left", "left", bucket_left_probs),
        ("validation_right", "right", bucket_right_probs),
    ]:
        directory = tmp_path / name
        bucket_dirs.append(directory)
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=probabilities,
                logits=np.log(np.clip(probabilities, 1e-6, 1.0) / np.clip(1.0 - probabilities, 1e-6, 1.0)),
                sample_ids=sample_ids,
                metadata={
                    "cipher": "PRESENT-80",
                    "rounds": 8,
                    "validation_samples_per_class": 4,
                    "pairs_per_sample": 16,
                    "feature_encoding": metadata["feature_encoding"],
                    "negative_mode": metadata["negative_mode"],
                    "sample_structure": metadata["sample_structure"],
                    "difference_profile": metadata["difference_profile"],
                    "difference_member": 0,
                    "validation_key": "0x1",
                    "model_key": model_key,
                    "run_id": name,
                },
            ),
        )
    output_validation_dir = tmp_path / "bucket_expert_validation"
    output_train_dir = tmp_path / "bucket_expert_train"
    output_report = tmp_path / "bucket_expert_report.json"

    status = fit_bucket_conditioned_feature_main(
        [
            "--train-feature-dir",
            str(train_feature_dir),
            "--validation-feature-dir",
            str(validation_feature_dir),
            "--train-bucket-artifacts",
            str(bucket_dirs[0]),
            str(bucket_dirs[1]),
            "--validation-bucket-artifacts",
            str(bucket_dirs[2]),
            str(bucket_dirs[3]),
            "--bucket-feature",
            "logit_gap_abs",
            "--bucket-count",
            "2",
            "--include-feature-prefix",
            "primary_depth_trailword_",
            "--include-feature-prefix",
            "aux_depth_cell_",
            "--include-feature-prefix",
            "aux_depth_word_",
            "--include-feature-prefix",
            "aux_word_global_",
            "--steps",
            "50",
            "--learning-rate",
            "0.1",
            "--output-train-dir",
            str(output_train_dir),
            "--output-validation-dir",
            str(output_validation_dir),
            "--output-report",
            str(output_report),
        ]
    )

    report = json.loads(output_report.read_text(encoding="utf-8"))
    loaded = load_score_artifact(output_validation_dir)
    assert status == 0
    assert report["status"] == "pass"
    assert report["bucket_feature"] == "logit_gap_abs"
    assert report["bucket_count"] == 2
    assert report["feature_count"] == 4
    assert (output_train_dir / "models.json").exists()
    assert loaded.metadata["feature_model"] == "bucket_conditioned_logistic"
    assert loaded.metadata["expert_family"] == "bucket_conditioned_spn_residual"
    assert loaded.metadata["bucket_train_values_shuffled"] is False
    assert loaded.metadata["bucket_validation_values_shuffled"] is False
    assert np.array_equal(loaded.labels, labels)
    assert "train-derived" in loaded.metadata["claim_scope"]


def test_fit_bucket_conditioned_feature_expert_rejects_second_bucket_sample_mismatch(tmp_path):
    metadata = {
        "status": "pass",
        "kind": "bit_sensitivity_feature_matrix",
        "cipher": "PRESENT-80",
        "rounds": 8,
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x0",
        "validation_key": "0x1",
        "feature_view": "compressed_span_summary",
        "output_feature_bits": 2,
        "feature_view_metadata": {
            "feature_names": [
                "primary_depth_trailword_mean_depth0_trailword0",
                "aux_depth_cell_global_mean",
            ]
        },
    }
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    features = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]], dtype=np.float32)
    train_feature_dir = tmp_path / "train_features"
    validation_feature_dir = tmp_path / "validation_features"
    for directory, split in [(train_feature_dir, "train"), (validation_feature_dir, "validation")]:
        directory.mkdir()
        np.save(directory / "features.npy", features)
        np.save(directory / "labels.npy", labels)
        np.save(directory / "sample_ids.npy", sample_ids)
        (directory / "metadata.json").write_text(
            json.dumps({**metadata, "split": split, "samples_per_class": 2, "total_rows": 4}) + "\n",
            encoding="utf-8",
        )

    probabilities = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32)
    logits = np.log(np.clip(probabilities, 1e-6, 1.0) / np.clip(1.0 - probabilities, 1e-6, 1.0))
    bucket_dirs = []
    for name, model_key, ids in [
        ("train_left", "left", sample_ids),
        ("train_right", "right", sample_ids[::-1]),
        ("validation_left", "left", sample_ids),
        ("validation_right", "right", sample_ids),
    ]:
        directory = tmp_path / name
        bucket_dirs.append(directory)
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=probabilities,
                logits=logits,
                sample_ids=ids,
                metadata={**metadata, "model_key": model_key, "run_id": name},
            ),
        )

    try:
        fit_bucket_conditioned_feature_main(
            [
                "--train-feature-dir",
                str(train_feature_dir),
                "--validation-feature-dir",
                str(validation_feature_dir),
                "--train-bucket-artifacts",
                str(bucket_dirs[0]),
                str(bucket_dirs[1]),
                "--validation-bucket-artifacts",
                str(bucket_dirs[2]),
                str(bucket_dirs[3]),
                "--bucket-count",
                "2",
                "--steps",
                "10",
                "--output-validation-dir",
                str(tmp_path / "validation_scores"),
                "--output-report",
                str(tmp_path / "report.json"),
            ]
        )
    except ValueError as exc:
        assert "train bucket artifact 1 feature sample_ids differ" in str(exc)
    else:  # pragma: no cover - the assertion above is the behavior under test.
        raise AssertionError("expected second bucket artifact sample mismatch to be rejected")


def test_fit_residual_correction_feature_expert_writes_corrected_artifact(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_output = tmp_path / "residual_train_scores"
    validation_output = tmp_path / "residual_validation_scores"
    report_output = tmp_path / "residual_report.json"
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "feature_names": [
            "aux_word_global_mean",
            "primary_depth_mean_depth0",
        ]
    }
    train_features = np.array(
        [
            [-2.0, 0.0],
            [-1.5, 0.1],
            [-1.0, 0.2],
            [-0.8, 0.3],
            [0.8, 0.3],
            [1.0, 0.2],
            [1.5, 0.1],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=train_features.copy(),
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    left_train = tmp_path / "left_train"
    right_train = tmp_path / "right_train"
    left_validation = tmp_path / "left_validation"
    right_validation = tmp_path / "right_validation"
    left_probabilities = np.array([0.1, 0.2, 0.65, 0.7, 0.3, 0.35, 0.8, 0.9], dtype=np.float32)
    right_probabilities = np.array([0.15, 0.25, 0.55, 0.75, 0.25, 0.45, 0.75, 0.85], dtype=np.float32)
    for path, model_key, probabilities in [
        (left_train, "trail", left_probabilities),
        (right_train, "raw117", right_probabilities),
        (left_validation, "trail", left_probabilities),
        (right_validation, "raw117", right_probabilities),
    ]:
        _write_tiny_score_artifact(path, labels, probabilities, sample_ids, model_key=model_key)

    status = fit_residual_correction_feature_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--train-base-artifacts",
            str(left_train),
            str(right_train),
            "--validation-base-artifacts",
            str(left_validation),
            str(right_validation),
            "--include-feature-prefix",
            "aux_word_",
            "--bucket-feature",
            "logit_gap_abs",
            "--bucket-count",
            "2",
            "--steps",
            "500",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--residual-focus-fraction",
            "0.25",
            "--residual-focus-background-weight",
            "0.2",
            "--output-train-dir",
            str(train_output),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    validation_artifact = load_score_artifact(validation_output)
    assert status == 0
    assert report["status"] == "pass"
    assert report["feature_selection"]["include_feature_prefixes"] == ["aux_word_"]
    assert report["selected_feature_count"] == 1
    assert report["bucket_count"] == 2
    assert report["fit"]["residual_focus"]["mode"] == "top_base_residual_loss"
    assert report["fit"]["residual_focus"]["focused_rows"] == 2
    assert report["validation_metrics"]["auc"] > report["validation_base_logit_mean_metrics"]["auc"]
    assert report["delta_validation_corrected_vs_base_logit_mean_auc"] > 0.0
    assert validation_metadata["feature_model"] == "residual_logit_correction"
    assert validation_metadata["residual_focus_fraction"] == 0.25
    assert validation_metadata["residual_focus_background_weight"] == 0.2
    assert validation_metadata["score_split"] == "validation"
    assert validation_metadata["base_model_order"] == ["trail", "raw117"]
    assert validation_metadata["base_run_order"] == ["trail", "raw117"]
    assert np.array_equal(validation_artifact.sample_ids, sample_ids)


def test_evaluate_residual_slice_correction_uses_train_derived_threshold(tmp_path):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    left_train = tmp_path / "left_train"
    right_train = tmp_path / "right_train"
    left_validation = tmp_path / "left_validation"
    right_validation = tmp_path / "right_validation"
    corrected_validation = tmp_path / "corrected_validation"
    base_left = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9], dtype=np.float32)
    base_right = np.array([0.15, 0.25, 0.55, 0.65, 0.35, 0.45, 0.75, 0.85], dtype=np.float32)
    corrected = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9], dtype=np.float32)
    for path, model_key, probabilities in [
        (left_train, "trail", base_left),
        (right_train, "raw117", base_right),
        (left_validation, "trail", base_left),
        (right_validation, "raw117", base_right),
        (corrected_validation, "residual_focus", corrected),
    ]:
        _write_tiny_score_artifact(path, labels, probabilities, sample_ids, model_key=model_key)
    output = tmp_path / "slice_report.json"

    status = evaluate_residual_slice_correction_main(
        [
            "--train-base-artifacts",
            str(left_train),
            str(right_train),
            "--validation-base-artifacts",
            str(left_validation),
            str(right_validation),
            "--validation-corrected-artifact",
            str(corrected_validation),
            "--focus-fraction",
            "0.5",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["focus"]["mode"] == "train_derived_base_residual_loss_threshold"
    assert report["validation_focus_metrics"]["rows"] == 4
    assert report["validation_focus_delta"]["residual_loss_mean"] < 0.0
    assert report["validation_focus_corrected_metrics"]["auc"] > report["validation_focus_base_metrics"]["auc"]
    assert "validation is sliced by train-derived base residual threshold" in report["claim_scope"]


def test_evaluate_stacked_ensemble_rejects_mismatched_model_order(tmp_path):
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 8,
        "validation_samples_per_class": 2,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
    }
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    sample_ids = np.array(["0", "1", "2", "3"], dtype=str)
    dirs = [tmp_path / name for name in ["train_a", "train_b", "validation_a", "validation_b"]]
    for directory, model_key in zip(dirs, ["a", "b", "b", "a"], strict=True):
        write_score_artifact(
            directory,
            EnsembleScoreArtifact(
                labels=labels,
                probabilities=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
                logits=np.array([-2.2, -0.8, 0.8, 2.2], dtype=np.float32),
                sample_ids=sample_ids,
                metadata={**metadata, "model_key": model_key},
            ),
        )

    try:
        evaluate_stacked_ensemble_main(
            [
                "--train-artifacts",
                str(dirs[0]),
                str(dirs[1]),
                "--validation-artifacts",
                str(dirs[2]),
                str(dirs[3]),
                "--output",
                str(tmp_path / "summary.json"),
            ]
        )
    except ValueError as exc:
        assert "model order differs" in str(exc)
    else:
        raise AssertionError("expected mismatched model order to fail")


def test_summarize_stacked_selection_reports_stability(tmp_path):
    reports = []
    for index, delta in enumerate([0.1, 0.2, 0.3]):
        path = tmp_path / f"positive_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "stacked_ensemble_improves_validation_best_single",
                    "selection": {
                        "selection_seed": index,
                        "selected": {
                            "feature_space": "probabilities",
                            "l2": 0.0,
                            "standardize": False,
                        },
                    },
                    "fit": {"l2": 0.0, "standardize": False},
                    "validation_metrics": {"auc": 0.8 + delta},
                    "validation_best_single": {"metrics": {"auc": 0.8}},
                    "delta_stacked_vs_validation_best_single_auc": delta,
                    "delta_stacked_vs_validation_best_fixed_ensemble_auc": delta + 0.01,
                }
            ),
            encoding="utf-8",
        )
        reports.append(path)
    output = tmp_path / "summary.json"

    status = summarize_stacked_selection_main(
        [
            "--reports",
            *(str(path) for path in reports),
            "--output",
            str(output),
            "--require-same-selection",
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["decision"] == "stable_stacked_selection_improves_best_single"
    assert summary["positive_count"] == 3
    assert summary["same_selection"] is True
    assert summary["dominant_selection"]["feature_space"] == "probabilities"
    assert summary["delta_vs_best_single_auc"]["min"] == 0.1
    assert "does not train models" in summary["claim_scope"]


def test_summarize_stacked_selection_reports_mixed_diagnostic(tmp_path):
    reports = []
    for index, delta in enumerate([-0.1, 0.2]):
        path = tmp_path / f"mixed_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "diagnostic",
                    "selection": {
                        "selection_seed": index,
                        "selected": {
                            "feature_space": "logits" if index == 0 else "probabilities",
                            "l2": 0.0,
                            "standardize": False,
                        },
                    },
                    "validation_metrics": {"auc": 0.8 + delta},
                    "validation_best_single": {"metrics": {"auc": 0.8}},
                    "delta_stacked_vs_validation_best_single_auc": delta,
                    "delta_stacked_vs_validation_best_fixed_ensemble_auc": delta + 0.01,
                }
            ),
            encoding="utf-8",
        )
        reports.append(path)
    output = tmp_path / "summary.json"

    status = summarize_stacked_selection_main(
        [
            "--reports",
            *(str(path) for path in reports),
            "--output",
            str(output),
            "--require-same-selection",
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["decision"] == "mixed_or_unstable_stacked_selection_diagnostic"
    assert summary["positive_count"] == 1
    assert summary["same_selection"] is False
    assert summary["delta_vs_best_single_auc"]["max"] == 0.2


def test_summarize_stacked_route_reports_cross_seed_pass(tmp_path):
    summaries = []
    for index, delta_mean in enumerate([0.01, 0.02]):
        path = tmp_path / f"seed{index}_summary.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "stable_stacked_selection_improves_best_single",
                    "report_count": 5,
                    "positive_count": 5,
                    "positive_fraction": 1.0,
                    "delta_vs_best_single_auc": {
                        "min": delta_mean,
                        "max": delta_mean + 0.001,
                        "mean": delta_mean,
                    },
                    "same_selection": True,
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)
    output = tmp_path / "route_summary.json"

    status = summarize_stacked_route_main(
        [
            "--summaries",
            *(str(path) for path in summaries),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["decision"] == "stable_cross_seed_stacking_improves_best_single"
    assert summary["passed_seed_count"] == 2
    assert summary["positive_seed_fraction"] == 1.0
    assert summary["delta_mean_vs_best_single_auc"]["min"] == 0.01
    assert "route-level diagnostic" in summary["claim_scope"]


def test_summarize_stacked_route_reports_mixed_cross_seed_diagnostic(tmp_path):
    rows = [
        (
            "seed0_summary.json",
            "mixed_or_unstable_stacked_selection_diagnostic",
            0,
            0.0,
            -0.00004,
        ),
        (
            "seed1_summary.json",
            "stable_stacked_selection_improves_best_single",
            5,
            1.0,
            0.00012,
        ),
    ]
    summaries = []
    for name, decision, positive_count, positive_fraction, delta_mean in rows:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "decision": decision,
                    "report_count": 5,
                    "positive_count": positive_count,
                    "positive_fraction": positive_fraction,
                    "delta_vs_best_single_auc": {
                        "min": delta_mean,
                        "max": delta_mean,
                        "mean": delta_mean,
                    },
                    "same_selection": False,
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)
    output = tmp_path / "route_summary.json"

    status = summarize_stacked_route_main(
        [
            "--summaries",
            *(str(path) for path in summaries),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["decision"] == "stable_but_mixed_cross_seed_stacking_diagnostic"
    assert summary["passed_seed_count"] == 1
    assert summary["positive_seed_fraction"] == 0.5
    assert summary["delta_mean_vs_best_single_auc"]["min"] == -0.00004


def test_fit_compressed_feature_expert_writes_train_and_validation_artifacts(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    train_output = tmp_path / "train_scores"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    _write_feature_dir(
        train_dir,
        split="train",
        features=np.array(
            [
                [-2.0, 0.0],
                [-1.5, 0.2],
                [1.5, 0.1],
                [2.0, -0.1],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=np.array(
            [
                [-1.8, 0.1],
                [-1.2, -0.1],
                [1.2, 0.0],
                [1.8, 0.2],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
    )

    status = fit_compressed_feature_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-train-dir",
            str(train_output),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--steps",
            "400",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--run-id",
            "compressed_feature_test",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    train_metadata = json.loads((train_output / "models.json").read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    validation_labels = np.load(validation_output / "labels.npy")
    validation_probabilities = np.load(validation_output / "probabilities.npy")
    assert status == 0
    assert report["decision"] == "compressed_feature_expert_local_screen_positive_needs_controls"
    assert "strict_negative_mode_required" in report["guardrails"]
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["model_key"] == "compressed_feature_logistic_expert"
    assert validation_metadata["expert_family"] == "compressed_spn_structural_stats"
    assert validation_metadata["feature_fit_split"] == "train"
    assert validation_metadata["score_split"] == "validation"
    assert train_metadata["split"] == "train"
    assert train_metadata["score_split"] == "train"
    assert train_metadata["train_samples_per_class"] == 2
    assert np.array_equal(validation_labels, np.array([0, 0, 1, 1], dtype=np.float32))
    assert validation_metadata["validation_samples_per_class"] == 2
    assert validation_probabilities[2] > validation_probabilities[1]
    assert (train_output / "models.json").exists()


def test_fit_compressed_feature_expert_can_filter_span_feature_families(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    feature_count = 3708
    span_index = 2620
    train_features = np.zeros((4, feature_count), dtype=np.float32)
    validation_features = np.zeros((4, feature_count), dtype=np.float32)
    train_features[:, 0] = np.array([2.0, 1.5, -1.5, -2.0], dtype=np.float32)
    validation_features[:, 0] = np.array([1.8, 1.2, -1.2, -1.8], dtype=np.float32)
    train_features[:, span_index] = np.array([-2.0, -1.5, 1.5, 2.0], dtype=np.float32)
    validation_features[:, span_index] = np.array([-1.8, -1.2, 1.2, 1.8], dtype=np.float32)
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        feature_view_metadata={
            "words_per_pair": 39,
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "output_feature_bits": feature_count,
        },
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        feature_view_metadata={
            "words_per_pair": 39,
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "output_feature_bits": feature_count,
        },
    )

    status = fit_compressed_feature_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--steps",
            "400",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--include-feature-family",
            "depth_word_cell_span",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["feature_selection"]["include_feature_families"] == ["depth_word_cell_span"]
    assert report["feature_selection"]["original_feature_count"] == feature_count
    assert report["feature_count"] == 576
    assert span_index in report["feature_selection"]["selected_feature_indices"]
    assert 0 not in report["feature_selection"]["selected_feature_indices"]
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_selection"]["include_feature_families"] == ["depth_word_cell_span"]


def test_fit_compressed_feature_expert_can_filter_feature_name_prefixes(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, 2.0, -1.5],
            [-1.5, 1.5, -1.0],
            [1.5, -1.5, 1.0],
            [2.0, -2.0, 1.5],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, 1.8, -1.2],
            [-1.2, 1.2, -0.8],
            [1.2, -1.2, 0.8],
            [1.8, -1.8, 1.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 3,
        "feature_names": [
            "primary_depth_mean_depth0",
            "aux_cell_mean_cell0",
            "primary_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_feature_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--steps",
            "400",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--include-feature-prefix",
            "primary_",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["feature_selection"]["mode"] == "prefix_filter"
    assert report["feature_selection"]["include_feature_prefixes"] == ["primary_"]
    assert report["feature_selection"]["selected_feature_indices"] == [0, 2]
    assert report["feature_count"] == 2
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_selection"]["include_feature_prefixes"] == ["primary_"]


def test_fit_compressed_span_grouped_expert_combines_primary_and_aux_branches(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    train_output = tmp_path / "train_scores"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, 0.2, -1.8],
            [-1.5, 0.4, -1.2],
            [1.5, -0.4, 1.2],
            [2.0, -0.2, 1.8],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, 0.1, -1.5],
            [-1.2, 0.3, -1.0],
            [1.2, -0.3, 1.0],
            [1.8, -0.1, 1.5],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 3,
        "feature_names": [
            "primary_depth_mean_depth0",
            "aux_cell_mean_cell0",
            "primary_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_grouped_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-train-dir",
            str(train_output),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--branch-steps",
            "400",
            "--steps",
            "300",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    train_metadata = json.loads((train_output / "models.json").read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_grouped_expert_local_screen_positive_needs_controls"
    assert report["feature_count"] == 2
    assert report["branch_reports"]["primary"]["feature_count"] == 2
    assert report["branch_reports"]["auxiliary"]["feature_count"] == 1
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["model_key"] == "compressed_span_grouped_logistic_expert"
    assert validation_metadata["feature_model"] == "two_branch_logistic"
    assert validation_metadata["branch_prefixes"] == {"primary": "primary_", "auxiliary": "aux_"}
    assert train_metadata["score_split"] == "train"
    assert validation_metadata["score_split"] == "validation"
    assert np.array_equal(np.load(validation_output / "labels.npy"), labels)


def test_fit_compressed_span_grouped_expert_can_use_semantic_branch_groups(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.5, 0.2, 0.1],
            [-1.4, -1.0, 0.4, 0.3],
            [1.4, 1.0, -0.4, -0.3],
            [2.0, 1.5, -0.2, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -1.3, 0.1, 0.2],
            [-1.1, -0.9, 0.3, 0.4],
            [1.1, 0.9, -0.3, -0.4],
            [1.8, 1.3, -0.1, -0.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_grouped_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--group-mode",
            "semantic",
            "--branch-steps",
            "300",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["group_mode"] == "semantic"
    assert report["feature_count"] == 4
    assert report["branch_feature_counts"] == {
        "primary_depth": 1,
        "primary_cell": 1,
        "aux_depth_cell": 1,
        "aux_word_global": 1,
    }
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_model"] == "semantic_group_logistic"
    assert validation_metadata["branch_group_mode"] == "semantic"
    assert validation_metadata["branch_group_names"] == [
        "primary_depth",
        "primary_cell",
        "aux_depth_cell",
        "aux_word_global",
    ]


def test_fit_compressed_span_grouped_expert_can_use_hybrid_branch_groups(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.5, 0.2, 0.1],
            [-1.4, -1.0, 0.4, 0.3],
            [1.4, 1.0, -0.4, -0.3],
            [2.0, 1.5, -0.2, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -1.3, 0.1, 0.2],
            [-1.1, -0.9, 0.3, 0.4],
            [1.1, 0.9, -0.3, -0.4],
            [1.8, 1.3, -0.1, -0.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_grouped_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--group-mode",
            "hybrid",
            "--branch-steps",
            "300",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["group_mode"] == "hybrid"
    assert report["feature_count"] == 6
    assert report["branch_feature_counts"]["primary"] == 2
    assert report["branch_feature_counts"]["auxiliary"] == 2
    assert validation_metadata["feature_model"] == "hybrid_group_logistic"
    assert validation_metadata["branch_group_names"] == [
        "primary",
        "auxiliary",
        "primary_depth",
        "primary_cell",
        "aux_depth_cell",
        "aux_word_global",
    ]


def test_fit_compressed_span_interaction_expert_adds_train_selected_cross_group_terms(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--top-primary",
            "1",
            "--top-auxiliary",
            "2",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_interaction_local_screen_positive_needs_controls"
    assert report["feature_model"] == "raw_plus_primary_auxiliary_interactions_logistic"
    assert report["feature_count"] == 6
    assert report["interaction_count"] == 2
    assert report["interaction_selection"]["top_primary"] == 1
    assert report["interaction_selection"]["top_auxiliary"] == 2
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_model"] == "raw_plus_primary_auxiliary_interactions_logistic"
    assert validation_metadata["interaction_count"] == 2
    assert validation_metadata["feature_count"] == 6


def test_fit_compressed_span_interaction_expert_can_hold_out_selection_rows(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--top-primary",
            "1",
            "--top-auxiliary",
            "2",
            "--selection-holdout-fraction",
            "0.5",
            "--selection-seed",
            "7",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["selection_fit_split_mode"] == "train_internal_holdout"
    assert report["interaction_selection_rows"] == 2
    assert report["fit_rows"] == 2
    assert report["selection_holdout_fraction"] == 0.5
    assert report["selection_seed"] == 7
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["selection_fit_split_mode"] == "train_internal_holdout"
    assert validation_metadata["interaction_selection_rows"] == 2
    assert validation_metadata["fit_rows"] == 2


def test_fit_compressed_span_interaction_expert_can_select_raw_prefixes(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 4.0, 0.5, 0.1],
            [-1.5, -0.8, 3.5, 0.4, 0.2],
            [1.5, 0.8, 3.0, -0.4, -0.2],
            [2.0, 1.0, 2.5, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 5.0, 0.4, 0.1],
            [-1.2, -0.7, 4.5, 0.3, 0.2],
            [1.2, 0.7, 4.0, -0.3, -0.2],
            [1.8, 0.9, 3.5, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 5,
        "feature_names": [
            "primary_depth_trailword_mean_depth0_trailword0",
            "primary_depth_trailword_mean_depth0_trailword1",
            "drop_global_noise",
            "aux_depth_cell_depth_mean_depth0",
            "aux_word_global_mean",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--primary-prefix",
            "primary_depth_trailword_",
            "--auxiliary-prefix",
            "aux_depth_cell_",
            "--top-primary",
            "1",
            "--top-auxiliary",
            "1",
            "--include-raw-feature-prefix",
            "primary_depth_trailword_",
            "--include-raw-feature-prefix",
            "aux_depth_cell_",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["feature_model"] == "selected_raw_plus_primary_auxiliary_interactions_logistic"
    assert report["raw_feature_selection"] == "prefix_filter"
    assert report["selected_raw_feature_prefixes"] == ["aux_depth_cell_", "primary_depth_trailword_"]
    assert report["selected_raw_feature_count"] == 3
    assert report["selected_raw_feature_indices"] == [0, 1, 3]
    assert report["feature_count"] == 4
    assert report["interaction_count"] == 1
    assert validation_metadata["feature_model"] == "selected_raw_plus_primary_auxiliary_interactions_logistic"
    assert validation_metadata["selected_raw_feature_count"] == 3
    assert validation_metadata["feature_count"] == 4


def test_fit_compressed_span_block_interaction_expert_builds_semantic_block_terms(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_block_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_block_interaction_local_screen_positive_needs_controls"
    assert report["feature_model"] == "raw_plus_semantic_block_interactions_logistic"
    assert report["raw_feature_count"] == 4
    assert report["primary_group_count"] == 2
    assert report["auxiliary_group_count"] == 2
    assert report["block_pair_count"] == 4
    assert report["block_interaction_stat_count"] == 4
    assert report["feature_count"] == 20
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_model"] == "raw_plus_semantic_block_interactions_logistic"
    assert validation_metadata["block_pair_count"] == 4
    assert validation_metadata["feature_count"] == 20


def test_fit_compressed_span_low_rank_interaction_expert_builds_block_tensor_terms(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_low_rank_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--rank",
            "1",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_low_rank_interaction_local_screen_positive_needs_controls"
    assert report["feature_model"] == "raw_plus_semantic_low_rank_block_interactions_logistic"
    assert report["raw_feature_count"] == 4
    assert report["primary_group_count"] == 2
    assert report["auxiliary_group_count"] == 2
    assert report["low_rank_projection_rank"] == 1
    assert report["block_pair_count"] == 4
    assert report["low_rank_interaction_feature_count"] == 4
    assert report["feature_count"] == 8
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_model"] == "raw_plus_semantic_low_rank_block_interactions_logistic"
    assert validation_metadata["low_rank_interaction_feature_count"] == 4
    assert validation_metadata["feature_count"] == 8


def test_fit_compressed_span_low_rank_interaction_expert_can_use_interaction_only_features(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 1.0, 0.5],
            [-1.5, -0.8, 0.9, 0.4],
            [1.5, 0.8, 0.9, 0.4],
            [2.0, 1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.9, 0.4],
            [-1.2, -0.7, 0.8, 0.3],
            [1.2, 0.7, 0.8, 0.3],
            [1.8, 0.9, 0.9, 0.4],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_low_rank_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--rank",
            "1",
            "--interaction-only",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_low_rank_interaction_local_screen_positive_needs_controls"
    assert report["feature_model"] == "semantic_low_rank_block_interactions_only_logistic"
    assert report["raw_features_included"] is False
    assert report["raw_feature_count"] == 4
    assert report["low_rank_interaction_feature_count"] == 4
    assert report["feature_count"] == 4
    assert report["fit"]["weight_count"] == 4
    assert 0.0 <= report["validation_metrics"]["auc"] <= 1.0
    assert validation_metadata["feature_model"] == "semantic_low_rank_block_interactions_only_logistic"
    assert validation_metadata["raw_features_included"] is False
    assert validation_metadata["feature_count"] == 4


def test_fit_compressed_span_low_rank_interaction_expert_can_select_raw_prefixes_before_adding_interactions(
    tmp_path,
):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 1.0, 0.5],
            [-1.5, -0.8, 0.9, 0.4],
            [1.5, 0.8, 0.9, 0.4],
            [2.0, 1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.9, 0.4],
            [-1.2, -0.7, 0.8, 0.3],
            [1.2, 0.7, 0.8, 0.3],
            [1.8, 0.9, 0.9, 0.4],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_low_rank_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--rank",
            "1",
            "--include-raw-feature-prefix",
            "primary_depth_mean_",
            "--include-raw-feature-prefix",
            "aux_depth_cell_",
            "--steps",
            "250",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["feature_model"] == "selected_raw_plus_semantic_low_rank_block_interactions_logistic"
    assert report["raw_features_included"] is True
    assert report["raw_feature_count"] == 4
    assert report["raw_feature_selection"] == "prefix_filter"
    assert report["selected_raw_feature_count"] == 2
    assert report["selected_raw_feature_prefixes"] == ["aux_depth_cell_", "primary_depth_mean_"]
    assert report["selected_raw_feature_indices"] == [0, 2]
    assert report["low_rank_interaction_feature_count"] == 4
    assert report["feature_count"] == 6
    assert report["fit"]["weight_count"] == 6
    assert validation_metadata["feature_model"] == "selected_raw_plus_semantic_low_rank_block_interactions_logistic"
    assert validation_metadata["raw_feature_selection"] == "prefix_filter"
    assert validation_metadata["selected_raw_feature_count"] == 2
    assert validation_metadata["selected_raw_feature_prefixes"] == ["aux_depth_cell_", "primary_depth_mean_"]
    assert validation_metadata["feature_count"] == 6


def test_fit_compressed_span_learned_low_rank_interaction_expert_scores_block_tensor_model(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_learned_low_rank_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--rank",
            "1",
            "--steps",
            "250",
            "--learning-rate",
            "0.05",
            "--weight-decay",
            "0.0",
            "--seed",
            "7",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_learned_low_rank_interaction_local_diagnostic"
    assert report["feature_model"] == "raw_plus_learned_semantic_low_rank_block_interactions"
    assert report["raw_feature_count"] == 4
    assert report["primary_group_count"] == 2
    assert report["auxiliary_group_count"] == 2
    assert report["block_pair_count"] == 4
    assert report["learned_low_rank_interaction_count"] == 4
    assert report["low_rank_projection_rank"] == 1
    assert report["feature_count"] == 8
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["feature_model"] == "raw_plus_learned_semantic_low_rank_block_interactions"
    assert validation_metadata["learned_low_rank_interaction_count"] == 4
    assert validation_metadata["feature_count"] == 8


def test_fit_compressed_span_learned_low_rank_interaction_expert_can_freeze_svd_projections(tmp_path):
    train_dir = tmp_path / "train_summary_features"
    validation_dir = tmp_path / "validation_summary_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    train_features = np.array(
        [
            [-2.0, -1.0, 0.5, 0.1],
            [-1.5, -0.8, 0.4, 0.2],
            [1.5, 0.8, -0.4, -0.2],
            [2.0, 1.0, -0.5, -0.1],
        ],
        dtype=np.float32,
    )
    validation_features = np.array(
        [
            [-1.8, -0.9, 0.4, 0.1],
            [-1.2, -0.7, 0.3, 0.2],
            [1.2, 0.7, -0.3, -0.2],
            [1.8, 0.9, -0.4, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    feature_view_metadata = {
        "view": "compressed_span_summary",
        "output_feature_bits": 4,
        "feature_names": [
            "primary_depth_mean_depth0",
            "primary_cell_mean_cell0",
            "aux_depth_cell_mean_depth0_cell0",
            "aux_word_global_max",
        ],
    }
    _write_feature_dir(
        train_dir,
        split="train",
        features=train_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=validation_features,
        labels=labels,
        feature_view_metadata=feature_view_metadata,
    )

    status = fit_compressed_span_learned_low_rank_interaction_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--rank",
            "1",
            "--projection-init",
            "svd",
            "--freeze-projections",
            "--steps",
            "250",
            "--learning-rate",
            "0.05",
            "--weight-decay",
            "0.0",
            "--seed",
            "11",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    validation_metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["projection_initialization"] == "svd"
    assert report["freeze_projections"] is True
    assert report["fit"]["trainable_projection_parameter_count"] == 0
    assert report["fit"]["projection_parameter_count"] == 4
    assert report["validation_metrics"]["auc"] == 1.0
    assert validation_metadata["projection_initialization"] == "svd"
    assert validation_metadata["freeze_projections"] is True


def test_fit_compressed_feature_expert_requires_train_split(tmp_path):
    train_dir = tmp_path / "bad_train_features"
    validation_dir = tmp_path / "validation_features"
    _write_feature_dir(
        train_dir,
        split="validation",
        features=np.array([[0.0], [1.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.float32),
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=np.array([[0.0], [1.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.float32),
    )

    try:
        fit_compressed_feature_main(
            [
                "--train-feature-dir",
                str(train_dir),
                "--validation-feature-dir",
                str(validation_dir),
                "--output-validation-dir",
                str(tmp_path / "scores"),
                "--output-report",
                str(tmp_path / "report.json"),
            ]
        )
    except ValueError as exc:
        assert "train feature dir must have split=train" in str(exc)
    else:
        raise AssertionError("expected validation split to be rejected as train input")


def test_fit_compressed_feature_expert_requires_strict_negative_mode(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    _write_feature_dir(
        train_dir,
        split="train",
        features=np.array([[0.0], [1.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.float32),
        negative_mode="random_ciphertexts",
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=np.array([[0.0], [1.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.float32),
        negative_mode="random_ciphertexts",
    )

    try:
        fit_compressed_feature_main(
            [
                "--train-feature-dir",
                str(train_dir),
                "--validation-feature-dir",
                str(validation_dir),
                "--output-validation-dir",
                str(tmp_path / "scores"),
                "--output-report",
                str(tmp_path / "report.json"),
            ]
        )
    except ValueError as exc:
        assert "negative_mode must be encrypted_random_plaintexts" in str(exc)
    else:
        raise AssertionError("expected non-strict negative mode to be rejected")


def test_fit_compressed_feature_expert_can_run_shuffle_label_control(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    validation_output = tmp_path / "validation_scores"
    report_output = tmp_path / "report.json"
    features = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    _write_feature_dir(train_dir, split="train", features=features, labels=labels)
    _write_feature_dir(validation_dir, split="validation", features=features, labels=labels)

    status = fit_compressed_feature_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--output-validation-dir",
            str(validation_output),
            "--output-report",
            str(report_output),
            "--steps",
            "400",
            "--learning-rate",
            "0.1",
            "--l2",
            "0.0",
            "--shuffle-train-labels",
            "--shuffle-seed",
            "2",
        ]
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    metadata = json.loads((validation_output / "models.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_feature_expert_shuffle_train_labels_control"
    assert report["label_control"] == {"shuffle_train_labels": True, "shuffle_seed": 2}
    assert metadata["fit_train_labels_shuffled"] is True
    assert metadata["fit_label_shuffle_seed"] == 2
    assert report["validation_metrics"]["auc"] < 1.0


def test_summarize_compressed_feature_expert_reports_not_ensemble_gain(tmp_path):
    normal0 = _write_compressed_feature_report(
        tmp_path / "seed0_normal.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=1.0,
    )
    normal1 = _write_compressed_feature_report(
        tmp_path / "seed1_normal.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=1.0,
    )
    shuffle0 = _write_compressed_feature_report(
        tmp_path / "seed0_shuffle.json",
        decision="compressed_feature_expert_shuffle_train_labels_control",
        validation_auc=0.497,
    )
    shuffle1 = _write_compressed_feature_report(
        tmp_path / "seed1_shuffle.json",
        decision="compressed_feature_expert_shuffle_train_labels_control",
        validation_auc=0.525,
    )
    ensemble0 = _write_compressed_feature_ensemble(
        tmp_path / "seed0_ensemble.json",
        best_single_auc=1.0,
        best_ensemble_auc=1.0,
        delta=0.0,
    )
    ensemble1 = _write_compressed_feature_ensemble(
        tmp_path / "seed1_ensemble.json",
        best_single_auc=1.0,
        best_ensemble_auc=0.999995,
        delta=-0.000005,
    )
    output = tmp_path / "route_gate.json"

    status = summarize_compressed_feature_main(
        [
            "--normal-reports",
            str(normal0),
            str(normal1),
            "--shuffle-control-reports",
            str(shuffle0),
            str(shuffle1),
            "--ensemble-reports",
            str(ensemble0),
            str(ensemble1),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["decision"] == "compressed_feature_local_positive_controls_pass_not_ensemble_gain"
    assert summary["normal_passed_seed_count"] == 2
    assert summary["shuffle_control_passed_seed_count"] == 2
    assert summary["ensemble_gain_passed_seed_count"] == 0
    assert summary["all_normal_positive"] is True
    assert summary["all_shuffle_controls_random_like"] is True
    assert summary["all_ensemble_gains_positive"] is False
    assert summary["validation_auc"]["min"] == 1.0
    assert summary["shuffle_control_auc"]["max"] == 0.525
    assert summary["claim_scope"].startswith("route-level compressed SPN feature expert diagnostic")


def test_summarize_compressed_span_route_reports_compact_retention(tmp_path):
    flat0 = _write_compressed_feature_report(
        tmp_path / "seed0_flat.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=0.99997,
        feature_count=731,
    )
    flat1 = _write_compressed_feature_report(
        tmp_path / "seed1_flat.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=0.99995,
        feature_count=731,
    )
    summary0 = _write_compressed_feature_report(
        tmp_path / "seed0_summary.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=0.99991,
        feature_count=273,
    )
    summary1 = _write_compressed_feature_report(
        tmp_path / "seed1_summary.json",
        decision="compressed_feature_expert_local_screen_positive_needs_controls",
        validation_auc=0.99984,
        feature_count=273,
    )
    shuffle0 = _write_compressed_feature_report(
        tmp_path / "seed0_shuffle.json",
        decision="compressed_feature_expert_shuffle_train_labels_control",
        validation_auc=0.505,
        feature_count=273,
    )
    shuffle1 = _write_compressed_feature_report(
        tmp_path / "seed1_shuffle.json",
        decision="compressed_feature_expert_shuffle_train_labels_control",
        validation_auc=0.482,
        feature_count=273,
    )
    output = tmp_path / "span_route_summary.json"

    status = summarize_compressed_span_route_main(
        [
            "--flat-span-reports",
            str(flat0),
            str(flat1),
            "--summary-reports",
            str(summary0),
            str(summary1),
            "--summary-shuffle-control-reports",
            str(shuffle0),
            str(shuffle1),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "compressed_span_summary_retains_flat_signal_controls_pass"
    assert report["seed_count"] == 2
    assert report["feature_counts"] == {"flat_span": 731, "summary": 273}
    assert report["feature_reduction_ratio"] == 273 / 731
    assert report["summary_auc"]["min"] == 0.99984
    assert report["shuffle_control_auc"]["max"] == 0.505
    assert report["auc_drop_vs_flat"]["max"] < 0.001
    assert report["all_summary_positive"] is True
    assert report["all_shuffle_controls_random_like"] is True
    assert report["all_summary_retains_flat_signal"] is True
    assert report["claim_scope"].startswith("compressed span summary retention diagnostic")


def test_audit_compressed_feature_sparsity_selects_train_only_top_features(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    output = tmp_path / "sparse_audit.json"
    _write_feature_dir(
        train_dir,
        split="train",
        features=np.array(
            [
                [-3.0, 0.9, 0.0],
                [-2.0, 0.1, 0.0],
                [2.0, 0.8, 0.0],
                [3.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=np.array(
            [
                [-2.5, 0.2, 0.0],
                [-1.5, 0.8, 0.0],
                [1.5, 0.1, 0.0],
                [2.5, 0.9, 0.0],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
    )

    status = audit_compressed_feature_sparsity_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--top-k",
            "1",
            "2",
            "--steps",
            "400",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["decision"] == "sparse_compressed_feature_local_screen_positive"
    assert report["rows"][0]["top_k"] == 1
    assert report["rows"][0]["selected_feature_indices"] == [0]
    assert report["rows"][0]["validation_metrics"]["auc"] == 1.0
    assert report["best_row"]["top_k"] == 1
    assert report["claim_scope"].startswith("train-ranked compressed SPN feature sparsity diagnostic")


def test_audit_compressed_feature_sparsity_can_filter_feature_prefixes(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    output = tmp_path / "sparse_audit.json"
    feature_names = [
        "drop_global_noise",
        "keep_primary_weak",
        "keep_primary_strong",
        "drop_auxiliary_strong",
    ]
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    _write_feature_dir(
        train_dir,
        split="train",
        features=np.array(
            [
                [8.0, -0.5, -3.0, -4.0],
                [7.0, -0.2, -2.0, -3.5],
                [6.0, 0.3, 2.0, 3.5],
                [5.0, 0.5, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
        labels=labels,
        feature_view_metadata={"feature_names": feature_names},
    )
    _write_feature_dir(
        validation_dir,
        split="validation",
        features=np.array(
            [
                [4.0, -0.4, -2.5, -4.5],
                [3.0, -0.1, -1.5, -3.0],
                [2.0, 0.2, 1.5, 3.0],
                [1.0, 0.4, 2.5, 4.5],
            ],
            dtype=np.float32,
        ),
        labels=labels,
        feature_view_metadata={"feature_names": feature_names},
    )

    status = audit_compressed_feature_sparsity_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--top-k",
            "1",
            "--include-feature-prefix",
            "keep_primary_",
            "--steps",
            "400",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["rows"][0]
    assert status == 0
    assert report["feature_selection"]["mode"] == "prefix_filter"
    assert report["feature_selection"]["include_feature_prefixes"] == ["keep_primary_"]
    assert report["feature_selection"]["selected_feature_indices"] == [1, 2]
    assert row["selected_feature_indices"] == [2]
    assert row["selected_feature_names"] == ["keep_primary_strong"]
    assert row["validation_metrics"]["auc"] == 1.0


def test_decode_compressed_feature_sparsity_maps_indices_to_stat_families(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_view_metadata": {
                    "words_per_pair": 39,
                    "trail_depth": 4,
                    "trail_words_per_depth": 9,
                    "prefix_words": 3,
                    "output_feature_bits": 3708,
                }
            }
        ),
        encoding="utf-8",
    )
    sparse_report = tmp_path / "sparse.json"
    sparse_report.write_text(
        json.dumps(
            {
                "train_feature_dir": str(feature_dir),
                "feature_count": 3708,
                "rows": [
                    {
                        "top_k": 3,
                        "selected_feature_indices": [0, 624, 1248],
                        "validation_metrics": {"auc": 0.75},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decoded.json"

    status = decode_compressed_feature_sparsity_main(
        [
            "--sparse-report",
            str(sparse_report),
            "--output",
            str(output),
        ]
    )

    decoded = json.loads(output.read_text(encoding="utf-8"))
    row = decoded["rows"][0]
    assert status == 0
    assert decoded["status"] == "pass"
    assert row["top_k"] == 3
    assert row["decoded_features"][0]["name"] == "word_cell_mean_word0_cell0"
    assert row["decoded_features"][1]["name"] == "word_cell_std_word0_cell0"
    assert row["decoded_features"][2]["name"] == "word_mean_word0"
    assert row["family_counts"] == {"word_cell_mean": 1, "word_cell_std": 1, "word_mean": 1}
    assert row["validation_auc"] == 0.75


def test_audit_compressed_feature_families_reports_single_and_leave_one_out(tmp_path):
    train_dir = tmp_path / "train_features"
    validation_dir = tmp_path / "validation_features"
    output = tmp_path / "family_audit.json"
    feature_count = 3708
    depth_span_index = 2620
    word_span_index = 1365
    train_features = np.zeros((4, feature_count), dtype=np.float32)
    validation_features = np.zeros((4, feature_count), dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    train_features[:, depth_span_index] = np.array([-2.0, -1.5, 1.5, 2.0], dtype=np.float32)
    validation_features[:, depth_span_index] = np.array([-1.8, -1.2, 1.2, 1.8], dtype=np.float32)
    train_features[:, word_span_index] = np.array([-1.0, -0.8, 0.8, 1.0], dtype=np.float32)
    validation_features[:, word_span_index] = np.array([-0.9, -0.7, 0.7, 0.9], dtype=np.float32)
    for directory, split, features in (
        (train_dir, "train", train_features),
        (validation_dir, "validation", validation_features),
    ):
        _write_feature_dir(
            directory,
            split=split,
            features=features,
            labels=labels,
            feature_view_metadata={
                "words_per_pair": 39,
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "output_feature_bits": feature_count,
            },
        )

    status = audit_compressed_feature_families_main(
        [
            "--train-feature-dir",
            str(train_dir),
            "--validation-feature-dir",
            str(validation_dir),
            "--family",
            "depth_word_cell_span",
            "--family",
            "word_span",
            "--output",
            str(output),
            "--steps",
            "300",
            "--learning-rate",
            "0.2",
            "--l2",
            "0.0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    rows = {(row["mode"], row.get("family", row.get("left_out_family", ""))): row for row in report["rows"]}
    assert status == 0
    assert report["status"] == "pass"
    assert report["families"] == ["depth_word_cell_span", "word_span"]
    assert report["union_row"]["mode"] == "union"
    assert report["union_row"]["validation_metrics"]["auc"] == 1.0
    assert rows[("single_family", "depth_word_cell_span")]["feature_count"] == 576
    assert rows[("single_family", "word_span")]["feature_count"] == 39
    assert rows[("leave_one_out", "word_span")]["include_feature_families"] == ["depth_word_cell_span"]
    assert report["claim_scope"].startswith("compressed SPN feature-family attribution diagnostic")


def test_export_compressed_span_blocks_preserves_spn_coordinates(tmp_path):
    feature_dir = tmp_path / "features"
    output_dir = tmp_path / "span_blocks"
    feature_count = 3708
    features = np.zeros((2, feature_count), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.float32)
    features[:, 2620] = np.array([1.25, 2.25], dtype=np.float32)
    features[:, 2764] = np.array([3.25, 4.25], dtype=np.float32)
    features[:, 3388] = np.array([5.25, 6.25], dtype=np.float32)
    features[:, 1365] = np.array([7.25, 8.25], dtype=np.float32)
    features[:, 3560] = np.array([9.25, 10.25], dtype=np.float32)
    features[:, 1452] = np.array([11.25, 12.25], dtype=np.float32)
    _write_feature_dir(
        feature_dir,
        split="train",
        features=features,
        labels=labels,
        feature_view_metadata={
            "words_per_pair": 39,
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "output_feature_bits": feature_count,
        },
    )

    status = export_compressed_span_blocks_main(
        [
            "--feature-dir",
            str(feature_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    depth_word_cell = np.load(output_dir / "depth_word_cell_span.npy")
    depth_cell = np.load(output_dir / "depth_cell_span.npy")
    word = np.load(output_dir / "word_span.npy")
    depth_word = np.load(output_dir / "depth_word_span.npy")
    cell = np.load(output_dir / "cell_span.npy")
    assert status == 0
    assert manifest["status"] == "pass"
    assert manifest["source_feature_count"] == feature_count
    assert manifest["blocks"]["depth_word_cell_span"]["shape"] == [2, 4, 9, 16]
    assert manifest["blocks"]["depth_word_cell_span"]["role"] == "primary_backbone"
    assert manifest["blocks"]["depth_cell_span"]["role"] == "auxiliary_context"
    assert depth_word_cell.shape == (2, 4, 9, 16)
    assert depth_cell.shape == (2, 4, 16)
    assert word.shape == (2, 39)
    assert depth_word.shape == (2, 4, 9)
    assert cell.shape == (2, 16)
    assert depth_word_cell[0, 0, 0, 0] == np.float32(1.25)
    assert depth_word_cell[1, 1, 0, 0] == np.float32(4.25)
    assert depth_cell[0, 0, 0] == np.float32(5.25)
    assert word[1, 0] == np.float32(8.25)
    assert depth_word[0, 0, 0] == np.float32(9.25)
    assert cell[1, 0] == np.float32(12.25)
    assert np.array_equal(np.load(output_dir / "labels.npy"), labels)


def test_export_compressed_span_blocks_can_write_summary_feature_artifact(tmp_path):
    feature_dir = tmp_path / "features"
    output_dir = tmp_path / "span_blocks"
    summary_dir = tmp_path / "span_summary_features"
    feature_count = 3708
    features = np.zeros((2, feature_count), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.float32)
    features[:, 2620:3196] = 1.0
    features[:, 3388:3452] = 2.0
    features[:, 1365:1404] = 3.0
    features[:, 3560:3596] = 4.0
    features[:, 1452:1468] = 5.0
    _write_feature_dir(
        feature_dir,
        split="train",
        features=features,
        labels=labels,
        feature_view_metadata={
            "words_per_pair": 39,
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "output_feature_bits": feature_count,
        },
    )

    status = export_compressed_span_blocks_main(
        [
            "--feature-dir",
            str(feature_dir),
            "--output-dir",
            str(output_dir),
            "--output-summary-feature-dir",
            str(summary_dir),
        ]
    )

    summary = np.load(summary_dir / "features.npy")
    metadata = json.loads((summary_dir / "metadata.json").read_text(encoding="utf-8"))
    assert status == 0
    assert summary.shape == (2, 273)
    assert np.array_equal(np.load(summary_dir / "labels.npy"), labels)
    assert metadata["feature_view"] == "compressed_span_summary"
    assert metadata["feature_view_metadata"]["output_feature_bits"] == 273
    assert metadata["feature_view_metadata"]["source_kind"] == "compressed_spn_span_blocks"
    assert "primary_depth_mean_depth0" in metadata["feature_view_metadata"]["feature_names"]
    assert "aux_cell_global_max" in metadata["feature_view_metadata"]["feature_names"]


def _write_compressed_feature_report(
    path: Path,
    *,
    decision: str,
    validation_auc: float,
    feature_count: int = 3708,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "decision": decision,
                "feature_count": feature_count,
                "validation_rows": 2048,
                "validation_metrics": {
                    "auc": validation_auc,
                    "accuracy": 0.999 if validation_auc > 0.9 else 0.51,
                },
                "train_metrics": {
                    "auc": validation_auc,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_compressed_feature_ensemble(
    path: Path,
    *,
    best_single_auc: float,
    best_ensemble_auc: float,
    delta: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "best_single": {
                    "model_key": "compressed_feature_logistic_expert",
                    "metrics": {"auc": best_single_auc},
                },
                "best_ensemble": {
                    "mode": "logit_mean",
                    "metrics": {"auc": best_ensemble_auc},
                },
                "delta_best_ensemble_vs_single_auc": delta,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gate_bucket_residual_controls_keeps_bucket_route_when_controls_pass(tmp_path):
    candidate0 = _write_compressed_feature_report(tmp_path / "seed0_bucket.json", decision="keep", validation_auc=0.999936)
    candidate1 = _write_compressed_feature_report(tmp_path / "seed1_bucket.json", decision="keep", validation_auc=0.999925)
    nobucket0 = _write_compressed_feature_report(tmp_path / "seed0_nobucket.json", decision="keep", validation_auc=0.999907)
    nobucket1 = _write_compressed_feature_report(tmp_path / "seed1_nobucket.json", decision="keep", validation_auc=0.999885)
    shuffle0 = _write_compressed_feature_report(tmp_path / "seed0_shuffle_label.json", decision="hold", validation_auc=0.53722)
    shuffle1 = _write_compressed_feature_report(tmp_path / "seed1_shuffle_label.json", decision="hold", validation_auc=0.54351)
    trainshuffle0 = _write_compressed_feature_report(
        tmp_path / "seed0_trainbucket_shuffle.json", decision="hold", validation_auc=0.999860
    )
    trainshuffle1 = _write_compressed_feature_report(
        tmp_path / "seed1_trainbucket_shuffle.json", decision="hold", validation_auc=0.999746
    )
    valshuffle0 = _write_compressed_feature_report(
        tmp_path / "seed0_valbucket_shuffle.json", decision="hold", validation_auc=0.999318
    )
    valshuffle1 = _write_compressed_feature_report(
        tmp_path / "seed1_valbucket_shuffle.json", decision="hold", validation_auc=0.999369
    )
    two0 = _write_compressed_feature_ensemble(
        tmp_path / "seed0_two_score.json",
        best_single_auc=0.999925,
        best_ensemble_auc=0.999942,
        delta=0.000017,
    )
    two1 = _write_compressed_feature_ensemble(
        tmp_path / "seed1_two_score.json",
        best_single_auc=0.999910,
        best_ensemble_auc=0.999919,
        delta=0.000009,
    )
    three0 = _write_compressed_feature_ensemble(
        tmp_path / "seed0_three_score.json",
        best_single_auc=0.999936,
        best_ensemble_auc=0.999948,
        delta=0.000012,
    )
    three1 = _write_compressed_feature_ensemble(
        tmp_path / "seed1_three_score.json",
        best_single_auc=0.999925,
        best_ensemble_auc=0.999930,
        delta=0.000005,
    )
    trainshuffle_ensemble0 = _write_compressed_feature_ensemble(
        tmp_path / "seed0_trainshuffle_three_score.json",
        best_single_auc=0.999936,
        best_ensemble_auc=0.999922,
        delta=-0.000014,
    )
    trainshuffle_ensemble1 = _write_compressed_feature_ensemble(
        tmp_path / "seed1_trainshuffle_three_score.json",
        best_single_auc=0.999925,
        best_ensemble_auc=0.999875,
        delta=-0.000050,
    )
    valshuffle_ensemble0 = _write_compressed_feature_ensemble(
        tmp_path / "seed0_valshuffle_three_score.json",
        best_single_auc=0.999936,
        best_ensemble_auc=0.999866,
        delta=-0.000070,
    )
    valshuffle_ensemble1 = _write_compressed_feature_ensemble(
        tmp_path / "seed1_valshuffle_three_score.json",
        best_single_auc=0.999925,
        best_ensemble_auc=0.999849,
        delta=-0.000076,
    )
    output = tmp_path / "gate.json"

    status = gate_bucket_residual_controls_main(
        [
            "--candidate-report",
            str(candidate0),
            "--candidate-report",
            str(candidate1),
            "--two-score-ensemble",
            str(two0),
            "--two-score-ensemble",
            str(two1),
            "--three-score-ensemble",
            str(three0),
            "--three-score-ensemble",
            str(three1),
            "--shuffle-label-report",
            str(shuffle0),
            "--shuffle-label-report",
            str(shuffle1),
            "--train-bucket-shuffle-report",
            str(trainshuffle0),
            "--train-bucket-shuffle-report",
            str(trainshuffle1),
            "--train-bucket-shuffle-ensemble",
            str(trainshuffle_ensemble0),
            "--train-bucket-shuffle-ensemble",
            str(trainshuffle_ensemble1),
            "--validation-bucket-shuffle-report",
            str(valshuffle0),
            "--validation-bucket-shuffle-report",
            str(valshuffle1),
            "--validation-bucket-shuffle-ensemble",
            str(valshuffle_ensemble0),
            "--validation-bucket-shuffle-ensemble",
            str(valshuffle_ensemble1),
            "--no-bucket-report",
            str(nobucket0),
            "--no-bucket-report",
            str(nobucket1),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "bucket_conditioned_residual_controls_pass_local_diagnostic"
    assert report["action"] == "keep_as_262k_migration_candidate_wait_for_trail_position_artifacts"
    assert report["seed_count"] == 2
    assert report["min_bucket_vs_nobucket_auc_delta"] > 0.0
    assert report["min_three_vs_two_auc_delta"] > 0.0
    assert report["max_shuffle_label_validation_auc"] < 0.60
    assert report["max_trainbucket_shuffle_three_vs_two_delta"] <= 0.0
    assert report["max_valbucket_shuffle_three_vs_two_delta"] <= 0.0
    assert report["next_action"]["should_launch_remote"] is False
    assert "local 2048/class frozen-score control gate only" in report["claim_scope"]


def test_gate_bucket_residual_controls_holds_when_valbucket_shuffle_keeps_gain(tmp_path):
    candidate = _write_compressed_feature_report(tmp_path / "bucket.json", decision="keep", validation_auc=0.95)
    nobucket = _write_compressed_feature_report(tmp_path / "nobucket.json", decision="keep", validation_auc=0.94)
    shuffle = _write_compressed_feature_report(tmp_path / "shuffle_label.json", decision="hold", validation_auc=0.51)
    trainshuffle = _write_compressed_feature_report(tmp_path / "trainbucket_shuffle.json", decision="hold", validation_auc=0.93)
    valshuffle = _write_compressed_feature_report(tmp_path / "valbucket_shuffle.json", decision="hold", validation_auc=0.93)
    two = _write_compressed_feature_ensemble(
        tmp_path / "two_score.json",
        best_single_auc=0.94,
        best_ensemble_auc=0.945,
        delta=0.005,
    )
    three = _write_compressed_feature_ensemble(
        tmp_path / "three_score.json",
        best_single_auc=0.95,
        best_ensemble_auc=0.951,
        delta=0.001,
    )
    trainshuffle_ensemble = _write_compressed_feature_ensemble(
        tmp_path / "trainshuffle_three_score.json",
        best_single_auc=0.95,
        best_ensemble_auc=0.944,
        delta=-0.006,
    )
    valshuffle_ensemble = _write_compressed_feature_ensemble(
        tmp_path / "valshuffle_three_score.json",
        best_single_auc=0.95,
        best_ensemble_auc=0.946,
        delta=-0.004,
    )
    output = tmp_path / "gate.json"

    status = gate_bucket_residual_controls_main(
        [
            "--candidate-report",
            str(candidate),
            "--two-score-ensemble",
            str(two),
            "--three-score-ensemble",
            str(three),
            "--shuffle-label-report",
            str(shuffle),
            "--train-bucket-shuffle-report",
            str(trainshuffle),
            "--train-bucket-shuffle-ensemble",
            str(trainshuffle_ensemble),
            "--validation-bucket-shuffle-report",
            str(valshuffle),
            "--validation-bucket-shuffle-ensemble",
            str(valshuffle_ensemble),
            "--no-bucket-report",
            str(nobucket),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert report["status"] == "fail"
    assert report["decision"] == "hold_bucket_conditioned_residual_controls_failed"
    assert "seed0: validation_bucket_shuffle_three_score_not_below_two_score" in report["errors"]


def test_analyze_residual_bucket_axis_spectrum_ranks_grouped_residual_error_axes(tmp_path):
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    feature_dir = tmp_path / "features"
    feature_names = [
        "primary_depth_mean_depth0",
        "primary_depth_mean_depth1",
        "aux_word_mean_word0",
        "aux_word_mean_word1",
    ]
    _write_feature_dir(
        feature_dir,
        split="train",
        features=features,
        labels=labels,
        feature_view_metadata={"feature_names": feature_names},
    )
    left_probabilities = np.array([0.10, 0.40, 0.70, 0.90, 0.05, 0.95, 0.95, 0.05], dtype=np.float32)
    right_probabilities = np.array([0.12, 0.42, 0.72, 0.88, 0.15, 0.85, 0.65, 0.35], dtype=np.float32)
    left_scores = tmp_path / "left_scores"
    right_scores = tmp_path / "right_scores"
    _write_tiny_score_artifact(left_scores, labels, left_probabilities, sample_ids, model_key="trail")
    _write_tiny_score_artifact(right_scores, labels, right_probabilities, sample_ids, model_key="raw117")
    output = tmp_path / "spectrum.json"

    status = analyze_residual_bucket_axis_spectrum_main(
        [
            "--feature-dir",
            str(feature_dir),
            "--bucket-artifacts",
            str(left_scores),
            str(right_scores),
            "--bucket-count",
            "2",
            "--top-groups",
            "2",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    low_bucket = report["bucket_reports"][0]
    high_bucket = report["bucket_reports"][1]
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "residual_bucket_axis_spectrum_ready"
    assert report["row_count"] == 8
    assert report["bucket_feature"] == "logit_gap_abs"
    assert report["group_count"] == 2
    assert report["residual_error_rate_at_0_5"] == 0.5
    assert low_bucket["top_groups"][0]["group"] == "primary_depth_mean"
    assert low_bucket["top_groups"][0]["residual_error_auc"] == 1.0
    assert high_bucket["top_groups"][0]["group"] == "aux_word_mean"
    assert high_bucket["top_groups"][0]["residual_error_auc"] == 1.0
    assert "train-only or validation-only diagnostic" in report["claim_scope"]


def test_analyze_residual_bucket_axis_spectrum_supports_continuous_residual_loss(tmp_path):
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    features = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    feature_dir = tmp_path / "features"
    _write_feature_dir(
        feature_dir,
        split="validation",
        features=features,
        labels=labels,
        feature_view_metadata={
            "feature_names": [
                "uncertain_depth_mean_depth0",
                "uncertain_depth_mean_depth1",
                "confident_word_mean_word0",
                "confident_word_mean_word1",
            ]
        },
    )
    left_probabilities = np.array([0.44, 0.54, 0.40, 0.44, 0.01, 0.97, 0.01, 0.91], dtype=np.float32)
    right_probabilities = np.array([0.46, 0.56, 0.56, 0.60, 0.03, 0.99, 0.09, 0.99], dtype=np.float32)
    left_scores = tmp_path / "left_scores"
    right_scores = tmp_path / "right_scores"
    _write_tiny_score_artifact(left_scores, labels, left_probabilities, sample_ids, model_key="trail")
    _write_tiny_score_artifact(right_scores, labels, right_probabilities, sample_ids, model_key="raw117")
    output = tmp_path / "continuous_spectrum.json"

    status = analyze_residual_bucket_axis_spectrum_main(
        [
            "--feature-dir",
            str(feature_dir),
            "--bucket-artifacts",
            str(left_scores),
            str(right_scores),
            "--bucket-count",
            "2",
            "--top-groups",
            "2",
            "--target",
            "residual_loss",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["target"] == "residual_loss"
    assert report["residual_error_rate_at_0_5"] == 0.0
    assert report["target_mean"] > 0.0
    by_group = {group["group"]: group for group in report["global_top_groups"]}
    assert by_group["uncertain_depth_mean"]["target_auc"] == 1.0
    assert by_group["uncertain_depth_mean"]["target_score"] == 0.5


def _write_feature_dir(
    path: Path,
    *,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    negative_mode: str = "encrypted_random_plaintexts",
    feature_view_metadata: dict[str, int] | None = None,
) -> None:
    path.mkdir(parents=True)
    sample_ids = np.array([str(index) for index in range(len(labels))], dtype=str)
    np.save(path / "features.npy", features.astype(np.float32, copy=False))
    np.save(path / "labels.npy", labels.astype(np.float32, copy=False))
    np.save(path / "sample_ids.npy", sample_ids)
    metadata = {
        "status": "pass",
        "kind": "bit_sensitivity_feature_matrix",
        "split": split,
        "feature_view": "trail_position_stats",
        "cipher": "PRESENT-80",
        "rounds": 8,
        "samples_per_class": int(len(labels) // 2),
        "pairs_per_sample": 16,
        "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
        "negative_mode": negative_mode,
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x00000000000000000000",
        "validation_key": "0x11111111111111111111",
    }
    if feature_view_metadata is not None:
        metadata["feature_view_metadata"] = feature_view_metadata
    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _write_tiny_score_artifact(
    path: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_ids: np.ndarray,
    *,
    model_key: str,
) -> None:
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    write_score_artifact(
        path,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=probabilities,
            logits=logits.astype(np.float32),
            sample_ids=sample_ids,
            metadata={
                "model_key": model_key,
                "run_id": model_key,
                "expert_family": model_key,
                "candidate_status": "weak_positive",
                "cipher": "PRESENT-80",
                "rounds": 8,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
            },
        ),
    )


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


def test_plan_bucket_residual_262k_waits_for_trail_position_postprocess(tmp_path):
    postprocess = tmp_path / "postprocess.json"
    postprocess.write_text(
        json.dumps(
            {
                "status": "pending",
                "decision": "wait_for_trail_position_score_artifacts",
                "expected_matrix_rows": 2,
                "runs": [
                    {
                        "run_id": "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706",
                        "train_matrix": str(tmp_path / "run0" / "results" / "train_matrix.jsonl"),
                        "train_rows": 1,
                        "missing_score_files": ["score_artifacts/trail_position/models.json"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bucket_plan.json"

    status = plan_bucket_residual_262k_main(
        [
            "--postprocess-status",
            str(postprocess),
            "--artifact-root",
            str(tmp_path / "bucket262k"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_trail_position_262k_score_artifacts"
    assert report["should_run"] is False
    assert "score_artifacts/trail_position/models.json" in report["missing"]
    assert "not prove a breakthrough" in report["claim_scope"]


def test_plan_bucket_residual_262k_emits_same_protocol_v16_commands(tmp_path):
    run_root = tmp_path / "i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706"
    trail_scores = run_root / "score_artifacts" / "trail_position"
    trail_scores.mkdir(parents=True)
    (trail_scores / "models.json").write_text(
        json.dumps(
            {
                "checkpoint_path": (
                    "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
                    "i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706\\"
                    "checkpoints\\row0002_present_trail_position_stats_pairset_seed1.pt"
                ),
                "model_key": "present_trail_position_stats_pairset",
                "expert_family": "trail_position",
            }
        ),
        encoding="utf-8",
    )
    postprocess = tmp_path / "postprocess.json"
    postprocess.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "support_trail_position_score_residual_all_runs",
                "expected_score_rows": 262144,
                "runs": [
                    {
                        "status": "pass",
                        "run_id": run_root.name,
                        "run_root": str(run_root),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bucket_plan.json"

    status = plan_bucket_residual_262k_main(
        [
            "--postprocess-status",
            str(postprocess),
            "--artifact-root",
            str(tmp_path / "bucket262k"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    command_text = "\n".join(report["commands"])
    control_text = "\n".join(report["control_commands"])
    gate_command = report["gate_command"]
    seed = report["seeds"][0]
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "bucket_residual_262k_action_plan_ready"
    assert report["should_run"] is True
    assert report["expected_score_rows"] == 262144
    assert seed["seed"] == 1
    assert seed["eval_plan"].endswith("innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv")
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in seed["train_trail_position_checkpoint"]
    assert seed["remote_checkpoint_reference"] is True
    assert "remote Windows checkpoint path" in seed["warnings"][0]
    assert "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export-bit-sensitivity-features" in command_text
    assert "--feature-view trail_position_stats" in command_text
    assert "scripts/export-compressed-span-blocks" in command_text
    assert "scripts/export-checkpoint-scores" in command_text
    assert "--split train" in command_text
    assert "scripts/fit-compressed-feature-expert" in command_text
    assert "scripts/fit-bucket-conditioned-feature-expert" in command_text
    assert "--train-bucket-artifacts" in command_text
    assert "scripts/evaluate-neural-ensemble" in command_text
    assert "primary_depth_trailword_" in command_text
    assert "aux_depth_cell_" in command_text
    assert "aux_depth_word_" in command_text
    assert "aux_word_global_" in command_text
    assert "--shuffle-train-labels" in control_text
    assert "--shuffle-train-bucket-values" in control_text
    assert "--shuffle-validation-bucket-values" in control_text
    assert "bucket_raw117_logitgap_valbucket_shuffle" in control_text
    assert "trail_raw117_bucket_valshuffle_three_score_ensemble" in control_text
    assert "raw117_nobucket_l2_0p0003" in control_text
    assert report["gate_output"].endswith("bucket_residual_controls_gate.json")
    assert "UV_CACHE_DIR=/tmp/uv-cache uv run scripts/gate-bucket-residual-controls" in gate_command
    assert "--candidate-report" in gate_command
    assert "bucket_raw117_logitgap_report.json" in gate_command
    assert "--two-score-ensemble" in gate_command
    assert "trail_raw117_two_score_ensemble.json" in gate_command
    assert "--three-score-ensemble" in gate_command
    assert "trail_raw117_bucket_three_score_ensemble.json" in gate_command
    assert "--shuffle-label-report" in gate_command
    assert "bucket_raw117_logitgap_shuffle_labels_report.json" in gate_command
    assert "--train-bucket-shuffle-report" in gate_command
    assert "bucket_raw117_logitgap_trainbucket_shuffle_report.json" in gate_command
    assert "--train-bucket-shuffle-ensemble" in gate_command
    assert "trail_raw117_bucket_trainshuffle_three_score_ensemble.json" in gate_command
    assert "--validation-bucket-shuffle-report" in gate_command
    assert "bucket_raw117_logitgap_valbucket_shuffle_report.json" in gate_command
    assert "--validation-bucket-shuffle-ensemble" in gate_command
    assert "trail_raw117_bucket_valshuffle_three_score_ensemble.json" in gate_command
    assert "--no-bucket-report" in gate_command
    assert "raw117_nobucket_l2_0p0003_report.json" in gate_command
    assert "cmd.exe /k" not in command_text
    assert "SSH" not in command_text
    assert "not prove a breakthrough" in report["claim_scope"]


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


def test_select_bit_sensitivity_projection_can_select_grouped_axes(tmp_path):
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
            [0.0, 0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1, 0.0],
            [0.8, 0.9, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
            [0.9, 0.9, 0.1, 0.0],
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
            "--group-size",
            "2",
            "--top-groups",
            "1",
        ]
    )

    assert status == 0
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert mask["projection_unit"] == "contiguous_axis_group"
    assert mask["selected_axis_count"] == 2
    assert mask["selected_axes"] == [0, 1]
    assert mask["selected_groups"][0]["axes"] == [0, 1]
    assert mask["selected_groups"][0]["positive_mean"] > mask["selected_groups"][0]["negative_mean"]
    assert report["summary"]["selected_group_count"] == 1
    assert "group_selection_must_be_train_only" in report["guardrails"]


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
    assert report["metrics"]["auc"] == binary_auc(labels, probabilities)
    assert "not a trained neural model" in report["claim_scope"]


def test_apply_bit_sensitivity_projection_uses_grouped_axes(tmp_path):
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
            [0.0, 0.2, 0.9],
            [0.1, 0.1, 0.8],
            [0.8, 1.0, 0.1],
            [1.0, 0.9, 0.2],
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
                "projection_unit": "contiguous_axis_group",
                "selected_axes": [0, 1],
                "selected_groups": [
                    {
                        "group_id": 0,
                        "axes": [0, 1],
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
        ]
    )

    assert status == 0
    metadata_out = json.loads((output_dir / "models.json").read_text(encoding="utf-8"))
    probabilities = np.load(output_dir / "probabilities.npy")
    assert metadata_out["projection_unit"] == "contiguous_axis_group"
    assert metadata_out["projection_axis_count"] == 2
    assert metadata_out["projection_group_count"] == 1
    assert probabilities[2] > probabilities[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["group_count"] == 1
    assert report["metrics"]["auc"] == binary_auc(labels, probabilities)


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


def test_export_bit_sensitivity_features_can_write_trail_position_stats_view(tmp_path):
    plan = write_tiny_present_trail_position_plan(tmp_path / "present_plan.csv")
    output = tmp_path / "trail_stats_features"

    status = export_bit_sensitivity_features_main(
        [
            "--eval-plan",
            str(plan),
            "--eval-row-index",
            "0",
            "--split",
            "validation",
            "--samples-per-class",
            "2",
            "--feature-view",
            "trail_position_stats",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    features = np.load(output / "features.npy")
    labels = np.load(output / "labels.npy")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert features.shape[0] == labels.shape[0] == 4
    assert features.shape[1] == metadata["output_feature_bits"]
    assert metadata["feature_view"] == "trail_position_stats"
    assert metadata["feature_view_metadata"]["view"] == "trail_position_stats"
    assert metadata["feature_view_metadata"]["pair_bits"] == 2496
    assert metadata["feature_view_metadata"]["pairs_per_sample"] == 2
    assert metadata["feature_view_metadata"]["trail_depth"] == 4
    assert metadata["feature_view_metadata"]["trail_words_per_depth"] == 9
    assert metadata["output_feature_bits"] < metadata["input_bits"]


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
