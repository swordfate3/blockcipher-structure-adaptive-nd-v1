from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.evaluate_neural_ensemble import main as evaluate_ensemble_main
from blockcipher_nd.cli.export_checkpoint_scores import main as export_scores_main
from blockcipher_nd.cli.postprocess_neural_ensemble import main as postprocess_ensemble_main
from blockcipher_nd.cli.neural_ensemble_status import main as neural_ensemble_status_main
from blockcipher_nd.cli.train import main as train_main
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
