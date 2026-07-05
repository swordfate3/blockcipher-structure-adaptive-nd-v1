from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.evaluate_neural_ensemble import main as evaluate_ensemble_main
from blockcipher_nd.cli.export_checkpoint_scores import main as export_scores_main
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
