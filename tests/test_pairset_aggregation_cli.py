from __future__ import annotations

import json
from pathlib import Path

import pytest

from blockcipher_nd.cli.evaluate_pairset_aggregation import main as evaluate_main
from blockcipher_nd.cli.train import main as train_main


def test_evaluate_pairset_aggregation_cli_loads_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "single_pair.pt"
    train_output = tmp_path / "single_pair_results.jsonl"
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
    plan = write_tiny_speck_pairset_plan(tmp_path / "eval_plan.csv")
    summary_path = tmp_path / "aggregation_summary.json"

    status = evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--eval-plan",
            str(plan),
            "--samples-per-class",
            "8",
            "--pairs-per-sample",
            "2",
            "--scorer-model-key",
            "mlp",
            "--scorer-hidden-bits",
            "8",
            "--aggregation-mode",
            "sum_logodds",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["rows"] == 16
    assert summary["pair_bits"] == 64
    assert summary["pairs_per_sample"] == 2
    assert summary["scorer_pairs_per_sample"] == 1
    assert summary["aggregation"]["mode"] == "sum_logodds"
    assert set(summary["metrics"]) >= {"accuracy", "auc", "calibrated_accuracy"}
    assert summary["checkpoint_metadata"]["checkpoint_output"] == str(checkpoint)


def test_evaluate_pairset_aggregation_rejects_non_single_pair_scorer(tmp_path) -> None:
    checkpoint = tmp_path / "unused.pt"
    checkpoint.write_bytes(b"not loaded")
    plan = write_tiny_speck_pairset_plan(tmp_path / "eval_plan.csv")

    with pytest.raises(ValueError, match="single-pair scorer"):
        evaluate_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--eval-plan",
                str(plan),
                "--scorer-model-key",
                "mlp",
                "--scorer-hidden-bits",
                "8",
                "--scorer-pairs-per-sample",
                "2",
                "--output",
                str(tmp_path / "summary.json"),
            ]
        )


def write_tiny_speck_pairset_plan(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,validation_key,key_rotation_interval,sample_structure,integral_active_nibble,difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature",
                'SPECK32/64,ARX,Tiny-Speck-MLP,mlp,tiny,0,1,1,0,8,2,ciphertext_pair_bits,encrypted_random_plaintexts,0x1918111009080100,0x1918111009080101,0,independent_pairs,0,,,bce,0.001,adam,0,none,,val_auc,true,0,0.0,"{}","SMOKE only for pairset aggregation CLI","test"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
