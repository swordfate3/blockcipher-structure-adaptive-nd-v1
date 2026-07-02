from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.planning.pairset_aggregation_postprocess import (
    postprocess_pairset_aggregation_control,
)


def test_pairset_aggregation_postprocess_writes_summary_and_updates_plan_doc(tmp_path) -> None:
    plan = write_pairset_plan(tmp_path / "plan.csv")
    learned = write_learned_results(tmp_path / "learned.jsonl")
    frozen = write_frozen_summary(tmp_path / "frozen.json")
    plan_doc = tmp_path / "plan.md"
    plan_doc.write_text("# Pairset Plan\n", encoding="utf-8")

    report = postprocess_pairset_aggregation_control(
        plan_path=plan,
        learned_results_path=learned,
        frozen_summary_path=frozen,
        output_dir=tmp_path / "out",
        run_id="pairset_postprocess_smoke",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["pairset_aggregation_status"] == "pass"
    assert report["decision"] == "support_learned_pairset_consistency"
    assert report["next_action"]["branch"] == "pairset_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["stage_a_plan_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_seed1.csv"
    )
    assert report["next_action"]["suggested_plan_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_r7_262k_seed1.csv"
    )
    assert report["next_action"]["stage_a_remote_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_seed1_gpu1_20260702.json"
    )
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702.json"
    )
    assert report["next_action"]["run_id"] == "i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    assert Path(report["next_action_readiness"]).exists()
    assert Path(report["pairset_aggregation_gate"]).exists()
    assert Path(report["curves"]).exists()
    assert Path(report["history_csv"]).exists()
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "pairset_seed1_confirmation"
    assert readiness["should_launch_remote"] is True
    assert readiness["requires_implementation"] is False
    assert readiness["readiness_pass"] is True
    assert [item["role"] for item in readiness["readiness_reports"]] == ["stage_a", "primary"]
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert readiness["readiness_reports"][1]["readiness"]["status"] == "pass"
    assert readiness["errors"] == []
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "Retrieved Pair-Set Aggregation Control Result" in plan_text
    assert "pairset_postprocess_smoke" in plan_text
    assert "support_learned_pairset_consistency" in plan_text
    assert "Next action readiness" in plan_text


def write_pairset_plan(path: Path) -> Path:
    rows = [
        "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,validation_key,key_rotation_interval,sample_structure,integral_active_nibble,difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature",
        'PRESENT-80,SPN,InvPOnly,present_nibble_invp_only_spn_only,test,0,1,7,0,8,16,ciphertext_pair_bits,encrypted_random_plaintexts,0x00000000000000000000,0x11111111111111111111,0,zhang_wang_case2_official_mcnd,0,present_zhang_wang2022_mcnd,0,mse,0.0001,adam,0.00001,none,,val_auc,true,0,0.0,"{}","SMOKE only","test"',
        'PRESENT-80,SPN,PairConsistency,present_nibble_invp_pair_consistency_spn_only,test,1,1,7,0,8,16,ciphertext_pair_bits,encrypted_random_plaintexts,0x00000000000000000000,0x11111111111111111111,0,zhang_wang_case2_official_mcnd,0,present_zhang_wang2022_mcnd,0,mse,0.0001,adam,0.00001,none,,val_auc,true,0,0.0,"{}","SMOKE only","test"',
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_learned_results(path: Path) -> Path:
    rows = [
        result_row("present_nibble_invp_only_spn_only", auc=0.800, calibrated_accuracy=0.721),
        result_row("present_nibble_invp_pair_consistency_spn_only", auc=0.805, calibrated_accuracy=0.724),
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def result_row(model: str, *, auc: float, calibrated_accuracy: float) -> dict:
    return {
        "cipher": "PRESENT-80",
        "cipher_key": "present80",
        "structure": "SPN",
        "model": model,
        "selected_model": model,
        "architecture": model,
        "architecture_rank": 0,
        "matching_score": 1,
        "rounds": 7,
        "seed": 0,
        "train_key": 0,
        "validation_key": int("11111111111111111111", 16),
        "input_difference": 0x9,
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "samples_per_class": 8,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "integral_active_nibble": 0,
        "metrics": {
            "accuracy": 0.72,
            "auc": auc,
            "calibrated_accuracy": calibrated_accuracy,
            "loss": 0.5,
        },
        "history": [
            {
                "epoch": 1.0,
                "train_loss": 0.6,
                "train_eval_loss": 0.6,
                "train_accuracy": 0.7,
                "train_auc": 0.75,
                "val_loss": 0.5,
                "val_accuracy": 0.72,
                "val_auc": auc,
                "learning_rate": 0.0001,
            }
        ],
        "training": {
            "loss": "mse",
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
            "pretraining": {"epochs_ran": 0},
        },
        "validation": {
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "pairs_per_sample": 16,
            "samples_per_class": 8,
            "key_rotation_interval": 0,
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "integral_active_nibble": 0,
        },
    }


def write_frozen_summary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "claim_scope": "frozen single-pair score aggregation control; not a learned pair-set model",
                "metrics": {
                    "auc": 0.801,
                    "accuracy": 0.721,
                    "calibrated_accuracy": 0.722,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
