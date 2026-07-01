from __future__ import annotations

import json

from blockcipher_training_accelerator.quality_gate import compare_result_files


def test_compare_result_files_passes_for_small_metric_drift(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    row = {
        "cipher_key": "speck32",
        "model": "mlp",
        "rounds": 1,
        "seed": 0,
        "samples_per_class": 8,
        "pairs_per_sample": 1,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "validation_key": None,
        "metrics": {"auc": 0.7000, "calibrated_accuracy": 0.6500, "accuracy": 0.6250},
    }
    baseline.write_text(json.dumps(row) + "\n", encoding="utf-8")
    drifted = {**row, "metrics": {"auc": 0.6995, "calibrated_accuracy": 0.6495, "accuracy": 0.6250}}
    candidate.write_text(json.dumps(drifted) + "\n", encoding="utf-8")

    report = compare_result_files(
        baseline,
        candidate,
        max_auc_drop=0.002,
        max_calibrated_accuracy_drop=0.002,
    )

    assert report.status == "passed"
    assert report.rows_compared == 1
    assert report.failures == []


def test_compare_result_files_fails_on_protocol_mismatch(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    row = {
        "cipher_key": "speck32",
        "model": "mlp",
        "rounds": 1,
        "seed": 0,
        "samples_per_class": 8,
        "pairs_per_sample": 1,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "validation_key": None,
        "metrics": {"auc": 0.7000, "calibrated_accuracy": 0.6500, "accuracy": 0.6250},
    }
    baseline.write_text(json.dumps(row) + "\n", encoding="utf-8")
    mismatched = {**row, "negative_mode": "random_ciphertext"}
    candidate.write_text(json.dumps(mismatched) + "\n", encoding="utf-8")

    report = compare_result_files(baseline, candidate)

    assert report.status == "failed"
    assert "row 0 identity mismatch" in report.failures[0]
