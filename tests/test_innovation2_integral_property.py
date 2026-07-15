from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.run_innovation2_integral_property import main
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    INPUT_BITS,
    IntegralExperimentConfig,
    IntegralStructure,
    adjudicate,
    build_integral_split,
    integral_mask_parity,
    summarize_splits,
)


def test_integral_mask_parity_matches_manual_masked_output_xor() -> None:
    structure = IntegralStructure(
        structure_id="manual",
        active_nibble=13,
        output_nibble=8,
        output_mask=0b1111,
        fixed_plaintext=0x120456789ABCDEF0,
    )
    cipher = Present80(rounds=5, key=0x00010203040506070809)

    manual = 0
    for value in range(16):
        ciphertext = cipher.encrypt(structure.plaintext(value))
        nibble = (ciphertext >> (4 * structure.output_nibble)) & 0xF
        manual ^= (nibble & structure.output_mask).bit_count() & 1

    assert integral_mask_parity(cipher, structure) == manual == 1
    features = structure.feature_vector()
    assert features.shape == (INPUT_BITS,)
    assert int(features.sum()) >= 3


def test_integral_splits_repeat_structure_over_keys_and_remain_disjoint() -> None:
    train = build_integral_split(
        name="train",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=101,
        key_seed=201,
    )
    validation = build_integral_split(
        name="validation",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=301,
        key_seed=401,
    )
    test = build_integral_split(
        name="test",
        rounds=5,
        structure_count=8,
        key_count=4,
        structure_seed=501,
        key_seed=601,
    )
    summary = summarize_splits(
        {"train": train, "validation": validation, "test": test}
    )

    assert train.dataset.features.shape == (32, INPUT_BITS)
    assert train.dataset.labels.shape == (32,)
    np.testing.assert_array_equal(
        train.dataset.features[0],
        train.dataset.features[3],
    )
    assert summary["status"] == "pass"
    assert summary["structure_splits_disjoint"] is True
    assert summary["key_splits_disjoint"] is True


def test_diagnostic_gate_requires_candidate_to_beat_linear_and_shuffle() -> None:
    config = IntegralExperimentConfig(
        run_id="gate",
        train_structures=1,
        validation_structures=1,
        test_structures=1,
        train_keys=1,
        validation_keys=1,
        test_keys=1,
        epochs=1,
    )
    rows = [
        {
            "role": "anchor",
            "test_auc": 0.58,
            "test_structure_rate_mae": 0.24,
        },
        {
            "role": "candidate",
            "test_auc": 0.63,
            "test_structure_rate_mae": 0.20,
        },
        {
            "role": "control",
            "test_auc": 0.50,
            "test_structure_rate_mae": 0.26,
        },
    ]
    dataset_summary = {
        "structure_splits_disjoint": True,
        "key_splits_disjoint": True,
        "splits": {
            name: {"q0_rows": 1, "q1_rows": 1}
            for name in ("train", "validation", "test")
        },
    }

    gate = adjudicate(config, rows, dataset_summary)

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_integral_property_advance_multiseed"


def test_cli_writes_complete_smoke_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    status = main(
        [
            "--run-id",
            "i2_test_smoke",
            "--output-root",
            str(output),
            "--train-structures",
            "32",
            "--validation-structures",
            "16",
            "--test-structures",
            "16",
            "--train-keys",
            "4",
            "--validation-keys",
            "8",
            "--test-keys",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--hidden-bits",
            "16",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--gate-mode",
            "smoke",
        ]
    )

    assert status == 0
    assert (output / "results.jsonl").is_file()
    assert (output / "progress.jsonl").is_file()
    assert (output / "dataset_summary.json").is_file()
    assert (output / "structure_rates.csv").is_file()
    assert (output / "gate.json").is_file()
    assert (output / "curves.svg").is_file()
    assert (output / "history.csv").is_file()
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert [row["role"] for row in rows] == ["anchor", "candidate", "control"]
    assert gate["status"] == "pass"
    assert gate["readiness_checks"]["all_splits_have_both_labels"] is True
    assert "训练 32 个结构" in svg
    assert "每个积分集合 16 个明文" in svg
    assert "同输入线性基线" in svg
    assert "训练标签打乱 MLP 控制" in svg
    assert "训练 0/类" not in svg
