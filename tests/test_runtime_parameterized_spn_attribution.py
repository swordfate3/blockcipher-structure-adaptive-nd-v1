from __future__ import annotations

from copy import deepcopy

from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_attribution import (
    adjudicate_runtime_spn_r1,
    adjudicate_runtime_spn_r1a_cell_token,
)


def _row(model: str, auc: float) -> dict:
    return {
        "model": model,
        "cipher": "GIFT-64",
        "rounds": 6,
        "seed": 0,
        "samples_per_class": 8192,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "difference_profile": "gift64_shen2024_spn_screen",
        "difference_member": 0,
        "train_key": 0,
        "validation_key": 1,
        "parameter_count": 100,
        "metrics": {"auc": auc},
        "training": {
            "epochs": 10,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "train_rows": 16384,
            "validation_rows": 8192,
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
        },
    }


def _inputs() -> tuple[list[dict], list[dict], dict]:
    runtime = [
        _row("gift64_runtime_spn_true", 0.56),
        _row("gift64_runtime_spn_corrupted", 0.54),
        _row("gift64_runtime_spn_independent", 0.53),
    ]
    anchor = [_row("gift_cross_spn_typed_cell_true", 0.55)]
    r0 = {
        "status": "pass",
        "decision": "innovation1_runtime_spn_r0_readiness_passed",
    }
    return runtime, anchor, r0


def test_runtime_spn_r1_gate_passes_only_attributed_same_protocol_result() -> None:
    runtime, anchor, r0 = _inputs()
    gate = adjudicate_runtime_spn_r1(
        run_id="pass",
        cipher="GIFT-64",
        runtime_rows=runtime,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_runtime_spn_r1_gate_holds_random_or_misordered_candidate() -> None:
    runtime, anchor, r0 = _inputs()
    runtime[0]["metrics"]["auc"] = 0.506
    runtime[1]["metrics"]["auc"] = 0.508
    runtime[2]["metrics"]["auc"] = 0.507
    gate = adjudicate_runtime_spn_r1(
        run_id="hold",
        cipher="GIFT-64",
        runtime_rows=runtime,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "hold"
    assert not any(gate["research_checks"].values())
    assert "cell-token" in gate["next_action"]


def test_runtime_spn_r1_gate_fails_protocol_mismatch() -> None:
    runtime, anchor, r0 = _inputs()
    broken = deepcopy(runtime)
    broken[1]["negative_mode"] = "random_ciphertext"
    gate = adjudicate_runtime_spn_r1(
        run_id="invalid",
        cipher="GIFT-64",
        runtime_rows=broken,
        anchor_rows=anchor,
        r0_gate=r0,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_data_protocol_as_anchor"] is False
    assert gate["protocol_checks"]["encrypted_random_plaintext_negatives"] is False


def test_cell_token_calibration_gate_rejects_random_misordered_candidate() -> None:
    current = _row("gift64_runtime_spn_true", 0.5024)
    cell_true = _row("gift64_runtime_cell_token_true", 0.5019)
    cell_corrupted = _row("gift64_runtime_cell_token_corrupted", 0.5095)
    for row in (current, cell_true, cell_corrupted):
        row["samples_per_class"] = 2048
        row["training"]["epochs"] = 5
        row["training"]["train_rows"] = 4096
        row["training"]["validation_rows"] = 2048
    cell_true["parameter_count"] = 200
    cell_corrupted["parameter_count"] = 200

    gate = adjudicate_runtime_spn_r1a_cell_token(
        run_id="r1a",
        rows=[current, cell_true, cell_corrupted],
        r1_gate={
            "status": "hold",
            "decision": "innovation1_runtime_spn_r1_seed0_not_supported",
        },
    )

    assert gate["status"] == "hold"
    assert all(gate["protocol_checks"].values())
    assert not any(gate["research_checks"].values())
    assert "position-identifiability" in gate["next_action"]
