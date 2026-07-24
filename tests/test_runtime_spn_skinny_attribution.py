from __future__ import annotations

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import (
    MODELS,
    adjudicate_runtime_spn_skinny_attribution,
)


def _rows(seed: int, aucs: dict[str, float]) -> list[dict[str, object]]:
    common_training = {
        "epochs": 5,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "train_rows": 4096,
        "validation_rows": 2048,
        "model_options": {
            "processor_steps": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
            "sbox_context_mode": "late_pair",
        },
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
    }
    return [
        {
            "cipher": "SKINNY-64/64",
            "model": model,
            "rounds": 7,
            "seed": seed,
            "samples_per_class": 2048,
            "dataset_label_mode": "balanced_per_class",
            "pairs_per_sample": 4,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "difference_profile": "skinny64_gohr2022_single_key",
            "difference_member": 0,
            "input_difference": 0x2000,
            "train_key": 0,
            "validation_key": 0x1111111111111111,
            "parameter_count": 442466,
            "trainable_parameter_count": 442466,
            "input_bit_order": "project_msb_to_runtime_lsb",
            "metrics": {"auc": aucs[role]},
            "training": common_training.copy(),
        }
        for role, model in MODELS.items()
    ]


def test_skinny_general_gf2_attribution_passes_both_control_margins() -> None:
    gate = adjudicate_runtime_spn_skinny_attribution(
        run_id="seed0",
        rows=_rows(0, {"true": 0.61, "corrupted": 0.57, "independent": 0.56}),
        expected_seed=0,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_runtime_spn_skinny_attribution_seed0_supported"
    assert gate["margins"]["true_minus_corrupted"] > 0.005
    assert gate["margins"]["true_minus_independent"] > 0.005


def test_skinny_general_gf2_attribution_holds_when_control_wins() -> None:
    gate = adjudicate_runtime_spn_skinny_attribution(
        run_id="held",
        rows=_rows(1, {"true": 0.58, "corrupted": 0.59, "independent": 0.54}),
        expected_seed=1,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_runtime_spn_skinny_attribution_not_supported"


def test_skinny_general_gf2_attribution_protocol_mismatch_fails_closed() -> None:
    rows = _rows(0, {"true": 0.61, "corrupted": 0.57, "independent": 0.56})
    rows[1]["negative_mode"] = "random_ciphertext"

    gate = adjudicate_runtime_spn_skinny_attribution(
        run_id="invalid",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_runtime_spn_skinny_attribution_protocol_invalid"
