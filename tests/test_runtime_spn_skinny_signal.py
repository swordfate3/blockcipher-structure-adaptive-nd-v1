from __future__ import annotations

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_signal import (
    adjudicate_runtime_spn_skinny_signal,
)


def _row(rounds: int, auc: float, seed: int = 0) -> dict[str, object]:
    training = {
        "epochs": 5,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "train_rows": 1024,
        "validation_rows": 512,
        "model_options": {
            "processor_steps": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
            "sbox_context_mode": "late_pair",
        },
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
    }
    return {
        "cipher": "SKINNY-64/64",
        "model": "skinny64_runtime_e4_equivariant_true",
        "rounds": rounds,
        "seed": seed,
        "samples_per_class": 512,
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
        "metrics": {"auc": auc},
        "training": training,
    }


def test_skinny_signal_screen_selects_deepest_round_above_floor() -> None:
    gate = adjudicate_runtime_spn_skinny_signal(
        run_id="screen",
        rows=[_row(6, 1.0), _row(7, 0.566), _row(8, 0.513)],
        expected_rounds=(6, 7, 8),
        expected_seed=0,
        phase="screen",
    )

    assert gate["status"] == "pass"
    assert gate["selected_round"] == 7
    assert gate["decision"] == "innovation1_runtime_spn_skinny_signal_anchor_selected"


def test_skinny_signal_confirmation_opens_attribution_only_after_fresh_seed() -> None:
    gate = adjudicate_runtime_spn_skinny_signal(
        run_id="confirmation",
        rows=[_row(7, 0.571, seed=1)],
        expected_rounds=(7,),
        expected_seed=1,
        phase="confirmation",
    )

    assert gate["status"] == "pass"
    assert gate["selected_round"] == 7
    assert gate["decision"] == "innovation1_runtime_spn_skinny_signal_anchor_confirmed"


def test_skinny_signal_protocol_mismatch_fails_closed() -> None:
    row = _row(7, 0.7)
    row["negative_mode"] = "random_ciphertext"

    gate = adjudicate_runtime_spn_skinny_signal(
        run_id="invalid",
        rows=[row],
        expected_rounds=(7,),
        expected_seed=0,
        phase="confirmation",
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_runtime_spn_skinny_signal_protocol_invalid"
