from __future__ import annotations

from blockcipher_nd.cli.gate_runtime_spn_present_transfer import (
    render_present_transfer_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_present_transfer import (
    adjudicate_runtime_spn_present_transfer,
)


def _row(
    model: str,
    auc: float,
    *,
    seed: int = 0,
    samples_per_class: int = 2048,
) -> dict:
    return {
        "model": model,
        "cipher": "PRESENT-80",
        "rounds": 7,
        "seed": seed,
        "samples_per_class": samples_per_class,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": 0,
        "validation_key": 1,
        "input_bit_order": "project_msb_to_runtime_lsb",
        "parameter_count": 442466,
        "trainable_parameter_count": 442466,
        "metrics": {"auc": auc},
        "training": {
            "epochs": 5,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "train_rows": 2 * samples_per_class,
            "validation_rows": samples_per_class,
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
            "model_options": {
                "processor_steps": 2,
                "pair_embedding_dim": 128,
                "dropout": 0.0,
                "sbox_context_mode": "late_pair",
            },
        },
    }


def _rows(*, seed: int = 0) -> list[dict]:
    return [
        _row("present_runtime_e4_equivariant_true", 0.66, seed=seed),
        _row("present_runtime_e4_equivariant_corrupted", 0.57, seed=seed),
        _row("present_runtime_e4_equivariant_independent", 0.55, seed=seed),
    ]


def test_present_runtime_transfer_passes_signal_and_two_control_gates() -> None:
    gate = adjudicate_runtime_spn_present_transfer(
        run_id="unit", rows=_rows(), expected_seed=0
    )

    assert gate["status"] == "pass"
    assert (
        gate["decision"] == "innovation1_runtime_spn_present_transfer_seed0_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_present_runtime_transfer_seed1_plot_has_final_next_action(tmp_path) -> None:
    gate = adjudicate_runtime_spn_present_transfer(
        run_id="unit-seed1", rows=_rows(seed=1), expected_seed=1
    )
    output = tmp_path / "curves.svg"
    render_present_transfer_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "seed1" in svg
    assert "PRESENT 两颗 seed 均通过" in svg
    assert "下一步只做 seed1" not in svg


def test_present_runtime_formal_gate_scales_only_the_row_contract() -> None:
    rows = [
        _row(
            "present_runtime_e4_equivariant_true",
            0.66,
            samples_per_class=1_000_000,
        ),
        _row(
            "present_runtime_e4_equivariant_corrupted",
            0.60,
            samples_per_class=1_000_000,
        ),
        _row(
            "present_runtime_e4_equivariant_independent",
            0.52,
            samples_per_class=1_000_000,
        ),
    ]

    gate = adjudicate_runtime_spn_present_transfer(
        run_id="formal-unit",
        rows=rows,
        expected_seed=0,
        expected_samples_per_class=1_000_000,
        phase="rtg3b",
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_runtime_spn_present_formal_seed0_supported"
    assert gate["samples_per_class"] == 1_000_000
    assert gate["validation_samples_per_class"] == 500_000
    assert all(gate["protocol_checks"].values())


def test_present_runtime_formal_gate_rejects_local_scale_rows() -> None:
    gate = adjudicate_runtime_spn_present_transfer(
        run_id="formal-invalid",
        rows=_rows(),
        expected_seed=0,
        expected_samples_per_class=1_000_000,
        phase="rtg3b",
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["frozen_transfer_scale"] is False
